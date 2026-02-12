#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import seed_all

from regmixer.aliases import ExperimentConfig
from regmixer.model.transformer import TransformerConfigBuilder
from regmixer.utils import mk_source_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check gradient flow for one regmixer variant.")
    parser.add_argument("--config", required=True, help="Path to regmixer YAML config.")
    parser.add_argument("--mix-file", required=True, help="Path to generated mixes JSON.")
    parser.add_argument("--mix-index", type=int, default=0, help="Index of mix to run.")
    parser.add_argument("--run-name", required=True, help="Run name.")
    parser.add_argument("--group-id", default="local-grad-check", help="Group id.")
    parser.add_argument("--beaker-user", default="local", help="User tag for output paths.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=1,
        help="Global batch size in sequences (before sequence length multiplication).",
    )
    parser.add_argument("--steps", type=int, default=2, help="How many train steps to run.")
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional path to write gradient-flow summary JSON.",
    )
    return parser.parse_args()


def ensure_single_process_dist_env() -> None:
    defaults = {
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "NUM_NODES": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29551",
        "TORCH_DISTRIBUTED_DEFAULT_PORT": "29552",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("WANDB_MODE", "disabled")


def pick_probe_parameter(model: torch.nn.Module, optim: torch.optim.Optimizer):
    name_by_id = {id(param): name for name, param in model.named_parameters()}

    for group_idx, group in enumerate(optim.param_groups):
        for param_idx, param in enumerate(group["params"]):
            if not param.requires_grad:
                continue
            if param.numel() <= 1_000_000:
                name = name_by_id.get(id(param), f"<optim_param_{group_idx}_{param_idx}>")
                return name, param

    for group_idx, group in enumerate(optim.param_groups):
        for param_idx, param in enumerate(group["params"]):
            if not param.requires_grad:
                continue
            name = name_by_id.get(id(param), f"<optim_param_{group_idx}_{param_idx}>")
            return name, param

    for name, param in model.named_parameters():
        if param.requires_grad:
            return name, param
    raise RuntimeError("No trainable parameter found for gradient-flow probing.")


def metric_to_float(metric):
    if metric is None:
        return None
    if isinstance(metric, torch.Tensor):
        return float(metric.detach().cpu().item())
    return float(metric)


def get_optimizer_step(optim: torch.optim.Optimizer) -> int:
    max_step = 0
    for state in optim.state.values():
        step = state.get("step")
        if step is None:
            continue
        if isinstance(step, torch.Tensor):
            step_val = int(step.detach().cpu().item())
        else:
            step_val = int(step)
        max_step = max(max_step, step_val)
    return max_step


def main() -> None:
    args = parse_args()
    ensure_single_process_dist_env()

    with Path(args.config).open("r") as f:
        config = ExperimentConfig(**yaml.safe_load(f))
    with Path(args.mix_file).open("r") as f:
        mixes = json.load(f)["mixes"]

    if args.mix_index < 0 or args.mix_index >= len(mixes):
        raise IndexError(f"mix-index {args.mix_index} out of range; available={len(mixes)}")

    mix = mixes[args.mix_index]
    mix_map = {k: (float(v[0]), float(v[1])) for k, v in mix.items()}
    sources = mk_source_instances(config.sources, mix_map)

    max_sequences_per_step = max(1, config.max_tokens // config.sequence_length)
    global_batch_size = max(1, args.global_batch_size)
    if global_batch_size > max_sequences_per_step:
        global_batch_size = max_sequences_per_step

    prepare_training_environment()
    try:
        built = TransformerConfigBuilder(
            beaker_user=args.beaker_user,
            cluster=config.cluster,
            group_id=args.group_id,
            run_name=args.run_name,
            max_tokens=config.max_tokens,
            sources=sources,
            sequence_length=config.sequence_length,
            seed=config.seed,
            dtype=str(config.dtype.value),
            tokenizer=config.tokenizer,
            model_identifier=config.proxy_model_id,
            weka=config.weka,
            device_batch_size=config.device_batch_size,
            global_batch_size=global_batch_size,
        ).build()

        # For gradient diagnostics we prefer faster startup over maximum throughput.
        built.train_module.compile_model = False

        dataset = built.dataset.build()
        seed_all(built.init_seed)
        model = built.model.build(init_device="meta")
        train_module = built.train_module.build(model)
        data_loader = built.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = built.trainer.build(train_module, data_loader)

        if "wandb" in trainer.callbacks:
            trainer.callbacks.pop("wandb")

        probe_name, probe_param = pick_probe_parameter(train_module.model, train_module.optim)
        current_epoch = 1
        data_loader.reshuffle(epoch=current_epoch)
        loader_iter = iter(data_loader)
        steps = max(1, args.steps)
        step_summaries = []

        print(f"effective_global_batch_size={global_batch_size}")
        print(
            f"tokenizer_v={built.dataset.tokenizer.vocab_size} "
            f"eos={built.dataset.tokenizer.eos_token_id} "
            f"bos={built.dataset.tokenizer.bos_token_id} "
            f"pad={built.dataset.tokenizer.pad_token_id}"
        )
        print(f"probe_parameter={probe_name} numel={probe_param.numel()}")

        for step_idx in range(steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                current_epoch += 1
                data_loader.reshuffle(epoch=current_epoch)
                loader_iter = iter(data_loader)
                batch = next(loader_iter)

            input_ids = batch["input_ids"]
            vocab_size = int(built.dataset.tokenizer.vocab_size)
            input_min = int(input_ids.min().item())
            input_max = int(input_ids.max().item())
            bad_token_mask = (input_ids < 0) | (input_ids >= vocab_size)
            bad_token_count = int(bad_token_mask.sum().item())
            print(
                f"batch_{step_idx}_input_min={input_min} "
                f"input_max={input_max} vocab={vocab_size} bad_token_count={bad_token_count}"
            )
            if bad_token_count > 0:
                bad_vals = input_ids[bad_token_mask]
                print(
                    f"batch_{step_idx}_bad_token_min={int(bad_vals.min().item())} "
                    f"bad_token_max={int(bad_vals.max().item())}"
                )
                raise ValueError(f"Found {bad_token_count} out-of-vocab token ids in batch {step_idx}")

            probe_before = probe_param.detach().float().clone()
            train_module.zero_grads()
            train_module.train_batch(batch)

            grad_param_count = 0
            grad_total_sq = 0.0
            grad_max_abs = 0.0
            for param in train_module.model.parameters():
                grad = param.grad
                if grad is None:
                    continue
                grad_param_count += 1
                grad_float = grad.detach().float()
                grad_total_sq += float(torch.sum(grad_float * grad_float).item())
                grad_max_abs = max(grad_max_abs, float(torch.max(torch.abs(grad_float)).item()))

            total_grad_norm = grad_total_sq ** 0.5
            ce_loss = metric_to_float(trainer.get_metric("CE loss", namespace="train"))
            z_loss = metric_to_float(trainer.get_metric("Z loss", namespace="train"))
            optim_step_before = get_optimizer_step(train_module.optim)

            train_module.optim_step()
            optim_step_after = get_optimizer_step(train_module.optim)
            optim_step_skipped = bool(getattr(train_module.optim, "step_skipped", False))
            probe_after = probe_param.detach().float()
            update_abs_mean = float(torch.mean(torch.abs(probe_after - probe_before)).item())
            update_abs_max = float(torch.max(torch.abs(probe_after - probe_before)).item())
            train_module.zero_grads()

            step_summary = {
                "step": step_idx,
                "ce_loss": ce_loss,
                "z_loss": z_loss,
                "grad_param_count": grad_param_count,
                "total_grad_norm": total_grad_norm,
                "grad_max_abs": grad_max_abs,
                "probe_update_abs_mean": update_abs_mean,
                "probe_update_abs_max": update_abs_max,
                "batch_input_min": input_min,
                "batch_input_max": input_max,
                "batch_bad_token_count": bad_token_count,
                "optim_step_before": optim_step_before,
                "optim_step_after": optim_step_after,
                "optim_step_skipped": optim_step_skipped,
            }
            step_summaries.append(step_summary)
            print(json.dumps(step_summary, ensure_ascii=True))

        summary = {
            "run_name": args.run_name,
            "group_id": args.group_id,
            "effective_global_batch_size": global_batch_size,
            "tokenizer_vocab_size": built.dataset.tokenizer.vocab_size,
            "tokenizer_eos": built.dataset.tokenizer.eos_token_id,
            "tokenizer_bos": built.dataset.tokenizer.bos_token_id,
            "tokenizer_pad": built.dataset.tokenizer.pad_token_id,
            "probe_parameter": probe_name,
            "steps": step_summaries,
        }
        print(json.dumps(summary, ensure_ascii=True))

        if args.summary_out:
            out_path = Path(args.summary_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
