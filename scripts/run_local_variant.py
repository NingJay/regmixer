#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import yaml
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import seed_all

from regmixer.aliases import ExperimentConfig
from regmixer.model.transformer import TransformerConfigBuilder
from regmixer.utils import mk_source_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one regmixer variant locally.")
    parser.add_argument("--config", required=True, help="Path to regmixer YAML config.")
    parser.add_argument("--mix-file", required=True, help="Path to generated mixes JSON.")
    parser.add_argument("--mix-index", type=int, default=0, help="Index of mix to run.")
    parser.add_argument("--run-name", required=True, help="Run name for training.")
    parser.add_argument("--group-id", default="local-group", help="Group id for wandb grouping.")
    parser.add_argument("--beaker-user", default="local", help="User tag for output paths.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=1,
        help="Global batch size in sequences (before multiplying sequence length).",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional path to write a JSON summary for this local run.",
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
        "MASTER_PORT": "29541",
        "TORCH_DISTRIBUTED_DEFAULT_PORT": "29542",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("WANDB_MODE", "disabled")


def main() -> None:
    args = parse_args()
    ensure_single_process_dist_env()

    with Path(args.config).open("r") as f:
        config = ExperimentConfig(**yaml.safe_load(f))
    with Path(args.mix_file).open("r") as f:
        mixes = json.load(f)["mixes"]

    if args.mix_index < 0 or args.mix_index >= len(mixes):
        raise IndexError(
            f"mix-index {args.mix_index} out of range; available mixes={len(mixes)}"
        )

    mix = mixes[args.mix_index]
    mix_map = {k: (float(v[0]), float(v[1])) for k, v in mix.items()}
    sources = mk_source_instances(config.sources, mix_map)
    max_sequences_per_step = max(1, config.max_tokens // config.sequence_length)
    global_batch_size = max(1, args.global_batch_size)
    if global_batch_size > max_sequences_per_step:
        print(
            f"requested global_batch_size={global_batch_size} is too large for "
            f"max_tokens={config.max_tokens} and sequence_length={config.sequence_length}; "
            f"clamping to {max_sequences_per_step}"
        )
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
        print(f"effective_global_batch_size={global_batch_size}")
        print(
            f"tokenizer_v={built.dataset.tokenizer.vocab_size} "
            f"eos={built.dataset.tokenizer.eos_token_id} "
            f"bos={built.dataset.tokenizer.bos_token_id} "
            f"pad={built.dataset.tokenizer.pad_token_id}"
        )

        dataset = built.dataset.build()
        seed_all(built.init_seed)
        model = built.model.build(init_device="meta")
        train_module = built.train_module.build(model)
        data_loader = built.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = built.trainer.build(train_module, data_loader)
        if "wandb" in trainer.callbacks:
            trainer.callbacks.pop("wandb")

        print(f"trainer_max_steps={trainer.max_steps}")
        print(f"trainer_max_tokens={trainer.max_tokens}")
        print(f"save_folder={trainer.save_folder}")

        trainer.fit()

        summary = {
            "train_done": True,
            "run_name": args.run_name,
            "group_id": args.group_id,
            "global_step": int(trainer.global_step),
            "global_train_tokens_seen": int(trainer.global_train_tokens_seen),
            "save_folder": str(trainer.save_folder),
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
