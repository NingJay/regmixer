from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train.checkpoint import load_state_dict

from regmixer.aliases import ExperimentConfig
from regmixer.eval.constants import ALL_CORE_TASKS, ALL_MMLU_TASKS
from regmixer.model.aliases import SupportedModels, SupportedTokenizers

logger = logging.getLogger(__name__)


TASK_ALIASES = {
    "csqa": "commonsense_qa",
    "socialiqa": "social_iqa",
}

METRIC_PRIORITY = (
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,none",
    "f1,none",
)

DEFAULT_OPENSEEK_TOKENIZER_DIR = Path("/home/staff/jiayining/OpenSeek/hf_openseek/tokenizer")


@dataclass(frozen=True)
class SummaryRecord:
    run_name: str
    mix_index: int
    global_step: int
    step_dir: Path


def _canonical_task(task: str) -> str:
    task = task.strip()
    return TASK_ALIASES.get(task, task)


def _parse_tasks(task_groups: str, tasks: Optional[str]) -> list[str]:
    if tasks:
        requested = [_canonical_task(task) for task in tasks.split(",") if task.strip()]
    else:
        requested = []
        groups = [group.strip().lower() for group in task_groups.split(",") if group.strip()]
        if not groups:
            groups = ["core"]

        for group in groups:
            if group == "core":
                requested.extend(_canonical_task(task) for task in ALL_CORE_TASKS)
            elif group == "mmlu":
                requested.extend(ALL_MMLU_TASKS)
            else:
                raise ValueError(
                    f"Unsupported task group '{group}'. Only 'core' and 'mmlu' are supported locally."
                )

    deduped: list[str] = []
    seen: set[str] = set()
    for task in requested:
        if task not in seen:
            seen.add(task)
            deduped.append(task)
    return deduped


def _validate_tasks_exist(tasks: list[str]) -> None:
    try:
        from lm_eval.tasks import TaskManager
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "lm-eval-harness is required for local eval. Install it in the active environment."
        ) from exc

    manager = TaskManager()
    missing = [task for task in tasks if task not in manager.all_tasks]
    if missing:
        raise RuntimeError(
            "The following lm_eval tasks are unavailable in this environment: "
            + ", ".join(missing)
        )


def _load_config(config_path: Path) -> ExperimentConfig:
    with config_path.open("r", encoding="utf-8") as f:
        return ExperimentConfig(**yaml.safe_load(f))


def _infer_mix_file(output_dir: Path, mix_file: Optional[Path]) -> Path:
    if mix_file:
        return mix_file

    candidates = sorted(output_dir.glob("*mixes.json"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Could not infer mix file under {output_dir}. Provide --mix-file explicitly."
        )
    raise ValueError(
        "Found multiple mix files; provide --mix-file explicitly: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_mix_weights(mix_file: Path) -> dict[int, dict[str, float]]:
    with mix_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    result: dict[int, dict[str, float]] = {}
    for idx, mix in enumerate(payload["mixes"]):
        result[idx] = {name: float(value[0]) for name, value in mix.items()}
    return result


def _load_summaries(summary_dir: Path) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []

    for summary_path in sorted(summary_dir.glob("*.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        if not summary.get("train_done", False):
            continue

        run_name = str(summary["run_name"])
        match = re.search(r"(\d+)$", run_name)
        if not match:
            raise ValueError(f"Unable to infer mix index from run name: {run_name}")

        mix_index = int(match.group(1))
        global_step = int(summary["global_step"])
        save_folder = Path(summary["save_folder"])
        step_dir = save_folder / f"step{global_step}"
        if not step_dir.exists():
            raise FileNotFoundError(f"Checkpoint step directory not found: {step_dir}")

        records.append(
            SummaryRecord(
                run_name=run_name,
                mix_index=mix_index,
                global_step=global_step,
                step_dir=step_dir,
            )
        )

    if not records:
        raise RuntimeError(f"No completed training summaries found in {summary_dir}")

    records.sort(key=lambda record: record.mix_index)
    return records


def _build_model(config: ExperimentConfig):
    model_config = SupportedModels[config.proxy_model_id].value
    tokenizer_config = SupportedTokenizers[config.tokenizer].value

    transformer = TransformerConfig.llama_like(
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        vocab_size=tokenizer_config.padded_vocab_size(),
        rope_theta=model_config.rope_theta,
        layer_norm_eps=model_config.layer_norm_eps,
        qk_norm=model_config.qk_norm,
        block_name=model_config.block_type,
    )
    return transformer.build(init_device="cpu")


def _attach_openseek_tokenizer(hf_dir: Path, tokenizer_dir: Path) -> None:
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_dir}")

    required_files = ("qwen.tiktoken", "tokenization_qwen.py", "qwen_generation_utils.py")
    for filename in required_files:
        src = tokenizer_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Required tokenizer file missing: {src}")
        shutil.copy2(src, hf_dir / filename)

    tokenization_file = hf_dir / "tokenization_qwen.py"
    with tokenization_file.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n"
            "def _regmixer_encode(self, text, allowed_special='all', disallowed_special=(), **kwargs):\n"
            "    add_special_tokens = bool(kwargs.pop('add_special_tokens', False))\n"
            "    token_ids = self.tokenizer.encode(\n"
            "        text,\n"
            "        allowed_special=allowed_special,\n"
            "        disallowed_special=disallowed_special,\n"
            "    )\n"
            "    if add_special_tokens:\n"
            "        prefix_id = self.im_start_id if hasattr(self, 'im_start_id') else self.eod_id\n"
            "        token_ids = [prefix_id] + token_ids\n"
            "    return token_ids\n\n"
            "QWenTokenizer.encode = _regmixer_encode\n"
        )

    src_cfg_path = tokenizer_dir / "tokenizer_config.json"
    src_cfg = {}
    if src_cfg_path.exists():
        with src_cfg_path.open("r", encoding="utf-8") as f:
            src_cfg = json.load(f)

    tokenizer_config = {
        "model_max_length": int(src_cfg.get("model_max_length", 8192)),
        "tokenizer_class": "QWenTokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_qwen.QWenTokenizer", None]},
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "bos_token": "<|im_start|>",
    }
    if "chat_template" in src_cfg:
        tokenizer_config["chat_template"] = src_cfg["chat_template"]

    with (hf_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)

    special_tokens_map = {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "bos_token": "<|im_start|>",
    }
    with (hf_dir / "special_tokens_map.json").open("w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2)


def _convert_checkpoint_to_hf(
    config: ExperimentConfig,
    record: SummaryRecord,
    hf_cache_dir: Path,
    tokenizer_dir: Path,
    force_convert: bool = False,
) -> Path:
    output_dir = hf_cache_dir / f"{record.run_name}-step{record.global_step}-hf"
    config_file = output_dir / "config.json"
    model_file = output_dir / "model.safetensors"

    if not force_convert and config_file.exists() and model_file.exists():
        logger.info("HF checkpoint already exists, reusing %s", output_dir)
        _attach_openseek_tokenizer(output_dir, tokenizer_dir)
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Converting %s to HF at %s", record.step_dir, output_dir)

    model = _build_model(config)
    state = {"model": model.state_dict()}
    load_state_dict(record.step_dir / "model_and_optim", state)
    model.load_state_dict(state["model"])
    save_hf_model(str(output_dir), state["model"], model, save_overwrite=True)
    _attach_openseek_tokenizer(output_dir, tokenizer_dir)
    return output_dir


def _find_results_file(results_dir: Path) -> Path:
    direct = results_dir / "results.json"
    if direct.exists():
        return direct

    # lm_eval may write nested result files under a sanitized model-path directory.
    json_candidates = sorted(
        (
            path
            for path in results_dir.rglob("*.json")
            if path.name == "results.json" or path.name.startswith("results_")
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if json_candidates:
        return json_candidates[-1]

    raise FileNotFoundError(f"No lm_eval JSON results were found under {results_dir}")


def _pick_metric(task_metrics: dict) -> tuple[str, float]:
    for metric_name in METRIC_PRIORITY:
        value = task_metrics.get(metric_name)
        if isinstance(value, (int, float)):
            return metric_name, float(value)

    for metric_name, value in task_metrics.items():
        if metric_name.endswith("_stderr,none"):
            continue
        if isinstance(value, (int, float)):
            return metric_name, float(value)

    raise ValueError(f"No scalar metric found in task metrics: {task_metrics}")


def _read_lm_eval_metrics(results_path: Path, tasks: list[str]) -> tuple[dict[str, float], dict[str, str]]:
    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise ValueError(f"Invalid lm_eval results payload in {results_path}")

    metric_values: dict[str, float] = {}
    metric_names: dict[str, str] = {}
    for task in tasks:
        if task not in results:
            raise KeyError(f"Task '{task}' not present in lm_eval output {results_path}")
        metric_name, metric_value = _pick_metric(results[task])
        metric_names[task] = metric_name
        metric_values[task] = metric_value

    return metric_values, metric_names


def _run_lm_eval(
    hf_model_dir: Path,
    tasks: list[str],
    results_dir: Path,
    *,
    num_fewshot: int,
    batch_size: int,
    limit: Optional[float],
    force_eval: bool,
) -> tuple[dict[str, float], dict[str, str], Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    existing: Optional[Path] = None
    if not force_eval:
        try:
            existing = _find_results_file(results_dir)
        except FileNotFoundError:
            existing = None

    if existing is not None:
        logger.info("Found cached lm_eval results at %s", existing)
        try:
            values, metric_names = _read_lm_eval_metrics(existing, tasks)
            logger.info("Reusing cached lm_eval results at %s", existing)
            return values, metric_names, existing
        except (KeyError, ValueError) as exc:
            logger.warning(
                "Cached lm_eval results at %s do not match requested tasks (%s); rerunning.",
                existing,
                exc,
            )

    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--trust_remote_code",
        "--model_args",
        (
            f"pretrained={hf_model_dir},tokenizer={hf_model_dir},"
            "trust_remote_code=True,add_bos_token=True,max_length=8192"
        ),
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(results_dir),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    logger.info("Running lm_eval for %s", hf_model_dir.name)
    subprocess.run(cmd, check=True)

    result_file = _find_results_file(results_dir)
    values, metric_names = _read_lm_eval_metrics(result_file, tasks)
    return values, metric_names, result_file


def run_local_hf_conversion(
    *,
    config_path: Path,
    log_dir: Path,
    summary_dir: Optional[Path],
    tokenizer_dir: Optional[Path],
    hf_cache_dir: Optional[Path],
    mix_start: Optional[int],
    mix_end: Optional[int],
    force_convert: bool,
    output_manifest: Optional[Path],
) -> Path:
    config_path = Path(config_path)
    log_dir = Path(log_dir)
    summary_dir = Path(summary_dir) if summary_dir is not None else None
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir is not None else None
    hf_cache_dir = Path(hf_cache_dir) if hf_cache_dir is not None else None
    output_manifest = Path(output_manifest) if output_manifest is not None else None

    config = _load_config(config_path)
    output_dir = log_dir.resolve().parent
    resolved_summary_dir = summary_dir or (output_dir / "summaries")
    resolved_tokenizer_dir = tokenizer_dir or DEFAULT_OPENSEEK_TOKENIZER_DIR
    resolved_hf_cache_dir = hf_cache_dir or (output_dir / "eval" / "hf_models")
    resolved_hf_cache_dir.mkdir(parents=True, exist_ok=True)

    records = _load_summaries(resolved_summary_dir)
    if mix_start is not None:
        records = [record for record in records if record.mix_index >= mix_start]
    if mix_end is not None:
        records = [record for record in records if record.mix_index <= mix_end]
    if not records:
        raise RuntimeError(
            f"No runs selected after applying mix range filters mix_start={mix_start}, mix_end={mix_end}."
        )

    logger.info("Preparing HF exports for %d runs", len(records))
    rows: list[dict[str, object]] = []
    for record in records:
        hf_dir = _convert_checkpoint_to_hf(
            config=config,
            record=record,
            hf_cache_dir=resolved_hf_cache_dir,
            tokenizer_dir=resolved_tokenizer_dir,
            force_convert=force_convert,
        )
        rows.append(
            {
                "run_name": record.run_name,
                "mix_index": record.mix_index,
                "global_step": record.global_step,
                "checkpoint_step_dir": str(record.step_dir),
                "hf_model_dir": str(hf_dir),
            }
        )

    manifest_path = output_manifest or (resolved_hf_cache_dir.parent / "hf_models_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "mix_index",
                "global_step",
                "checkpoint_step_dir",
                "hf_model_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("HF conversion manifest written to %s", manifest_path)
    return manifest_path


def run_local_eval(
    *,
    config_path: Path,
    log_dir: Path,
    output_csv: Path,
    summary_dir: Optional[Path],
    mix_file: Optional[Path],
    tokenizer_dir: Optional[Path],
    hf_cache_dir: Optional[Path],
    results_dir: Optional[Path],
    task_groups: str,
    tasks: Optional[str],
    num_fewshot: int,
    batch_size: int,
    limit: Optional[float],
    mix_start: Optional[int],
    mix_end: Optional[int],
    force_convert: bool,
    force_eval: bool,
) -> Path:
    config_path = Path(config_path)
    log_dir = Path(log_dir)
    output_csv = Path(output_csv)
    summary_dir = Path(summary_dir) if summary_dir is not None else None
    mix_file = Path(mix_file) if mix_file is not None else None
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir is not None else None
    hf_cache_dir = Path(hf_cache_dir) if hf_cache_dir is not None else None
    results_dir = Path(results_dir) if results_dir is not None else None

    config = _load_config(config_path)
    tasks_to_run = _parse_tasks(task_groups, tasks)
    _validate_tasks_exist(tasks_to_run)

    output_dir = output_csv.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_summary_dir = summary_dir or (log_dir.resolve().parent / "summaries")
    resolved_mix_file = _infer_mix_file(output_dir, mix_file)
    resolved_tokenizer_dir = tokenizer_dir or DEFAULT_OPENSEEK_TOKENIZER_DIR
    resolved_hf_cache_dir = hf_cache_dir or (output_dir / "eval" / "hf_models")
    resolved_results_dir = results_dir or (output_dir / "eval" / "raw_lm_eval")

    mix_weights = _load_mix_weights(resolved_mix_file)
    records = _load_summaries(resolved_summary_dir)
    if mix_start is not None:
        records = [record for record in records if record.mix_index >= mix_start]
    if mix_end is not None:
        records = [record for record in records if record.mix_index <= mix_end]
    if not records:
        raise RuntimeError(
            f"No runs selected after applying mix range filters mix_start={mix_start}, mix_end={mix_end}."
        )

    logger.info("Evaluating %d runs across %d tasks", len(records), len(tasks_to_run))
    rows: list[dict[str, object]] = []
    metric_name_map: dict[str, str] = {}

    for record in records:
        if record.mix_index not in mix_weights:
            raise KeyError(f"Mix index {record.mix_index} not found in {resolved_mix_file}")

        hf_dir = _convert_checkpoint_to_hf(
            config,
            record,
            resolved_hf_cache_dir,
            resolved_tokenizer_dir,
            force_convert,
        )
        run_results_dir = resolved_results_dir / record.run_name
        metrics, metric_names, metrics_file = _run_lm_eval(
            hf_dir,
            tasks_to_run,
            run_results_dir,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
            force_eval=force_eval,
        )
        metric_name_map.update(metric_names)

        row: dict[str, object] = {
            "run_name": record.run_name,
            "mix_index": record.mix_index,
            "global_step": record.global_step,
            "checkpoint_step_dir": str(record.step_dir),
            "hf_model_dir": str(hf_dir),
            "raw_metrics_file": str(metrics_file),
        }

        for source_name, weight in sorted(mix_weights[record.mix_index].items()):
            row[f"weight__{source_name}"] = weight
        for task_name, value in sorted(metrics.items()):
            row[f"task__{task_name}"] = value

        task_values = [float(metrics[task]) for task in tasks_to_run]
        row["task_mean"] = sum(task_values) / len(task_values)
        rows.append(row)

    all_columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in all_columns:
                all_columns.append(key)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)

    metrics_legend = output_csv.with_suffix(".metrics.json")
    with metrics_legend.open("w", encoding="utf-8") as f:
        json.dump(metric_name_map, f, indent=2, sort_keys=True)

    logger.info("Local eval metrics written to %s", output_csv)
    logger.info("Metric name mapping written to %s", metrics_legend)
    return output_csv
