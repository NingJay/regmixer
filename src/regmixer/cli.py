import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Optional

import click
import yaml
from olmo_core.utils import generate_uuid, prepare_cli_environment
from tqdm import tqdm
from yaspin import yaspin

from regmixer.aliases import ExperimentConfig, LaunchGroup
from regmixer.model.transformer import TransformerConfigBuilder
from regmixer.utils import (
    config_from_path,
    mk_experiment_group,
    mk_instance_cmd,
    mk_launch_configs,
    mk_mixes,
)

logger = logging.getLogger(__name__)


def _require_beaker_client():
    try:
        from beaker import Beaker
        from beaker.services.job import JobClient
    except ModuleNotFoundError as exc:
        raise click.ClickException(
            "This command requires the 'beaker' package. "
            "Install beaker first, or use local-only commands such as generate-mixes."
        ) from exc
    return Beaker, JobClient


@click.group()
def cli():
    prepare_cli_environment()


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
@click.option(
    "-m",
    "--mixture-file",
    help="(Optional) Relative path to a mixture configuration file.",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the experiment group configurations without launching.",
)
@click.option(
    "--no-cache",
    "-n",
    is_flag=True,
    default=False,
    help="Do not cache sources for this experiment group.",
)
def launch(config: Path, mixture_file: Optional[Path], dry_run: bool, no_cache: bool):
    """Launch an experiment."""
    Beaker, _ = _require_beaker_client()

    with open(config, "r") as f:
        data = yaml.safe_load(f)

    experiment_config = ExperimentConfig(**data)
    group_uuid = generate_uuid()[:8]

    beaker_user = (Beaker.from_env().account.whoami().name).upper()
    logger.info(f"Launching experiment group '{group_uuid}' as user '{beaker_user}'")

    logger.info("Generating experiment group from the following config...")
    logger.info(experiment_config)

    if not click.confirm("Proceed with this configuration?", default=False):
        logger.info("Launch cancelled!")
        return

    launch_group = None

    if mixture_file:
        with open(mixture_file, "r") as f:
            predefined_mixes = json.load(f)

        logger.info(predefined_mixes)
        if click.confirm(f"Launch experiment {group_uuid} with this set of mixtures?", default=False):
            with yaspin(text="Building experiment group...", color="yellow") as spinner:
                launch_group = LaunchGroup(
                    instances=mk_launch_configs(
                        group=mk_experiment_group(
                            config=experiment_config,
                            mixes=predefined_mixes["mixes"],
                            group_uuid=group_uuid,
                        ),
                        beaker_user=beaker_user,
                    )
                )
                spinner.ok("✔")
    else:
        mixes = mk_mixes(config, group_uuid, use_cache=(no_cache == False))
        if click.confirm(f"Launch experiment {group_uuid} with this set of mixtures?", default=False):
            with yaspin(text="Building experiment group...", color="yellow") as spinner:
                launch_group = LaunchGroup(
                    instances=mk_launch_configs(
                        group=mk_experiment_group(
                            experiment_config, mixes=mixes, group_uuid=group_uuid
                        ),
                        beaker_user=beaker_user,
                    )
                )
                spinner.ok("✔")
        else:
            logger.info("Launch cancelled!")
            return

    if not launch_group:
        logger.info("Launch cancelled!")
        return

    with yaspin(text="Launching experiment group...", color="yellow") as spinner:
        try:
            if dry_run:
                logger.info("Dry run mode enabled. Printing experiment configurations...")
                for experiment in launch_group.instances:
                    logger.info(experiment.build_experiment_spec())
                return

            results = []            
            torchrun = True if experiment_config.gpus > 1 else False 
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(experiment.launch, torchrun=torchrun) for experiment in launch_group.instances
                ]

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Launching experiments",
                ):
                    results.append(future.result())

            spinner.ok("✔")
            logger.info(results)
            logger.info(f"Experiment group '{group_uuid}' launched successfully!")
        except KeyboardInterrupt:
            logger.warning(
                "\nAborting experiment group launch! You may need to manually stop the launched experiments."
            )


def status_for_group(path: Path, group_id: str):
    Beaker, JobClient = _require_beaker_client()
    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    config = config_from_path(path)
    cluster = beaker.cluster.get(config.cluster)
    jobs = client.list(cluster=cluster)

    statuses = [
        {"status": job.status, "display_name": job.display_name}
        for job in jobs
        if job.display_name.startswith(f"{config.name}-{group_id}")
    ]
    statuses.sort(key=lambda x: x["display_name"])
    logger.info(statuses)


def stop_for_group(path: Path, group_id: str):
    Beaker, JobClient = _require_beaker_client()
    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    config = config_from_path(path)
    cluster = beaker.cluster.get(config.cluster)
    jobs = [
        {"id": job.id, "display_name": job.display_name, "status": job.status}
        for job in client.list(cluster=cluster)
        if job.display_name.startswith(f"{config.name}-{group_id}")
    ]

    if len(jobs) == 0:
        logger.info(f"No jobs found for group {group_id}")
        return

    jobs.sort(key=lambda x: x["display_name"])
    logger.info("Jobs to cancel:")
    logger.info(jobs)
    if click.confirm("Cancel these jobs?", default=False):
        for job in jobs:
            logger.info(f"Stopping job {job['display_name']}...")
            client.stop(job["id"])


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path for the generated mixes (defaults to generated_mix.json if not specified)",
)
def generate_mixes(config: Path, output: Optional[Path] = None):
    """Generate a set of mixtures based on a provided config"""
    # mk_mixes signature is (config_file, group_uuid, output, use_cache).
    # Use keyword arguments to avoid accidental positional mismatch.
    mk_mixes(
        config_file=config,
        group_uuid=generate_uuid()[:8],
        output=output,
    )


@cli.command(name="convert-hf")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment configuration file.",
)
@click.option(
    "--log-dir",
    type=click.Path(exists=True),
    required=True,
    help="Log directory for the training run group (used to infer summaries/output roots).",
)
@click.option(
    "--summary-dir",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Optional override for summary directory.",
)
@click.option(
    "--tokenizer-dir",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Optional tokenizer directory copied into converted HF checkpoints.",
)
@click.option(
    "--hf-cache-dir",
    type=click.Path(),
    required=False,
    default=None,
    help="Optional directory for converted HF checkpoints.",
)
@click.option(
    "--output-manifest",
    type=click.Path(),
    required=False,
    default=None,
    help="Optional manifest CSV path listing converted runs and HF directories.",
)
@click.option(
    "--mix-start",
    type=int,
    default=None,
    required=False,
    help="Optional first mix index (inclusive) to convert.",
)
@click.option(
    "--mix-end",
    type=int,
    default=None,
    required=False,
    help="Optional last mix index (inclusive) to convert.",
)
@click.option(
    "--force-convert",
    is_flag=True,
    default=False,
    help="Re-convert checkpoints even if HF export already exists.",
)
def convert_hf_command(
    config: Path,
    log_dir: Path,
    summary_dir: Optional[Path],
    tokenizer_dir: Optional[Path],
    hf_cache_dir: Optional[Path],
    output_manifest: Optional[Path],
    mix_start: Optional[int],
    mix_end: Optional[int],
    force_convert: bool,
):
    """Convert local checkpoints to HF model directories and write a manifest."""
    from regmixer.local_eval import run_local_hf_conversion

    run_local_hf_conversion(
        config_path=config,
        log_dir=log_dir,
        summary_dir=summary_dir,
        tokenizer_dir=tokenizer_dir,
        hf_cache_dir=hf_cache_dir,
        mix_start=mix_start,
        mix_end=mix_end,
        force_convert=force_convert,
        output_manifest=output_manifest,
    )


@cli.command(name="eval")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment configuration file.",
)
@click.option(
    "--log-dir",
    type=click.Path(exists=True),
    required=True,
    help="Log directory for the training run group (used to infer summaries/output roots).",
)
@click.option(
    "--output-csv",
    type=click.Path(),
    required=True,
    help="Path to write aggregated local eval metrics CSV.",
)
@click.option(
    "--summary-dir",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Optional override for summary directory.",
)
@click.option(
    "--mix-file",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Optional override for mix JSON file.",
)
@click.option(
    "--tokenizer-dir",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Optional tokenizer directory copied into converted HF checkpoints.",
)
@click.option(
    "--hf-cache-dir",
    type=click.Path(),
    required=False,
    default=None,
    help="Optional directory for converted HF checkpoints.",
)
@click.option(
    "--results-dir",
    type=click.Path(),
    required=False,
    default=None,
    help="Optional directory for raw lm_eval outputs.",
)
@click.option(
    "--task-groups",
    type=str,
    default="core,mmlu",
    show_default=True,
    help="Comma-separated task groups to evaluate locally. Supported: core,mmlu.",
)
@click.option(
    "--tasks",
    type=str,
    default=None,
    help="Optional explicit comma-separated lm_eval task list; overrides --task-groups.",
)
@click.option(
    "--num-fewshot",
    type=int,
    default=5,
    show_default=True,
    help="Few-shot count passed to lm_eval.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Batch size passed to lm_eval.",
)
@click.option(
    "--limit",
    type=float,
    default=None,
    required=False,
    help="Optional lm_eval limit for partial runs (testing only).",
)
@click.option(
    "--mix-start",
    type=int,
    default=None,
    required=False,
    help="Optional first mix index (inclusive) to evaluate.",
)
@click.option(
    "--mix-end",
    type=int,
    default=None,
    required=False,
    help="Optional last mix index (inclusive) to evaluate.",
)
@click.option(
    "--force-convert",
    is_flag=True,
    default=False,
    help="Re-convert checkpoints even if HF export already exists.",
)
@click.option(
    "--force-eval",
    is_flag=True,
    default=False,
    help="Re-run lm_eval even if cached raw results exist.",
)
def eval_command(
    config: Path,
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
):
    """Run local full eval from local checkpoints and write eval metrics CSV."""
    from regmixer.local_eval import run_local_eval

    run_local_eval(
        config_path=config,
        log_dir=log_dir,
        output_csv=output_csv,
        summary_dir=summary_dir,
        mix_file=mix_file,
        tokenizer_dir=tokenizer_dir,
        hf_cache_dir=hf_cache_dir,
        results_dir=results_dir,
        task_groups=task_groups,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        mix_start=mix_start,
        mix_end=mix_end,
        force_convert=force_convert,
        force_eval=force_eval,
    )


@cli.command(name="fit-mixture")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to experiment configuration file (kept for script compatibility).",
)
@click.option(
    "--eval-metrics",
    type=click.Path(exists=True),
    required=True,
    help="CSV produced by local eval step.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Output JSON path for proposed mixture weights.",
)
@click.option(
    "--mix-file",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Optional mix JSON file override when eval CSV lacks weight columns.",
)
@click.option(
    "--metric-columns",
    type=str,
    required=False,
    default=None,
    help="Optional comma-separated subset of metric columns from eval CSV.",
)
@click.option(
    "--search-samples",
    type=int,
    required=False,
    default=200000,
    show_default=True,
    help="Number of Dirichlet samples for proposed-mixture search.",
)
@click.option(
    "--seed",
    type=int,
    required=False,
    default=42,
    show_default=True,
    help="Random seed for search.",
)
def fit_mixture_command(
    config: Path,
    eval_metrics: Path,
    output: Path,
    mix_file: Optional[Path],
    metric_columns: Optional[str],
    search_samples: int,
    seed: int,
):
    """Fit a local regression surface and propose p* weights from eval CSV."""
    from regmixer.local_fit import run_local_fit

    run_local_fit(
        config_path=config,
        eval_metrics=eval_metrics,
        output=output,
        mix_file=mix_file,
        metric_columns=metric_columns,
        search_samples=search_samples,
        seed=seed,
    )


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def validate(config: Path):
    """Validate an experiment configuration."""
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    mixes = mk_mixes(config_file=config, group_uuid=generate_uuid()[:8])
    experiment_group = mk_experiment_group(ExperimentConfig(**data), mixes, generate_uuid()[:8])
    beaker_user = "validate-no-op"

    for experiment in experiment_group.instances:
        logger.info(
            mk_instance_cmd(
                experiment, experiment_group.config, experiment_group.group_id, beaker_user
            )
        )
        transformer = TransformerConfigBuilder(
            cluster=experiment_group.config.cluster,
            beaker_user="validate-no-op",
            group_id="validate-no-op",
            run_name="validate-no-op",
            max_tokens=experiment_group.config.max_tokens,
            sources=experiment.sources,
            sequence_length=experiment_group.config.sequence_length,
            seed=experiment_group.config.seed,
            tokenizer=experiment_group.config.tokenizer,
            dtype=experiment_group.config.dtype,
            model_identifier=experiment_group.config.proxy_model_id,
            weka=experiment_group.config.weka,
            device_batch_size=experiment_group.config.device_batch_size,
        ).build()
        dataset = transformer.dataset.build()
        dataset.prepare()


@cli.command()
@click.option(
    "-g",
    "--group-id",
    required=True,
    help="The group ID of the experiment group to stop.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def status(config: Path, group_id: str):
    """Get the status of a launched experiment group."""

    status_for_group(config, group_id)


@cli.command()
@click.option(
    "-g",
    "--group-id",
    required=True,
    help="The group ID of the experiment group to stop.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def cancel(config: Path, group_id: str):
    """Cancel all running jobs for an experiment group."""

    stop_for_group(config, group_id)


if __name__ == "__main__":
    cli()
