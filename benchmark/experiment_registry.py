"""Lightweight benchmark experiment registry."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class SlurmSpec:
    """Default SLURM resources for one experiment."""

    partition: str | None = None
    account: str | None = None
    time: str = "02:00:00"
    cpus_per_task: int = 25
    mem: str = "16G"
    gres: str | None = None
    nodes: int = 1
    ntasks: int = 1
    job_name: str | None = None
    qos: str | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    """Metadata needed to list and launch one benchmark experiment."""

    name: str
    description: str
    tags: tuple[str, ...]
    default_args: tuple[str, ...] = ()
    module: str = "benchmark.benchmark_tasks"
    function: str = "main"
    config_path: str = "benchmark/configs/config.yml"
    output_dir: str = "_results"
    default_seed: int = 4578902
    slurm: SlurmSpec | None = field(default_factory=SlurmSpec)
    sweep: str | None = None
    notes: str = ""


EXPERIMENTS: Mapping[str, ExperimentSpec] = {
    "bench_01_delta_regr_illus_synth": ExperimentSpec(
        name="bench_01_delta_regr_illus_synth",
        description="Regression collaboration-criteria illustration.",
        tags=("benchmark", "regression", "synthetic", "illustration"),
        default_args=("--eps", "0.5"),
        sweep="l_eps_regr_mono",
    ),
    "bench_02_sigma_infl_synth": ExperimentSpec(
        name="bench_02_sigma_infl_synth",
        description="Regression sigma influence benchmark.",
        tags=("benchmark", "regression", "synthetic", "influence"),
        default_args=("--eps", "0.5"),
        sweep="l_eps_regr_mono",
    ),
    "bench_03_eps_infl_synth": ExperimentSpec(
        name="bench_03_eps_infl_synth",
        description="Regression epsilon influence benchmark.",
        tags=("benchmark", "regression", "synthetic", "influence"),
    ),
    "bench_04_perf_regr_synth": ExperimentSpec(
        name="bench_04_perf_regr_synth",
        description="Synthetic regression performance comparison.",
        tags=("benchmark", "regression", "synthetic", "performance"),
        default_args=("--eps", "0.5"),
        sweep="l_eps_regr_mono",
    ),
    "bench_05_perf_regr_real": ExperimentSpec(
        name="bench_05_perf_regr_real",
        description="Real-data regression performance comparison.",
        tags=("benchmark", "regression", "real", "performance"),
        default_args=("--dataset-name", "lipophilicity_chemberta"),
        sweep="l_datasets",
    ),
    "bench_06_lbda_infl_synth": ExperimentSpec(
        name="bench_06_lbda_infl_synth",
        description="Regression lambda influence benchmark.",
        tags=("benchmark", "regression", "synthetic", "influence"),
    ),
    "bench_07_intro_regr_illus": ExperimentSpec(
        name="bench_07_intro_regr_illus",
        description="Introductory regression illustration.",
        tags=("benchmark", "regression", "synthetic", "illustration"),
    ),
    "bench_08_perf_regr_multi_synth": ExperimentSpec(
        name="bench_08_perf_regr_multi_synth",
        description="Synthetic multi-source regression performance comparison.",
        tags=("benchmark", "regression", "synthetic", "multi-source", "performance"),
        default_args=("--eps", "1.0"),
        sweep="l_eps_regr_multi",
    ),
    "bench_09_perf_classif_synth": ExperimentSpec(
        name="bench_09_perf_classif_synth",
        description="Synthetic classification performance comparison.",
        tags=("benchmark", "classification", "synthetic", "performance"),
        default_args=("--eps", "1.0"),
        sweep="l_eps_classif_mono",
    ),
    "bench_10_perf_classif_multi_synth": ExperimentSpec(
        name="bench_10_perf_classif_multi_synth",
        description="Synthetic multi-source classification performance comparison.",
        tags=("benchmark", "classification", "synthetic", "multi-source", "performance"),
        default_args=("--eps", "1.0"),
        sweep="l_eps_classif_multi",
    ),
    "bench_11_delta_classif_illus_synth": ExperimentSpec(
        name="bench_11_delta_classif_illus_synth",
        description="Classification collaboration-criteria illustration.",
        tags=("benchmark", "classification", "synthetic", "illustration"),
        default_args=("--eps", "1.0"),
        sweep="l_eps_classif_mono",
    ),
    "bench_12_perf_regr_lognormal_synth": ExperimentSpec(
        name="bench_12_perf_regr_lognormal_synth",
        description="Synthetic regression comparison with centered log-normal residuals.",
        tags=("benchmark", "regression", "synthetic", "performance", "lognormal"),
        default_args=("--eps", "0.5"),
        sweep="l_eps_regr_mono",
    ),
}


def available_experiments() -> tuple[str, ...]:
    """Return experiment names in stable display order."""
    return tuple(EXPERIMENTS)


def get_experiment(name: str) -> ExperimentSpec:
    """Return an experiment spec or raise a clear error."""
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        available = "\n".join(f"- {key}" for key in available_experiments())
        raise ValueError(
            f"Unknown experiment: {name}\n"
            "Available experiments:\n"
            f"{available}\n"
            "Use `python -m benchmark_tasks list`."
        ) from exc
