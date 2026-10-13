"""Command-line entry point for benchmark experiments."""
from __future__ import annotations
import argparse
import importlib
import os
import shlex
import sys
from pathlib import Path
from benchmark.experiment_registry import EXPERIMENTS, ExperimentSpec, get_experiment
from benchmark.slurm import build_submission, submit_slurm


def _has_option(args: tuple[str, ...], option: str) -> bool:
    """Return whether an argv tuple contains an option."""
    return option in args or any(arg.startswith(f"{option}=") for arg in args)


def _defaulted_args(spec: ExperimentSpec, extra_args: tuple[str, ...]) -> list[str]:
    """Prepend experiment defaults that were not supplied explicitly."""
    out: list[str] = []
    for i in range(0, len(spec.default_args), 2):
        option = spec.default_args[i]
        value = spec.default_args[i + 1]
        if not _has_option(extra_args, option):
            out.extend([option, value])
    out.extend(extra_args)
    return out


def _print_list(tag: str | None = None) -> None:
    """Print available experiments, optionally filtered by tag."""
    for spec in EXPERIMENTS.values():
        if tag is not None and tag not in spec.tags:
            continue
        tags = ", ".join(spec.tags)
        print(f"{spec.name}\t{spec.description} [{tags}]")


def _print_info(name: str) -> None:
    """Print detailed metadata for one experiment."""
    spec = get_experiment(name)
    slurm = spec.slurm
    print(f"name: {spec.name}")
    print(f"description: {spec.description}")
    print(f"module: {spec.module}")
    print(f"function: {spec.function}")
    print(f"config_path: {spec.config_path}")
    print(f"output_dir: {spec.output_dir}")
    print(f"default_seed: {spec.default_seed}")
    print(f"tags: {', '.join(spec.tags)}")
    print(f"default_args: {' '.join(spec.default_args) if spec.default_args else '<none>'}")
    if slurm is not None:
        print(
            "slurm: "
            f"time={slurm.time}, cpus_per_task={slurm.cpus_per_task}, "
            f"mem={slurm.mem}, gres={slurm.gres or '<none>'}"
        )
    if spec.notes:
        print(f"notes: {spec.notes}")


def _build_run_parser() -> argparse.ArgumentParser:
    """Build the parser for the SLURM worker command."""
    parser = argparse.ArgumentParser(prog="python -m benchmark_tasks run")
    parser.add_argument("experiment_name", nargs="?")
    parser.add_argument("--mode", choices=("compute", "plot"), default="compute")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=None, help="Optional benchmark config path.")
    parser.add_argument("--dry-run", action="store_true", help="Show the command without running it.")
    return parser


def _build_submit_parser() -> argparse.ArgumentParser:
    """Build the parser for the submit command."""
    parser = argparse.ArgumentParser(prog="python -m benchmark_tasks submit")
    parser.add_argument("--seed", type=int, default=None, help="Override the experiment seed.")
    parser.add_argument("--mode", choices=("compute", "plot"), default="compute")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=None, help="Optional benchmark config path.")
    parser.add_argument("--log-dir", default="runs/slurm_logs")
    parser.add_argument("--script-dir", default="runs/slurm_scripts")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", default=None)
    parser.add_argument("--gres", default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--ntasks", type=int, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _build_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmark_tasks plot")
    parser.add_argument("--output-dir", default=None, help="Directory containing computed curves.")
    parser.add_argument("--config", default=None, help="Benchmark config path.")
    return parser


def _optional_experiment_name(argv: list[str]) -> tuple[str | None, list[str]]:
    if argv and not argv[0].startswith("-"):
        return argv[0], argv[1:]
    return None, argv


def _load_experiment_main(spec: ExperimentSpec):
    """Load the experiment runner callable lazily."""
    module = importlib.import_module(spec.module)
    return getattr(module, spec.function)


def run_experiment(
    name: str,
    *,
    mode: str = "compute",
    seed: int | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
    dry_run: bool = False,
    extra_args: tuple[str, ...] = (),
) -> int:
    """Run or display one SLURM worker invocation."""
    spec = get_experiment(name)
    task_args = [mode, *_defaulted_args(spec, extra_args)]
    if output_dir is not None and not _has_option(tuple(task_args), "--results-dir"):
        task_args.extend(["--results-dir", output_dir])

    if dry_run:
        seed_part = f" --seed {seed}" if seed is not None else ""
        output_part = f" --output-dir {output_dir}" if output_dir is not None else ""
        config_part = f" --config {config_path}" if config_path is not None else ""
        display_args = " ".join(_defaulted_args(spec, extra_args))
        print(f"experiment: {spec.name}")
        print(f"description: {spec.description}")
        print(
            f"command: python -m benchmark_tasks run {spec.name} "
            f"--mode {mode}{seed_part}{output_part}{config_part} {display_args}".rstrip()
        )
        return 0

    if config_path is not None:
        os.environ["DYNTR_CONFIG"] = config_path
    runner = _load_experiment_main(spec)
    runner(spec.name, argv=task_args, seed=seed)
    return 0


def _configured_runs(name: str | None, config_path: str | None, extra_args: tuple[str, ...]):
    """Resolve benchmark variants from the same config for submit and plot."""
    if name is not None:
        return [(get_experiment(name), extra_args)]
    if any(_has_option(extra_args, option) for option in ("--eps", "--dataset-name")):
        raise ValueError("Select one experiment to override --eps or --dataset-name.")
    import yaml

    path = Path(config_path) if config_path else Path(__file__).parent / "benchmark/configs/config.yml"
    with path.open(encoding="utf-8") as stream:
        launcher = (yaml.safe_load(stream) or {}).get("launcher", {})
    runs = []
    for spec in EXPERIMENTS.values():
        values = launcher.get(spec.sweep) if spec.sweep else [None]
        if not isinstance(values, list) or not values:
            raise ValueError(f"launcher.{spec.sweep} must be a nonempty list in {path}")
        for value in values:
            option = "--dataset-name" if spec.sweep == "l_datasets" else "--eps"
            args = (option, str(value)) if spec.sweep else ()
            runs.append((spec, (*args, *extra_args)))
    return runs


def _slurm_overrides(args) -> dict[str, object | None]:
    """Return SLURM resource overrides from parsed CLI args."""
    return {
        "partition": args.partition,
        "account": args.account,
        "time": args.time,
        "cpus_per_task": args.cpus_per_task,
        "mem": args.mem,
        "gres": args.gres,
        "nodes": args.nodes,
        "ntasks": args.ntasks,
        "job_name": args.job_name,
        "qos": args.qos,
    }


def submit_experiments(args, extra_args: tuple[str, ...] = ()) -> int:
    """Submit selected experiments to SLURM or print dry-run scripts."""
    for option in ("--tag", "--array", "--seeds"):
        if _has_option(extra_args, option):
            raise ValueError(f"{option} is not supported by submit.")
    runs = _configured_runs(args.experiment_name, args.config, extra_args)
    for index, (spec, task_args) in enumerate(runs):
        submission = build_submission(
            spec,
            mode=args.mode,
            seed=args.seed,
            output_dir=args.output_dir,
            config_path=args.config,
            log_dir=args.log_dir,
            script_dir=args.script_dir,
            extra_args=task_args,
            slurm_overrides=_slurm_overrides(args),
        )
        if args.dry_run:
            if index:
                print()
            print(f"experiment: {spec.name}")
            print("sbatch command: " + shlex.join(submission.sbatch_command))
            print(submission.script)
            continue
        result = submit_slurm(submission, log_dir=args.log_dir)
        print(result)
    return 0


def plot_experiments(args, extra_args: tuple[str, ...] = ()) -> int:
    """Plot the configured variants from saved benchmark curves."""
    for spec, task_args in _configured_runs(args.experiment_name, args.config, extra_args):
        run_experiment(spec.name, mode="plot", output_dir=args.output_dir, config_path=args.config, extra_args=task_args)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level benchmark task parser."""
    parser = argparse.ArgumentParser(prog="python -m benchmark_tasks")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List experiments.")
    list_parser.add_argument("--tag", default=None, help="Filter experiments by tag.")

    info_parser = subparsers.add_parser("info", help="Show experiment metadata.")
    info_parser.add_argument("experiment_name")

    subparsers.add_parser("run", help="Compute one experiment locally or in a SLURM job.")
    subparsers.add_parser("submit", help="Submit experiments to SLURM.")
    subparsers.add_parser("plot", help="Plot saved results locally.")
    return parser


def cli(argv: list[str] | None = None) -> int:
    """Run the benchmark task command-line interface."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not argv:
        parser.print_help()
        return 2

    if argv[0] == "run":
        run_parser = _build_run_parser()
        run_args, extra_args = run_parser.parse_known_args(argv[1:])
        if run_args.experiment_name is None:
            run_parser.print_help()
            return 2
        try:
            return run_experiment(
                run_args.experiment_name,
                mode=run_args.mode,
                seed=run_args.seed,
                output_dir=run_args.output_dir,
                config_path=run_args.config,
                dry_run=run_args.dry_run,
                extra_args=tuple(extra_args),
            )
        except (RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if argv[0] == "submit":
        submit_parser = _build_submit_parser()
        name, remaining = _optional_experiment_name(argv[1:])
        submit_args, extra_args = submit_parser.parse_known_args(remaining)
        submit_args.experiment_name = name
        try:
            return submit_experiments(submit_args, extra_args=tuple(extra_args))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if argv[0] == "plot":
        plot_parser = _build_plot_parser()
        name, remaining = _optional_experiment_name(argv[1:])
        plot_args, extra_args = plot_parser.parse_known_args(remaining)
        plot_args.experiment_name = name
        try:
            return plot_experiments(plot_args, extra_args=tuple(extra_args))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            _print_list(tag=args.tag)
            return 0
        if args.command == "info":
            _print_info(args.experiment_name)
            return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(cli())
