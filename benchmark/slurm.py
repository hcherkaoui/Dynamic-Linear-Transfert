"""SLURM helpers for benchmark experiment submission."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import shlex
import shutil
import subprocess
from benchmark.experiment_registry import ExperimentSpec, SlurmSpec


@dataclass(frozen=True)
class SlurmSubmission:
    """Generated SLURM script and associated sbatch command."""

    script: str
    sbatch_command: tuple[str, ...]
    script_path: Path | None


def _repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def _safe_job_name(name: str) -> str:
    """Return a conservative SLURM job name."""
    clean = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name)
    return clean[:120] or "dyntr_benchmark"


def _script_path(script_dir: Path, job_name: str) -> Path:
    """Return a timestamped script path."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return script_dir / f"{job_name}_{stamp}.slurm.sh"


def _slurm_spec(spec: ExperimentSpec, overrides: dict[str, object | None]) -> SlurmSpec:
    """Merge registry SLURM defaults with CLI overrides."""
    base = spec.slurm or SlurmSpec()
    return SlurmSpec(
        partition=overrides.get("partition") or base.partition,
        account=overrides.get("account") or base.account,
        time=str(overrides.get("time") or base.time),
        cpus_per_task=int(overrides.get("cpus_per_task") or base.cpus_per_task),
        mem=str(overrides.get("mem") or base.mem),
        gres=overrides.get("gres") or base.gres,
        nodes=int(overrides.get("nodes") or base.nodes),
        ntasks=int(overrides.get("ntasks") or base.ntasks),
        job_name=overrides.get("job_name") or base.job_name,
        qos=overrides.get("qos") or base.qos,
    )


def _sbatch_lines(slurm: SlurmSpec, *, job_name: str, log_dir: Path) -> list[str]:
    """Return SBATCH directive lines."""
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir / '%x_%j.out'}",
        f"#SBATCH --error={log_dir / '%x_%j.err'}",
        f"#SBATCH --nodes={slurm.nodes}",
        f"#SBATCH --ntasks={slurm.ntasks}",
        f"#SBATCH --cpus-per-task={slurm.cpus_per_task}",
        f"#SBATCH --mem={slurm.mem}",
        f"#SBATCH --time={slurm.time}",
    ]
    if slurm.partition:
        lines.append(f"#SBATCH --partition={slurm.partition}")
    if slurm.account:
        lines.append(f"#SBATCH --account={slurm.account}")
    if slurm.gres:
        lines.append(f"#SBATCH --gres={slurm.gres}")
    if slurm.qos:
        lines.append(f"#SBATCH --qos={slurm.qos}")
    return lines


def _append_command_array(lines: list[str], command: list[str]) -> None:
    """Append a bash array command to script lines."""
    lines.append("CMD=(")
    for arg in command:
        if arg in {'"${PYTHON_BIN}"', '"${SEED}"'}:
            lines.append(f"  {arg}")
        else:
            lines.append(f"  {shlex.quote(arg)}")
    lines.append(")")


def build_slurm_script(
    spec: ExperimentSpec,
    *,
    mode: str = "compute",
    seed: int | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
    log_dir: str | Path = "runs/slurm_logs",
    extra_args: tuple[str, ...] = (),
    slurm_overrides: dict[str, object | None] | None = None,
) -> str:
    """Build a SLURM script for one benchmark experiment."""
    slurm = _slurm_spec(spec, slurm_overrides or {})
    job_name = _safe_job_name(str(slurm.job_name or f"dyntr_{spec.name}"))
    log_path = Path(log_dir)
    repo_root = _repo_root()
    resolved_output_dir = output_dir or spec.output_dir
    lines = _sbatch_lines(slurm, job_name=job_name, log_dir=log_path)
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            f'PROJECT_ROOT="${{PROJECT_ROOT:-{shlex.quote(str(repo_root))}}}"',
            'cd "${PROJECT_ROOT}"',
            f"mkdir -p {shlex.quote(str(log_path))}",
            f"mkdir -p {shlex.quote(str(resolved_output_dir))}",
            'export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"',
            'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"',
            'export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"',
            'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"',
            'export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"',
            "",
            'if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/activate" ]]; then',
            '  source "${VIRTUAL_ENV}/bin/activate"',
            'elif [[ -f ".venv/bin/activate" ]]; then',
            '  source ".venv/bin/activate"',
            "fi",
            "",
            'PYTHON_BIN="${PYTHON:-python}"',
            'GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || true)"',
            'echo "[slurm] host: $(hostname)"',
            'echo "[slurm] date: $(date -Is)"',
            'echo "[slurm] project_root: ${PROJECT_ROOT}"',
            'echo "[slurm] python: $(${PYTHON_BIN} --version 2>&1)"',
            'echo "[slurm] git_commit: ${GIT_COMMIT:-<unknown>}"',
            'echo "[slurm] job_id: ${SLURM_JOB_ID:-<none>}"',
            f'echo "[slurm] experiment: {spec.name}"',
        ]
    )
    if config_path is not None:
        lines.append(f"export DYNTR_CONFIG={shlex.quote(config_path)}")
    lines.append(f'SEED="{spec.default_seed if seed is None else seed}"')
    command = ['"${PYTHON_BIN}"', "-m", "benchmark_tasks", "run", spec.name, "--mode", mode, "--seed", '"${SEED}"']
    if output_dir is not None:
        command.extend(["--output-dir", output_dir])
    if config_path is not None:
        command.extend(["--config", config_path])
    command.extend(extra_args)
    preview = ["python", "-m", "benchmark_tasks", "run", spec.name, "--mode", mode, "--seed", "$SEED"]
    if output_dir is not None:
        preview.extend(["--output-dir", output_dir])
    if config_path is not None:
        preview.extend(["--config", config_path])
    preview.extend(extra_args)
    lines.append("# run command: " + shlex.join(preview))
    _append_command_array(lines, command)
    lines.extend(
        [
            'echo "[slurm] command: ${CMD[*]}"',
            '"${CMD[@]}"',
            "",
        ]
    )
    return "\n".join(lines)


def build_submission(
    spec: ExperimentSpec,
    *,
    script_dir: str | Path = "runs/slurm_scripts",
    **kwargs,
) -> SlurmSubmission:
    """Build a SLURM submission object without writing files."""
    slurm = _slurm_spec(spec, kwargs.get("slurm_overrides") or {})
    job_name = _safe_job_name(str(slurm.job_name or f"dyntr_{spec.name}"))
    path = _script_path(Path(script_dir), job_name)
    script = build_slurm_script(spec, **kwargs)
    return SlurmSubmission(script=script, sbatch_command=("sbatch", str(path)), script_path=path)


def submit_slurm(submission: SlurmSubmission, *, log_dir: str | Path = "runs/slurm_logs") -> str:
    """Write one script and submit it with sbatch."""
    if shutil.which("sbatch") is None:
        raise FileNotFoundError("sbatch was not found. Run this on a SLURM login node.")
    if submission.script_path is None:
        raise ValueError("A script path is required for non-dry-run submission.")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    submission.script_path.parent.mkdir(parents=True, exist_ok=True)
    submission.script_path.write_text(submission.script, encoding="utf-8")
    os.chmod(submission.script_path, 0o755)
    result = subprocess.run(submission.sbatch_command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "sbatch failed."
        raise RuntimeError(message)
    return result.stdout.strip()
