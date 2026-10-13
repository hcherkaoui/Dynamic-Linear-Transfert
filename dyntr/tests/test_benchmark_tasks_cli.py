"""Tests for centralized benchmark experiment launching."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import benchmark_tasks
import benchmark.slurm as slurm
from benchmark.experiment_registry import EXPERIMENTS


def test_experiment_registry_contains_expected_names():
    """Test that the lightweight registry contains core benchmark names."""
    expected = {
        "bench_01_delta_regr_illus_synth",
        "bench_05_perf_regr_real",
        "bench_12_perf_regr_lognormal_synth",
    }
    assert expected <= set(EXPERIMENTS)


def test_benchmark_tasks_list_command_runs(capsys):
    """Test that the list command prints registered experiments."""
    assert benchmark_tasks.cli(["list"]) == 0
    out = capsys.readouterr().out
    assert "bench_01_delta_regr_illus_synth" in out


def test_benchmark_tasks_info_command_runs(capsys):
    """Test that the info command prints experiment metadata."""
    assert benchmark_tasks.cli(["info", "bench_01_delta_regr_illus_synth"]) == 0
    out = capsys.readouterr().out
    assert "Regression collaboration-criteria illustration" in out
    assert "synthetic" in out


def test_benchmark_tasks_unknown_experiment_is_clear(capsys):
    """Test that unknown experiments report a clear error."""
    assert benchmark_tasks.cli(["info", "does_not_exist"]) == 2
    err = capsys.readouterr().err
    assert "Unknown experiment: does_not_exist" in err
    assert "Use `python -m benchmark_tasks list`" in err


def test_benchmark_tasks_run_compute_locally(monkeypatch):
    """Test that the worker can compute without SLURM."""
    for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_STEP_ID"):
        monkeypatch.delenv(name, raising=False)
    called = []
    monkeypatch.setattr(benchmark_tasks, "_load_experiment_main", lambda _spec: lambda *args, **kwargs: called.append((args, kwargs)))
    assert benchmark_tasks.cli(["run", "bench_01_delta_regr_illus_synth", "--seed", "0"]) == 0
    assert called[0][1]["seed"] == 0


def test_benchmark_tasks_submit_dry_run_prints_slurm(capsys):
    """Test that submit dry-run prints a SLURM script."""
    rc = benchmark_tasks.cli(
        ["submit", "bench_01_delta_regr_illus_synth", "--dry-run"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "#SBATCH --job-name=dyntr_bench_01_delta_regr_illus_synth" in out
    assert "python -m benchmark_tasks run" in out
    assert "bench_01_delta_regr_illus_synth" in out


def test_benchmark_tasks_submit_rejects_seed_arrays(capsys):
    """Several seeds would overwrite the same result file."""
    rc = benchmark_tasks.cli(
        [
            "submit",
            "bench_01_delta_regr_illus_synth",
            "--dry-run",
            "--array",
        ]
    )
    assert rc == 2
    assert "--array is not supported" in capsys.readouterr().err


def test_benchmark_tasks_submit_all_dry_run(capsys):
    """Test that submit expands the configured variants into distinct scripts."""
    rc = benchmark_tasks.cli(["submit", "--dry-run", "--config", "benchmark/configs/config.yml"])
    out = capsys.readouterr().out
    assert rc == 0
    scripts = [line for line in out.splitlines() if line.startswith("sbatch command:")]
    assert len(scripts) == 33
    assert len(set(scripts)) == len(scripts)
    assert out.splitlines().count("experiment: bench_05_perf_regr_real") == 2
    assert "--dataset-name\n  aqsoldb_chemberta" in out


def test_benchmark_tasks_submit_tag_is_rejected(capsys):
    """Test that tag-based submit selection is no longer accepted."""
    rc = benchmark_tasks.cli(["submit", "--tag", "classification", "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "--tag is not supported" in err


def test_benchmark_tasks_submit_rejects_sweep_override(capsys):
    """Overriding every variant with one epsilon would overwrite outputs."""
    assert benchmark_tasks.cli(["submit", "--dry-run", "--eps", "0.5"]) == 2
    assert "Select one experiment" in capsys.readouterr().err


def test_benchmark_tasks_plot_options_do_not_become_experiment_name(monkeypatch):
    """Numeric values of plot options must remain option arguments."""
    captured = []
    monkeypatch.setattr(benchmark_tasks, "plot_experiments", lambda args, extra_args: captured.append((args, extra_args)) or 0)
    assert benchmark_tasks.cli(["plot", "--real-ylim-quantile", "0.55"]) == 0
    assert captured[0][0].experiment_name is None
    assert captured[0][1] == ("--real-ylim-quantile", "0.55")


def test_benchmark_tasks_submit_unknown_experiment_is_clear(capsys):
    """Test that submit reports unknown experiments clearly."""
    rc = benchmark_tasks.cli(["submit", "does_not_exist", "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "Unknown experiment: does_not_exist" in err
    assert "Use `python -m benchmark_tasks list`" in err


def test_benchmark_tasks_submit_without_sbatch_is_clear(monkeypatch, capsys):
    """Test that missing sbatch reports a clear error."""
    monkeypatch.setattr(slurm.shutil, "which", lambda _name: None)
    rc = benchmark_tasks.cli(["submit", "bench_01_delta_regr_illus_synth"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "sbatch was not found" in err
    assert "SLURM login node" in err


def test_python_module_list_runs():
    """Test python -m benchmark_tasks list."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "benchmark_tasks", "list"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "bench_01_delta_regr_illus_synth" in result.stdout


def test_benchmark_module_computes_locally(tmp_path):
    """Test the small synthetic benchmark without a scheduler."""
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SLURM_"):
            env.pop(key)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark_tasks",
            "run",
            "bench_01_delta_regr_illus_synth",
            "--config",
            "benchmark/configs/config-blank.yml",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "01_delta_regr_illus_synth_eps_0.5.joblib").is_file()


def test_list_does_not_import_heavy_benchmark_module():
    """Test that listing experiments avoids the executable benchmark module."""
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys, benchmark_tasks; "
        "benchmark_tasks.cli(['list']); "
        "print('HEAVY=' + str('benchmark.benchmark_tasks' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "HEAVY=False" in result.stdout
