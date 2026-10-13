# Benchmarks

Run commands from the repository root. `benchmark/configs/config.yml` defines the full experiment sweeps; `config-blank.yml` is a small example. Both `submit` and `plot` read the same config, so they select the same dataset and epsilon variants.

## Local example

```bash
make smoke OUT=runs/smoke
```

This computes and plots one experiment without a scheduler. To run another variant directly:

```bash
python -m benchmark_tasks run bench_04_perf_regr_synth --config benchmark/configs/config-blank.yml --eps 0.5 --output-dir runs/smoke
python -m benchmark_tasks plot bench_04_perf_regr_synth --config benchmark/configs/config-blank.yml --eps 0.5 --output-dir runs/smoke
```

## SLURM

Set up Python on a shared filesystem. `bash benchmark/setup.sh` creates a virtual environment under `$WORK/venvs/dyntr` (or `.dyntr_cache/venvs/dyntr` when `WORK` is unset). A regular `.venv` in the repository also works.

```bash
python -m benchmark_tasks submit --config benchmark/configs/config.yml --output-dir runs/full --dry-run
python -m benchmark_tasks submit --config benchmark/configs/config.yml --output-dir runs/full
python -m benchmark_tasks plot --config benchmark/configs/config.yml --output-dir runs/full
```

Pass a single experiment name after `submit` to submit only that experiment. Without a name, each value in the config's `launcher.l_eps_*` and `launcher.l_datasets` lists is submitted. Use `--seed 0` to override the seed. The default request is 25 CPUs, matching the full config's `n_jobs`; pass `--cpus-per-task 1` with the blank config. Other resource options include `--time`, `--mem`, `--partition`, and `--account`. Logs and generated scripts default to `runs/slurm_logs` and `runs/slurm_scripts`.

The `benchmark/launcher.sh --run` and `benchmark/plot.sh --run` wrappers use the full config and write under `benchmark/_results/run`. Use `--blank` for the small config. Plot options include `--real-ylim-quantile 0.55` and `--real-error-ylim 0 1.2`.
