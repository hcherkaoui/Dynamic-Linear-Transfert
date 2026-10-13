# DynTr

Code and benchmarks for adaptive sample sharing between related regression and classification tasks. The repository accompanies [When to Transfer: Adaptive Source Selection for Positive Transfer in Linear Models](https://arxiv.org/abs/2510.16986).

## Install

Python 3.11 or 3.12 is recommended. From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
bash dyntr/datasets/extract_data.sh
```

The regression embedding caches used by the default benchmark are included in `dyntr/datasets/data/`. They record `DeepChem/ChemBERTa-100M-MLM` as the embedding model; the repository does not regenerate these two caches from raw molecules. The small tabular datasets are in `dyntr/datasets/data.zip`; the extraction command adds them to the same directory. Set `DYNTR_DATA_DIR` to another dataset directory if needed.

## Run an example locally

```bash
make smoke OUT=runs/smoke
```

This computes and plots one synthetic regression experiment using the small `config-blank.yml` settings. Results are saved in `runs/smoke`; plots go to `_figures` by default. Run `make check` for tests and lint.

To choose an experiment, run `make list`, then `make smoke EXP=bench_09_perf_classif_synth OUT=runs/classification`. Each experiment's default arguments are listed by `make info EXP=...`.

## Run the full benchmark

The full config uses many Monte Carlo runs and is intended for a SLURM cluster. Install the package in an environment visible to compute nodes, then preview and submit the configured experiment variants:

```bash
make send DRY=1 OUT=runs/full
make send OUT=runs/full
make plot OUT=runs/full
```

`make plot` runs locally after the jobs finish. For cluster options, alternate configs, and plotting controls, see [benchmark/README.md](benchmark/README.md).

## Prepare other datasets

Dataset preparation is optional and has separate dependencies:

```bash
python -m pip install -r requirements-datasets.txt
python scripts/prepare_embedding_regression_dataset.py --help
```

The resulting `.npz` file belongs in `$DYNTR_DATA_DIR/<dataset_name>/<dataset_name>.npz` and must contain `X_` and `y_` arrays for `target_train`, `source_train`, `target_test`, and `source_test`.

## Citation and license

If you use the code, cite Cherkaoui, Halconruy, and Petetin, *When to Transfer: Adaptive Source Selection for Positive Transfer in Linear Models*, arXiv:2510.16986 (2025). The code is under the [MIT License](LICENSE).
