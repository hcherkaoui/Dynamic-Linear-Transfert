[![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/hcherkaoui/Dynamic-Linear-Transfert/actions/workflows/ci.yml/badge.svg)](https://github.com/hcherkaoui/Dynamic-Linear-Transfert/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hcherkaoui/Dynamic-Linear-Transfert/graph/badge.svg)](https://codecov.io/gh/hcherkaoui/Dynamic-Linear-Transfert)
![maintenance-status](https://img.shields.io/badge/maintenance-passively--maintained-yellowgreen.svg)
![Lines of Code](badges/loc.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# DynTr

A collection of benchmarks for collaborative learning.

## Description:

**DynTR** is a research-oriented Python benchmark suite focused on **collaborative learning**, in which models enhance their predictive performance by leveraging training data from other related datasets to improve generalization.

## Install

Dependencies are managed externally (e.g., module environments on HPC).
Install the package without pulling dependencies:
```bash
git clone https://github.com/hcherkaoui/lin-dyn-transfert
cd lin-dyn-transfert
python -m pip install -e . --no-deps
```

## Testing

```bash
pytest --cov=dyntr --cov-report=term-missing --cov-report=xml --cov-fail-under=70
```

## Benchmarks

Jean Zay usage is documented in `benchmark/bench_00_usage_for_slurm.md`.

Optional:
- Enable Weights & Biases logging with `--wandb` (requires `wandb` auth). By default, logging is disabled.
- `pdftk` is required to merge PDFs into `_merge_figures/`.

Outputs are written to `_cache/`, `_figures/`, and `_merge_figures/` (all ignored by git).

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cherkaouitransfert2026,
  title={Adaptive Sample Sharing for Linear Regression},
  author={Cherkaoui, Hamza and Halconruy, Hélène and Petetin, Yohan},
  journal={arXiv preprint arXiv:2510.16986},
  year={2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
