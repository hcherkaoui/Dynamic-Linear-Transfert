#!/usr/bin/env bash

set -euo pipefail

# 1) create isolated venv to avoid mixing user-site packages
venv_dir="${HOME}/venvs/dyntr"
python -m venv "$venv_dir"
# shellcheck disable=SC1090
source "$venv_dir/bin/activate"

# avoid user-site leakage
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# 2) pip deps
python -m pip install --upgrade --no-cache-dir pip setuptools wheel
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"

python -m pip install --no-cache-dir "numpy<2"
python -m pip install --upgrade --no-cache-dir -e "$root_dir[bench]"

python -c "import dyntr, joblib, matplotlib, numpy, numba, sklearn, pandas; print('bench deps OK')"

echo "[done] dyntr installed in user site-packages."
