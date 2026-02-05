#!/usr/bin/env bash

set -euo pipefail

# 1) pip deps (user install)
python -m pip install --user --upgrade --no-cache-dir pip setuptools wheel
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"

python -m pip install --user --upgrade --no-cache-dir -e "$root_dir[bench]"

python -c "import dyntr, joblib, matplotlib, numpy, numba, sklearn, pandas; print('bench deps OK')"

echo "[done] dyntr installed in user site-packages."
