#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${WORK:-}" ]]; then
    echo "[error] WORK is not set; refusing to create venv under \$HOME." >&2
    exit 1
fi

venv_dir="${WORK}/venvs/dyntr"
if [[ ! -d "$venv_dir" ]]; then
    python -m venv "$venv_dir"
fi
py="$venv_dir/bin/python"

export PYTHONNOUSERSITE=1
unset PYTHONPATH

export SKLEARN_DATA="${WORK}/sklearn_data"
export XDG_CACHE_HOME="${WORK}/.cache"
mkdir -p "$SKLEARN_DATA" "$XDG_CACHE_HOME"

# Install dyntr only; dependencies are managed externally
"$py" -m pip install --upgrade --no-cache-dir pip setuptools wheel
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"

"$py" -m pip uninstall -y dyntr >/dev/null 2>&1 || true
# Force legacy setup.py install to avoid editable/PEP517 edge cases on HPC
PIP_USE_PEP517=0 "$py" -m pip install --no-cache-dir --no-deps "$root_dir"

"$py" -m pip show dyntr || { echo "[error] dyntr not found in venv." >&2; exit 1; }
"$py" -c "import sys; print('python:', sys.executable)"
"$py" -c "import dyntr; print('dyntr path:', dyntr.__file__)"

echo "[done] dyntr installed in $venv_dir"
