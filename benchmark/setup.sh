#!/usr/bin/env bash
set -euo pipefail

# Prepare the DynTR benchmark environment.
# By default, large caches and the venv live under $WORK. If WORK is not set,
# they fall back to a repo-local .dyntr_cache directory.

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERR ] %s\n' "$*" >&2; exit 1; }

run() {
  local title="$1"; shift
  log "$title"
  "$@"
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${DYNTR_CACHE_ROOT:-}" ]]; then
  cache_root="$DYNTR_CACHE_ROOT"
elif [[ -n "${WORK:-}" ]]; then
  cache_root="$WORK"
else
  cache_root="${repo_root}/.dyntr_cache"
  warn "WORK is not set; using ${cache_root}"
fi

python_bin="$(command -v python3.11 || command -v python3 || command -v python || true)"
[[ -n "$python_bin" ]] || die "No Python executable found."

venv_dir="${cache_root}/venvs/dyntr"
export PIP_CACHE_DIR="${cache_root}/.cache/pip"
export XDG_CACHE_HOME="${cache_root}/.cache"
export SKLEARN_DATA="${cache_root}/sklearn_data"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA"
export DYNTR_DATA_DIR="${DYNTR_DATA_DIR:-${repo_root}/dyntr/datasets/data}"

mkdir -p "$PIP_CACHE_DIR" "$SKLEARN_DATA" "$DYNTR_DATA_DIR"

log "repo_root      = $repo_root"
log "cache_root     = $cache_root"
log "venv_dir       = $venv_dir"
log "DYNTR_DATA_DIR = $DYNTR_DATA_DIR"

if [[ ! -d "$venv_dir" ]]; then
  run "Create virtual environment" "$python_bin" -m venv "$venv_dir"
else
  log "Reusing virtual environment"
fi

# shellcheck disable=SC1091
source "$venv_dir/bin/activate"
venv_python="$(command -v python)"

run "Install DynTR editable package" "$venv_python" -m pip install -e "$repo_root"

run "Extract bundled tabular datasets" env OUT_DIR_OVERRIDE="$DYNTR_DATA_DIR" \
  bash "$repo_root/dyntr/datasets/extract_data.sh"

for dataset in lipophilicity_chemberta aqsoldb_chemberta; do
  npz_path="${DYNTR_DATA_DIR}/${dataset}/${dataset}.npz"
  [[ -f "$npz_path" ]] || die "Missing precomputed embedding file: $npz_path"
  log "found $npz_path"
done

run "Sanity import" "$venv_python" -c "import dyntr, benchmark; print('dyntr:', dyntr.__file__)"
log "Setup completed."
