#!/usr/bin/env bash

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERR ] %s\n' "$*" >&2; exit 1; }

duration() {
  local total="$1"
  printf '%02dh %02dm %02ds' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

setup_benchmark_env() {
  local config_name="$1"

  benchmark_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${benchmark_dir}/.." && pwd)"
  cd "$benchmark_dir"

  cache_root="${DYNTR_CACHE_ROOT:-${WORK:-${repo_root}/.dyntr_cache}}"
  if [[ -z "${DYNTR_CACHE_ROOT:-}" && -z "${WORK:-}" ]]; then
    warn "WORK is not set; using ${cache_root}"
  fi

  local venv_activate="${cache_root}/venvs/dyntr/bin/activate"
  if [[ -f "$venv_activate" ]]; then
    # shellcheck disable=SC1090
    source "$venv_activate"
  else
    warn "Venv not found at $venv_activate; using current Python environment."
  fi

  export DYNTR_CONFIG="${benchmark_dir}/configs/${config_name}"
  export DYNTR_DATA_DIR="${DYNTR_DATA_DIR:-${repo_root}/dyntr/datasets/data}"
  export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
  export XDG_CACHE_HOME="${cache_root}/.cache"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${cache_root}/matplotlib}"
  export SKLEARN_DATA="${cache_root}/sklearn_data"
  export SCIKIT_LEARN_DATA="$SKLEARN_DATA"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${TMPDIR:-/tmp}}"
  mkdir -p "$DYNTR_DATA_DIR" "$SKLEARN_DATA" "$MPLCONFIGDIR" "$JOBLIB_TEMP_FOLDER"

  python_bin="$(command -v python || command -v python3 || true)"
  [[ -n "$python_bin" ]] || die "No Python executable found."

  log "config         = $DYNTR_CONFIG"
  log "cache_root     = $cache_root"
  log "DYNTR_DATA_DIR = $DYNTR_DATA_DIR"
  log "python         = $python_bin"
  log "PYTHONPATH     = $repo_root"
  log "BLAS threads   = OMP:$OMP_NUM_THREADS OPENBLAS:$OPENBLAS_NUM_THREADS MKL:$MKL_NUM_THREADS"
  log "joblib temp    = $JOBLIB_TEMP_FOLDER"
}

run_py() {
  log "python $*"
  "$python_bin" "$@"
}
