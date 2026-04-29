#!/usr/bin/env bash
set -euo pipefail

# Launch the DynTR benchmark suite using benchmark/configs/*.yml.

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERR ] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage:
  bash launcher.sh --run
  bash launcher.sh --blank
USAGE
}

duration() {
  local total="$1"
  printf '%02dh %02dm %02ds' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

case "${1:-}" in
  --run) config_name="config.yml" ;;
  --blank) config_name="config-blank.yml" ;;
  --help|-h) usage; exit 0 ;;
  *) usage; exit 1 ;;
esac

t_start="$(date +%s)"
benchmark_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${benchmark_dir}/.." && pwd)"
cd "$benchmark_dir"

cache_root="${DYNTR_CACHE_ROOT:-${WORK:-${repo_root}/.dyntr_cache}}"
if [[ -z "${DYNTR_CACHE_ROOT:-}" && -z "${WORK:-}" ]]; then
  warn "WORK is not set; using ${cache_root}"
fi

venv_activate="${cache_root}/venvs/dyntr/bin/activate"
if [[ -f "$venv_activate" ]]; then
  # shellcheck disable=SC1090
  source "$venv_activate"
else
  warn "Venv not found at $venv_activate; using current Python environment."
fi

export DYNTR_CONFIG="${benchmark_dir}/configs/${config_name}"
export DYNTR_DATA_DIR="${DYNTR_DATA_DIR:-${repo_root}/dyntr/datasets/data}"
export XDG_CACHE_HOME="${cache_root}/.cache"
export SKLEARN_DATA="${cache_root}/sklearn_data"
export SCIKIT_LEARN_DATA="$SKLEARN_DATA"
mkdir -p "$DYNTR_DATA_DIR" "$SKLEARN_DATA"

python_bin="$(command -v python || command -v python3 || true)"
[[ -n "$python_bin" ]] || die "No Python executable found."

log "config         = $DYNTR_CONFIG"
log "cache_root     = $cache_root"
log "DYNTR_DATA_DIR = $DYNTR_DATA_DIR"
log "python         = $python_bin"

yaml_list() {
  "$python_bin" - "$DYNTR_CONFIG" "$1" <<'PY'
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
value = cfg
for part in sys.argv[2].split("."):
    value = value.get(part) if isinstance(value, dict) else None
if isinstance(value, list):
    print(" ".join(str(item) for item in value))
PY
}

run_py() {
  log "python $*"
  "$python_bin" "$@"
}

run_eps() {
  local eps_values="$1"; shift
  local eps script

  for eps in $eps_values; do
    for script in "$@"; do
      run_py "$script" --eps "$eps"
    done
  done
}

datasets="$(yaml_list launcher.l_datasets)"
eps_classif_mono="$(yaml_list launcher.l_eps_classif_mono)"
eps_classif_multi="$(yaml_list launcher.l_eps_classif_multi)"
eps_regr_mono="$(yaml_list launcher.l_eps_regr_mono)"
eps_regr_multi="$(yaml_list launcher.l_eps_regr_multi)"

[[ -n "$datasets" ]] || die "No datasets configured in launcher.l_datasets."
log "datasets       = $datasets"
log "classif mono   = $eps_classif_mono"
log "classif multi  = $eps_classif_multi"
log "regr mono      = $eps_regr_mono"
log "regr multi     = $eps_regr_multi"

for dataset in $datasets; do
  case "$dataset" in
    lipophilicity_chemberta|aqsoldb_chemberta)
      npz_path="${DYNTR_DATA_DIR}/${dataset}/${dataset}.npz"
      [[ -f "$npz_path" ]] || die "Missing precomputed embedding file: $npz_path"
      ;;
  esac
done

run_eps "$eps_classif_mono" \
  bench_11_delta_classif_illus_synth.py \
  bench_09_perf_classif_synth.py

run_eps "$eps_classif_multi" \
  bench_10_perf_classif_multi_synth.py

run_eps "$eps_regr_mono" \
  bench_01_delta_regr_illus_synth.py \
  bench_02_sigma_infl_synth.py \
  bench_04_perf_regr_synth.py

run_eps "$eps_regr_multi" \
  bench_08_perf_regr_multi_synth.py

run_py bench_03_eps_infl_synth.py
run_py bench_06_lbda_infl_synth.py
run_py bench_07_intro_regr_illus.py

for dataset in $datasets; do
  run_py bench_05_perf_regr_real.py --dataset-name "$dataset"
done

total=$(( $(date +%s) - t_start ))
log "All benchmarks completed in $(duration "$total")."
