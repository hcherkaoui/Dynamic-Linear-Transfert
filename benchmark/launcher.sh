#!/usr/bin/env bash
set -euo pipefail

# Submit DynTR benchmark computations to SLURM.

usage() {
  cat <<'USAGE'
Usage:
  bash launcher.sh --run [--results-dir DIR] [submit options]
  bash launcher.sh --blank [--results-dir DIR] [submit options]

This script submits computation jobs only. It never computes locally.
Common submit options include --dry-run, --seed, --time, --mem,
--cpus-per-task, --partition, --account, and --gres.
USAGE
}

config_name=""
config_label=""
results_dir=""
submit_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) config_name="config.yml"; config_label="run"; shift ;;
    --blank) config_name="config-blank.yml"; config_label="blank"; shift ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) submit_args+=("$1"); shift ;;
  esac
done

[[ -n "$config_name" ]] || { usage; exit 1; }

# shellcheck source=_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

t_start="$(date +%s)"
setup_benchmark_env "$config_name"

if [[ -z "$results_dir" ]]; then
  results_dir="${DYNTR_RESULTS_DIR:-${benchmark_dir}/_results/${config_label}}"
fi
log_dir="${DYNTR_SLURM_LOG_DIR:-${benchmark_dir}/runs/slurm_logs}"
script_dir="${DYNTR_SLURM_SCRIPT_DIR:-${benchmark_dir}/runs/slurm_scripts}"
log "results_dir    = $results_dir"
log "log_dir        = $log_dir"
log "script_dir     = $script_dir"
log "submit_args    = ${submit_args[*]:-<none>}"

run_py -m benchmark_tasks submit \
  --config "$DYNTR_CONFIG" \
  --output-dir "$results_dir" \
  --log-dir "$log_dir" \
  --script-dir "$script_dir" \
  "${submit_args[@]}"

total=$(( $(date +%s) - t_start ))
log "Benchmark submissions completed in $(duration "$total")."
