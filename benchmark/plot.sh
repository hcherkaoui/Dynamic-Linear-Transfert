#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash plot.sh --run|--blank [--results-dir DIR] [plot options]"
}

config_name=""
config_label=""
results_dir=""
plot_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) config_name="config.yml"; config_label="run"; shift ;;
    --blank) config_name="config-blank.yml"; config_label="blank"; shift ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) plot_args+=("$1"); shift ;;
  esac
done

[[ -n "$config_name" ]] || { usage; exit 2; }

# shellcheck source=_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
setup_benchmark_env "$config_name"
results_dir="${results_dir:-${DYNTR_RESULTS_DIR:-${benchmark_dir}/_results/${config_label}}}"
run_py -m benchmark_tasks plot --config "$DYNTR_CONFIG" --output-dir "$results_dir" "${plot_args[@]}"
