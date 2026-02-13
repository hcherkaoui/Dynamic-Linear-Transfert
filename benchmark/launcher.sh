#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# DynTR benchmark launcher
# -----------------------------------------------------------------------------

# --- Colors (auto-disable if not a TTY) --------------------------------------
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold || true)"
  DIM="$(tput dim || true)"
  RESET="$(tput sgr0 || true)"
  RED="$(tput setaf 1 || true)"
  GREEN="$(tput setaf 2 || true)"
  YELLOW="$(tput setaf 3 || true)"
  BLUE="$(tput setaf 4 || true)"
else
  BOLD=""; DIM=""; RESET=""
  RED=""; GREEN=""; YELLOW=""; BLUE=""
fi

ts() { date +"%Y-%m-%d %H:%M:%S"; }
info() { echo -e "${BLUE}${BOLD}[$(ts)] [INFO]${RESET} $*"; }
warn() { echo -e "${YELLOW}${BOLD}[$(ts)] [WARN]${RESET} $*"; }
die()  { echo -e "${RED}${BOLD}[$(ts)] [ERR ]${RESET} $*" >&2; exit 1; }

fmt_dur() {
  local total="$1"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf "%02dh %02dm %02ds" "$h" "$m" "$s"
}

run() {
  local title="$1"; shift
  info "${BOLD}${title}${RESET}"
  echo -e "${DIM}cmd:${RESET} $*"
  local t0; t0="$(date +%s)"
  "$@"
  local t1; t1="$(date +%s)"
  echo -e "${GREEN}${BOLD}[$(ts)] [DONE]${RESET} ${title} (${t1}s - ${t0}s = $((t1 - t0))s)"
  echo
}

# --- CLI ---------------------------------------------------------------------
usage() {
  cat <<'USAGE'
DynTR benchmark launcher

Usage:
  bash launcher.sh --run    # use benchmark/configs/config.yml
  bash launcher.sh --blank  # use benchmark/configs/config-blank.yml
  bash launcher.sh --help
USAGE
}

mode=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) mode="run"; shift ;;
    --blank) mode="blank"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1. Use --help." ;;
  esac
done

[[ -n "$mode" ]] || { usage; exit 1; }

# Track total runtime
t_start="$(date +%s)"

# --- Guards ------------------------------------------------------------------
if [[ -z "${WORK:-}" ]]; then
  WORK="$HOME"
  warn "WORK not set; defaulting to HOME: $WORK"
fi

VENV_ACTIVATE="$WORK/venvs/dyntr/bin/activate"

# --- Environment -------------------------------------------------------------
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
else
  warn "Venv not found at $VENV_ACTIVATE; using current Python environment."
fi

export SKLEARN_DATA="${WORK}/sklearn_data"
mkdir -p "$SKLEARN_DATA"

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/configs"
if [[ "$mode" == "run" ]]; then
  export DYNTR_CONFIG="${CONFIG_DIR}/config.yml"
else
  export DYNTR_CONFIG="${CONFIG_DIR}/config-blank.yml"
fi

info "WORK           = $WORK"
info "PWD            = $(pwd)"
info "VENV           = ${VENV_ACTIVATE:-N/A}"
info "SKLEARN_DATA   = $SKLEARN_DATA"
info "DYNTR_CONFIG   = $DYNTR_CONFIG"
info "(The benchmark configuration file is determined with the env variable 'DYNTR_CONFIG', by default it is 'configs/config.yml')"

PYTHON_BIN="$(command -v python3.11 || command -v python3 || command -v python || true)"
[[ -n "$PYTHON_BIN" ]] || die "No python executable found (tried python3.11, python3, python)."
info "PYTHON_BIN     = $PYTHON_BIN"

# --- Parameters ---------------------------------------------------------------
yaml_val() {
  local key="$1"
  "$PYTHON_BIN" - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${DYNTR_CONFIG}").read_text()) or {}
cur = cfg
for part in "${key}".split("."):
    cur = cur.get(part, None) if isinstance(cur, dict) else None
    if cur is None:
        break
if isinstance(cur, list):
    print(" ".join(str(x) for x in cur))
elif cur is None:
    print("")
else:
    print(cur)
PY
}

read -r -a l_datasets <<<"$(yaml_val launcher.l_datasets)"
read -r -a l_eps_classif_mono <<<"$(yaml_val launcher.l_eps_classif_mono)"
read -r -a l_eps_classif_multi <<<"$(yaml_val launcher.l_eps_classif_multi)"
read -r -a l_eps_regr_mono <<<"$(yaml_val launcher.l_eps_regr_mono)"
read -r -a l_eps_regr_multi <<<"$(yaml_val launcher.l_eps_regr_multi)"

info "Parameters:"
info "  datasets         = ${l_datasets[*]}"
info "  classif mono eps = ${l_eps_classif_mono[*]}"
info "  classif multi eps= ${l_eps_classif_multi[*]}"
info "  regr mono eps    = ${l_eps_regr_mono[*]}"
info "  regr multi eps   = ${l_eps_regr_multi[*]}"
echo

# --- Classification benchmark runs --------------------------------------------
for eps in "${l_eps_classif_mono[@]}"; do
  run "bench_11 (synthetic / mono-source / classif) eps=${eps}" \
    "$PYTHON_BIN" bench_11_delta_classif_illus_synth.py \
      --eps "$eps"

  run "bench_09 (synthetic / mono-source / classif) eps=${eps}" \
    "$PYTHON_BIN" bench_09_perf_classif_synth.py \
      --eps "$eps"
done

for eps in "${l_eps_classif_multi[@]}"; do
  run "bench_10 (synthetic / multi-sources / classif)" \
    "$PYTHON_BIN" bench_10_perf_classif_multi_synth.py \
    --eps "$eps"
done

# --- Regression benchmark runs ------------------------------------------------
for eps in "${l_eps_regr_mono[@]}"; do
  run "bench_01 (synthetic / mono-source / regr) eps=${eps}" \
    "$PYTHON_BIN" bench_01_delta_regr_illus_synth.py \
      --eps "$eps"

  run "bench_02 (synthetic / mono-source / regr) eps=${eps}" \
    "$PYTHON_BIN" bench_02_sigma_infl_synth.py \
      --eps "$eps"

  run "bench_04 (synthetic / mono-source / regr) eps=${eps}" \
    "$PYTHON_BIN" bench_04_perf_regr_synth.py \
      --eps "$eps"
done

for eps in "${l_eps_regr_multi[@]}"; do
  run "bench_08 (synthetic / multi-sources / regr)" \
    "$PYTHON_BIN" bench_08_perf_regr_multi_synth.py \
    --eps "$eps"
done

# --- Additiional regression benchmark runs ------------------------------------
run "bench_03 (synthetic / mono-source / regr) epsilon influence" \
  "$PYTHON_BIN" bench_03_eps_infl_synth.py

run "bench_06 (synthetic / mono-source / regr) lambda_S influence" \
  "$PYTHON_BIN" bench_06_lbda_infl_synth.py

run "bench_07 (synthetic / mono-source / regr) intro illustration" \
  "$PYTHON_BIN" bench_07_intro_regr_illus.py

for ds in "${l_datasets[@]}"; do
  run "bench_05 (real / mono-source / regr) dataset=${ds}" \
    "$PYTHON_BIN" bench_05_perf_regr_real.py \
      --dataset-name "$ds"
done

total=$(( $(date +%s) - t_start ))
info "${GREEN}${BOLD}All benchmarks completed in $(fmt_dur "$total").${RESET}"
