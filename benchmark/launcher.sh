#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# DynTR benchmark launcher (Jean Zay-friendly)
# - Assumes the venv already exists at: $WORK/venvs/dyntr
# - Uses $WORK for scikit-learn dataset cache
# - Keeps your python3.11 invocations unchanged
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

# --- Guards ------------------------------------------------------------------
: "${WORK:?WORK is not set. Example: export WORK=/lustre/fswork/projects/rech/<project>/<login>}"

[[ -f "$WORK/venvs/dyntr/bin/activate" ]] || die "Missing venv: $WORK/venvs/dyntr (activate not found)."

# --- Environment -------------------------------------------------------------
# shellcheck disable=SC1090
source "$WORK/venvs/dyntr/bin/activate"

export SKLEARN_DATA="${WORK}/sklearn_data"
mkdir -p "$SKLEARN_DATA"

info "WORK           = $WORK"
info "PWD            = $(pwd)"
info "VENV           = $WORK/venvs/dyntr"
info "SKLEARN_DATA   = $SKLEARN_DATA"
info "python3.11     = $(command -v python3.11 || echo 'NOT FOUND')"
python3.11 -V || die "python3.11 is not available in PATH."

echo

# --- Parameters --------------------------------------------------------------
eps_list=(1.0 0.5 0.1)
alpha=0.05
n_t=1000
n_s=2000
joblib_verbose=10
datasets=(email boston)

info "Parameters: eps_list=(${eps_list[*]}), alpha=${alpha}, n_t=${n_t}, n_s=${n_s}, joblib_verbose=${joblib_verbose}"
info "Datasets:   (${datasets[*]})"
echo

# --- Benchmark runs -----------------------------------------------------------
for eps in "${eps_list[@]}"; do
  run "bench_1 (synthetic) eps=${eps}" \
    python3.11 bench_1_delta_illustration_synthetic_data.py \
      --eps "$eps" -n_t "$n_t" -n_s "$n_s" --joblib-verbose "$joblib_verbose"

  run "bench_2 (synthetic) eps=${eps}" \
    python3.11 bench_2_sigma_influence_synthetic_data.py \
      --eps "$eps" --alpha "$alpha" --joblib-verbose "$joblib_verbose"

  run "bench_4 (synthetic) eps=${eps}" \
    python3.11 bench_4_performance_synthetic_data.py \
      --eps "$eps" -n_t "$n_t" -n_s "$n_s" --alpha "$alpha" --joblib-verbose "$joblib_verbose"
done

run "bench_3 (synthetic) epsilon influence" \
  python3.11 bench_3_epsilon_influence_synthetic_data.py \
    --alpha "$alpha" --joblib-verbose "$joblib_verbose"

run "bench_6 (synthetic) lambda_S influence" \
  python3.11 bench_6_add_lbda_influence_synthetic_data.py \
    --alpha "$alpha" --joblib-verbose "$joblib_verbose"

run "bench_7 (synthetic) intro illustration" \
  python3.11 bench_7_intro_illustration.py \
    --eps 0.5 -n_t 1500 -n_s 4000 --alpha "$alpha" --joblib-verbose "$joblib_verbose"

for ds in "${datasets[@]}"; do
  run "bench_5 (real) dataset=${ds}" \
    python3.11 bench_5_performance_real_data.py \
      --dataset-name "$ds" -n_t "$n_t" -n_s "$n_s" --alpha "$alpha" --joblib-verbose "$joblib_verbose"
done

info "${GREEN}${BOLD}All benchmarks completed.${RESET}"
