#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# DynTR setup (login node)
# - Creates/updates the venv in $WORK/venvs/dyntr
# - Installs dependencies + editable install of the repo
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
  "$@"
  echo -e "${GREEN}${BOLD}[$(ts)] [DONE]${RESET} ${title}"
  echo
}

# --- Guards ------------------------------------------------------------------
if [[ -z "${WORK:-}" ]]; then
  WORK="$HOME"
  warn "WORK not set; defaulting to HOME: $WORK"
fi

PYTHON_BIN="$(command -v python3.11 || command -v python3 || command -v python || true)"
[[ -n "$PYTHON_BIN" ]] || die "No python executable found (tried python3.11, python3, python)."

"$PYTHON_BIN" -V >/dev/null 2>&1 || die "Python not usable from PATH."

# --- Paths -------------------------------------------------------------------
venv_dir="$WORK/venvs/dyntr"

# This script is assumed to live in benchmark/, so repo root is ..
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

info "WORK      = $WORK"
info "Repo root = $repo_root"
info "Venv dir  = $venv_dir"
info "Python    = $PYTHON_BIN"
echo

# --- Create/activate venv ----------------------------------------------------
if [[ -d "$venv_dir" ]]; then
  warn "Venv already exists: $venv_dir (reusing it)"
else
  run "Create venv" "$PYTHON_BIN" -m venv "$venv_dir"
fi

# shellcheck disable=SC1090
source "$venv_dir/bin/activate"

VENV_PYTHON="$(command -v python)"
[[ -n "$VENV_PYTHON" ]] || die "Activated venv but could not resolve python."

info "Venv python = $VENV_PYTHON"

run "Upgrade pip/setuptools/wheel" "$VENV_PYTHON" -m pip install -U pip setuptools wheel

# --- Install deps + editable package ----------------------------------------
run "Install requirements" "$VENV_PYTHON" -m pip install -r "$repo_root/requirements.txt"
run "Editable install (pip -e .)" "$VENV_PYTHON" -m pip install -e "$repo_root"
run "Ensure joblib" "$VENV_PYTHON" -m pip install -U joblib

# --- Pre-download datasets ---------------------------------------------------
run "Extract bundled datasets" bash "$repo_root/dyntr/datasets/extract_data.sh"
run "Download California Housing dataset" "$VENV_PYTHON" -c "from sklearn.datasets import fetch_california_housing; fetch_california_housing(); print('california_housing: ok')"

# --- Sanity check ------------------------------------------------------------
run "Sanity import" "$VENV_PYTHON" -c "import dyntr, benchmark; print('dyntr:', dyntr.__file__)"
info "${GREEN}${BOLD}Setup completed.${RESET}"
