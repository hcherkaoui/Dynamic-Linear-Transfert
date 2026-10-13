#!/usr/bin/env bash
set -euo pipefail

#===============================================================================
# Script to manage transfers with Jean Zay.
# Usage:
#   ./transfer.sh send        # Send code to Jean Zay
#   ./transfer.sh fetch       # Fetch benchmark/_figures from Jean Zay
#   ./transfer.sh supp        # Build code.zip supplementary package
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}"
WORK_DIR="/tmp/lin_dyn_transfert_jz_send"
SUPP_DIR="/tmp/lin_dyn_transfert_supp"
REMOTE_ROOT="jz:/lustre/fswork/projects/rech/jcx/uor49lv/src"
REMOTE_PROJECT="${REMOTE_ROOT}/lin-dyn-transfert"
REMOTE_FIGURES="${REMOTE_PROJECT}/benchmark/_figures"
LOCAL_BENCH_DIR="${SRC_DIR}/benchmark"
ZIP_NAME="code.zip"
MODE="${1:-}"
SCRIPT_NAME="$(basename "$0")"

if [[ -t 1 ]]; then
    BLUE="\033[94m"
    CYAN="\033[96m"
    GREEN="\033[92m"
    YELLOW="\033[93m"
    RED="\033[91m"
    BOLD="\033[1m"
    RESET="\033[0m"
else
    BLUE=""; CYAN=""; GREEN=""; YELLOW=""; RED=""; BOLD=""; RESET=""
fi

if [[ ! -d "${SRC_DIR}" ]]; then
    echo -e "${RED}Source directory does not exist:${RESET} ${SRC_DIR}" >&2
    exit 1
fi

if [[ -z "${MODE}" ]]; then
    echo -e "${RED}Missing option.${RESET} Choose one of: send, fetch, supp." >&2
    echo "Usage: ${SCRIPT_NAME} [send|fetch|supp]" >&2
    exit 2
fi

ensure_safe_dir() {
    local dir="$1"
    if [[ -z "${dir}" || "${dir}" == "/" || "${dir}" == "${HOME}" ]]; then
        echo -e "${RED}Unsafe directory value:${RESET} ${dir}" >&2
        exit 1
    fi
}

prepare_common_staging() {
    local target_dir="$1"
    ensure_safe_dir "${target_dir}"
    rm -rf "${target_dir}"
    cp -a "${SRC_DIR}" "${target_dir}"
    cd "${target_dir}"

    rm -rf .coveragerc .coverage .codecov.yml .flake8
    rm -rf .git .github .gitignore .pytest_cache .claude dyntr.egg-info
    rm -rf transfer.sh badges MANIFEST.in

    cd benchmark
    rm -rf _figures
    rm -rf _results
    cd ..

    if find . -type l -print -quit | grep -q .; then
        echo -e "${YELLOW}  - Removing symlinks from staging copy${RESET}"
        find . -type l -print -delete
    fi
    find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*~" \) -delete
    find . -type d -name "__pycache__" -prune -exec rm -rf {} +
}

send_code() {
    echo -e "${BLUE}${BOLD}Send to Jean Zay${RESET}"
    echo -e "${BLUE}Source:${RESET} ${SRC_DIR}"
    echo -e "${BLUE}Staging:${RESET} ${WORK_DIR}"
    echo -e "${BLUE}Remote:${RESET} ${REMOTE_ROOT}/"
    echo ""

    echo -e "${CYAN}[1/3] Preparing code...${RESET}"
    prepare_common_staging "${WORK_DIR}"

    echo -e "${CYAN}[2/3] Sending code...${RESET}"
    rsync -avh --info=stats2,progress2 "${WORK_DIR}" "${REMOTE_ROOT}/"

    echo -e "${CYAN}[3/3] Cleanup...${RESET}"
    rm -rf "${WORK_DIR}"

    echo -e "${GREEN}${BOLD}Done.${RESET}"
}

fetch_figures() {
    echo -e "${BLUE}${BOLD}Fetch figures from Jean Zay${RESET}"
    echo -e "${BLUE}Remote figures:${RESET} ${REMOTE_FIGURES}"
    echo -e "${BLUE}Local target:${RESET} ${LOCAL_BENCH_DIR}"
    echo ""

    mkdir -p "${LOCAL_BENCH_DIR}"
    (
        cd "${LOCAL_BENCH_DIR}"
        rsync -var --progress "${REMOTE_FIGURES}" .
    )

    echo -e "${GREEN}${BOLD}Figures synced.${RESET}"
}

build_supp_zip() {
    echo -e "${BLUE}${BOLD}Build supplementary package${RESET}"
    echo -e "${BLUE}Source:${RESET} ${SRC_DIR}"
    echo -e "${BLUE}Staging:${RESET} ${SUPP_DIR}"
    echo -e "${BLUE}Output:${RESET} ${SRC_DIR}/${ZIP_NAME}"
    echo ""

    echo -e "${CYAN}[1/3] Preparing package...${RESET}"
    prepare_common_staging "${SUPP_DIR}"
    cd "${SUPP_DIR}"
    rm -rf benchmark/_results benchmark/_tables benchmark/_reports

    cat > README.md <<'EOF'
# Anonymous Code Supplement

This archive contains code and benchmark scripts for anonymous peer review.
EOF

    cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "anonymous_package"
version = "0.0.0"
description = "Anonymous benchmark package"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }

[tool.setuptools]
packages = ["dyntr", "dyntr.datasets", "benchmark"]

[tool.setuptools.package-data]
"dyntr.datasets" = ["data.zip", "extract_data.sh", "data/**/*"]
EOF

    cat > benchmark/README.md <<'EOF'
# Benchmark Usage

## 1) Install
```bash
bash setup.sh
```

## 2) Submit benchmark jobs
```bash
bash launcher.sh --run
```
EOF

    cat > LICENSE <<'EOF'
MIT License
EOF

    # Fail fast if identity-bearing markers are still present in text files.
    local leaked
    leaked="$(
        grep -R -n -E \
        'Hamza|Cherkaoui|telecom-sudparis|hcherkaoui|uor49lv|github.com/hcherkaoui|lin-dyn-transfert|Dynamic-Linear-Transfert|NIduUWTCXT' \
        --binary-files=without-match \
        --exclude-dir=.git \
        . || true
    )"
    if [[ -n "${leaked}" ]]; then
        echo -e "${RED}Anonymization check failed. Leaked markers found:${RESET}" >&2
        echo "${leaked}" >&2
        rm -rf "${SUPP_DIR}"
        exit 1
    fi

    echo -e "${CYAN}[2/3] Building ${ZIP_NAME}...${RESET}"
    rm -f "${SRC_DIR}/${ZIP_NAME}"
    (
        cd "${SUPP_DIR}"
        zip -qr "${SRC_DIR}/${ZIP_NAME}" .
    )

    echo -e "${CYAN}[3/3] Cleanup...${RESET}"
    rm -rf "${SUPP_DIR}"
    echo -e "${GREEN}${BOLD}Created:${RESET} ${SRC_DIR}/${ZIP_NAME}"
}

case "${MODE}" in
    send)
        send_code
        ;;
    fetch)
        fetch_figures
        ;;
    supp)
        build_supp_zip
        ;;
    *)
        echo "Usage: ${SCRIPT_NAME} [send|fetch|supp]" >&2
        exit 2
        ;;
esac
