#!/usr/bin/env bash
#===============================================================================
# Script to send the current version of lin-dyn-transfert to Jean Zay
# Usage:
#   ./send2jz.sh              # Send code
#===============================================================================

SRC_DIR="${HOME}/src/lin-dyn-transfert"
WORK_DIR="${HOME}/lin-dyn-transfert"
REMOTE="jz:/lustre/fswork/projects/rech/jcx/uor49lv/src/"

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

echo -e "${BLUE}${BOLD}Send to Jean Zay${RESET}"
echo -e "${BLUE}Source:${RESET} ${SRC_DIR}"
echo -e "${BLUE}Staging:${RESET} ${WORK_DIR}"
echo -e "${BLUE}Remote:${RESET} ${REMOTE}"
echo ""

echo -e "${CYAN}[1/4] Preparing code...${RESET}"
rm -rf "${WORK_DIR}"
cp -a "${SRC_DIR}" "${WORK_DIR}"

echo -e "${CYAN}[2/4] Cleaning code...${RESET}"
cd "${WORK_DIR}"
rm -rf _results _figures .coveragerc .codecov.yml .flake8
rm -rf .claude .git .github .gitignore .pytest_cache dyntr.egg-info send2jz.sh
rm -rf LICENCE MANIFEST.in badges make_supp_code.sh
if find . -type l -print -quit | grep -q .; then
    echo -e "${YELLOW}  - Removing symlinks from staging copy${RESET}"
    find . -type l -print -delete
fi
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*~" \) -delete
find . -type d -name "__pycache__" -prune -exec rm -rf {} +

echo -e "${CYAN}[3/4] Sending code...${RESET}"
rsync -avh --info=stats2,progress2 "${WORK_DIR}" "${REMOTE}"

echo -e "${CYAN}[4/4] Cleanup...${RESET}"
rm -rf "${WORK_DIR}"
echo -e "${GREEN}${BOLD}Done.${RESET}"
