#!/usr/bin/env bash
# mak_supp_code.sh
# Create a clean "conference code" appendix from a Python package.
# Usage:
#   bash make_conference_code.sh /path/to/src [--dest "$HOME/code"] [--force] \
#        [--redact "hamza,cherkaoui,telecom"] [--replace ""] [--extra-exclude "docs/,*.ipynb"]
#
# Notes:
# - By default, redaction terms are: hamza,cherkaoui,telecom
# - By default, strings are REMOVED (replacement is empty); set --replace "REDACTED" to mask
# - Requires: rsync, grep, sed, tar, zip. (sha256sum or shasum optional)

set -euo pipefail

# ---------- defaults ----------
SRC_DIR=""
DEST_DIR="${HOME}/code"
FORCE_OVERWRITE=0
REDACT="hamza,cherkaoui,telecom-sudparis.eu"
REPLACEMENT="***"
EXTRA_EXCLUDES=""
# Common dev/build/cache junk to exclude
READONLY_EXCLUDES=(
  ".git/" ".github/" ".gitignore" ".gitattributes" ".gitmodules"
  ".DS_Store" "__pycache__/" "*.pyc" ".pytest_cache/" ".mypy_cache/" ".ruff_cache/"
  ".ipynb_checkpoints/" ".idea/" ".vscode/" ".venv/" "venv/" ".tox/"
  "dist/" "build/" "*.egg-info/" ".coverage" "coverage.xml" "htmlcov/"
  ".cache/" "node_modules/" "package-lock.json" "poetry.lock" ".python-version"
  "*.log" "*.tmp" "*.swp" "*.swo"
)

print_usage() {
  cat <<EOF
Usage:
  bash $0 /path/to/src [--dest "$HOME/code"] [--force] \\
       [--redact "hamza,cherkaoui,telecom"] [--replace ""] \\
       [--extra-exclude "docs/,*.ipynb"]

Options:
  --dest DIR           Destination folder (default: \$HOME/code).
  --force              Overwrite destination if it exists.
  --redact STRS        Comma-separated strings to remove (default: "$REDACT").
  --replace STR        Replacement text for redacted strings (default: empty = delete).
  --extra-exclude PATS Comma-separated additional rsync exclude patterns.

Examples:
  bash $0 ./foo
  bash $0 ./foo --dest "\$HOME/submission/code" --force --replace "REDACTED"
  bash $0 ./foo --redact "hamza,cherkaoui,telecom,companyX" --extra-exclude "docs/,*.ipynb"
EOF
}

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1"; exit 1; }; }

# ---------- parse args ----------
if [[ $# -lt 1 ]]; then
  print_usage; exit 1
fi

SRC_DIR="$1"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST_DIR="$2"; shift 2 ;;
    --force) FORCE_OVERWRITE=1; shift ;;
    --redact) REDACT="$2"; shift 2 ;;
    --replace) REPLACEMENT="$2"; shift 2 ;;
    --extra-exclude) EXTRA_EXCLUDES="$2"; shift 2 ;;
    -h|--help) print_usage; exit 0 ;;
    *) echo "Unknown arg: $1"; print_usage; exit 1 ;;
  esac
done

# ---------- checks ----------
need rsync
need grep
need sed
need tar
need zip
[[ -d "$SRC_DIR" ]] || { echo "Source directory not found: $SRC_DIR"; exit 1; }

if [[ -e "$DEST_DIR" ]]; then
  if [[ "$FORCE_OVERWRITE" -eq 1 ]]; then
    rm -rf "$DEST_DIR"
  else
    echo "Destination exists: $DEST_DIR (use --force to overwrite)"; exit 1
  fi
fi

mkdir -p "$(dirname "$DEST_DIR")"

# ---------- build rsync exclude list ----------
RSYNC_EXCLUDES=()
for pat in "${READONLY_EXCLUDES[@]}"; do
  RSYNC_EXCLUDES+=(--exclude "$pat")
done

# user extra excludes
IFS=',' read -r -a EXTRA_ARR <<< "${EXTRA_EXCLUDES}"
for pat in "${EXTRA_ARR[@]}"; do
  [[ -n "$pat" ]] && RSYNC_EXCLUDES+=(--exclude "$pat")
done

# ---------- copy with rsync ----------
echo "[1/5] Copying to $DEST_DIR (excluding common dev files)..."
rsync -a "${RSYNC_EXCLUDES[@]}" "$SRC_DIR/" "$DEST_DIR/"

# ---------- safety cleanup (in case some slipped through) ----------
echo "[2/5] Pruning common dev/build/cache artifacts..."
# dirs
PRUNE_DIRS=( ".git" ".github" "__pycache__" ".pytest_cache" ".mypy_cache" ".ruff_cache"
             ".ipynb_checkpoints" ".idea" ".vscode" ".venv" "venv" ".tox" "dist" "build"
             "*.egg-info" "htmlcov" ".cache" "node_modules" "tests" "_figures" "figures"
             "_cache" "_merge_figures")
for d in "${PRUNE_DIRS[@]}"; do
  find "$DEST_DIR" -type d -name "$d" -prune -exec rm -rf {} + 2>/dev/null || true
done
# files
PRUNE_FILES=( ".gitignore" ".gitattributes" ".gitmodules" ".DS_Store" "*.pyc" "MANIFEST.in" "make_supp_code.sh"
              ".coverage" "coverage.xml" "package-lock.json" "poetry.lock" "*.log" "*.tmp" "*.swp" "*.swo" )
for f in "${PRUNE_FILES[@]}"; do
  find "$DEST_DIR" -type f -name "$f" -exec rm -f {} + 2>/dev/null || true
done

# ---------- redact strings in text files ----------
echo "[3/5] Redacting strings in text files..."
IFS=',' read -r -a TERMS <<< "$REDACT"

# Build a sed script to apply all replacements in one pass.
SED_SCRIPT=""
for term in "${TERMS[@]}"; do
  t="$(printf '%s' "$term" | sed -e 's/[\/&]/\\&/g')"
  r="$(printf '%s' "$REPLACEMENT" | sed -e 's/[\/&]/\\&/g')"
  # Case-insensitive removal/replacement
  SED_SCRIPT+="s/${t}/${r}/Ig;"
done

if [[ -n "$SED_SCRIPT" ]]; then
  # Find likely text files (grep -Il skips binaries). Exclude big/vendor dirs just in case.
  mapfile -t TEXT_FILES < <(grep -Ilr . "$DEST_DIR" \
    --exclude-dir=.git --exclude-dir=.github --exclude-dir=__pycache__ \
    --exclude-dir=.pytest_cache --exclude-dir=.mypy_cache --exclude-dir=.ruff_cache \
    --exclude-dir=.ipynb_checkpoints --exclude-dir=node_modules --exclude-dir=.venv \
    --exclude-dir=venv --exclude-dir=dist --exclude-dir=build 2>/dev/null || true)

  if (( ${#TEXT_FILES[@]} )); then
    # Use -i with backup then remove backups (portable across sed variants)
    printf "%s\0" "${TEXT_FILES[@]}" | xargs -0 sed -i.bak -E "$SED_SCRIPT" || true
    find "$DEST_DIR" -type f -name "*.bak" -delete || true
  fi
fi

# ---------- verify size and list ----------
echo "[4/5] Summary of cleaned tree:"
du -sh "$DEST_DIR" || true
# show top-level contents
find "$DEST_DIR" -maxdepth 2 -mindepth 1 -print | head -n 50 || true

# ---------- compress ----------
PARENT="$(dirname "$DEST_DIR")"
BASENAME="$(basename "$DEST_DIR")"
TGZ="${PARENT}/${BASENAME}.tar.gz"
ZIP="${PARENT}/${BASENAME}.zip"

echo "[5/5] Creating archives:"
tar -C "$PARENT" -czf "$TGZ" "$BASENAME"
( cd "$PARENT" && zip -qr "$ZIP" "$BASENAME" )

# checksums (best-effort)
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$TGZ" "$ZIP"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$TGZ" "$ZIP"
fi

echo
echo "Done."
echo "Clean copy:     $DEST_DIR"
echo "Archive (tgz):  $TGZ"
echo "Archive (zip):  $ZIP"
