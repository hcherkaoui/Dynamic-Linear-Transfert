#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
zip_path="${ZIP_PATH_OVERRIDE:-${script_dir}/data.zip}"
out_dir="${OUT_DIR_OVERRIDE:-${script_dir}/data}"

if [[ ! -f "$zip_path" ]]; then
  echo "[ERR] data.zip not found at ${zip_path}" >&2
  exit 1
fi

if [[ -d "$out_dir" ]] && [[ -n "$(ls -A "$out_dir" 2>/dev/null)" ]]; then
  echo "[INFO] data already extracted at ${out_dir}"
  exit 0
fi

mkdir -p "$out_dir"

ZIP_PATH="$zip_path" OUT_DIR="$out_dir" python - <<'PY'
from pathlib import Path
import os
import shutil
import tempfile
import zipfile

zip_path = Path(os.environ["ZIP_PATH"])
out_dir = Path(os.environ["OUT_DIR"])

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    nested = tmpdir / "data"
    if nested.is_dir():
        for item in nested.iterdir():
            target = out_dir / item.name
            if target.exists():
                continue
            shutil.move(str(item), str(target))
    else:
        for item in tmpdir.iterdir():
            target = out_dir / item.name
            if target.exists():
                continue
            shutil.move(str(item), str(target))

print(f"[INFO] Extracted data.zip to {out_dir}")
PY
