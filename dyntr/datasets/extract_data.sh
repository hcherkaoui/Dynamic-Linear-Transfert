#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
zip_path="${ZIP_PATH_OVERRIDE:-${script_dir}/data.zip}"
out_dir="${OUT_DIR_OVERRIDE:-${DYNTR_DATA_DIR:-${script_dir}/data}}"

if [[ ! -f "$zip_path" ]]; then
  echo "[ERR] data.zip not found at ${zip_path}" >&2
  exit 1
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
missing = []

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
            missing.append(item.name)
            shutil.move(str(item), str(target))
    else:
        for item in tmpdir.iterdir():
            target = out_dir / item.name
            if target.exists():
                continue
            missing.append(item.name)
            shutil.move(str(item), str(target))

if missing:
    print(f"[INFO] Extracted missing data entries to {out_dir}: {missing}")
else:
    print(f"[INFO] data.zip entries already available at {out_dir}")
PY
