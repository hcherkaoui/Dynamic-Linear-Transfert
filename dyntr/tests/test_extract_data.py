"""Smoke test for dataset extraction."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_extract_data_smoke(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "datasets" / "extract_data.sh"
    zip_path = Path(__file__).resolve().parents[1] / "datasets" / "data.zip"
    assert script.exists()
    assert zip_path.exists()

    target = tmp_path / "data"
    cmd = ["bash", str(script)]
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["ZIP_PATH_OVERRIDE"] = str(zip_path)
    env["OUT_DIR_OVERRIDE"] = str(target)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert (target / "email" / "Hillstrom.csv").exists()
