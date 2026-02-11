"""Pytest configuration for dyntr tests."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import numba
import pytest


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
numba.config.DISABLE_JIT = 1


@pytest.fixture(autouse=True, scope="session")
def _cleanup_figures_dir():
    yield
    roots = [
        Path.cwd() / "_figures",
        Path(__file__).resolve().parents[2] / "_figures",
    ]
    for root in roots:
        if root.exists() and root.is_dir():
            shutil.rmtree(root, ignore_errors=True)
