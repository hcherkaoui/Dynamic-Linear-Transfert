"""Pytest configuration for dyntr tests."""
from __future__ import annotations
import os
import numba


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
numba.config.DISABLE_JIT = 1
