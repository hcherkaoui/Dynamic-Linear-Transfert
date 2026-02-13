"""Tests for benchmark.config."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.experiments import get_config  # noqa


def test_config_has_expected_types():
    cfg = get_config()
    assert isinstance(cfg.N_JOBS, int)
    assert cfg.N_JOBS > 0
    assert isinstance(cfg.SEED, int)
    assert cfg.D > 0
    assert cfg.SIGMA >= 0
