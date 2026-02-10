"""Tests for benchmark.config."""
from benchmark.config import get_config


def test_config_has_expected_types():
    cfg = get_config()
    assert isinstance(cfg.N_JOBS, int)
    assert cfg.N_JOBS > 0
    assert isinstance(cfg.SEED, int)
    assert cfg.D > 0
    assert cfg.SIGMA >= 0
