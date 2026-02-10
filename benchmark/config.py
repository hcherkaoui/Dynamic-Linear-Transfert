"""Configuration loader."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
from typing import Any, Mapping
import numpy as np
import yaml


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _ROOT / "benchmark" / "configs" / "config.yml"


@dataclass(frozen=True)
class Config:
    FIGURES_DIR: str
    FONTSIZE: int
    LW: float
    ALPHA: float
    N_JOBS: int
    N_RUNS: int
    SEED: int
    L_EPS: np.ndarray
    L_SIGMA_S: np.ndarray
    N_TRAIN_SAMPLES: int
    N_T: int
    N_POINTS: int
    SIGMA: float
    D: int
    EPS: list[float]
    LBDA_DEFAULT: float


def _logspace(spec: Any) -> np.ndarray:
    if isinstance(spec, Mapping):
        start = float(spec["start"])
        stop = float(spec["stop"])
        num = int(spec["num"])
        return np.logspace(start=start, stop=stop, num=num)
    return np.asarray(spec, dtype=float)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=None)
def get_config(path: str | None = None) -> Config:
    config_path = Path(path or os.environ.get("DYNTR_CONFIG", _DEFAULT_CONFIG)).resolve()
    data = _load_yaml(config_path)

    plotting = data.get("plotting", {})
    launcher = data.get("launcher", {})
    data_cfg = data.get("data", {})

    figures_dir = str(plotting.get("figures_dir", "_figures"))
    os.makedirs(figures_dir, exist_ok=True)

    return Config(
        FIGURES_DIR=figures_dir,
        FONTSIZE=int(plotting.get("fontsize", 18)),
        LW=float(plotting.get("lw", 5)),
        ALPHA=float(plotting.get("alpha", 0.4)),
        N_JOBS=int(launcher.get("n_jobs", 1)),
        N_RUNS=int(launcher.get("n_runs", 1)),
        SEED=int(launcher.get("seed", 0)),
        L_EPS=_logspace(data_cfg.get("l_eps", {"start": -3, "stop": 3, "num": 50})),
        L_SIGMA_S=_logspace(data_cfg.get("l_sigma_s", {"start": -3, "stop": 3, "num": 50})),
        N_TRAIN_SAMPLES=int(data_cfg.get("n_train_samples", 2000)),
        N_T=int(data_cfg.get("n_t", 1000)),
        N_POINTS=int(data_cfg.get("n_points", 20)),
        SIGMA=float(data_cfg.get("sigma", 1.0)),
        D=int(data_cfg.get("d", 50)),
        EPS=list(data_cfg.get("eps", [0.5])),
        LBDA_DEFAULT=float(data_cfg.get("lbda_default", 1.0)),
    )
