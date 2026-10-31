"""Feature engineering from machining physics.

Three features a mechanical engineer would compute by hand:
  power_w        P = T * omega = torque * rpm * 2*pi/60      (spindle power, W)
  temp_diff_k    process temperature - air temperature      (how well heat leaves the process, K)
  strain_min_nm  tool wear * torque                          (cumulative load on a worn tool, min*Nm)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import RAW_NUMERIC

ENGINEERED = ["power_w", "temp_diff_k", "strain_min_nm"]
CATEGORICAL = ["type"]

FEATURE_SETS = {
    "raw": RAW_NUMERIC + CATEGORICAL,
    "engineered": RAW_NUMERIC + ENGINEERED + CATEGORICAL,
}


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the physics features added. Row-wise only, so no leakage."""
    out = df.copy()
    out["power_w"] = out["torque_nm"] * out["rpm"] * 2 * np.pi / 60
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["strain_min_nm"] = out["tool_wear_min"] * out["torque_nm"]
    return out


def split_columns(cols: list[str]) -> tuple[list[str], list[str]]:
    """Separate numeric and categorical feature names."""
    cat = [c for c in cols if c in CATEGORICAL]
    return [c for c in cols if c not in cat], cat


def xy(df: pd.DataFrame, feature_set: str = "engineered") -> tuple[pd.DataFrame, pd.Series]:
    cols = FEATURE_SETS[feature_set]
    return df[cols], df[config.TARGET]
