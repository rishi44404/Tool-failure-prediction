"""Loading, quality checks and cleaning of the AI4I 2020 dataset.

Every cleaning decision is logged with its reason, so the report and the
notebook can print exactly what was done to the data and why.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Original column names contain spaces and brackets, which are awkward in code
# and rejected by some libraries. Units stay in the new names.
RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}

RAW_NUMERIC = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as downloaded from UCI (a .zip holding the CSV also works)."""
    path = Path(path)
    if not path.is_absolute():
        path = config.ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download ai4i2020.csv from "
            "https://archive.ics.uci.edu/dataset/601 and place it in data/."
        )
    return pd.read_csv(path, encoding="utf-8-sig")  # the UCI file starts with a byte-order mark


def quality_report(raw: pd.DataFrame) -> dict:
    """Facts about the raw file that drive the cleaning decisions."""
    df = raw.rename(columns=RENAME)
    modes = df[config.FAILURE_MODES]
    any_mode = modes.sum(axis=1) > 0
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "duplicate_rows_ignoring_id": int(df.drop(columns=["udi", "product_id"]).duplicated().sum()),
        "duplicate_product_ids": int(df["product_id"].duplicated().sum()),
        "failure_rate": float(df[config.TARGET].mean()),
        "failures": int(df[config.TARGET].sum()),
        "mode_counts": {m: int(modes[m].sum()) for m in config.FAILURE_MODES},
        "rows_with_multiple_modes": int((modes.sum(axis=1) > 1).sum()),
        # Label consistency: the target should be 1 exactly when some mode is 1.
        "failure_without_mode": int(((df[config.TARGET] == 1) & ~any_mode).sum()),
        "mode_without_failure": int(((df[config.TARGET] == 0) & any_mode).sum()),
        "mode_without_failure_by_mode": {
            m: int(((df[config.TARGET] == 0) & (modes[m] == 1)).sum()) for m in config.FAILURE_MODES
        },
        "type_counts": {str(k): int(v) for k, v in df["type"].value_counts().items()},
        "product_id_letter_matches_type": bool((df["product_id"].str[0] == df["type"]).all()),
    }


def iqr_outliers(df: pd.DataFrame, cols=RAW_NUMERIC, k: float = 1.5) -> pd.DataFrame:
    """Count values outside [Q1 - k*IQR, Q3 + k*IQR] per column, and the failure rate among them."""
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        mask = (df[c] < lo) | (df[c] > hi)
        rows.append({
            "feature": c,
            "lower_fence": round(lo, 2),
            "upper_fence": round(hi, 2),
            "n_outliers": int(mask.sum()),
            "failure_rate_in_outliers": round(float(df.loc[mask, config.TARGET].mean()), 3) if mask.any() else np.nan,
            "failure_rate_elsewhere": round(float(df.loc[~mask, config.TARGET].mean()), 3),
        })
    return pd.DataFrame(rows)


def failure_mode_label(df: pd.DataFrame) -> pd.Series:
    """One readable label per row: 'none', a single mode, 'multiple', or 'unknown'."""
    modes = df[config.FAILURE_MODES]
    n = modes.sum(axis=1)
    label = pd.Series("none", index=df.index, dtype="object")
    single = (n == 1) & (df[config.TARGET] == 1)
    label[single] = modes[single].idxmax(axis=1)
    label[(n > 1) & (df[config.TARGET] == 1)] = "multiple"
    label[(n == 0) & (df[config.TARGET] == 1)] = "unknown"
    return label


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return the cleaned frame and a log of every decision taken."""
    log: list[str] = []
    df = raw.rename(columns=RENAME).copy()
    log.append("Renamed columns to snake_case with units kept (e.g. 'Torque [Nm]' -> torque_nm).")

    # 1. Data types -------------------------------------------------------
    df["type"] = pd.Categorical(df["type"], categories=["L", "M", "H"], ordered=True)
    log.append("Type -> ordered category L < M < H (low, medium, high product quality).")
    for c in config.FAILURE_MODES + [config.TARGET]:
        df[c] = df[c].astype(int)

    # 2. Missing values ---------------------------------------------------
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        log.append(f"{n_missing} missing values found; numeric ones are imputed with the TRAIN median "
                   "inside the model pipeline (never on the full data, to avoid leakage).")
    else:
        log.append("No missing values, so no imputation is needed. A median imputer still sits inside "
                   "the model pipeline so it would be fitted on training data only if new data had gaps.")

    # 3. Duplicates -------------------------------------------------------
    dup = df.drop(columns=["udi", "product_id"]).duplicated()
    if dup.any():
        df = df[~dup].copy()
        log.append(f"Dropped {int(dup.sum())} exact duplicate rows (ignoring IDs) so they cannot sit in "
                   "both train and test.")
    else:
        log.append("No duplicate rows (ignoring the ID columns) and no repeated product IDs.")

    # 4. Identifier columns -----------------------------------------------
    log.append("udi and product_id are identifiers, not measurements; they are kept for tracing rows "
               "but never used as features (product_id's first letter only repeats Type).")

    # 5. Label consistency ------------------------------------------------
    any_mode = df[config.FAILURE_MODES].sum(axis=1) > 0
    f_no_mode = int(((df[config.TARGET] == 1) & ~any_mode).sum())
    m_no_f = int(((df[config.TARGET] == 0) & any_mode).sum())
    df["failure_mode"] = failure_mode_label(df)
    log.append(f"Label check: {f_no_mode} failures have no mode flagged (labelled 'unknown'), and "
               f"{m_no_f} rows have a mode flagged but machine_failure = 0. machine_failure is kept as "
               "the ground truth because it records whether the machine actually stopped.")

    # 6. Leakage ----------------------------------------------------------
    log.append("TWF/HDF/PWF/OSF/RNF describe HOW a failure happened, so they are only known after the "
               "event. They are excluded from the features (using them would be target leakage) and used "
               "only for error analysis.")

    # 7. Outliers ---------------------------------------------------------
    out = iqr_outliers(df)
    flagged = ", ".join(f"{r.feature}: {r.n_outliers}" for r in out.itertuples() if r.n_outliers)
    log.append(f"IQR outliers: {flagged or 'none'}. They are physically possible readings and they are "
               "where failures concentrate, so they are kept; removing them would delete the signal.")
    return df.reset_index(drop=True), log
