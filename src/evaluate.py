"""Metrics, cost-based threshold selection and error analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config


def total_cost(y_true, y_pred, c_fn=config.COST_MISSED_FAILURE, c_fp=config.COST_FALSE_ALARM) -> float:
    """Rupees lost: each missed failure costs c_fn, each false alarm costs c_fp."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return fn * c_fn + fp * c_fp


def cost_curve(y_true, proba, c_fn=config.COST_MISSED_FAILURE, c_fp=config.COST_FALSE_ALARM) -> pd.DataFrame:
    """Cost at every candidate threshold (each distinct predicted probability, plus 0 and 1)."""
    thresholds = np.unique(np.concatenate([[0.0, 1.0001], np.round(proba, 4)]))
    rows = [(t, total_cost(y_true, (proba >= t).astype(int), c_fn, c_fp)) for t in thresholds]
    return pd.DataFrame(rows, columns=["threshold", "cost"])


def best_threshold(y_true, proba, c_fn=config.COST_MISSED_FAILURE, c_fp=config.COST_FALSE_ALARM) -> float:
    """Threshold with the lowest total cost. Ties go to the middle of the cheapest flat stretch."""
    curve = cost_curve(y_true, proba, c_fn, c_fp)
    best = curve[curve["cost"] == curve["cost"].min()]["threshold"]
    return float(best.iloc[len(best) // 2])


def summarise(y_true, proba, threshold: float, c_fn=config.COST_MISSED_FAILURE,
              c_fp=config.COST_FALSE_ALARM) -> dict:
    """Everything we report for one model on one dataset."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    ranked = len(np.unique(proba)) > 2  # AUCs mean little for a hard 0/1 rule
    return {
        "threshold": round(float(threshold), 4),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": float((pred == y_true).mean()),
        "pr_auc": average_precision_score(y_true, proba) if ranked else np.nan,
        "roc_auc": roc_auc_score(y_true, proba) if ranked else np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "cost_rs": total_cost(y_true, pred, c_fn, c_fp),
    }


def recall_by_mode(df_part: pd.DataFrame, pred) -> pd.DataFrame:
    """For actual failures only: how many of each failure mode the model caught."""
    d = df_part.assign(pred=np.asarray(pred))
    d = d[d[config.TARGET] == 1]
    out = d.groupby("failure_mode").agg(failures=("pred", "size"), caught=("pred", "sum"))
    out["recall"] = out["caught"] / out["failures"]
    names = {**config.FAILURE_MODE_NAMES, "multiple": "Several modes at once", "unknown": "No mode recorded"}
    out.index = [f"{names.get(i, i)} ({i})" for i in out.index]
    return out.sort_values("failures", ascending=False)


def cost_sensitivity(y_train, oof, y_test, test_proba, baseline_preds: dict,
                     ratios=(5, 10, 25, 50, 100)) -> pd.DataFrame:
    """Re-pick the threshold on training out-of-fold predictions for each cost ratio, then
    report the test cost of the model next to the baselines. c_fp is held at its configured value."""
    rows = []
    for r in ratios:
        c_fp, c_fn = config.COST_FALSE_ALARM, r * config.COST_FALSE_ALARM
        t = best_threshold(y_train, oof, c_fn, c_fp)
        row = {"cost_ratio_fn_to_fp": f"{r}:1", "threshold": round(t, 4),
               "model_cost_rs": total_cost(y_test, (test_proba >= t).astype(int), c_fn, c_fp)}
        for name, pred in baseline_preds.items():
            row[f"{name} cost_rs"] = total_cost(y_test, pred, c_fn, c_fp)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(y_true, proba, threshold: float, n_boot: int = 1000, seed: int = config.SEED) -> pd.DataFrame:
    """95% bootstrap intervals for the test-set numbers: how much they could move with another sample
    of 2,000 jobs. Resamples test rows with replacement; the model and threshold stay fixed."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    rows = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        yt, pt = y_true[i], pred[i]
        rows.append({"recall": recall_score(yt, pt, zero_division=0),
                     "precision": precision_score(yt, pt, zero_division=0),
                     "cost_rs": total_cost(yt, pt)})
    b = pd.DataFrame(rows)
    point = {"recall": recall_score(y_true, pred), "precision": precision_score(y_true, pred),
             "cost_rs": total_cost(y_true, pred)}
    return pd.DataFrame({"estimate": point, "ci_low_2.5%": b.quantile(0.025), "ci_high_97.5%": b.quantile(0.975)})
