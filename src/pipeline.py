"""The data science pipeline, one function per step.

run_pipeline.py calls run() to do everything headless; the notebook calls the
steps one at a time and explains each result.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from . import config, data, evaluate, features, models, plots


# 1. Data ----------------------------------------------------------------------
def load_and_clean(path=config.DATA_PATH):
    raw = data.load_raw(path)
    report = data.quality_report(raw)
    df, log = data.clean(raw)
    df = features.add_engineered(df)
    return raw, df, report, log


# 2. EDA -------------------------------------------------------------------------
def eda(df: pd.DataFrame) -> dict:
    num = data.RAW_NUMERIC + features.ENGINEERED
    plots.class_balance(df)
    plots.distributions(df, num)
    plots.correlation(df, num)
    plots.rpm_torque(df)
    plots.failure_by_type(df)
    plots.physics_views(df)
    plots.tool_wear_rate(df)
    corr = df[num + [config.TARGET]].corr()[config.TARGET].drop(config.TARGET).sort_values()
    return {
        "describe": df[num].describe().T.round(2),
        "outliers": data.iqr_outliers(df),
        "corr_with_target": corr.round(3),
        "failure_rate_by_type": df.groupby("type", observed=True)[config.TARGET].mean().round(4),
    }


# 3. Split -----------------------------------------------------------------------
def split(df: pd.DataFrame):
    """80/20 stratified split. The 20% test part is not touched until final evaluation."""
    train, test = train_test_split(df, test_size=config.TEST_SIZE, stratify=df[config.TARGET],
                                   random_state=config.SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# 4. Feature ablation: does feature engineering help? -----------------------------
def ablation(train: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for fs in ["raw", "engineered"]:
        X, y = features.xy(train, fs)
        for name, m in models.make_models(features.FEATURE_SETS[fs]).items():
            rows.setdefault(name, {})[fs] = models.cv_scores(m, X, y)["pr_auc"]
    return pd.DataFrame(rows).T[["raw", "engineered"]]


# 5. Tuning ------------------------------------------------------------------------
def tune(train: pd.DataFrame, feature_set: str = "engineered"):
    X, y = features.xy(train, feature_set)
    searches, rows = {}, []
    for name, m in models.make_models(features.FEATURE_SETS[feature_set]).items():
        s = models.tune(name, m, X, y)
        searches[name] = s
        rows.append({"model": name, "cv_pr_auc": s.best_score_,
                     "best_params": {k.replace("clf__", ""): v for k, v in s.best_params_.items()}})
    return searches, pd.DataFrame(rows).set_index("model")


# 6. Thresholds chosen by cost on out-of-fold predictions --------------------------
def thresholds(train: pd.DataFrame, searches: dict, feature_set: str = "engineered"):
    X, y = features.xy(train, feature_set)
    oof, rows = {}, []
    for name, s in searches.items():
        p = models.oof_proba(s.best_estimator_, X, y)
        t = evaluate.best_threshold(y, p)
        oof[name] = p
        s_at_t = evaluate.summarise(y, p, t)
        rows.append({"model": name, "threshold": t, "oof_cost_rs": s_at_t["cost_rs"],
                     "oof_cost_at_0.5_rs": evaluate.summarise(y, p, 0.5)["cost_rs"],
                     "oof_recall": s_at_t["recall"], "oof_precision": s_at_t["precision"]})
    table = pd.DataFrame(rows).set_index("model")
    best = table["oof_cost_rs"].idxmin()  # chosen on TRAINING folds, never on test
    return oof, table, best


# 7. Final test evaluation -------------------------------------------------------------
def test_eval(train, test, searches, thr_table, feature_set: str = "engineered"):
    Xtr, ytr = features.xy(train, feature_set)
    Xte, yte = features.xy(test, feature_set)
    rows, probas, preds = {}, {}, {}
    for name, b in models.make_baselines().items():
        b.fit(Xtr, ytr)
        p = b.predict_proba(Xte)[:, 1]
        rows[name] = evaluate.summarise(yte, p, 0.5)
        preds[name] = (p >= 0.5).astype(int)
    for name, s in searches.items():
        p = s.best_estimator_.predict_proba(Xte)[:, 1]
        t = thr_table.loc[name, "threshold"]
        rows[name] = evaluate.summarise(yte, p, t)
        rows[f"{name} @ 0.5"] = evaluate.summarise(yte, p, 0.5)
        probas[name] = p
        preds[name] = (p >= t).astype(int)
    return pd.DataFrame(rows).T, probas, preds


# 8. Error analysis and interpretation ---------------------------------------------------
def error_analysis(test, best_name, searches, probas, preds, feature_set: str = "engineered"):
    Xte, yte = features.xy(test, feature_set)
    pred = preds[best_name]
    by_mode = evaluate.recall_by_mode(test, pred)
    show = ["type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min",
            "power_w", "temp_diff_k", "strain_min_nm", "failure_mode"]
    missed = test.loc[(yte == 1) & (pred == 0), show].assign(p_fail=probas[best_name][(yte == 1) & (pred == 0)])
    false_alarms = test.loc[(yte == 0) & (pred == 1), show].assign(p_fail=probas[best_name][(yte == 0) & (pred == 1)])
    imp = permutation_importance(searches[best_name].best_estimator_, Xte, yte, scoring="average_precision",
                                 n_repeats=10, random_state=config.SEED, n_jobs=-1)
    importance = pd.DataFrame({"mean": imp.importances_mean, "std": imp.importances_std},
                              index=Xte.columns).sort_values("mean", ascending=False)
    return by_mode, missed.round(2), false_alarms.round(2), importance


def ablation_title(abl: pd.DataFrame) -> str:
    gain = abl["engineered"] - abl["raw"]
    return (f"Physics features raise CV PR-AUC by {gain.min():+.2f} to {gain.max():+.2f} "
            f"(largest for {gain.idxmax().lower()})")


def sensitivity_title(sens: pd.DataFrame) -> str:
    base_cols = [c for c in sens.columns if c.endswith(" cost_rs")]
    if (sens["model_cost_rs"].to_numpy()[:, None] < sens[base_cols].to_numpy()).all():
        return "The model is the cheapest option at every cost ratio tested"
    return "Test-set cost across cost assumptions"


def sensitivity(train, test, best_name, oof, probas, preds):
    _, ytr = features.xy(train)
    _, yte = features.xy(test)
    base = {k: v for k, v in preds.items() if k in models.make_baselines()}
    return evaluate.cost_sensitivity(ytr, oof[best_name], yte, probas[best_name], base)


# Everything, headless ------------------------------------------------------------------
def run(path=config.DATA_PATH, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    raw, df, report, log = load_and_clean(path)
    say(f"[1/8] Loaded {report['rows']:,} rows; failure rate {report['failure_rate']:.2%}")
    for line in log:
        say("      - " + line)

    e = eda(df)
    e["outliers"].to_csv(config.TABLE_DIR / "outliers_iqr.csv", index=False)
    say("[2/8] EDA figures saved")

    train, test = split(df)
    say(f"[3/8] Split: train {len(train):,} ({train[config.TARGET].sum()} failures), "
        f"test {len(test):,} ({test[config.TARGET].sum()} failures)")

    abl = ablation(train)
    abl.to_csv(config.TABLE_DIR / "feature_ablation_cv_pr_auc.csv")
    plots.feature_ablation(abl, ablation_title(abl))
    say("[4/8] Feature ablation (CV PR-AUC):\n" + abl.round(3).to_string())

    searches, tuned = tune(train)
    tuned.to_csv(config.TABLE_DIR / "tuning_results.csv")
    say("[5/8] Tuned:\n" + tuned.to_string())

    oof, thr, best = thresholds(train, searches)
    thr.to_csv(config.TABLE_DIR / "thresholds_oof.csv")
    plots.cost_vs_threshold(evaluate.cost_curve(train[config.TARGET], oof[best]), thr.loc[best, "threshold"], best)
    say(f"[6/8] Thresholds (chosen on training folds); best = {best}\n" + thr.round(3).to_string())

    res, probas, preds = test_eval(train, test, searches, thr)
    res.to_csv(config.TABLE_DIR / "test_results.csv")
    bl = models.make_baselines()
    plots.pr_curves(test[config.TARGET], probas,
                    {n: (res.loc[n, "recall"], res.loc[n, "precision"]) for n in bl})
    r = res.loc[best]
    plots.confusion(r["tp"], r["fp"], r["fn"], r["tn"], best)
    ci = evaluate.bootstrap_ci(test[config.TARGET], probas[best], thr.loc[best, "threshold"])
    ci.to_csv(config.TABLE_DIR / "bootstrap_ci_best_model.csv")
    say("      Bootstrap 95% CI for the final model:\n" + ci.round(3).to_string())
    say("[7/8] Test results:\n" + res[["threshold", "precision", "recall", "f1", "pr_auc", "roc_auc",
                                       "fn", "fp", "cost_rs"]].round(3).to_string())

    by_mode, missed, fa, imp = error_analysis(test, best, searches, probas, preds)
    by_mode.to_csv(config.TABLE_DIR / "recall_by_failure_mode.csv")
    missed.to_csv(config.TABLE_DIR / "missed_failures.csv", index=False)
    fa.to_csv(config.TABLE_DIR / "false_alarms.csv", index=False)
    imp.to_csv(config.TABLE_DIR / "permutation_importance.csv")
    plots.recall_modes(by_mode)
    plots.importance(imp)
    plots.risk_map(searches[best].best_estimator_, df, thr.loc[best, "threshold"],
                   features.FEATURE_SETS["engineered"], features.add_engineered)
    sens = sensitivity(train, test, best, oof, probas, preds)
    sens.to_csv(config.TABLE_DIR / "cost_sensitivity.csv", index=False)
    plots.sensitivity(sens, "model_cost_rs", {n: f"{n} cost_rs" for n in bl}, sensitivity_title(sens))
    say("[8/8] Recall by failure mode:\n" + by_mode.round(3).to_string())
    say("Cost sensitivity:\n" + sens.to_string())

    metrics = {
        "data_quality": report,
        "cleaning_log": log,
        "split": {"train": len(train), "test": len(test),
                  "train_failures": int(train[config.TARGET].sum()), "test_failures": int(test[config.TARGET].sum())},
        "ablation_cv_pr_auc": abl.round(4).to_dict(orient="index"),
        "tuning": {k: {"cv_pr_auc": round(v["cv_pr_auc"], 4), "best_params": v["best_params"]}
                   for k, v in tuned.to_dict(orient="index").items()},
        "thresholds": thr.round(4).to_dict(orient="index"),
        "best_model": best,
        "test_results": res.round(4).to_dict(orient="index"),
        "bootstrap_ci_best_model": ci.round(4).to_dict(orient="index"),
        "recall_by_failure_mode": by_mode.round(3).to_dict(orient="index"),
        "permutation_importance": imp.round(4).to_dict(orient="index"),
        "cost_sensitivity": sens.to_dict(orient="records"),
        "assumed_costs_rs": {"missed_failure": config.COST_MISSED_FAILURE, "false_alarm": config.COST_FALSE_ALARM},
        "seed": config.SEED,
    }
    with open(config.OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    say(f"Saved figures to {config.FIG_DIR} and tables to {config.TABLE_DIR}")
    return {"df": df, "train": train, "test": test, "searches": searches, "results": res, "best": best,
            "metrics": metrics}
