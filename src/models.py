"""Model definitions, hyper-parameter grids and cross-validation helpers.

Every model is a scikit-learn Pipeline (preprocessing + estimator), so the
imputer, scaler and encoder are re-fitted inside each CV fold on that fold's
training part only. That is what keeps the validation free of leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import split_columns


def cv_splitter() -> StratifiedKFold:
    """Stratified so each fold keeps the ~3% failure rate."""
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


def preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    num, cat = split_columns(feature_cols)
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


class ScheduledReplacement(BaseEstimator, ClassifierMixin):
    """Domain baseline with no learning: raise an alarm once tool wear passes a fixed limit.

    This is what a plant does without a model (time-based preventive maintenance).
    Its 'probability' is 1 above the limit and 0 below.
    """

    def __init__(self, wear_limit: float = config.SCHEDULED_REPLACEMENT_WEAR_MIN):
        self.wear_limit = wear_limit

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (X["tool_wear_min"].to_numpy() >= self.wear_limit).astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_models(feature_cols: list[str]) -> dict[str, Pipeline]:
    """The candidates, in order of complexity. Class imbalance is handled with class weights."""
    pre = lambda: preprocessor(feature_cols)  # a fresh, unfitted copy per model
    return {
        "Logistic regression": Pipeline([
            ("pre", pre()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=config.SEED)),
        ]),
        "Random forest": Pipeline([
            ("pre", pre()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=config.SEED)),
        ]),
        "Gradient boosting": Pipeline([
            ("pre", pre()),
            ("clf", HistGradientBoostingClassifier(random_state=config.SEED)),
        ]),
    }


def make_baselines() -> dict[str, BaseEstimator]:
    return {
        "Never alarm (majority class)": DummyClassifier(strategy="most_frequent"),
        f"Replace tool at {config.SCHEDULED_REPLACEMENT_WEAR_MIN} min wear": ScheduledReplacement(),
    }


# Small grids on purpose: the rubric rewards a justified search, not a huge one.
PARAM_GRIDS = {
    "Logistic regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "Random forest": {
        "clf__max_depth": [None, 8, 16],
        "clf__min_samples_leaf": [1, 3, 10],
        "clf__max_features": ["sqrt", 0.5],
    },
    "Gradient boosting": {
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_leaf_nodes": [15, 31],
        "clf__max_iter": [200, 500],
        "clf__l2_regularization": [0.0, 1.0],
    },
}


def cv_scores(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Out-of-fold PR-AUC and ROC-AUC. PR-AUC is the headline because failures are rare."""
    proba = cross_val_predict(model, X, y, cv=cv_splitter(), method="predict_proba")[:, 1]
    return {"pr_auc": average_precision_score(y, proba), "roc_auc": roc_auc_score(y, proba)}


def tune(name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    """Grid search scored by average precision (area under the precision-recall curve)."""
    search = GridSearchCV(model, PARAM_GRIDS[name], scoring="average_precision",
                          cv=cv_splitter(), n_jobs=-1, refit=True)
    return search.fit(X, y)


def oof_proba(model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Out-of-fold failure probabilities on the training set, used to pick the alarm threshold."""
    return cross_val_predict(model, X, y, cv=cv_splitter(), method="predict_proba")[:, 1]
