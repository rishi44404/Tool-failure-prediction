"""All figures. Each function draws one figure, saves it to outputs/figures and returns it.

Colour rules used throughout: blue = no failure / model, orange = failure /
second series, grey = baselines and reference lines. One measure per axis.
"""
from __future__ import annotations

import matplotlib

try:  # inside Jupyter keep the inline backend; elsewhere draw off-screen (no display needed)
    get_ipython  # noqa: F821  (defined only inside IPython/Jupyter)
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve

from . import config

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GREY, INK, INK2, GRID, SURFACE = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
SERIES = [BLUE, ORANGE, AQUA, VIOLET]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", [BLUE, "#f0efec", "#e34948"])
RISK = LinearSegmentedColormap.from_list("risk", ["#f0efec", "#f9c9b4", ORANGE, "#a3360f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 10, "figure.dpi": 110,
    "savefig.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2,
})

LABELS = {
    "air_temp_k": "Air temperature (K)", "process_temp_k": "Process temperature (K)",
    "rpm": "Rotational speed (rpm)", "torque_nm": "Torque (Nm)", "tool_wear_min": "Tool wear (min)",
    "power_w": "Power (W)", "temp_diff_k": "Process − air temperature (K)",
    "strain_min_nm": "Tool wear × torque (min·Nm)", "machine_failure": "Machine failure",
}


def _save(fig, name: str):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / f"{name}.png")
    return fig


def class_balance(df: pd.DataFrame):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1.4]})
    counts = df[config.TARGET].value_counts().reindex([0, 1])
    a.barh(["No failure", "Failure"], counts.values, color=[BLUE, ORANGE], height=0.55)
    for i, v in enumerate(counts.values):
        a.text(v, i, f"  {v:,} ({v / len(df):.1%})", va="center", color=INK2, fontsize=9)
    a.set_title(f"Only {counts[1] / len(df):.1%} of rows are failures")
    a.set_xlim(0, counts.max() * 1.35)
    a.grid(axis="y", visible=False)

    modes = df[config.FAILURE_MODES].sum().sort_values()
    b.barh([f"{config.FAILURE_MODE_NAMES[m]} ({m})" for m in modes.index], modes.values, color=ORANGE, height=0.55)
    for i, v in enumerate(modes.values):
        b.text(v, i, f"  {v}", va="center", color=INK2, fontsize=9)
    b.set_title("Failure-mode flags (a row can have several)")
    b.set_xlim(0, modes.max() * 1.2)
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "01_class_balance")


def distributions(df: pd.DataFrame, cols: list[str]):
    n = len(cols)
    ncol = 4
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(13, 2.9 * np.ceil(n / ncol)))
    for ax, c in zip(axes.flat, cols):
        bins = np.histogram_bin_edges(df[c], bins=40)
        for val, colr, lab in [(0, BLUE, "No failure"), (1, ORANGE, "Failure")]:
            ax.hist(df.loc[df[config.TARGET] == val, c], bins=bins, density=True, histtype="step",
                    color=colr, linewidth=1.8, label=lab)
        ax.set_title(LABELS.get(c, c), fontsize=10)
        ax.set_yticks([])
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(loc="upper left")
    fig.suptitle("Feature distributions by outcome (each curve scaled to the same area)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "02_distributions")


def correlation(df: pd.DataFrame, cols: list[str]):
    corr = df[cols + [config.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    names = [LABELS.get(c, c) for c in corr.columns]
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Correlation between features and failure")
    fig.tight_layout()
    return _save(fig, "03_correlation")


def rpm_torque(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ok, bad = df[df[config.TARGET] == 0], df[df[config.TARGET] == 1]
    ax.scatter(ok["rpm"], ok["torque_nm"], s=4, color=BLUE, alpha=0.25, label="No failure", linewidths=0)
    ax.scatter(bad["rpm"], bad["torque_nm"], s=14, color=ORANGE, alpha=0.9, label="Failure",
               edgecolors=SURFACE, linewidths=0.5)
    rpm = np.linspace(df["rpm"].min(), df["rpm"].max(), 200)
    for p in (3500, 9000):
        ax.plot(rpm, p / (rpm * 2 * np.pi / 60), color=GREY, linestyle="--", linewidth=1.2)
        x = rpm[-1]
        ax.text(x, p / (x * 2 * np.pi / 60) + 1.5, f"P = {p:,} W", color=INK2, fontsize=8.5, ha="right",
                va="bottom")
    ax.set_ylim(0, df["torque_nm"].max() * 1.05)
    ax.set_xlabel(LABELS["rpm"]); ax.set_ylabel(LABELS["torque_nm"])
    ax.set_title("Failures sit at the extremes of the speed–torque curve")
    ax.legend(loc="upper right", markerscale=2)
    fig.tight_layout()
    return _save(fig, "04_rpm_vs_torque")


def failure_by_type(df: pd.DataFrame):
    g = df.groupby("type", observed=True)[config.TARGET].agg(["mean", "size", "sum"])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(g.index.astype(str), g["mean"] * 100, color=BLUE, width=0.55)
    for i, (t, r) in enumerate(g.iterrows()):
        ax.text(i, r["mean"] * 100, f"{r['mean']:.1%}\n{int(r['sum'])} of {int(r['size']):,}",
                ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_ylabel("Failure rate (%)")
    ax.set_xticks(range(3), ["L (low quality)", "M (medium)", "H (high)"])
    ax.set_ylim(0, g["mean"].max() * 100 * 1.45)
    ax.set_title("Failure rate by product type")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, "05_failure_by_type")


def physics_views(df: pd.DataFrame):
    """The three engineered features, each against the mode it explains."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ok = df["failure_mode"] == "none"

    a = axes[0]
    hdf = df["HDF"] == 1
    a.scatter(df.loc[ok, "rpm"], df.loc[ok, "temp_diff_k"], s=3, color=BLUE, alpha=0.2, linewidths=0, label="No failure")
    a.scatter(df.loc[hdf, "rpm"], df.loc[hdf, "temp_diff_k"], s=14, color=ORANGE, label="Heat dissipation failure",
              edgecolors=SURFACE, linewidths=0.5)
    a.set_xlabel(LABELS["rpm"]); a.set_ylabel(LABELS["temp_diff_k"])
    a.set_title("Heat failures: small ΔT and low speed")
    a.legend(loc="upper right", markerscale=2)

    b = axes[1]
    bins = np.linspace(df["power_w"].min(), df["power_w"].max(), 60)
    b.hist(df.loc[ok, "power_w"], bins=bins, color=BLUE, alpha=0.8, label="No failure")
    b.hist(df.loc[df["PWF"] == 1, "power_w"], bins=bins, color=ORANGE, label="Power failure")
    b.set_yscale("log")
    b.set_xlabel(LABELS["power_w"]); b.set_ylabel("Rows (log scale)")
    b.set_title("Power failures: outside a safe power band")
    b.legend(loc="upper right")

    c = axes[2]
    for t, colr in zip(["L", "M", "H"], SERIES):
        sub = df[(df["type"] == t) & ok]
        c.scatter(sub["tool_wear_min"], sub["torque_nm"], s=3, color=GREY, alpha=0.15, linewidths=0)
    osf = df["OSF"] == 1
    for t, colr in zip(["L", "M", "H"], SERIES):
        sub = df[osf & (df["type"] == t)]
        c.scatter(sub["tool_wear_min"], sub["torque_nm"], s=16, color=colr, label=f"Overstrain, type {t}",
                  edgecolors=SURFACE, linewidths=0.5)
    c.set_xlabel(LABELS["tool_wear_min"]); c.set_ylabel(LABELS["torque_nm"])
    c.set_title("Overstrain: worn tool under high torque")
    c.legend(loc="lower left", markerscale=1.5)
    fig.tight_layout()
    return _save(fig, "06_physics_views")


def tool_wear_rate(df: pd.DataFrame, min_rows: int = 30):
    """Failure rate per 10-min tool-wear bin; bins with fewer than min_rows rows are dropped as noise."""
    bins = np.arange(0, df["tool_wear_min"].max() + 10, 10)
    d = df.assign(bin=pd.cut(df["tool_wear_min"], bins, include_lowest=True))
    g = d.groupby("bin", observed=True).agg(n=(config.TARGET, "size"), all_f=(config.TARGET, "sum"),
                                            twf=("TWF", "sum"))
    g = g[g["n"] >= min_rows]
    mids = [iv.mid for iv in g.index]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(mids, g["all_f"] / g["n"] * 100, color=BLUE, marker="o", markersize=4, label="All failures")
    ax.plot(mids, g["twf"] / g["n"] * 100, color=ORANGE, marker="o", markersize=4, label="Tool wear failures (TWF)")
    ax.set_xlabel(LABELS["tool_wear_min"]); ax.set_ylabel("Failure rate in 10-min bin (%)")
    ax.set_title("Failure risk climbs once tool wear passes ~200 min")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "07_tool_wear_rate")


def feature_ablation(table: pd.DataFrame, title: str = "Effect of the physics features on each model"):
    """table: rows = model, columns = raw / engineered CV PR-AUC."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(table))
    h = 0.36
    ax.barh(y + h / 2, table["raw"], height=h, color=GREY, label="Raw sensor features")
    ax.barh(y - h / 2, table["engineered"], height=h, color=BLUE, label="+ physics features")
    for i, (r, e) in enumerate(zip(table["raw"], table["engineered"])):
        ax.text(r, i + h / 2, f" {r:.3f}", va="center", fontsize=8.5, color=INK2)
        ax.text(e, i - h / 2, f" {e:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, table.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.4)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("5-fold CV PR-AUC on the training set (higher is better)")
    ax.set_title(title)
    ax.legend(loc="center right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "08_feature_ablation")


def pr_curves(y_test, probas: dict, points: dict):
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for (name, p), colr in zip(probas.items(), SERIES):
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, color=colr, label=name)
    for (name, (r, p)), mk in zip(points.items(), ["s", "D", "^"]):
        ax.scatter([r], [p], color=GREY, marker=mk, s=60, zorder=5, label=name, edgecolors=SURFACE)
    ax.axhline(np.mean(y_test), color=GREY, linestyle=":", linewidth=1.2)
    ax.text(0.55, np.mean(y_test) + 0.015, "random guessing (= failure rate)", fontsize=8, color=INK2)
    ax.set_xlabel("Recall (share of failures caught)"); ax.set_ylabel("Precision (share of alarms that are real)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.set_title("Precision–recall on the held-out test set")
    ax.legend(loc="lower left", bbox_to_anchor=(0.3, 0.5), fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "09_pr_curves")


def cost_vs_threshold(curve: pd.DataFrame, chosen: float, model_name: str):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    c = curve[curve["threshold"] <= 1]
    ax.plot(c["threshold"], c["cost"] / 1e5, color=BLUE)
    best = c.loc[(c["threshold"] - chosen).abs().idxmin()]
    at05 = c.loc[(c["threshold"] - 0.5).abs().idxmin()]
    ax.scatter([best["threshold"]], [best["cost"] / 1e5], color=ORANGE, s=60, zorder=5)
    ax.annotate(f"chosen t = {chosen:.2f}\n₹{best['cost'] / 1e5:.1f} lakh", (best["threshold"], best["cost"] / 1e5),
                xytext=(12, 22), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.scatter([0.5], [at05["cost"] / 1e5], color=GREY, s=50, zorder=5)
    ax.annotate(f"default t = 0.50\n₹{at05['cost'] / 1e5:.1f} lakh", (0.5, at05["cost"] / 1e5),
                xytext=(12, 10), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlabel("Alarm threshold on predicted failure probability")
    ax.set_ylabel("Total cost on training folds (₹ lakh)")
    ax.set_title(f"Choosing the alarm threshold by cost ({model_name})")
    fig.tight_layout()
    return _save(fig, "10_cost_vs_threshold")


def confusion(tp, fp, fn, tn, model_name: str):
    m = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(m > 0, cmap=LinearSegmentedColormap.from_list("x", ["#f0efec", "#f0efec"]))
    labels = [["Correct: no alarm", "False alarm"], ["Missed failure", "Caught failure"]]
    colors = [[INK2, BLUE], [ORANGE, INK2]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}\n{labels[i][j]}", ha="center", va="center", fontsize=10,
                    color=colors[i][j], fontweight="bold" if (i, j) in [(0, 1), (1, 0)] else "normal")
    ax.set_xticks([0, 1], ["Predicted OK", "Predicted failure"])
    ax.set_yticks([0, 1], ["Actually OK", "Actually failed"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Test set: {model_name}")
    fig.tight_layout()
    return _save(fig, "11_confusion_matrix")


def recall_modes(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(table) + 1.4))
    ax.barh(table.index, table["recall"] * 100, color=BLUE, height=0.55)
    for i, r in enumerate(table.itertuples()):
        ax.text(r.recall * 100, i, f"  {int(r.caught)} of {int(r.failures)}", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 118)
    ax.invert_yaxis()
    ax.set_xlabel("Failures caught on the test set (%)")
    ax.set_title("Which kinds of failure the model catches")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "12_recall_by_failure_mode")


def importance(table: pd.DataFrame):
    t = table.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(t) + 1.3))
    ax.barh([LABELS.get(i, i) for i in t.index], t["mean"], xerr=t["std"], color=BLUE, height=0.55,
            error_kw={"ecolor": GREY, "elinewidth": 1})
    ax.set_xlabel("Drop in test PR-AUC when the feature is shuffled")
    ax.set_title("What the model relies on (permutation importance)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "13_permutation_importance")


def risk_map(model, df: pd.DataFrame, threshold: float, feature_cols: list[str], add_features):
    """Predicted failure risk over the speed-torque plane, for a fresh and a worn tool (type L)."""
    rpm = np.linspace(df["rpm"].quantile(0.001), df["rpm"].max(), 120)
    tq = np.linspace(df["torque_nm"].min(), df["torque_nm"].max(), 120)
    R, T = np.meshgrid(rpm, tq)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, wear in zip(axes, [30, 210]):
        grid = pd.DataFrame({"rpm": R.ravel(), "torque_nm": T.ravel(), "tool_wear_min": wear,
                             "air_temp_k": df["air_temp_k"].median(),
                             "process_temp_k": df["process_temp_k"].median(),
                             "type": pd.Categorical(["L"] * R.size, categories=["L", "M", "H"], ordered=True)})
        grid = add_features(grid)
        p = model.predict_proba(grid[feature_cols])[:, 1].reshape(R.shape)
        im = ax.contourf(R, T, p, levels=np.linspace(0, 1, 11), cmap=RISK)
        ax.contour(R, T, p, levels=[threshold], colors=INK, linewidths=1.5)
        ax.scatter(df["rpm"].sample(1500, random_state=config.SEED),
                   df["torque_nm"].sample(1500, random_state=config.SEED), s=2, color=BLUE, alpha=0.35, linewidths=0)
        ax.set_title(f"Type L, tool wear {wear} min")
        ax.set_xlabel(LABELS["rpm"])
        ax.grid(False)
    axes[0].set_ylabel(LABELS["torque_nm"])
    fig.colorbar(im, ax=axes, shrink=0.85, label="Predicted failure probability")
    fig.suptitle("Predicted risk over speed and torque (black line = alarm threshold, blue dots = real operating "
                 "points; regions with no dots are extrapolation)",
                 x=0.01, ha="left", fontweight="bold", color=INK, y=1.02)
    return _save(fig, "14_risk_map")


def sensitivity(table: pd.DataFrame, model_col: str, baseline_cols: dict,
                title: str = "Test-set cost across cost assumptions"):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(table))
    ax.plot(x, table[model_col] / 1e5, color=BLUE, marker="o", label="Tuned model (threshold re-picked per ratio)")
    for (label, col), colr, ls in zip(baseline_cols.items(), [GREY, ORANGE], ["--", ":"]):
        ax.plot(x, table[col] / 1e5, color=colr, linestyle=ls, marker="o", markersize=4, label=label)
    ax.set_xticks(x, table["cost_ratio_fn_to_fp"])
    ax.set_yscale("log")
    ax.set_xlabel("Cost of a missed failure : cost of a false alarm")
    ax.set_ylabel("Test-set cost (₹ lakh, log scale)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "15_cost_sensitivity")
