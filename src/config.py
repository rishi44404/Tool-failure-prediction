"""Every setting the pipeline uses, in one place.

Change a number here and re-run; nothing else in the code hard-codes these.
"""
from pathlib import Path

# ---- Reproducibility -------------------------------------------------------
SEED = 42  # used for the train/test split, CV shuffling and every model

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

# ---- Problem definition ----------------------------------------------------
TARGET = "machine_failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_MODE_NAMES = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}

# ---- Business assumption: what each kind of mistake costs ------------------
# These are ASSUMED figures for a mid-size Indian machine shop, stated in the
# report. The sensitivity analysis varies their ratio from 5:1 to 100:1, so the
# conclusions do not hang on these exact values.
COST_MISSED_FAILURE = 50_000   # Rs, false negative: unplanned breakdown, scrap, downtime
COST_FALSE_ALARM = 2_000       # Rs, false positive: stop the machine and inspect it

# ---- Validation design -----------------------------------------------------
TEST_SIZE = 0.20   # 20% locked away until the very end
CV_FOLDS = 5       # stratified folds on the 80% training part

# ---- Domain baseline -------------------------------------------------------
# A plant without any model typically replaces tools on a fixed schedule.
SCHEDULED_REPLACEMENT_WEAR_MIN = 200
