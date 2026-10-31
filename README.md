# Predictive maintenance on a milling machine (AI4I 2020)

This project predicts which machining jobs will end in a machine failure and picks the alarm threshold that keeps the cost of mistakes lowest. It is the IDSE course project, Part A.

**Result:** a tuned gradient boosting model catches **65 of 68** failures in unseen data and cuts the cost of mistakes from ₹34.0 lakh (no model) and ₹24.0 lakh (replace tools at 200 min) to **₹3.9 lakh**. The full write-up is in [REPORT.md](REPORT.md) (also as `REPORT.pdf`), and the step-by-step analysis is in [notebooks/AI4I_pipeline.ipynb](notebooks/AI4I_pipeline.ipynb).

## Problem

A missed failure costs an unplanned breakdown, assumed at ₹50,000. A false alarm costs a stop and inspection, assumed at ₹2,000. Given the sensor readings at the start of a job (temperatures, speed, torque, tool wear, product type), the project asks two things:

1. Which jobs should trigger an alarm?
2. How does that compare with doing nothing, or with the usual "replace the tool at 200 min" rule?

## Dataset

The data is the AI4I 2020 Predictive Maintenance Dataset from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (CC BY 4.0). It has 10,000 rows, and 339 of them (3.39%) are failures. Download `ai4i2020.csv` and place it in `data/` (see [data/README.md](data/README.md)).

## How to run

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py             # about 2 minutes; writes outputs/
jupyter notebook notebooks/AI4I_pipeline.ipynb
```

`run_pipeline.py` reruns everything from the raw CSV: cleaning, all 15 figures, all tables and `outputs/metrics.json` (every number in the report). The random seed is fixed at 42 in `src/config.py`, so repeated runs give identical results.

## Project structure

```
├── README.md
├── REPORT.md / REPORT.pdf         written report with figures
├── requirements.txt
├── run_pipeline.py                one command, end to end
├── data/                          put ai4i2020.csv here
├── notebooks/AI4I_pipeline.ipynb  the analysis, step by step, with written insights
├── outputs/
│   ├── figures/                   01–15 PNG figures
│   ├── tables/                    CSV results (tuning, test scores, error analysis, ...)
│   └── metrics.json               every number quoted in the report
└── src/
    ├── config.py                  seed, paths, assumed costs, split sizes
    ├── data.py                    loading, quality checks, cleaning (with a log of decisions)
    ├── features.py                physics features: power, temperature difference, wear × torque
    ├── models.py                  pipelines, baselines, tuning grids, cross-validation
    ├── evaluate.py                metrics, cost-based threshold, error analysis, bootstrap
    ├── plots.py                   all figures
    └── pipeline.py                the steps in order, shared by the script and the notebook
```

## Method in brief

1. **Clean.** Check types, missing values, duplicates, outliers and label consistency. Each decision is logged with its reason. Failure-mode flags are dropped from the features, since they are known only after a failure (target leakage).
2. **Explore.** Seven EDA figures, each with a written insight.
3. **Engineer features.** Spindle power, process − air temperature, and tool wear × torque.
4. **Split.** Stratified 80/20. The test set is touched once, at the end.
5. **Model.** Two baselines, then logistic regression, random forest and gradient boosting. Tuned by grid search with 5-fold stratified CV, scored by PR-AUC.
6. **Choose the threshold by cost.** Using out-of-fold training predictions, then pick the model with the lowest cross-validated cost.
7. **Evaluate on the test set.** Precision, recall, PR-AUC, cost, bootstrap confidence intervals, recall per failure mode, false-alarm analysis, permutation importance, a risk map of the speed–torque plane, and cost-ratio sensitivity.

## Where each marking criterion is addressed

| Criterion | Where |
| --- | --- |
| Problem and data | REPORT §1–2; notebook §1 (question, measurable goals, dataset justification) |
| Data preparation and EDA | REPORT §3–5; notebook §2–5; `src/data.py` cleaning log; figures 01–08 |
| Modelling | REPORT §6; notebook §6–8: two baselines + three models, locked test set, pipelines (no leakage), 5-fold CV, grid search |
| Evaluation and interpretation | REPORT §7–9; notebook §9–10: cost metric, model comparison, bootstrap CIs, error analysis by failure mode, limitations, bias, next steps |
| Reproducibility and code quality | This README, `requirements.txt`, seed 42, `run_pipeline.py` runs end to end, documented modules |
| Report and presentation | `REPORT.md` / `REPORT.pdf`, figures in `outputs/figures/` |

## Authors
Lakshay Vardhan - ME24B1001 |
Praneeth Bollu - ME24B1005 |
Chanakya Swaraj - EC24B1113

## Reference

S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", *Third International Conference on Artificial Intelligence for Industries (AI4I 2020)*.
