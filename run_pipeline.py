"""Run the whole AI4I pipeline end to end.

    python run_pipeline.py                 # uses data/ai4i2020.csv
    python run_pipeline.py path/to/file.csv

Writes figures to outputs/figures, tables to outputs/tables and every number
quoted in the report to outputs/metrics.json.
"""
import sys
import time

from src import pipeline

if __name__ == "__main__":
    start = time.time()
    pipeline.run(*sys.argv[1:2])
    print(f"Done in {time.time() - start:.0f} s")
