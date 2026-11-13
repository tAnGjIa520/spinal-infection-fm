"""Repeated stratified patient-level cross-validation of the framework and its ablations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spinefm import AnatomyGuidedLDA
from spinefm.evaluation import cross_validate, make_splits, summarize
from spinefm.features import load_cohort

VARIANTS = {"framework": dict(), "without_regional": dict(use_regional=False), "without_global": dict(use_global=False)}

ap = argparse.ArgumentParser()
ap.add_argument("--features", type=Path, required=True)
ap.add_argument("--splits", type=Path, help="CSV with patient_id, repeat, fold; generated if omitted")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--bootstrap", type=int, default=10000)
args = ap.parse_args()

cohort = load_cohort(sorted(args.features.glob("*.npz")))
splits = (pd.read_csv(args.splits, dtype={"patient_id": str}) if args.splits
          else make_splits(cohort["ids"], cohort["y"]))
args.out.mkdir(parents=True, exist_ok=True)
splits.to_csv(args.out / "splits.csv", index=False)
scores = np.stack([cross_validate(cohort, splits, lambda kw=kw: AnatomyGuidedLDA(**kw)) for kw in VARIANTS.values()])
for name, s in zip(VARIANTS, scores):
    pd.DataFrame(s.T, index=cohort["ids"], columns=[f"repeat{r}" for r in range(len(s))]).to_csv(args.out / f"oof_{name}.csv")
table = summarize(cohort["y"], scores, list(VARIANTS), args.bootstrap)
table.to_csv(args.out / "summary.csv", index=False)
print(table.round(3).to_string(index=False))
