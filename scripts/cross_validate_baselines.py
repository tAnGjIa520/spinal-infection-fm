"""Cross-validate the radiomics, DL and DLR comparators with the framework's splits."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from spinefm.baselines import RadiomicsStyleModel
from spinefm.evaluation import summarize

ap = argparse.ArgumentParser()
ap.add_argument("--features", type=Path, required=True, help="output of extract_baseline_features.py")
ap.add_argument("--splits", type=Path, required=True, help="splits.csv written by cross_validate.py")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mri-only", action="store_true", help="use T1 and T2 features only")
args = ap.parse_args()

mods = ("t1", "t2") if args.mri_only else ("ct", "t1", "t2")
records = [dict(np.load(p, allow_pickle=False)) for p in sorted(args.features.glob("*.npz"))]
ids = np.array([str(r["patient_id"]) for r in records])
y = np.array([int(r["label"]) for r in records])
rad = [json.loads(str(r["radiomics"])) for r in records]
names = {m: sorted(next(d[m] for d in rad if d[m] is not None)) for m in mods}
# Patients without an ROI in a modality get zero features, as in the reported analysis.
radiomics = np.concatenate([np.array([[d[m][k] if d[m] else 0. for k in names[m]] for d in rad]) for m in mods], 1)
deep = np.concatenate([np.stack([r[f"dl_{m}"] for r in records]) for m in mods], 1)
models = {"Radiomics": ([radiomics], "random_forest"), "DL": ([deep], "extra_trees"),
          "DLR": ([radiomics, deep], "extra_trees")}

splits = pd.read_csv(args.splits, dtype={"patient_id": str})
repeats = sorted(splits.repeat.unique())
scores = np.full((len(models), len(repeats), len(y)), np.nan)
for r in repeats:
    fold = splits[splits.repeat == r].set_index("patient_id").loc[ids, "fold"].to_numpy()
    for f in sorted(set(fold)):
        train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        inner = [(train[a], train[b]) for a, b in
                 StratifiedKFold(3, shuffle=True, random_state=20261028 + 100 * r + f).split(train, y[train])]
        for k, (blocks, clf) in enumerate(models.values()):
            scores[k, r, test] = RadiomicsStyleModel(clf).fit(blocks, y, train, inner).decision_function(blocks, test)
args.out.mkdir(parents=True, exist_ok=True)
table = summarize(y, scores, list(models))
table.to_csv(args.out / f"summary_baselines{'_mri' if args.mri_only else ''}.csv", index=False)
print(table.round(3).to_string(index=False))
