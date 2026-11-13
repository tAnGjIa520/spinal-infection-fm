"""Repeated patient-level cross-validation, metrics and conditional bootstrap intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .framework import AnatomyGuidedLDA, Seeds

METRICS = ("auc", "accuracy", "sensitivity", "specificity")


def metric(y, score) -> dict:
    """Threshold 0; sensitivity for tuberculosis (1), specificity for pyogenic infection (0)."""
    y, score = np.asarray(y), np.asarray(score)
    tb, pred = y == 1, score >= 0
    return dict(auc=float(roc_auc_score(y, score)), accuracy=float((pred == y).mean()),
                sensitivity=float(pred[tb].mean()), specificity=float((~pred[~tb]).mean()))


def make_splits(ids, y, repeats: int = 3, folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Stratified patient-level folds; one row per patient and repetition."""
    rows = []
    for r in range(repeats):
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed + r)
        for f, (_, test) in enumerate(skf.split(np.zeros(len(y)), y)):
            rows += [dict(patient_id=str(ids[i]), label=int(y[i]), repeat=r, fold=f) for i in test]
    return pd.DataFrame(rows)


def cross_validate(cohort, splits: pd.DataFrame, model_factory=AnatomyGuidedLDA, seeds=Seeds.for_fold):
    """Out-of-fold scores, shape (repeats, patients)."""
    ids = cohort["ids"].astype(str)
    repeats = sorted(splits.repeat.unique())
    scores = np.full((len(repeats), len(ids)), np.nan)
    for r in repeats:
        fold = splits[splits.repeat == r].set_index("patient_id").loc[ids, "fold"].to_numpy()
        for f in sorted(set(fold)):
            train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            model = model_factory().fit(cohort, train, seeds(r, f))
            scores[r, test] = model.decision_function(cohort, test)
    assert np.isfinite(scores).all()
    return scores


def bootstrap(y, scores, n: int = 10000, seed: int = 20261029) -> np.ndarray:
    """Class-stratified patient bootstrap of repetition-averaged metrics.

    ``scores`` has shape (models, repeats, patients). Resampling is expressed as
    multinomial patient weights within each class, shared by every model and
    repetition, so paired differences use identical resamples. Intervals are
    conditional on the saved out-of-fold predictions and do not include
    model-refitting uncertainty.
    """
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    wp = rng.multinomial(len(pos), np.ones(len(pos)) / len(pos), size=n)
    wn = rng.multinomial(len(neg), np.ones(len(neg)) / len(neg), size=n)
    out = np.empty((n, len(scores), len(METRICS)))
    for m, per_repeat in enumerate(scores):
        auc, se, sp = [], [], []
        for s in per_repeat:
            u = (s[pos][:, None] > s[neg][None, :]) + .5 * (s[pos][:, None] == s[neg][None, :])
            auc.append(np.sum((wp @ u) * wn, axis=1) / (len(pos) * len(neg)))
            se.append(wp @ (s[pos] >= 0) / len(pos))
            sp.append(wn @ (s[neg] < 0) / len(neg))
        se, sp = np.mean(se, 0), np.mean(sp, 0)
        out[:, m] = np.stack([np.mean(auc, 0), (se * len(pos) + sp * len(neg)) / len(y), se, sp], 1)
    return out


def summarize(y, scores, names, n_boot: int = 10000) -> pd.DataFrame:
    boot = bootstrap(y, scores, n_boot)
    rows = []
    for m, name in enumerate(names):
        row = dict(model=name)
        for k, key in enumerate(METRICS):
            row[key] = float(np.mean([metric(y, s)[key] for s in scores[m]]))
            row[f"{key}_low"], row[f"{key}_high"] = np.quantile(boot[:, m, k], [.025, .975])
        rows.append(row)
    return pd.DataFrame(rows)
