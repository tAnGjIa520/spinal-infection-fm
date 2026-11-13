"""Training-fold feature reduction, fixed MIL pooling and shrinkage LDA.

Every data-driven step (scalers, PCA, instance sampling, LDA) is fitted on the
training patients passed to :meth:`AnatomyGuidedLDA.fit` only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

GLOBAL_GROUPS = ("ct_rad", "ct_gmean", "ct_gmax", "med_t1_mean", "med_t1_max", "med_t2_mean", "med_t2_max")
MODALITIES = ("ct", "t1", "t2")
GLOBAL_COMPONENTS = 16
REGION_COMPONENTS = 8
INSTANCES_PER_PATIENT = 96


def derived_seed(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


@dataclass
class Seeds:
    """Seeds for instance sampling and randomised PCA within one training set."""
    instance: dict = field(default_factory=lambda: {"ct": 0, "t1": 1, "t2": 2})
    region: dict = field(default_factory=lambda: {"t1": 11, "t2": 12})

    @classmethod
    def for_fold(cls, repeat: int, fold: int) -> "Seeds":
        """Seeds used in the reported cross-validation (outer training sets)."""
        a = 20261029 + 1000 * repeat + 100 * fold
        b = 20261029 + 10000 * repeat + 1000 * fold
        return cls(instance={m: a + j for j, m in enumerate(MODALITIES)}, region={"t1": b + 1, "t2": b + 2})


# ---- regional instances ------------------------------------------------------------

def _sample(bags, train, seed):
    """Up to 96 instances per training patient (with replacement), so each patient weighs equally."""
    rng = np.random.default_rng(seed)
    values = [bags[i][rng.choice(len(bags[i]), INSTANCES_PER_PATIENT, replace=True)] for i in train if len(bags[i])]
    return np.concatenate(values).astype(np.float64)


def _fit_pca(values, seed):
    scale = StandardScaler().fit(values)
    pca = PCA(n_components=REGION_COMPONENTS, svd_solver="randomized", random_state=seed).fit(scale.transform(values))
    return dict(scale=scale, pca=pca, sd=np.sqrt(pca.explained_variance_).clip(1e-6))


def _apply_pca(model, bags, round_scores=True):
    """Whitened PCA scores per bag as float32; ``round_scores`` stores scores as float32 before whitening."""
    out = []
    for b in bags:
        if not len(b):
            out.append(np.zeros((0, REGION_COMPONENTS), np.float32))
            continue
        z = model["pca"].transform(model["scale"].transform(b.astype(np.float64)))
        if round_scores:
            z = z.astype(np.float32).astype(np.float64)
        out.append((z / model["sd"]).astype(np.float32))
    return out


def moments(bags) -> np.ndarray:
    """Distribution pooling: per-dimension mean and shrinkage variance sum((x-mean)^2)/(n+2)."""
    rows = []
    for b in bags:
        n = len(b)
        if n == 0:
            rows.append(np.zeros(2 * REGION_COMPONENTS, np.float32))
            continue
        mu = b.sum(0, dtype=np.float32) / np.float32(n)
        var = ((b - mu) ** 2).sum(0, dtype=np.float32) / np.float32(n + 2)
        rows.append(np.concatenate([mu, var]).astype(np.float32))
    return np.stack(rows)


# ---- classifier ----------------------------------------------------------------------

def fit_shrinkage_lda(x, y):
    """Two-class LDA with Ledoit-Wolf class covariances, equal priors, threshold 0."""
    scale = StandardScaler().fit(x)
    z = scale.transform(x)
    means = np.stack([z[y == c].mean(0) for c in (0, 1)])
    cov = np.stack([LedoitWolf().fit(z[y == c]).covariance_ for c in (0, 1)]).mean(0)
    ridge = max(float(np.trace(cov) / len(cov)) * 1e-6, 1e-8)
    w = np.linalg.solve(cov + ridge * np.eye(len(cov)), means[1] - means[0])
    return dict(scale=scale, w=w, b=float(-.5 * (means[1] + means[0]) @ w))


def lda_score(model, x):
    return model["scale"].transform(np.asarray(x, np.float64)) @ model["w"] + model["b"]


class AnatomyGuidedLDA:
    """Global (7 x PCA16 = 112) plus regional (3 modalities x mean/variance of PCA8 = 48) features.

    Positive scores favour tuberculosis (label 1); the decision threshold is 0.
    ``use_global``/``use_regional`` give the ablations reported in the paper.
    """

    def __init__(self, use_global: bool = True, use_regional: bool = True):
        assert use_global or use_regional
        self.use_global, self.use_regional = use_global, use_regional

    def _global(self, cohort, idx):
        tokens = [m["pca"].transform(m["scale"].transform(cohort[m["group"]][idx])) for m in self.global_models]
        g = np.stack(tokens, 1).astype(np.float32).reshape(len(idx), -1).astype(np.float64)
        return ((g - self.global_mean) / self.global_std).astype(np.float32)

    def _regional(self, cohort, idx):
        pooled = []
        for mod in MODALITIES:
            bags = [cohort[f"{mod}_bags"][i] for i in idx]
            if mod == "ct":
                z = _apply_pca(self.instance_models[mod], bags)
            else:
                # Each MRI region has three context scales: reduce each scale, then the region.
                z = _apply_pca(self.instance_models[mod], [b.reshape(-1, b.shape[-1]) for b in bags])
                z = _apply_pca(self.region_models[mod], [s.reshape(-1, 3 * REGION_COMPONENTS) for s in z], False)
            pooled.append(moments(z))
        return np.concatenate(pooled, 1)

    def features(self, cohort, idx):
        idx = np.asarray(idx)
        parts = []
        if self.use_global:
            parts.append(self._global(cohort, idx))
        if self.use_regional:
            parts.append(self._regional(cohort, idx))
        return np.concatenate(parts, 1).astype(np.float64)

    def fit(self, cohort, train, seeds: Seeds | None = None):
        train = np.asarray(train)
        seeds = seeds or Seeds()
        self.train_ = train
        if self.use_global:
            self.global_models = []
            for group in GLOBAL_GROUPS:
                raw = cohort[group]
                scale = StandardScaler().fit(raw[train])
                n = min(GLOBAL_COMPONENTS, len(train) - 2, raw.shape[1])
                pca = PCA(n_components=n, random_state=derived_seed("transform", train.tolist(), group, "pca16"))
                pca.fit(scale.transform(raw)[train])
                self.global_models.append(dict(group=group, scale=scale, pca=pca))
            tokens = [m["pca"].transform(m["scale"].transform(cohort[m["group"]][train])) for m in self.global_models]
            g = np.stack(tokens, 1).astype(np.float32).reshape(len(train), -1).astype(np.float64)
            self.global_mean, self.global_std = g.mean(0), g.std(0).clip(1e-6)
        if self.use_regional:
            self.instance_models, self.region_models = {}, {}
            for mod in MODALITIES:
                bags = cohort[f"{mod}_bags"]
                flat = bags if mod == "ct" else [b.reshape(-1, b.shape[-1]) for b in bags]
                self.instance_models[mod] = _fit_pca(_sample(flat, train, seeds.instance[mod]), seeds.instance[mod])
                if mod != "ct":
                    scaled = _apply_pca(self.instance_models[mod], [flat[i] for i in range(len(flat))])
                    regions = [s.reshape(-1, 3 * REGION_COMPONENTS) for s in scaled]
                    self.region_models[mod] = _fit_pca(_sample(regions, train, seeds.region[mod]), seeds.region[mod])
        self.head = fit_shrinkage_lda(self.features(cohort, train), cohort["y"][train])
        return self

    def decision_function(self, cohort, idx):
        return lda_score(self.head, self.features(cohort, idx))
