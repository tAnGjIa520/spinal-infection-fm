"""Comparator models adapted from a published deep-learning radiomics study.

Radiomics (PyRadiomics), DL (frozen ImageNet ResNet101) and DLR (both) features are
extracted from the same anatomical proxy ROIs as the framework. Within each training
set: median imputation and standardisation, Mann-Whitney U filtering, Spearman
redundancy removal, L1-logistic (LASSO) selection with the penalty chosen by inner
cross-validation, then a fixed tree ensemble. Tuberculosis is predicted when the
probability is at least the training prevalence (equal-prior rule).
"""
from __future__ import annotations

import warnings

import nibabel as nib
import numpy as np
from scipy.special import logit
from scipy.stats import mannwhitneyu, rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from .anatomy import CT_LABEL, affected_vertebrae, annotation_regions

LASSO_CS = np.logspace(-2, 1, 20)
SEED = 20261113


# ---- ROIs -----------------------------------------------------------------------------

def ct_roi_masks(lab: np.ndarray, annotation: str) -> list[np.ndarray]:
    """One mask per affected vertebra in the (2 mm) TotalSegmentator label map."""
    masks = [lab == CT_LABEL[n] for n in affected_vertebrae(annotation)]
    return [m for m in masks if m.any()]


def mri_roi_masks(seg: np.ndarray, annotation: str) -> list[np.ndarray]:
    masks = [np.isin(seg, labels) for labels in annotation_regions(annotation)]
    return [m for m in masks if m.any()]


# ---- feature extraction ----------------------------------------------------------------

def _to_sitk(arr, affine):
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    lps = np.diag([-1., -1., 1.])
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(lps @ affine[:3, 3]))
    image.SetDirection(tuple((lps @ affine[:3, :3] / spacing).ravel()))
    return image


def radiomics_features(image: nib.Nifti1Image, masks: list[np.ndarray], modality: str) -> dict | None:
    """PyRadiomics features of the ROI union (None when no ROI is available)."""
    import logging
    from radiomics import featureextractor

    if not masks:
        return None
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    arr = np.asarray(image.dataobj, np.float32).copy()
    union = np.logical_or.reduce(masks)
    if modality == "ct":
        arr = np.clip(arr, -1000, 2000)
    else:
        values = arr[union]
        arr = np.clip((arr - values.mean()) / max(float(values.std()), 1e-6), -3, 3) * 100
    ex = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25 if modality == "ct" else 5, resampledPixelSpacing=[2, 2, 2], interpolator="sitkBSpline",
        normalize=False, preCrop=True, padDistance=10, minimumROIDimensions=2, minimumROISize=8,
        correctMask=False, geometryTolerance=1e-3, voxelArrayShift=0, distances=[1],
        symmetricalGLCM=True, force2D=False, additionalInfo=False)
    ex.disableAllImageTypes()
    ex.enableImageTypeByName("Original")
    ex.enableImageTypeByName("LoG", customArgs={"sigma": [2., 4.]})
    ex.enableImageTypeByName("Wavelet", customArgs={"wavelet": "coif1", "level": 1, "start_level": 0})
    ex.disableAllFeatures()
    for name in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        ex.enableFeatureClassByName(name)
    result = ex.execute(_to_sitk(arr, image.affine), _to_sitk(union.astype(np.uint8), image.affine), label=1)
    return {k: float(v) for k, v in result.items() if not k.startswith("diagnostics")}


class ResNetExtractor:
    """Frozen ImageNet ResNet101; 2048-dimensional pooled features."""

    def __init__(self, device: str = "cuda"):
        import torch
        from torchvision.models import ResNet101_Weights, resnet101

        self.torch, self.device = torch, device
        weights = ResNet101_Weights.IMAGENET1K_V1
        self.net = resnet101(weights=weights)
        self.net.fc = torch.nn.Identity()
        self.net.eval().to(device)
        self.transform = weights.transforms()

    def __call__(self, image: nib.Nifti1Image, masks: list[np.ndarray], modality: str,
                 max_slices: int = 7, margin_mm: float = 10.) -> np.ndarray:
        """Masked sagittal slices per ROI -> mean over slices -> mean over ROIs (zeros if none)."""
        from PIL import Image

        if not masks:
            return np.zeros(2048, np.float32)
        canon = nib.as_closest_canonical(image)
        arr = np.asarray(canon.dataobj, np.float32)
        masks = [np.asarray(nib.as_closest_canonical(nib.Nifti1Image(m.astype(np.uint8), image.affine)).dataobj, bool)
                 for m in masks]
        spacing = np.linalg.norm(canon.affine[:3, :3], axis=0)
        if modality == "ct":
            arr = np.clip((arr + 160) / 400, 0, 1)  # level 40, width 400 HU
        else:
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0, 1)
        rois = []
        for mask in masks:
            coords = np.argwhere(mask)
            pad = np.ceil(margin_mm / spacing).astype(int)
            a, b = np.maximum(coords.min(0) - pad, 0), np.minimum(coords.max(0) + 1 + pad, arr.shape)
            area = mask.sum((1, 2))
            valid = np.flatnonzero(area > 0)
            chosen = np.sort(valid[np.argsort(-area[valid], kind="stable")[:max_slices]])
            vectors = []
            for x in chosen:  # sagittal slices in RAS orientation
                crop = arr[x, a[1]:b[1], a[2]:b[2]].copy()
                crop[~mask[x, a[1]:b[1], a[2]:b[2]]] = 0
                h, w = crop.shape
                side = int(np.ceil(max(h, w) * 8 / 7))  # letterbox so the centre crop keeps the ROI
                canvas = np.zeros((side, side), np.float32)
                canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
                rgb = Image.fromarray(np.round(canvas * 255).astype(np.uint8)).convert("RGB")
                with self.torch.inference_mode():
                    vectors.append(self.net(self.transform(rgb)[None].to(self.device)).float().cpu().numpy()[0])
            rois.append(np.mean(vectors, axis=0))
        return np.mean(rois, axis=0).astype(np.float32)


# ---- training-fold selection and classification -----------------------------------------

def _filter_block(x, y, train):
    """Imputation, scaling, U-test filter and Spearman redundancy removal, fitted on ``train``."""
    a = np.asarray(x[train], float)
    median = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in a.T])
    a = np.where(np.isfinite(a), a, median)
    scale = StandardScaler().fit(a)
    z = scale.transform(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = mannwhitneyu(z[y[train] == 0], z[y[train] == 1], axis=0).pvalue
    p = np.nan_to_num(p, nan=1.)
    order = np.argsort(p, kind="stable")
    passed = order[p[order] < .05]
    candidates = passed if len(passed) else order[:10]
    r = rankdata(z[:, candidates], axis=0)
    r -= r.mean(0)
    sd = np.sqrt((r * r).sum(0))
    usable = np.flatnonzero(sd > 1e-12)
    r = r[:, usable] / sd[usable]
    keep = []
    for j in range(r.shape[1]):  # in order of increasing P
        if not keep or np.max(np.abs(r[:, keep].T @ r[:, j])) <= .9:
            keep.append(j)
    return dict(median=median, scale=scale, columns=candidates[usable[keep]], n_passed=int(len(passed)))


def _apply_block(x, m):
    a = np.asarray(x, float)
    a = np.where(np.isfinite(a), a, m["median"])
    return m["scale"].transform(a)[:, m["columns"]]


def _prepare(blocks, y, train):
    models = [_filter_block(x, y, train) for x in blocks]
    return models, np.concatenate([_apply_block(x, m) for x, m in zip(blocks, models)], 1)


def _lasso(z, y, train, C):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return LogisticRegression(penalty="l1", C=float(C), solver="liblinear", max_iter=20000,
                                  tol=1e-6, random_state=SEED).fit(z[train], y[train])


class RadiomicsStyleModel:
    """``blocks`` is a list of (patients x features) arrays: [radiomics], [dl] or [radiomics, dl]."""

    def __init__(self, classifier: str = "random_forest"):
        assert classifier in ("random_forest", "extra_trees")
        self.classifier = classifier

    def fit(self, blocks, y, train, inner_splits):
        """``inner_splits``: list of (fit_idx, val_idx) partitions of ``train`` used to choose C."""
        y, train = np.asarray(y), np.asarray(train)
        prepared = [_prepare(blocks, y, a)[1] for a, _ in inner_splits]
        losses = []
        for C in LASSO_CS:
            oof = np.full(len(y), np.nan)
            try:
                for (a, b), z in zip(inner_splits, prepared):
                    m = _lasso(z, y, a, C)
                    if not np.any(np.abs(m.coef_[0]) > 1e-8):
                        raise ValueError("no feature selected")
                    oof[b] = m.predict_proba(z[b])[:, 1]
                losses.append(log_loss(y[train], oof[train], labels=[0, 1]))
            except (ValueError, ConvergenceWarning):
                losses.append(np.inf)  # invalid candidate
        if not np.isfinite(losses).any():
            raise RuntimeError("no penalty selected features in every inner fold")
        self.C = float(LASSO_CS[next(i for i, v in enumerate(losses) if v <= min(losses) + 1e-7)])
        self.filters, z = _prepare(blocks, y, train)
        self.lasso = _lasso(z, y, train, self.C)
        self.selected = np.flatnonzero(np.abs(self.lasso.coef_[0]) > 1e-8)
        if not len(self.selected):
            raise RuntimeError("selected penalty gives no features on the full training set")
        cls = RandomForestClassifier if self.classifier == "random_forest" else ExtraTreesClassifier
        self.trees = cls(n_estimators=500, random_state=SEED).fit(z[train][:, self.selected], y[train])
        self.prior_logit = float(logit(y[train].mean()))
        return self

    def predict_proba(self, blocks, idx):
        z = np.concatenate([_apply_block(x[idx], m) for x, m in zip(blocks, self.filters)], 1)
        return self.trees.predict_proba(z[:, self.selected])[:, 1]

    def decision_function(self, blocks, idx):
        """Equal-prior score: >= 0 predicts tuberculosis."""
        p = np.clip(self.predict_proba(blocks, idx), 1e-6, 1 - 1e-6)
        return logit(p) - self.prior_logit
