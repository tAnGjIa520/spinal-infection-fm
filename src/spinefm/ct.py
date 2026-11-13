"""CT: DICOM loading, vertebra segmentation, CT-FM encoding and hand-crafted descriptors."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

from .anatomy import CT_LABEL, CT_VERTEBRAE, LEVELS, affected_vertebrae

MIN_VERTEBRA_VOXELS = 50
WINDOW = 64          # slices per global CT-FM window (2 mm voxels)
MAX_WINDOWS = 12


def read_ct_zip(path: Path) -> nib.Nifti1Image:
    """Read a single-series CT DICOM ZIP into a RAS NIfTI image.

    One missing slice is linearly interpolated; other spacing irregularities fail.
    """
    import pydicom

    series: dict[str, list] = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            try:
                d = pydicom.dcmread(io.BytesIO(z.read(name)))
            except pydicom.errors.InvalidDicomError:
                continue
            if hasattr(d, "PixelData") and hasattr(d, "ImagePositionPatient"):
                series.setdefault(str(d.SeriesInstanceUID), []).append(d)
    if len(series) != 1:
        raise ValueError(f"expected a single CT series, found {len(series)}")
    ds = next(iter(series.values()))
    orient = np.array(ds[0].ImageOrientationPatient, float)
    normal = np.cross(orient[:3], orient[3:])
    ds.sort(key=lambda d: np.dot(np.array(d.ImagePositionPatient, float), normal))
    pos = np.array([d.ImagePositionPatient for d in ds], float)
    dif = np.diff(pos, axis=0)
    step = np.median(dif, axis=0)
    if len(ds) < 3 or np.linalg.norm(step) < .1:
        raise ValueError("insufficient or duplicate slices")
    gaps = np.flatnonzero(np.linalg.norm(dif - step, axis=1) > .2)
    if len(gaps) and not (len(gaps) == 1 and np.linalg.norm(dif[gaps[0]] - 2 * step) < .2):
        raise ValueError("non-uniform slice positions beyond a single missing slice")
    ps = np.array(ds[0].PixelSpacing, float)
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1))
                    + float(getattr(d, "RescaleIntercept", 0)) for d in ds], axis=2)
    vol = vol.transpose(1, 0, 2)  # row, column, slice -> column, row, slice
    if len(gaps):
        g = int(gaps[0])
        vol = np.insert(vol, g + 1, (vol[:, :, g] + vol[:, :, g + 1]) / 2, axis=2)
    aff = np.eye(4)
    aff[:3, 0] = orient[:3] * ps[1]
    aff[:3, 1] = orient[3:] * ps[0]
    aff[:3, 2] = step
    aff[:3, 3] = pos[0]
    return nib.Nifti1Image(vol, np.diag([-1, -1, 1, 1]) @ aff)  # LPS -> RAS


def segment_vertebrae(image: nib.Nifti1Image, device: str = "cuda") -> nib.Nifti1Image:
    """TotalSegmentator vertebra labels (task 292, 1.5 mm, fold 0, no TTA)."""
    from totalsegmentator.nnunet import nnUNet_predict_image

    seg, _, _ = nnUNet_predict_image(
        image, None, 292, model="3d_fullres", folds=[0], trainer="nnUNetTrainerNoMirroring",
        tta=False, multilabel_image=True, resample=1.5, crop=None, task_name="total",
        save_binary=False, nr_threads_resampling=2, nr_threads_saving=1, quiet=True,
        verbose=False, skip_saving=True, device=device)
    return seg


def resample_2mm(image: nib.Nifti1Image, seg: nib.Nifti1Image):
    ct = resample_to_output(image, voxel_sizes=(2, 2, 2), order=1, cval=-1024)
    return ct, resample_from_to(seg, ct, order=0)


def crop(v: np.ndarray, center, shape) -> np.ndarray:
    out = np.full(shape, -1024, np.float32)
    starts = np.round(np.asarray(center) - np.asarray(shape) / 2).astype(int)
    src, dst = [], []
    for n, s, k in zip(v.shape, starts, shape):
        lo, hi = max(s, 0), min(s + k, n)
        if hi <= lo:
            return out
        src.append(slice(lo, hi))
        dst.append(slice(lo - s, hi - s))
    out[tuple(dst)] = v[tuple(src)]
    return out


class CTFMEncoder:
    """Frozen CT-FM SegResEncoder; features are globally pooled last-stage activations."""

    output_dim = 512

    def __init__(self, weights_dir: Path, device: str = "cuda"):
        import torch
        from lighter_zoo import SegResEncoder
        from safetensors.torch import load_file

        weights_dir = Path(weights_dir)
        self.torch, self.device = torch, device
        self.model = SegResEncoder(**json.loads((weights_dir / "config.json").read_text()))
        self.model.load_state_dict(load_file(str(weights_dir / "model.safetensors")))
        self.model.to(device).eval().requires_grad_(False)

    def __call__(self, patches) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as F

        out = []
        with torch.inference_mode():
            for p in patches:
                x = torch.from_numpy(np.clip(p, -1024, 2048).copy()).to(self.device)[None, None]
                x = (F.interpolate(x, size=(64, 192, 192), mode="trilinear", align_corners=False) + 1024) / 3072
                with torch.autocast(self.device.split(":")[0], dtype=torch.float16):
                    f = F.adaptive_avg_pool3d(self.model(x)[-1], 1).flatten(1)
                out.append(f.float().cpu().numpy()[0])
        return np.stack(out)


def window_starts(length: int) -> list[int]:
    starts = sorted(set(list(range(0, max(length - WINDOW, 0) + 1, WINDOW // 2)) + [max(length - WINDOW, 0)]))
    if len(starts) > MAX_WINDOWS:
        starts = [starts[i] for i in np.unique(np.linspace(0, len(starts) - 1, MAX_WINDOWS).round().astype(int))]
    return starts


def detect_vertebrae(lab_zyx: np.ndarray):
    names, centers = [], []
    for name, k in CT_LABEL.items():
        vox = np.argwhere(lab_zyx == k)
        if len(vox) >= MIN_VERTEBRA_VOXELS:
            names.append(name)
            centers.append(np.median(vox, axis=0))
    return names, centers


def ct_deep_features(encoder: CTFMEncoder, ct2: nib.Nifti1Image, lab2: nib.Nifti1Image):
    """Global window embeddings and one embedding per detected vertebra."""
    v = ct2.get_fdata(dtype=np.float32).transpose(2, 1, 0)
    m = np.asarray(lab2.dataobj).astype(np.uint8).transpose(2, 1, 0)
    windows = encoder([crop(v, (s + WINDOW // 2, v.shape[1] / 2, v.shape[2] / 2), (WINDOW, v.shape[1], v.shape[2]))
                       for s in window_starts(len(v))])
    names, centers = detect_vertebrae(m)
    if not centers:
        raise ValueError("no vertebra detected")
    vertebrae = encoder([crop(v, c, (64, 96, 96)) for c in centers])
    return windows, vertebrae, names


# ---- hand-crafted CT descriptors (333 values) --------------------------------------

def _stats(v, region, prefix, denom):
    x = np.asarray(v[region], np.float64)
    x = x[np.isfinite(x) & (x >= -1024) & (x <= 3071)]
    keys = ["mean", "std", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "iqr", "entropy",
            "lt_m30", "m30_80", "ge100", "ge300"]
    vals = [np.log1p(len(x)), len(x) / max(denom, 1)]
    if not len(x):
        vals += [0.] * len(keys)
    else:
        q = np.percentile(x, [5, 10, 25, 50, 75, 90, 95])
        hist = np.histogram(np.clip(x, -1024, 1200), bins=32, range=(-1024, 1200))[0].astype(float)
        p = hist / hist.sum()
        vals += [x.mean(), x.std(), *q, q[4] - q[2], -(p[p > 0] * np.log(p[p > 0])).sum(),
                 np.mean(x < -30), np.mean((x >= -30) & (x < 80)), np.mean(x >= 100), np.mean(x >= 300)]
    return [float(z) for z in vals], [f"{prefix}_{n}" for n in ["log_n", "fraction", *keys]]


def _disc_between(shape, c1, c2, spacing, half_parallel=8.0, radius=32.0):
    """Cylinder between two adjacent vertebral centres approximating the disc/endplates."""
    c = (c1 + c2) / 2
    u = (c2 - c1) * spacing
    u = u / max(np.linalg.norm(u), 1e-6)
    rad = np.ceil((radius + half_parallel) / spacing).astype(int)
    lo = np.maximum(np.floor(c).astype(int) - rad, 0)
    hi = np.minimum(np.ceil(c).astype(int) + rad + 1, shape)
    grid = np.indices(tuple(hi - lo)).reshape(3, -1).T + lo
    delta = (grid - c) * spacing
    parallel = delta @ u
    perp = np.sqrt(np.maximum((delta * delta).sum(1) - parallel * parallel, 0))
    sub = np.zeros(tuple(hi - lo), bool)
    sub.reshape(-1)[(np.abs(parallel) <= half_parallel) & (perp <= radius)] = True
    mask = np.zeros(shape, bool)
    mask[tuple(slice(a, b) for a, b in zip(lo, hi))] = sub
    return mask


def ct_descriptors(ct2: nib.Nifti1Image, lab2: nib.Nifti1Image, annotation: str, annotation_reliable: bool = True):
    """Intensity, texture-proxy and geometry descriptors around the affected vertebrae."""
    from scipy.ndimage import distance_transform_edt, gaussian_gradient_magnitude, laplace

    v = np.asarray(ct2.dataobj, np.float32)
    lab = np.rint(np.asarray(lab2.dataobj)).astype(np.uint8)
    spacing = np.asarray(ct2.header.get_zooms()[:3], float)
    req = affected_vertebrae(annotation)
    available = [n for n in req if np.sum(lab == CT_LABEL[n]) >= MIN_VERTEBRA_VOXELS]
    reliable = annotation_reliable and len(available) == len(req) and len(req) > 0
    selected = available or [n for n in CT_VERTEBRAE if np.sum(lab == CT_LABEL[n]) >= MIN_VERTEBRA_VOXELS]
    target = np.isin(lab, [CT_LABEL[n] for n in selected])
    denom = int(np.sum(lab > 1))
    inside = distance_transform_edt(target, sampling=spacing)
    outside = distance_transform_edt(~target, sampling=spacing)
    regions = {"target": target, "target_surface3": target & (inside <= 3), "target_core": target & (inside > 3),
               "ring0_5": ~target & (outside > 0) & (outside <= 5),
               "ring5_10": ~target & (outside > 5) & (outside <= 10),
               "ring10_20": ~target & (outside > 10) & (outside <= 20)}
    centers = {n: np.median(np.argwhere(lab == CT_LABEL[n]), axis=0) for n in selected if np.any(lab == CT_LABEL[n])}
    disc, pairs = np.zeros(v.shape, bool), []
    for a, b in zip(req[:-1], req[1:]):
        if a in centers and b in centers and abs(LEVELS.index(a) - LEVELS.index(b)) == 1:
            disc |= _disc_between(v.shape, centers[a], centers[b], spacing)
            pairs.append((a, b))
    regions.update(disc_endplate=disc, disc_soft=disc & ~target, disc_bone=disc & target)
    feats, names = [], []
    for name, mask in regions.items():
        q, n = _stats(v, mask, name, denom)
        feats += q
        names += n
    clipped = np.clip(v, -1024, 1500)
    grad, lapv = gaussian_gradient_magnitude(clipped, sigma=1), np.abs(laplace(clipped))
    for name in ["target", "ring0_5", "ring5_10", "disc_endplate", "disc_soft"]:
        for arr, suffix in [(grad, "grad"), (lapv, "lap")]:
            q, n = _stats(arr, regions[name], f"{name}_{suffix}", denom)
            feats += q
            names += n
    cent = np.asarray(list(centers.values())) if centers else np.zeros((0, 3))
    extent = np.ptp(cent, axis=0) * spacing if len(cent) > 1 else np.zeros(3)
    feats += [len(req), len(available), len(selected), len(pairs), float(reliable),
              np.log1p(target.sum()), np.log1p(disc.sum()), *extent]
    names += ["requested_count", "available_count", "selected_count", "disc_pair_count", "reliable",
              "target_log_voxels", "disc_log_voxels", "center_extent_x", "center_extent_y", "center_extent_z"]
    return np.asarray(feats, np.float32), names
