"""Per-patient feature extraction: seven global feature groups and three regional bags."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .anatomy import affected_vertebrae, annotation_regions, region_vertebrae, VERTEBRA_LABEL
from .ct import CTFMEncoder, ct_deep_features, ct_descriptors, resample_2mm
from .framework import GLOBAL_GROUPS, MODALITIES
from .mri import M3DCLIPEncoder, mri_region_embeddings


@dataclass
class PatientInputs:
    patient_id: str
    annotation: str                  # affected levels, e.g. "T11-L1;L3"
    ct: Path                         # CT NIfTI
    ct_seg: Path                     # TotalSegmentator vertebra labels for that CT
    t1: Path
    t1_seg: Path                     # TotalSpineSeg labels for T1
    t2: Path
    t2_seg: Path
    annotation_reliable: bool = True  # set False to disable regional features for this patient


def extract_patient(p: PatientInputs, ct_encoder: CTFMEncoder, mri_encoder: M3DCLIPEncoder) -> dict:
    out: dict[str, np.ndarray] = {}
    ct2, lab2 = resample_2mm(nib.load(p.ct), nib.load(p.ct_seg))
    windows, vertebrae, names = ct_deep_features(ct_encoder, ct2, lab2)
    out["ct_rad"], _ = ct_descriptors(ct2, lab2, p.annotation, p.annotation_reliable)
    out["ct_gmean"], out["ct_gmax"] = windows.mean(0), windows.max(0)

    requested = affected_vertebrae(p.annotation)
    complete = bool(requested) and all(n in names for n in requested)
    # Regional features require a usable level annotation that CT segmentation fully localises.
    quality = bool(p.annotation_reliable and complete)
    regions = annotation_regions(p.annotation)

    ct_needed = set()
    for region in regions:
        bones = region_vertebrae(region)
        if quality and bones and all(b in names for b in bones):
            ct_needed.update(bones)
    out["ct_bag"] = vertebrae[[i for i, n in enumerate(names) if n in ct_needed]].astype(np.float32).reshape(-1, 512)

    for mod, image, seg in (("t1", p.t1, p.t1_seg), ("t2", p.t2, p.t2_seg)):
        emb, index, fallback, labels = mri_region_embeddings(mri_encoder, nib.load(image), nib.load(seg), p.annotation)
        out[f"med_{mod}_mean"] = emb.mean(0).reshape(-1)
        out[f"med_{mod}_max"] = emb.max(0).reshape(-1)
        keep = []
        if not fallback:
            for row, j in enumerate(index):
                bones = [VERTEBRA_LABEL[b] for b in region_vertebrae(regions[j])]
                if quality and bones and all(np.any(labels == v) for v in bones):
                    keep.append(row)
        out[f"{mod}_bag"] = emb[keep].reshape(-1, emb.shape[1], emb.shape[2])
    out["quality"] = np.array(quality)
    return out


def save_patient(path: Path, features: dict, label: int, patient_id: str) -> None:
    np.savez_compressed(path, label=label, patient_id=patient_id, **features)


def load_cohort(paths: list[Path]):
    """Stack per-patient files into global matrices and ragged regional bags."""
    records = [dict(np.load(p, allow_pickle=False)) for p in paths]
    cohort = {g: np.stack([r[g] for r in records]).astype(np.float32) for g in GLOBAL_GROUPS}
    for mod in MODALITIES:
        cohort[f"{mod}_bags"] = [r[f"{mod}_bag"].astype(np.float32) for r in records]
    cohort["y"] = np.array([int(r["label"]) for r in records])
    cohort["ids"] = np.array([str(r["patient_id"]) for r in records])
    return cohort
