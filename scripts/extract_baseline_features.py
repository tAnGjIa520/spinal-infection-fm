"""Radiomics and ResNet101 features for the comparator models (same inputs as extract_features.py)."""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from spinefm.baselines import ResNetExtractor, ct_roi_masks, mri_roi_masks, radiomics_features
from spinefm.ct import resample_2mm

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", type=Path, required=True)
ap.add_argument("--nifti", type=Path, required=True)
ap.add_argument("--ct-seg", type=Path, required=True)
ap.add_argument("--mri-seg", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

resnet = ResNetExtractor(args.device)
args.out.mkdir(parents=True, exist_ok=True)
for row in pd.read_csv(args.cohort, dtype={"patient_id": str}).itertuples(index=False):
    pid, target = row.patient_id, args.out / f"{row.patient_id}.npz"
    if target.exists():
        continue
    use = bool(getattr(row, "annotation_reliable", 1))
    ct, lab = resample_2mm(nib.load(args.nifti / f"{pid}_CT.nii.gz"), nib.load(args.ct_seg / f"{pid}_CT_vertebrae.nii.gz"))
    images = {"ct": (ct, ct_roi_masks(np.rint(np.asarray(lab.dataobj)).astype(np.uint8), row.levels) if use else [])}
    for mod, seq in (("t1", "T1w"), ("t2", "T2w")):
        image = nib.load(args.nifti / f"{pid}_{seq}.nii.gz")
        seg = np.asarray(nib.load(args.mri_seg / seq / "step2_output" / f"{pid}_{seq}.nii.gz").dataobj, np.int16)
        images[mod] = (image, mri_roi_masks(seg, row.levels) if use else [])
    rad = {mod: radiomics_features(img, masks, mod) for mod, (img, masks) in images.items()}
    dl = {mod: resnet(img, masks, mod) for mod, (img, masks) in images.items()}
    np.savez_compressed(target, label=int(row.label), patient_id=pid, radiomics=json.dumps(rad),
                        **{f"dl_{m}": v for m, v in dl.items()})
    print("extracted", pid, flush=True)
