"""Extract global and regional features for every patient.

cohort.csv columns: patient_id, label (1 = tuberculosis, 0 = pyogenic),
levels (affected levels, e.g. "T11-L1;L3"), optional annotation_reliable (0/1).
"""
import argparse
from pathlib import Path

import pandas as pd

from spinefm.ct import CTFMEncoder
from spinefm.features import PatientInputs, extract_patient, save_patient
from spinefm.mri import M3DCLIPEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", type=Path, required=True)
ap.add_argument("--nifti", type=Path, required=True, help="output of convert_dicom.py")
ap.add_argument("--ct-seg", type=Path, required=True, help="output of segment_ct.py")
ap.add_argument("--mri-seg", type=Path, required=True, help="output of segment_mri.sh")
ap.add_argument("--ctfm", type=Path, required=True, help="CT-FM feature extractor weights directory")
ap.add_argument("--m3dclip", type=Path, required=True, help="M3D-CLIP model directory")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

ct_encoder, mri_encoder = CTFMEncoder(args.ctfm, args.device), M3DCLIPEncoder(args.m3dclip, args.device)
args.out.mkdir(parents=True, exist_ok=True)
for row in pd.read_csv(args.cohort, dtype={"patient_id": str}).itertuples(index=False):
    pid, target = row.patient_id, args.out / f"{row.patient_id}.npz"
    if target.exists():
        continue
    p = PatientInputs(
        patient_id=pid, annotation=row.levels,
        ct=args.nifti / f"{pid}_CT.nii.gz", ct_seg=args.ct_seg / f"{pid}_CT_vertebrae.nii.gz",
        t1=args.nifti / f"{pid}_T1w.nii.gz", t1_seg=args.mri_seg / "T1w/step2_output" / f"{pid}_T1w.nii.gz",
        t2=args.nifti / f"{pid}_T2w.nii.gz", t2_seg=args.mri_seg / "T2w/step2_output" / f"{pid}_T2w.nii.gz",
        annotation_reliable=bool(getattr(row, "annotation_reliable", 1)))
    save_patient(target, extract_patient(p, ct_encoder, mri_encoder), int(row.label), pid)
    print("extracted", pid, flush=True)
