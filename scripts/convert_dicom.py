"""Convert CT / T1 / T2 DICOM ZIP archives to NIfTI.

manifest.csv columns: patient_id, ct_zip, t1_zip, t2_zip
Writes <out>/<patient_id>_CT.nii.gz, _T1w.nii.gz and _T2w.nii.gz.
"""
import argparse
from pathlib import Path

import nibabel as nib
import pandas as pd

from spinefm.ct import read_ct_zip
from spinefm.mri import read_mri_zip

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
for row in pd.read_csv(args.manifest, dtype={"patient_id": str}).itertuples(index=False):
    for column, suffix, reader in (("ct_zip", "CT", read_ct_zip), ("t1_zip", "T1w", read_mri_zip), ("t2_zip", "T2w", read_mri_zip)):
        target = args.out / f"{row.patient_id}_{suffix}.nii.gz"
        if not target.exists():
            nib.save(reader(Path(getattr(row, column))), target)
    print("converted", row.patient_id, flush=True)
