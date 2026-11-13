"""TotalSegmentator vertebra labels for every <id>_CT.nii.gz in a directory."""
import argparse
from pathlib import Path

import nibabel as nib

from spinefm.ct import segment_vertebrae

ap = argparse.ArgumentParser()
ap.add_argument("--nifti", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
for path in sorted(args.nifti.glob("*_CT.nii.gz")):
    target = args.out / path.name.replace("_CT.nii.gz", "_CT_vertebrae.nii.gz")
    if not target.exists():
        nib.save(segment_vertebrae(nib.load(path), args.device), target)
        print("segmented", path.name, flush=True)
