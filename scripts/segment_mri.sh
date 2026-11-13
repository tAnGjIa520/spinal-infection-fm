#!/usr/bin/env bash
# TotalSpineSeg for T1 and T2 volumes produced by convert_dicom.py.
# Usage: bash scripts/segment_mri.sh <nifti_dir> <out_dir>
# Labels used downstream are read from <out_dir>/<T1w|T2w>/step2_output/<id>_<T1w|T2w>.nii.gz
set -euo pipefail
NIFTI=$1; OUT=$2
for SEQ in T1w T2w; do
  mkdir -p "$OUT/$SEQ/input"
  for f in "$NIFTI"/*_"$SEQ".nii.gz; do ln -sf "$(realpath "$f")" "$OUT/$SEQ/input/"; done
  totalspineseg "$OUT/$SEQ/input" "$OUT/$SEQ"
done
