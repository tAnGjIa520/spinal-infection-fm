<div align="center">

# spinal-infection-fm

**Anatomy-guided CT + MRI framework on frozen foundation models for differentiating tuberculous from pyogenic spondylitis**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/paper-under%20review-orange.svg)](#citation)

**English** | [简体中文](README_zh.md)

<img src="docs/framework.png" width="820" alt="Framework overview">

</div>

---

## TL;DR

Tuberculous and pyogenic spinal infections need different treatment, but their imaging appearances overlap. This repository contains the code for a compact, fully reproducible framework that:

- uses **CT, T1-weighted and T2-weighted MRI** together,
- localises affected vertebrae **automatically** (TotalSegmentator / TotalSpineSeg) from a level annotation such as `T11-L1`,
- extracts features with **frozen 3D foundation models** (CT-FM, M3D-CLIP) — no neural network is trained on study labels,
- summarises regional features with **fixed multiple-instance learning (MIL) pooling** (mean + shrinkage variance), and
- classifies with **Ledoit–Wolf shrinkage linear discriminant analysis**, which stays stable when features outnumber patients.

Only a small linear stage is fitted to local data, so the whole pipeline runs on a few hundred patients and every decision can be decomposed exactly into feature-group contributions.

## Highlights

| | |
|---|---|
| 🧊 **Frozen encoders** | CT-FM (SegResEncoder) for CT, M3D-CLIP for MRI; no fine-tuning |
| 🦴 **Anatomy-guided ROIs** | Affected-level annotation + automated segmentation, no manual delineation |
| 🧩 **Fixed MIL pooling** | Per-modality mean and shrinkage variance Σ(x−μ)²/(n+2) |
| 📐 **Shrinkage LDA** | Equal priors, threshold fixed at 0 |
| 🔒 **Leakage-safe** | Every scaler, PCA, sampler and classifier is fitted on training patients only |
| 📊 **Comparators included** | Radiomics, DL and DLR models adapted from a published ESJ study |

## Results

Three repetitions of stratified, patient-level 5-fold cross-validation on 145 patients (50 tuberculous, 95 pyogenic). 95% CIs from a class-stratified patient bootstrap of the saved out-of-fold predictions.

| Model | AUC [95% CI] | Accuracy, % | TB sensitivity, % | PY specificity, % |
|---|---|---|---|---|
| **Proposed framework** | **0.824** [0.754–0.887] | **77.2** | 72.0 | 80.0 |
| CT only | 0.800 [0.725–0.871] | 74.7 | 72.0 | 76.1 |
| MRI only | 0.717 [0.639–0.794] | 66.0 | 57.3 | 70.5 |
| Radiomics (adapted) | 0.729 [0.650–0.802] | 66.4 | 71.3 | 63.9 |
| DL (adapted) | 0.668 [0.579–0.750] | 62.8 | 67.3 | 60.4 |
| DLR (adapted) | 0.695 [0.610–0.774] | 66.2 | 68.7 | 64.9 |

| Ablation | AUC |
|---|---|
| Without regional features | 0.818 |
| Without global features | 0.743 |

> These are internal cross-validation results on a development cohort, **not** an external validation. The ROIs are anatomical proxies derived from level annotations and automated segmentation, **not** manual lesion masks.

## Installation

```bash
git clone https://github.com/tAnGjIa520/spinal-infection-fm.git
cd spinal-infection-fm
conda create -n spinefm python=3.10 -y && conda activate spinefm
pip install -e ".[extract,baselines,test]"
```

[TotalSpineSeg](https://github.com/neuropoly/totalspineseg) is installed separately and provides the `totalspineseg` command. The classifier and evaluation run on CPU; feature extraction expects a CUDA GPU.

**Pretrained weights** (download separately; nothing is fine-tuned):

| Model | Source | Expected local folder |
|---|---|---|
| CT-FM feature extractor | Hugging Face `project-lighter/ct_fm_feature_extractor` | `config.json`, `model.safetensors` |
| M3D-CLIP | Hugging Face `GoodBaiBai88/M3D-CLIP` | `config.json`, `configuration_m3d_clip.py`, `modeling_m3d_clip.py`, `model.safetensors` |
| TotalSegmentator | downloaded automatically on first use | — |
| ResNet101 (comparators) | torchvision ImageNet weights, downloaded automatically | — |

## Data preparation

Prepare one row per patient in `cohort.csv`:

| column | description |
|---|---|
| `patient_id` | pseudonymous identifier |
| `label` | `1` = tuberculous, `0` = pyogenic |
| `levels` | affected levels, e.g. `T11-L1;L3` (`;` separates groups, `A-B` is an inclusive range) |
| `annotation_reliable` | optional `0/1`; `0` disables regional features for that patient |
| `ct_zip`, `t1_zip`, `t2_zip` | paths to one DICOM ZIP per series (only for step 1) |

```
data/
├── nifti/       <id>_CT.nii.gz, <id>_T1w.nii.gz, <id>_T2w.nii.gz
├── ct_seg/      <id>_CT_vertebrae.nii.gz
└── mri_seg/     T1w/step2_output/<id>_T1w.nii.gz, T2w/step2_output/<id>_T2w.nii.gz
```

## Usage

**Step 1 — Convert DICOM to NIfTI**
```bash
python scripts/convert_dicom.py --manifest cohort.csv --out data/nifti
```

**Step 2 — Segment the spine**
```bash
python scripts/segment_ct.py --nifti data/nifti --out data/ct_seg      # TotalSegmentator
bash   scripts/segment_mri.sh data/nifti data/mri_seg                    # TotalSpineSeg
```

**Step 3 — Extract features with the frozen encoders**
```bash
python scripts/extract_features.py --cohort cohort.csv --nifti data/nifti \
    --ct-seg data/ct_seg --mri-seg data/mri_seg \
    --ctfm weights/ct_fm_feature_extractor --m3dclip weights/m3d_clip --out features
```

**Step 4 — Cross-validate the framework and its ablations**
```bash
python scripts/cross_validate.py --features features --out results
```
Writes out-of-fold scores, the splits used (`splits.csv`) and `summary.csv` with bootstrap 95% CIs.

**Step 5 (optional) — Comparator models**
```bash
python scripts/extract_baseline_features.py --cohort cohort.csv --nifti data/nifti \
    --ct-seg data/ct_seg --mri-seg data/mri_seg --out baseline_features
python scripts/cross_validate_baselines.py --features baseline_features \
    --splits results/splits.csv --out results            # add --mri-only for T1+T2 only
```

**Python API**
```python
from pathlib import Path
from spinefm import AnatomyGuidedLDA, Seeds
from spinefm.features import load_cohort

cohort = load_cohort(sorted(Path("features").glob("*.npz")))
model = AnatomyGuidedLDA().fit(cohort, train_idx, Seeds.for_fold(repeat=0, fold=0))
scores = model.decision_function(cohort, test_idx)   # >= 0 -> tuberculosis
```

## Repository structure

```
src/spinefm/
├── anatomy.py      level parsing, segmentation label maps, ROI geometry
├── ct.py           CT loading, TotalSegmentator, CT-FM encoding, CT descriptors
├── mri.py          MRI loading, M3D-CLIP region embeddings
├── features.py     per-patient global features and regional bags
├── framework.py    training-fold PCA, MIL pooling, shrinkage LDA
├── evaluation.py   repeated cross-validation, metrics, bootstrap
└── baselines.py    radiomics / DL / DLR comparators
scripts/            command-line pipeline (steps 1–5)
tests/              synthetic unit tests (`pytest -q`)
```

## Reproducibility

Using the cached features and original splits of the study cohort, this package reproduces the reported out-of-fold scores (maximum difference 2 × 10⁻⁵, no changed prediction), the reported confidence intervals, the ablations and the comparator predictions (difference 0). Re-running the frozen encoders reproduces the cached embeddings exactly with PyTorch 2.9 / CUDA 12.8; other software stacks can introduce small numerical differences (cosine similarity ≥ 0.9999).

## Data availability

The patient images, features and predictions used in the study are **not publicly available**. This repository contains code only.

## Citation

The paper is under review. Please cite it once published; a BibTeX entry will be added here.

## Acknowledgements

This work builds on [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), [TotalSpineSeg](https://github.com/neuropoly/totalspineseg), [CT-FM](https://huggingface.co/project-lighter/ct_fm_feature_extractor), [M3D](https://github.com/BAAI-DCAI/M3D), [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) and [scikit-learn](https://scikit-learn.org/). The README layout is inspired by [CLAM](https://github.com/mahmoodlab/CLAM).

## License

Released under the [MIT License](LICENSE). The pretrained models are subject to their own licences.
