"""Spinal level parsing and segmentation label maps.

Affected levels are given as a string such as ``"T11-L1;L3"``: groups are separated
by ``;`` and a group is either one level or an inclusive range of two levels.
The resulting regions are anatomical proxy regions derived from automated
segmentation; they are not manual lesion masks.
"""
from __future__ import annotations

import re

import numpy as np

LEVELS = (*[f"C{i}" for i in range(1, 8)], *[f"T{i}" for i in range(1, 13)],
          *[f"L{i}" for i in range(1, 6)], "S1")
LEVEL_INDEX = {name: i for i, name in enumerate(LEVELS)}

# TotalSpineSeg (MRI) labels.
VERTEBRA_LABEL = {**{f"C{i}": 10 + i for i in range(1, 8)},
                  **{f"T{i}": 20 + i for i in range(1, 13)},
                  **{f"L{i}": 40 + i for i in range(1, 6)},
                  "S1": 50}
DISC_LABEL = {**{(f"C{i}", f"C{i + 1}"): 61 + i for i in range(2, 7)},
              ("C7", "T1"): 71,
              **{(f"T{i}", f"T{i + 1}"): 71 + i for i in range(1, 12)},
              ("T12", "L1"): 91,
              **{(f"L{i}", f"L{i + 1}"): 91 + i for i in range(1, 5)},
              ("L5", "S1"): 100}
LABEL_LEVEL = {v: k for k, v in VERTEBRA_LABEL.items()}
SPINE_STRUCTURES = ((11, 50), (63, 100))  # vertebra and disc label ranges

# TotalSegmentator (CT) vertebra labels in the task-292 output.
CT_VERTEBRAE = ["S1", "L5", "L4", "L3", "L2", "L1", *[f"T{i}" for i in range(12, 0, -1)],
                *[f"C{i}" for i in range(7, 0, -1)]]
CT_LABEL = {name: i + 2 for i, name in enumerate(CT_VERTEBRAE)}

# MRI context scales: (in-plane margin mm, cranio-caudal margin mm).
VIEW_MARGINS_MM = ((8.0, 8.0), (25.0, 15.0), (45.0, 30.0))


def normalize_level(text: str) -> str:
    text = text.strip().upper()
    if text in {"S", "S1"}:
        return "S1"
    if text not in LEVEL_INDEX:
        raise ValueError(f"unsupported spine level {text!r}")
    return text


def annotation_regions(annotation: str) -> list[tuple[int, ...]]:
    """MRI regions as TotalSpineSeg label tuples, one per affected motion segment.

    A single level gives the vertebra plus its adjacent discs; a range is split into
    adjacent vertebra pairs with the intervening disc.
    """
    regions: list[tuple[int, ...]] = []
    for group in str(annotation).split(";"):
        tokens = [normalize_level(x) for x in group.strip().split("-") if x.strip()]
        if len(tokens) == 1:
            center = LEVEL_INDEX[tokens[0]]
            labels = {VERTEBRA_LABEL[tokens[0]]}
            for j in (center - 1, center):
                if 0 <= j < len(LEVELS) - 1 and (LEVELS[j], LEVELS[j + 1]) in DISC_LABEL:
                    labels.add(DISC_LABEL[(LEVELS[j], LEVELS[j + 1])])
            regions.append(tuple(sorted(labels)))
            continue
        if len(tokens) != 2:
            raise ValueError(f"invalid annotation group {group!r}")
        lo, hi = sorted((LEVEL_INDEX[tokens[0]], LEVEL_INDEX[tokens[1]]))
        if lo == hi:
            regions.append((VERTEBRA_LABEL[LEVELS[lo]],))
            continue
        for j in range(lo, hi):
            pair = (LEVELS[j], LEVELS[j + 1])
            labels = {VERTEBRA_LABEL[pair[0]], VERTEBRA_LABEL[pair[1]]}
            if pair in DISC_LABEL:
                labels.add(DISC_LABEL[pair])
            regions.append(tuple(sorted(labels)))
    return list(dict.fromkeys(regions))


def affected_vertebrae(annotation: str) -> list[str]:
    """All vertebrae covered by the annotation, in cranio-caudal order (used for CT)."""
    out = set()
    for part in str(annotation).split(";"):
        tokens = re.findall(r"[CTLS]\d+", part)
        if len(tokens) == 2 and all(t in LEVEL_INDEX for t in tokens):
            a, b = sorted((LEVEL_INDEX[tokens[0]], LEVEL_INDEX[tokens[1]]))
            out.update(LEVELS[a:b + 1])
        else:
            out.update(t for t in tokens if t in LEVEL_INDEX)
    return sorted(out, key=LEVEL_INDEX.get)


def region_vertebrae(region: tuple[int, ...]) -> list[str]:
    return [LABEL_LEVEL[v] for v in region if v in LABEL_LEVEL]


def bbox_with_margin(mask: np.ndarray, zooms, margin_mm: tuple[float, float]):
    """Bounding box of ``mask`` enlarged in the first two (sagittal-plane) axes."""
    coords = np.argwhere(mask)
    if not len(coords):
        raise ValueError("empty target mask")
    starts, stops = coords.min(0), coords.max(0) + 1
    margin = np.asarray([margin_mm[0] / zooms[0], margin_mm[1] / zooms[1]], float)
    starts[:2] = np.floor(starts[:2] - margin).astype(int)
    stops[:2] = np.ceil(stops[:2] + margin).astype(int)
    return np.maximum(starts, 0), np.minimum(stops, np.asarray(mask.shape))


def spine_structure_mask(seg: np.ndarray) -> np.ndarray:
    (a, b), (c, d) = SPINE_STRUCTURES
    return ((seg >= a) & (seg <= b)) | ((seg >= c) & (seg <= d))
