"""MRI: DICOM conversion and frozen M3D-CLIP region embeddings.

Spine structures are segmented beforehand with the TotalSpineSeg command-line tool;
see ``scripts/segment_mri.sh``.
"""
from __future__ import annotations

import importlib.util
import inspect
import io
import sys
import types
import zipfile
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from .anatomy import VIEW_MARGINS_MM, annotation_regions, bbox_with_margin, spine_structure_mask


def read_mri_zip(path: Path) -> nib.Nifti1Image:
    """Read the dominant series of an MRI DICOM ZIP into a RAS NIfTI image."""
    import pydicom

    datasets = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            try:
                ds = pydicom.dcmread(io.BytesIO(archive.read(name)), force=True)
            except Exception:
                continue
            if all(hasattr(ds, k) for k in ("PixelData", "ImageOrientationPatient", "ImagePositionPatient")):
                datasets.append(ds)
    if not datasets:
        raise RuntimeError(f"no spatial DICOM images in {path}")
    uid, _ = Counter(str(ds.SeriesInstanceUID) for ds in datasets).most_common(1)[0]
    datasets = [ds for ds in datasets if str(ds.SeriesInstanceUID) == uid]
    orientation = np.asarray(datasets[0].ImageOrientationPatient, float)
    col_dir, row_dir = orientation[:3], orientation[3:]
    normal = np.cross(col_dir, row_dir)
    datasets.sort(key=lambda ds: float(np.dot(np.asarray(ds.ImagePositionPatient, float), normal)))
    positions = np.stack([np.asarray(ds.ImagePositionPatient, float) for ds in datasets])
    arrays = []
    for ds in datasets:
        image = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0)) \
            + float(getattr(ds, "RescaleIntercept", 0.0))
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            image = image.max() + image.min() - image
        arrays.append(image)
    volume = np.stack(arrays, axis=-1).transpose(1, 0, 2)
    row_spacing, col_spacing = map(float, datasets[0].PixelSpacing)
    if len(datasets) > 1:
        slice_vector = np.median(np.diff(positions, axis=0), axis=0)
    else:
        ds = datasets[0]
        slice_vector = normal * float(getattr(ds, "SpacingBetweenSlices", getattr(ds, "SliceThickness", 1.0)))
    affine = np.eye(4)
    affine[:3, 0] = col_dir * col_spacing
    affine[:3, 1] = row_dir * row_spacing
    affine[:3, 2] = slice_vector
    affine[:3, 3] = positions[0]
    image = nib.Nifti1Image(volume, np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine)
    image.header.set_xyzt_units("mm")
    return image


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class M3DCLIPEncoder:
    """Frozen M3D-CLIP vision encoder; returns the L2-normalised projected CLS token.

    ``model_dir`` is a local copy of the official M3D-CLIP release
    (config.json, configuration_m3d_clip.py, modeling_m3d_clip.py, model.safetensors).
    """

    output_dim = 768

    def __init__(self, model_dir: Path, device: str = "cuda"):
        import torch
        from safetensors.torch import load_file
        import monai.networks.blocks.patchembedding as patch_module

        self.torch, self.device = torch, device
        model_dir = Path(model_dir)
        package = f"_m3dclip_{abs(hash(str(model_dir.resolve())))}"
        module = types.ModuleType(package)
        module.__path__ = [str(model_dir)]
        sys.modules[package] = module
        configuration = _load_module(f"{package}.configuration_m3d_clip", model_dir / "configuration_m3d_clip.py")
        # M3D-CLIP targets the older MONAI argument ``pos_embed`` (now ``proj_type``).
        original = patch_module.PatchEmbeddingBlock
        if "pos_embed" not in inspect.signature(original).parameters:
            class Compatible(original):
                def __init__(self, in_channels, img_size, patch_size, hidden_size, num_heads,
                             pos_embed="conv", dropout_rate=0.0, spatial_dims=3):
                    super().__init__(in_channels=in_channels, img_size=img_size, patch_size=patch_size,
                                     hidden_size=hidden_size, num_heads=num_heads, proj_type=pos_embed,
                                     pos_embed_type="learnable", dropout_rate=dropout_rate,
                                     spatial_dims=spatial_dims)
            patch_module.PatchEmbeddingBlock = Compatible
        try:
            modeling = _load_module(f"{package}.modeling_m3d_clip", model_dir / "modeling_m3d_clip.py")
        finally:
            patch_module.PatchEmbeddingBlock = original
        config = configuration.M3DCLIPConfig.from_json_file(model_dir / "config.json")
        self.vision = modeling.ViT(
            in_channels=config.in_channels, img_size=config.img_size, patch_size=config.patch_size,
            hidden_size=config.hidden_size, mlp_dim=config.mlp_dim, num_layers=config.num_layers,
            num_heads=config.num_heads, pos_embed=config.pos_embed, dropout_rate=config.dropout_rate,
            spatial_dims=config.spatial_dims, classification=True)
        self.projection = torch.nn.Linear(config.hidden_size, config.hidden_size)
        state = load_file(str(model_dir / "model.safetensors"), device="cpu")
        vision = {k.removeprefix("vision_encoder."): v for k, v in state.items() if k.startswith("vision_encoder.")}
        projection = {k.removeprefix("mm_vision_proj."): v for k, v in state.items() if k.startswith("mm_vision_proj.")}
        missing, unexpected = self.vision.load_state_dict(vision, strict=False)
        # Newer MONAI adds unused optional cross-attention parameters.
        unsafe = [k for k in missing if ".cross_attn." not in k and ".norm_cross_attn." not in k]
        if unsafe or unexpected:
            raise RuntimeError(f"M3D-CLIP checkpoint mismatch: missing={unsafe}, unexpected={unexpected}")
        self.projection.load_state_dict(projection, strict=True)
        for m in (self.vision, self.projection):
            m.to(device).eval().requires_grad_(False)

    def __call__(self, batch):
        torch = self.torch
        with torch.inference_mode(), torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16):
            tokens, _ = self.vision(batch.to(self.device))
            z = torch.nn.functional.normalize(self.projection(tokens[:, 0]), dim=-1)
        return z.float().cpu().numpy()


def normalized_crop(image: np.ndarray, starts: np.ndarray, stops: np.ndarray):
    """Crop (one extra slice each side), resize to 32x256x256 and min-max normalise."""
    import torch
    import torch.nn.functional as F

    starts, stops = starts.copy(), stops.copy()
    starts[2] = max(0, starts[2] - 1)
    stops[2] = min(image.shape[2], stops[2] + 1)
    crop = np.asarray(image[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]], np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 1, 0)))[None, None]
    tensor = F.interpolate(tensor, size=(32, 256, 256), mode="trilinear", align_corners=False)
    low, high = tensor.amin(), tensor.amax()
    return ((tensor - low) / (high - low).clamp_min(1e-6))[0]


def mri_region_embeddings(encoder: M3DCLIPEncoder, image_nii: nib.Nifti1Image, seg_nii: nib.Nifti1Image,
                          annotation: str, batch_size: int = 2):
    """Embed every localised affected region at three context scales.

    Returns ``(embeddings, region_index, fallback)`` where ``embeddings`` has shape
    (regions, 3 scales, 768) and ``region_index`` maps rows to annotation regions.
    If no affected region can be localised, the whole segmented spine is used as a
    single region (``fallback=True``); such a patient contributes global but no
    regional features.
    """
    import torch

    image = np.asarray(image_nii.dataobj, np.float32)
    seg = np.asarray(seg_nii.dataobj, np.int16)
    if image.shape != seg.shape or not np.allclose(image_nii.affine, seg_nii.affine, atol=1e-3):
        raise ValueError("image/segmentation geometry mismatch")
    lo, hi = np.percentile(image[np.isfinite(image)], [1.0, 99.0])
    image = (np.clip(image, lo, hi) - lo) / max(float(hi - lo), 1e-6)
    found = [(j, np.isin(seg, labels)) for j, labels in enumerate(annotation_regions(annotation))]
    found = [(j, t) for j, t in found if t.any()]
    fallback = not found
    if fallback:
        target = spine_structure_mask(seg)
        if not target.any():
            raise RuntimeError("no spine structures localised")
        found = [(-1, target)]
    out = np.zeros((len(found), len(VIEW_MARGINS_MM), encoder.output_dim), np.float32)
    crops, index = [], []

    def flush():
        if crops:
            for (j, k), z in zip(index, encoder(torch.stack(crops))):
                out[j, k] = z
            crops.clear()
            index.clear()

    # Scale-major order with batches of two, as in the reference extraction.
    for k, margin in enumerate(VIEW_MARGINS_MM):
        for j, (_, target) in enumerate(found):
            a, b = bbox_with_margin(target, image_nii.header.get_zooms(), margin)
            crops.append(normalized_crop(image, a, b))
            index.append((j, k))
            if len(crops) == batch_size:
                flush()
    flush()
    return out, np.array([j for j, _ in found]), fallback, seg
