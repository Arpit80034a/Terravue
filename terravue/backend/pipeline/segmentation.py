"""
backend/pipeline/segmentation.py
SegFormer B2 semantic segmentation on ADE20K (150 classes).
Returns a class-ID array and a coloured overlay PIL image.

ADE20K groups used by TerraVue:
  buildings  : [1, 25, 48]
  vegetation : [4, 9, 17, 68]
  road       : [6, 11, 52]
  water      : [21, 26, 60]
  terrain    : [13, 16, 29]
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
from PIL import Image

SEG_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"

# ADE20K class groups (0-indexed as returned by the model)
CLASS_GROUPS: Dict[str, list] = {
    "building":   [1, 25, 48],
    "vegetation": [4, 9, 17, 68],
    "road":       [6, 11, 52],
    "water":      [21, 26, 60],
    "terrain":    [13, 16, 29],
}

# Overlay colours (RGB)
GROUP_COLORS: Dict[str, Tuple[int, int, int]] = {
    "building":   (255, 165,  40),   # amber
    "vegetation": ( 34, 197,  94),   # green
    "road":       (148, 163, 184),   # slate
    "water":      ( 56, 189, 248),   # sky blue
    "terrain":    (161, 102,  47),   # brown
}


@lru_cache(maxsize=1)
def _get_model_and_processor():
    """Load and cache SegFormer model + processor (lazy, first call only)."""
    from transformers import (
        SegformerImageProcessor,
        SegformerForSemanticSegmentation,
    )
    import torch

    processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return processor, model


def run_segmentation(
    image_rgb: Image.Image,
) -> Tuple[np.ndarray, Image.Image, Dict[str, int]]:
    """
    Run SegFormer segmentation on an RGB PIL image.

    Parameters
    ----------
    image_rgb : PIL.Image.Image

    Returns
    -------
    seg_map   : np.ndarray shape (H, W), int64  — per-pixel ADE20K class ID
    overlay   : PIL.Image.Image  — semi-transparent coloured overlay on input
    class_counts : dict  — pixel count per semantic group
    """
    import torch
    import torch.nn.functional as F

    processor, model = _get_model_and_processor()
    device = next(model.parameters()).device

    inputs = processor(images=image_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Upsample logits to original image size
    orig_h, orig_w = image_rgb.height, image_rgb.width
    upsampled_logits = F.interpolate(
        outputs.logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )
    seg_map = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

    overlay = _make_overlay(image_rgb, seg_map)
    class_counts = _count_groups(seg_map)

    return seg_map, overlay, class_counts


def _make_overlay(
    image_rgb: Image.Image,
    seg_map: np.ndarray,
    alpha: float = 0.55,
) -> Image.Image:
    """Blend group-coloured mask with the original image."""
    h, w = seg_map.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for group, ids in CLASS_GROUPS.items():
        color = GROUP_COLORS[group]
        mask = np.isin(seg_map, ids)
        color_mask[mask] = color

    color_pil = Image.fromarray(color_mask, mode="RGB")
    base = image_rgb.resize((w, h)).convert("RGB")
    overlay = Image.blend(base, color_pil, alpha=alpha)
    return overlay


def _count_groups(seg_map: np.ndarray) -> Dict[str, int]:
    """Return pixel count per semantic group."""
    counts: Dict[str, int] = {}
    for group, ids in CLASS_GROUPS.items():
        counts[group] = int(np.isin(seg_map, ids).sum())
    return counts