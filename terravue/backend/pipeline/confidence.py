"""
backend/pipeline/confidence.py
Generates a per-pixel confidence map that combines:
  - Depth gradient magnitude  (high gradient = uncertain boundary)
  - Semantic consistency      (pixels matching a known ADE20K group get +bonus)
  - Depth range normalisation (extreme min/max edges penalised)

Output is a float32 array in [0, 1] and a coloured PIL image.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from .segmentation import CLASS_GROUPS


def compute_confidence(
    depth_arr: np.ndarray,
    seg_map: np.ndarray,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Compute a per-pixel confidence score in [0, 1].

    Parameters
    ----------
    depth_arr : np.ndarray (H, W), float32
        Metric depth map in metres.
    seg_map   : np.ndarray (H, W), int64
        ADE20K class ID per pixel.

    Returns
    -------
    conf      : np.ndarray (H, W), float32  in [0, 1]
    conf_img  : PIL.Image.Image  — amber/teal heatmap
    """
    h, w = depth_arr.shape

    # 1. Depth smoothness: low gradient = high confidence
    gy, gx = np.gradient(depth_arr)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_max = grad_mag.max() if grad_mag.max() > 0 else 1.0
    smoothness = 1.0 - np.clip(grad_mag / grad_max, 0.0, 1.0)

    # 2. Semantic known-class bonus
    known_ids = []
    for ids in CLASS_GROUPS.values():
        known_ids.extend(ids)
    semantic_known = np.isin(seg_map, known_ids).astype(np.float32)
    # Scale bonus so it does not dominate (0 = unknown, 0.3 = known)
    semantic_bonus = semantic_known * 0.3

    # 3. Depth range penalty: pixels near absolute min/max are less reliable
    d_norm = np.zeros_like(depth_arr)
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        d_norm = (depth_arr - d_min) / (d_max - d_min)
    # Penalise very-near (0) and very-far (1) extremes
    range_conf = 1.0 - np.abs(d_norm - 0.5) * 0.4   # [0.8 .. 1.0]

    # Combine (weights sum to roughly 1)
    conf = (smoothness * 0.55 + semantic_bonus + range_conf * 0.15)
    conf = np.clip(conf, 0.0, 1.0).astype(np.float32)

    conf_img = _confidence_to_image(conf)
    return conf, conf_img


def _confidence_to_image(conf: np.ndarray) -> Image.Image:
    """
    Map confidence to a teal (high) -> amber (low) colour scale.
    High confidence: #00d4aa  Low: #f59e0b
    """
    HIGH = np.array([0,  212, 170], dtype=np.float32)   # teal  #00d4aa
    LOW  = np.array([245, 158,  11], dtype=np.float32)  # amber #f59e0b

    c = conf[:, :, np.newaxis]          # H x W x 1
    rgb = (c * HIGH + (1 - c) * LOW).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")