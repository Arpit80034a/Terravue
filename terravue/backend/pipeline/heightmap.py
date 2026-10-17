"""
backend/pipeline/heightmap.py
Assembles the final elevation array from metric depth and the semantic map,
then down-samples to a mesh-friendly resolution for Three.js.

Elevation model:
  H(x, y) = metric_depth(x, y)
  i.e. we treat the metric depth as height above the camera (above-ground
  proxy). Water pixels are clamped to 0 m (sea level). Building pixels get
  a small upward bias to emphasise vertical structure.

Output is a 2-D float32 array and basic statistics.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .segmentation import CLASS_GROUPS

# Maximum mesh resolution sent to the frontend (width x height vertices)
MAX_MESH_W = 256
MAX_MESH_H = 256


def build_heightmap(
    metric_depth: np.ndarray,
    seg_map: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Build a height map suitable for Three.js PlaneGeometry.

    Parameters
    ----------
    metric_depth : np.ndarray (H, W) float32  — metric depth in metres
    seg_map      : np.ndarray (H, W) int64    — ADE20K class ID per pixel

    Returns
    -------
    heightmap : np.ndarray (Hm, Wm) float32
        Down-sampled elevation array. Wm <= MAX_MESH_W, Hm <= MAX_MESH_H.
    stats : dict
        min_elev, max_elev, mean_elev (all floats in metres)
    """
    h, w = metric_depth.shape

    # --- Semantic adjustments ---
    adjusted = metric_depth.copy()

    # Water pixels -> clamp near zero (flat water surface)
    water_mask = np.isin(seg_map, CLASS_GROUPS["water"])
    adjusted[water_mask] = np.clip(adjusted[water_mask], 0.0, 0.5)

    # Building pixels get a slight upward bias (they appear taller)
    building_mask = np.isin(seg_map, CLASS_GROUPS["building"])
    adjusted[building_mask] = adjusted[building_mask] * 1.25

    # --- Down-sample to mesh resolution ---
    scale_w = min(1.0, MAX_MESH_W / w)
    scale_h = min(1.0, MAX_MESH_H / h)
    scale = min(scale_w, scale_h)
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))

    # PIL resize with LANCZOS for smooth down-sampling
    adj_pil = Image.fromarray(adjusted, mode="F")   # float32 single-channel
    adj_small = adj_pil.resize((new_w, new_h), Image.LANCZOS)
    heightmap = np.array(adj_small, dtype=np.float32)

    # Clip negatives (artefacts from resize)
    heightmap = np.clip(heightmap, 0.0, None)

    stats = {
        "min_elev":  float(heightmap.min()),
        "max_elev":  float(heightmap.max()),
        "mean_elev": float(heightmap.mean()),
    }

    return heightmap, stats