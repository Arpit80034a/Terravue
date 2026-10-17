"""
backend/pipeline/calibration.py
Converts relative inverse-depth (from Depth Anything V2) into metric depth
in metres using either:
  1. SRTM ground-truth (for georeferenced inputs) — uses the srtm.py library
  2. Heuristic scaling   — 95th-percentile relative depth is mapped to 12 m

The final output is a float32 array where values represent metres above the
local ground plane.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Target scale for heuristic calibration (metres)
HEURISTIC_P95_TARGET_M = 12.0


def calibrate(
    depth_arr: np.ndarray,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Tuple[np.ndarray, str]:
    """
    Calibrate a relative depth map to metric depth (metres).

    Depth Anything V2 outputs *inverse* relative depth: larger value = closer.
    We first convert to relative distance (invert), then scale to metres.

    Parameters
    ----------
    depth_arr : np.ndarray shape (H, W), float32
        Raw relative inverse-depth from Depth Anything V2.
    lat, lon : float, optional
        Centre coordinates of the image (WGS-84 decimal degrees).
        If provided, SRTM is attempted for scale calibration.

    Returns
    -------
    metric_depth : np.ndarray shape (H, W), float32
        Absolute depth in metres (closer pixels have smaller values).
    method : str
        'srtm' or 'heuristic'
    """
    depth = depth_arr.astype(np.float32)

    # --- Invert to get relative distance (depth increases away from camera) ---
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        # Normalise 0-1 then invert
        depth_norm = (depth - d_min) / (d_max - d_min)
        depth_dist = 1.0 - depth_norm          # 0=close, 1=far
    else:
        depth_dist = np.zeros_like(depth)

    # --- Try SRTM calibration ---
    if lat is not None and lon is not None:
        srtm_scale = _srtm_scale(lat, lon)
        if srtm_scale is not None:
            metric = depth_dist * srtm_scale
            return metric.astype(np.float32), "srtm"

    # --- Heuristic calibration: map p95 -> HEURISTIC_P95_TARGET_M ---
    p95 = float(np.percentile(depth_dist, 95))
    if p95 > 0:
        scale = HEURISTIC_P95_TARGET_M / p95
    else:
        scale = HEURISTIC_P95_TARGET_M

    metric = depth_dist * scale
    return metric.astype(np.float32), "heuristic"


def _srtm_scale(lat: float, lon: float) -> Optional[float]:
    """
    Query SRTM for a ground elevation at (lat, lon) and derive a scale factor.

    We assume the ground plane in the image corresponds to the SRTM elevation,
    and that the maximum depth in the scene is ~2x the SRTM elevation above
    sea level (a rough but usable heuristic for aerial/overhead shots).

    Returns None if SRTM lookup fails.
    """
    try:
        import srtm
        elevation_data = srtm.get_data()
        elev = elevation_data.get_elevation(lat, lon)
        if elev is None or elev <= 0:
            return None
        # Use SRTM elevation as the scale target for p95 depth
        return float(elev)
    except Exception:
        return None