"""
backend/pipeline/preprocessor.py
Handles all input types: plain images, GeoTIFF (with georef extraction),
and video files (extracts the middle frame).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class PreprocessResult:
    image_rgb: Image.Image          # PIL RGB image ready for inference
    width: int
    height: int
    is_georeferenced: bool = False
    lat: Optional[float] = None     # centre latitude  (georef only)
    lon: Optional[float] = None     # centre longitude (georef only)
    original_path: str = ""


def preprocess(file_path: str, kind: str) -> PreprocessResult:
    """
    Load and normalise any supported input file.

    Parameters
    ----------
    file_path : str
        Absolute path to the uploaded file.
    kind : str
        One of 'image', 'geotiff', 'video'  (from file_utils.detect_kind).

    Returns
    -------
    PreprocessResult
    """
    if kind == "geotiff":
        return _load_geotiff(file_path)
    if kind == "video":
        return _load_video(file_path)
    return _load_image(file_path)


# ---------------------------------------------------------------------------
# Plain image
# ---------------------------------------------------------------------------

def _load_image(path: str) -> PreprocessResult:
    img = Image.open(path).convert("RGB")
    return PreprocessResult(
        image_rgb=img,
        width=img.width,
        height=img.height,
        original_path=path,
    )


# ---------------------------------------------------------------------------
# GeoTIFF
# ---------------------------------------------------------------------------

def _load_geotiff(path: str) -> PreprocessResult:
    try:
        import rasterio
        from rasterio.crs import CRS
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError(
            "rasterio and pyproj are required for GeoTIFF support. "
            "Install them with: pip install rasterio pyproj"
        ) from exc

    with rasterio.open(path) as src:
        # Read up to 3 bands as RGB
        band_count = min(src.count, 3)
        bands = [src.read(i + 1) for i in range(band_count)]

        # Pad to exactly 3 bands if necessary (e.g. single-band DEM)
        while len(bands) < 3:
            bands.append(bands[0])

        # Normalise each band to 0-255 uint8
        norm_bands = []
        for b in bands:
            b = b.astype(np.float32)
            lo, hi = b.min(), b.max()
            if hi > lo:
                b = (b - lo) / (hi - lo) * 255
            else:
                b = np.zeros_like(b)
            norm_bands.append(b.astype(np.uint8))

        rgb_arr = np.stack(norm_bands, axis=-1)  # H x W x 3
        img = Image.fromarray(rgb_arr, mode="RGB")

        # Try to extract a centre lat/lon
        lat, lon = None, None
        is_georef = False
        try:
            bounds = src.bounds
            cx = (bounds.left + bounds.right) / 2
            cy = (bounds.bottom + bounds.top) / 2
            if src.crs:
                transformer = Transformer.from_crs(
                    src.crs, "EPSG:4326", always_xy=True
                )
                lon, lat = transformer.transform(cx, cy)
                is_georef = True
        except Exception:
            pass

    return PreprocessResult(
        image_rgb=img,
        width=img.width,
        height=img.height,
        is_georeferenced=is_georef,
        lat=lat,
        lon=lon,
        original_path=path,
    )


# ---------------------------------------------------------------------------
# Video  (extract middle frame)
# ---------------------------------------------------------------------------

def _load_video(path: str) -> PreprocessResult:
    mid_frame = _read_video_frame_cv2(path)
    if mid_frame is None:
        raise RuntimeError(f"Could not read any frames from video: {path}")
    if mid_frame.ndim == 2:
        mid_frame = np.stack([mid_frame] * 3, axis=-1)
    elif mid_frame.shape[2] == 4:
        mid_frame = mid_frame[:, :, :3]
    img = Image.fromarray(mid_frame.astype(np.uint8), mode="RGB")
    return PreprocessResult(
        image_rgb=img, width=img.width, height=img.height, original_path=path,
    )

def _read_video_frame_cv2(path: str):
    """Extract middle frame using cv2 (reliable on Windows)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
