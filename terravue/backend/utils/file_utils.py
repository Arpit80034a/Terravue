"""
backend/utils/file_utils.py
File-type detection, temporary I/O helpers, and base64 encoding utilities.
"""
from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from PIL import Image

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_EXTS = ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS

InputKind = Literal["image", "geotiff", "video"]


def detect_kind(filename: str) -> InputKind:
    """Return the high-level kind of an uploaded file."""
    ext = Path(filename).suffix.lower()
    if ext in (".tif", ".tiff"):
        return "geotiff"
    if ext in ALLOWED_VIDEO_EXTS:
        return "video"
    if ext in ALLOWED_IMAGE_EXTS:
        return "image"
    raise ValueError(f"Unsupported file extension: {ext!r}")


def validate_extension(filename: str) -> None:
    """Raise ValueError if the extension is not in ALLOWED_EXTS."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError(
            f"File type {ext!r} is not supported. "
            f"Allowed: {sorted(ALLOWED_EXTS)}"
        )


def make_temp_dir() -> str:
    """Create and return a unique temporary directory path."""
    tmp = os.path.join(tempfile.gettempdir(), "terravue", str(uuid.uuid4()))
    os.makedirs(tmp, exist_ok=True)
    return tmp


def cleanup_temp_dir(path: str) -> None:
    """Remove a temporary directory tree, silently ignoring errors."""
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def save_upload(data: bytes, filename: str, dest_dir: str) -> str:
    """Write data to dest_dir/filename and return the full path."""
    dest = os.path.join(dest_dir, Path(filename).name)
    with open(dest, "wb") as fh:
        fh.write(data)
    return dest


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def numpy_to_base64(arr, fmt: str = "PNG") -> str:
    """Encode a numpy uint8 H x W x C array to base64 PNG string."""
    import numpy as np
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return pil_to_base64(Image.fromarray(arr), fmt)


def load_pil_rgb(path: str) -> Image.Image:
    """Open any supported raster file as an RGB PIL image."""
    return Image.open(path).convert("RGB")