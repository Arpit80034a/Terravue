"""
backend/pipeline/depth.py  — fixed v2
Manually upsamples model output to avoid DPTImageProcessor version issues.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Tuple
import numpy as np
from PIL import Image

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

@lru_cache(maxsize=1)
def _get_model():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return processor, model, device

def run_depth(image_rgb: Image.Image) -> Tuple[np.ndarray, Image.Image]:
    """
    Run Depth Anything V2 on an RGB PIL image.
    Returns (depth_arr: float32 HxW numpy, heatmap: PIL RGB image).
    """
    import torch
    import torch.nn.functional as F

    processor, model, device = _get_model()
    orig_h, orig_w = image_rgb.height, image_rgb.width

    inputs = processor(images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.predicted_depth shape: (batch, H', W') — model output resolution
    # Manually upsample to original image size (avoids processor version issues)
    pred = outputs.predicted_depth  # (1, H', W')
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)    # add batch dim if missing

    pred_up = F.interpolate(
        pred.unsqueeze(1).float(),  # (1, 1, H', W')
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()  # → (H, W)

    depth_arr = pred_up.detach().cpu().numpy().astype(np.float32)

    # Safety: squeeze any remaining extra dims
    while depth_arr.ndim > 2:
        depth_arr = depth_arr.squeeze(0)

    heatmap = _depth_to_heatmap(depth_arr)
    return depth_arr, heatmap

def _depth_to_heatmap(depth_arr: np.ndarray) -> Image.Image:
    """Float32 HxW → plasma colourmap RGB PIL image using cv2."""
    import cv2
    d = depth_arr.copy().astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    if hi > lo:
        d = (d - lo) / (hi - lo)
    else:
        d = np.zeros_like(d)
    d_uint8 = (d * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(d_uint8, cv2.COLORMAP_PLASMA)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_rgb, mode="RGB")
