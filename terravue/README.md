# TerraVue

**Single-View Metric Elevation and 3D Flythrough — SIH 2026 Pixel2Peak**

TerraVue converts any single photo, GeoTIFF, or video frame into a calibrated
elevation map and an interactive 3D flythrough — no stereo cameras, no LiDAR,
no GPS required.

---

## Pipeline

`
Input (image / GeoTIFF / video frame / webcam)
  Preprocessor     normalise, handle GeoTIFF georef, extract video frame
  Depth Anything V2  relative inverse-depth map (HuggingFace transformers)
  Metric Calibration  SRTM lookup (georef) or heuristic (95th-pct = 12 m)
  SegFormer B2 ADE20K  semantic classes (buildings, veg, road, water, terrain)
  Confidence Map  per-pixel quality score
  Height Map  metric elevation array (m)
  Three.js 3D Mesh  PlaneGeometry + OrbitControls + flythrough animation
`

## Quick Start

`ash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
uvicorn backend.main:app --reload
# Open http://localhost:8000
`

## Tech Stack

- Backend: FastAPI + Uvicorn
- Depth: depth-anything/Depth-Anything-V2-Small-hf (transformers pipeline)
- Segmentation: nvidia/segformer-b2-finetuned-ade-512-512
- SRTM: srtm.py library
- GeoTIFF: rasterio
- 3D: Three.js r165 + OrbitControls (CDN)

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/process | Upload file, returns pipeline JSON |
| GET  | /api/health  | Health check |
| GET  | /            | Serves frontend/index.html |

## ADE20K Class Groups

| Group | IDs |
|-------|-----|
| buildings | 1, 25, 48 |
| vegetation | 4, 9, 17, 68 |
| road | 6, 11, 52 |
| water | 21, 26, 60 |
| terrain | 13, 16, 29 |

## License

MIT