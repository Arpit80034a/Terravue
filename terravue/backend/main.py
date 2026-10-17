import os,sys,logging,traceback
from pathlib import Path
from fastapi import FastAPI,File,HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from fastapi.staticfiles import StaticFiles
from .pipeline.calibration import calibrate
from .pipeline.confidence import compute_confidence
from .pipeline.depth import run_depth
from .pipeline.heightmap import build_heightmap
from .pipeline.preprocessor import preprocess
from .pipeline.segmentation import run_segmentation
from .utils.file_utils import (
    cleanup_temp_dir,detect_kind,make_temp_dir,
    numpy_to_base64,pil_to_base64,save_upload,validate_extension,
)
logging.basicConfig(level=logging.INFO)
log=logging.getLogger("terravue")
app=FastAPI(title="TerraVue",description="Single-View Metric Elevation API",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
FRONTEND_DIR=Path(__file__).parent.parent/"frontend"
if FRONTEND_DIR.exists():
    app.mount("/static",StaticFiles(directory=str(FRONTEND_DIR)),name="static")

@app.get("/",include_in_schema=False)
async def serve_index():
    idx=FRONTEND_DIR/"index.html"
    if idx.exists():return FileResponse(str(idx))
    return JSONResponse({"message":"TerraVue API running."})

@app.get("/api/health")
async def health():
    return {"status":"ok","service":"TerraVue"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/api/process")
async def process_file(file:UploadFile=File(...)):
    filename=file.filename or "upload.jpg"
    try:
        validate_extension(filename)
    except ValueError as exc:
        raise HTTPException(status_code=415,detail=str(exc))
    tmp_dir=make_temp_dir()
    try:
        raw=await file.read()
        file_path=save_upload(raw,filename,tmp_dir)
        kind=detect_kind(filename)
        log.info("Processing %s kind=%s size=%d",filename,kind,len(raw))
        prep=preprocess(file_path,kind)
        img=prep.image_rgb
        depth_arr,depth_heatmap=run_depth(img)
        metric_depth,calib_method=calibrate(depth_arr,lat=prep.lat,lon=prep.lon)
        seg_map,seg_overlay,seg_counts=run_segmentation(img)
        conf_arr,conf_img=compute_confidence(metric_depth,seg_map)
        heightmap,elev_stats=build_heightmap(metric_depth,seg_map)
        response={
            "width":prep.width,"height":prep.height,
            "depth_heatmap":pil_to_base64(depth_heatmap),
            "seg_overlay":pil_to_base64(seg_overlay),
            "confidence_map":pil_to_base64(conf_img),
            "heightmap":heightmap.tolist(),
            "stats":{**elev_stats,
                "calibration":calib_method,
                "mean_confidence":float(conf_arr.mean()),
                "seg_classes":seg_counts},
        }
        log.info("Done calib=%s elev_max=%.1fm",calib_method,elev_stats["max_elev"])
        return JSONResponse(response)
    except Exception as exc:
        log.error("Pipeline error: %s",traceback.format_exc())
        raise HTTPException(status_code=500,detail="Pipeline error: "+str(exc))
    finally:
        cleanup_temp_dir(tmp_dir)