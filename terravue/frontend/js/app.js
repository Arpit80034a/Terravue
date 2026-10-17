/**
 * frontend/js/app.js
 * Main controller: handles file upload, camera capture,
 * API calls, overlay toggles, stats, and wires everything together.
 */

import { initScene }                              from './scene3d.js';
import { drawOverlay, setOverlayVisible, resizeOverlays } from './overlays.js';

// ---- DOM refs ----
const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const previewImg   = document.getElementById('preview-img');
const btnProcess   = document.getElementById('btn-process');
const btnCamera    = document.getElementById('btn-camera');
const btnCapture   = document.getElementById('btn-capture');
const btnStopCam   = document.getElementById('btn-stop-camera');
const cameraWrap   = document.getElementById('camera-wrap');
const cameraVideo  = document.getElementById('camera-video');
const progressWrap = document.getElementById('progress-wrap');
const progressFill = document.getElementById('progress-bar-fill');
const progressLbl  = document.getElementById('progress-label');
const statusDot    = document.getElementById('status-dot');
const statusLabel  = document.getElementById('status-label');
const toast        = document.getElementById('toast');

const canvasWrap   = document.getElementById('canvas-wrap');
const emptyState   = document.getElementById('empty-state');
const flyCtrls     = document.getElementById('flythrough-controls');
const btnFly       = document.getElementById('btn-fly');
const btnResetCam  = document.getElementById('btn-reset-cam');
const btnWireframe = document.getElementById('btn-wireframe');

const thumbDepth   = document.getElementById('thumb-depth');
const thumbSeg     = document.getElementById('thumb-seg');
const thumbConf    = document.getElementById('thumb-conf');

const canvDepth    = document.getElementById('overlay-depth');
const canvSeg      = document.getElementById('overlay-seg');
const canvConf     = document.getElementById('overlay-conf');

const togDepth     = document.getElementById('tog-depth');
const togSeg       = document.getElementById('tog-seg');
const togConf      = document.getElementById('tog-conf');

const chipCalib    = document.getElementById('chip-calib');
const chipSize     = document.getElementById('chip-size');

// Stats
const statMin   = document.getElementById('stat-min');
const statMax   = document.getElementById('stat-max');
const statMean  = document.getElementById('stat-mean');
const statConf  = document.getElementById('stat-conf');
const statCalib = document.getElementById('stat-calib');
const statBld   = document.getElementById('stat-bld');
const statVeg   = document.getElementById('stat-veg');
const statWater = document.getElementById('stat-water');
const statRoad  = document.getElementById('stat-road');

// ---- App state ----
let pendingFile   = null;       // File object ready to send
let cameraStream  = null;       // MediaStream from getUserMedia
let sceneApi      = null;       // Three.js scene API
let overlayData   = {};         // { depth: b64, seg: b64, conf: b64 }

// ---- Init Three.js scene ----
sceneApi = initScene(canvasWrap);

// ---- Resize ----
window.addEventListener('resize', () => {
  sceneApi.resize();
  resizeOverlays([canvDepth, canvSeg, canvConf]);
});

// ================================================================
// File input / drag-and-drop
// ================================================================

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  pendingFile = file;
  btnProcess.disabled = false;
  previewImg.src = URL.createObjectURL(file);
  previewImg.classList.add('visible');
  setStatus('ready', 'File ready');
}

// ================================================================
// Camera
// ================================================================

btnCamera.addEventListener('click', async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    cameraVideo.srcObject = cameraStream;
    cameraWrap.classList.add('active');
    setStatus('ready', 'Camera active');
  } catch (err) {
    showToast('Camera error: ' + err.message, 'error');
  }
});

btnCapture.addEventListener('click', () => {
  const canvas = document.createElement('canvas');
  canvas.width  = cameraVideo.videoWidth;
  canvas.height = cameraVideo.videoHeight;
  canvas.getContext('2d').drawImage(cameraVideo, 0, 0);
  canvas.toBlob(blob => {
    const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
    setFile(file);
  }, 'image/jpeg', 0.92);
});

btnStopCam.addEventListener('click', stopCamera);

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraWrap.classList.remove('active');
}

// ================================================================
// Process
// ================================================================

btnProcess.addEventListener('click', runPipeline);

async function runPipeline() {
  if (!pendingFile) return;
  setStatus('busy', 'Processing…');
  btnProcess.disabled = true;
  showProgress(true, 'Uploading…', 10);

  const fd = new FormData();
  fd.append('file', pendingFile);

  try {
    showProgress(true, 'Running depth estimation…', 30);
    const res = await fetch('/api/process', { method: 'POST', body: fd });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Server error');
    }

    showProgress(true, 'Rendering…', 80);
    const data = await res.json();
    showProgress(true, 'Done', 100);
    setTimeout(() => showProgress(false), 600);

    await applyResults(data);
    setStatus('ready', 'Ready');
    showToast('Scene analysed successfully', 'success');
    btnProcess.disabled = false;

  } catch (err) {
    showProgress(false);
    setStatus('error', 'Error');
    showToast('Error: ' + err.message, 'error');
    btnProcess.disabled = false;
  }
}

async function applyResults(data) {
  const { width, height, depth_heatmap, seg_overlay, confidence_map, heightmap, stats } = data;

  // Store for overlay toggles
  overlayData = {
    depth: depth_heatmap,
    seg:   seg_overlay,
    conf:  confidence_map,
  };

  // Update thumbnails
  thumbDepth.src = 'data:image/png;base64,' + depth_heatmap;
  thumbSeg.src   = 'data:image/png;base64,' + seg_overlay;
  thumbConf.src  = 'data:image/png;base64,' + confidence_map;

  // Draw overlays
  await drawOverlay(canvDepth, depth_heatmap);
  await drawOverlay(canvSeg,   seg_overlay);
  await drawOverlay(canvConf,  confidence_map);

  // Apply current toggle state
  setOverlayVisible(canvDepth, togDepth.classList.contains('on'));
  setOverlayVisible(canvSeg,   togSeg.classList.contains('on'));
  setOverlayVisible(canvConf,  togConf.classList.contains('on'));

  // Build 3D mesh — flatten 2D heightmap array
  const rows = heightmap.length;
  const cols = heightmap[0].length;
  const flat = heightmap.flat();
  sceneApi.buildMesh(flat, rows, cols, stats.max_elev);

  // Hide empty state, show controls
  emptyState.style.display = 'none';
  flyCtrls.style.display   = 'flex';

  // Chips
  chipCalib.textContent = stats.calibration === 'srtm' ? '🛰 SRTM' : '📐 Heuristic';
  chipCalib.style.display = '';
  chipSize.textContent  = width + ' × ' + height + ' px';
  chipSize.style.display = '';

  // Stats bar
  statMin.textContent   = stats.min_elev.toFixed(1)   + ' m';
  statMax.textContent   = stats.max_elev.toFixed(1)   + ' m';
  statMean.textContent  = stats.mean_elev.toFixed(1)  + ' m';
  statConf.textContent  = (stats.mean_confidence * 100).toFixed(0) + '%';
  statCalib.textContent = stats.calibration;
  const sc = stats.seg_classes || {};
  statBld.textContent   = (sc.building   || 0).toLocaleString();
  statVeg.textContent   = (sc.vegetation || 0).toLocaleString();
  statWater.textContent = (sc.water      || 0).toLocaleString();
  statRoad.textContent  = (sc.road       || 0).toLocaleString();
}

// ================================================================
// Overlay toggles
// ================================================================

function makeToggle(btn, canv) {
  btn.addEventListener('click', () => {
    btn.classList.toggle('on');
    setOverlayVisible(canv, btn.classList.contains('on'));
  });
}
makeToggle(togDepth, canvDepth);
makeToggle(togSeg,   canvSeg);
makeToggle(togConf,  canvConf);

// ================================================================
// Flythrough controls
// ================================================================

btnFly.addEventListener('click', () => {
  const active = btnFly.classList.toggle('active');
  if (active) {
    sceneApi.startFlythrough();
    btnFly.textContent = '⏹ Stop';
  } else {
    sceneApi.stopFlythrough();
    btnFly.textContent = '▶ Flythrough';
  }
});

btnResetCam.addEventListener('click', () => {
  sceneApi.resetCamera();
  btnFly.classList.remove('active');
  btnFly.textContent = '▶ Flythrough';
});

btnWireframe.addEventListener('click', () => {
  const on = sceneApi.toggleWireframe();
  btnWireframe.classList.toggle('active', on);
});

// ================================================================
// Helpers
// ================================================================

function setStatus(state, label) {
  statusDot.className = '';
  statusDot.classList.add(state);
  statusLabel.textContent = label;
}

function showProgress(visible, label, pct) {
  progressWrap.classList.toggle('visible', visible);
  if (label) progressLbl.textContent = label;
  if (pct   !== undefined) progressFill.style.width = pct + '%';
}

let toastTimer = null;
function showToast(msg, type) {
  toast.textContent = msg;
  toast.className   = 'show ' + (type || '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.className = ''; }, 3500);
}