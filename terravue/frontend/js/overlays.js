/**
 * frontend/js/overlays.js
 * Draws depth heatmap, semantic overlay, and confidence map
 * onto the overlay <canvas> elements sitting on top of the Three.js canvas.
 */

/**
 * Draw a base64-encoded PNG onto a canvas element.
 * @param {HTMLCanvasElement} canvas
 * @param {string} b64  - base64 PNG string (no data: prefix)
 */
export function drawOverlay(canvas, b64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      canvas.width  = canvas.offsetWidth  * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve();
    };
    img.src = 'data:image/png;base64,' + b64;
  });
}

/**
 * Clear a canvas element.
 * @param {HTMLCanvasElement} canvas
 */
export function clearOverlay(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Show or hide an overlay canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {boolean} visible
 */
export function setOverlayVisible(canvas, visible) {
  canvas.classList.toggle('visible', visible);
}

/**
 * Resize all overlay canvases to match the container.
 * Should be called on window resize.
 * @param {HTMLCanvasElement[]} canvases
 */
export function resizeOverlays(canvases) {
  canvases.forEach(c => {
    c.width  = c.offsetWidth  * window.devicePixelRatio;
    c.height = c.offsetHeight * window.devicePixelRatio;
  });
}