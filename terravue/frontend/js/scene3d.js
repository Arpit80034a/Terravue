/**
 * frontend/js/scene3d.js
 * Three.js 3D terrain mesh with OrbitControls and flythrough animation.
 *
 * Exports:
 *   initScene(container)  -> sceneApi
 *   sceneApi.buildMesh(heightmap, rows, cols, maxElev)
 *   sceneApi.startFlythrough()
 *   sceneApi.stopFlythrough()
 *   sceneApi.resetCamera()
 *   sceneApi.toggleWireframe()
 *   sceneApi.resize()
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function initScene(container) {
  // ---- Renderer ----
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setClearColor(0x080c12, 1);
  renderer.shadowMap.enabled = true;
  container.appendChild(renderer.domElement);

  // ---- Scene ----
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x080c12, 30, 120);

  // ---- Camera ----
  const camera = new THREE.PerspectiveCamera(
    55,
    container.clientWidth / container.clientHeight,
    0.1,
    500
  );
  camera.position.set(0, 18, 28);
  camera.lookAt(0, 0, 0);

  // ---- Lights ----
  const ambient = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff4e0, 1.2);
  sun.position.set(20, 40, 20);
  sun.castShadow = true;
  scene.add(sun);

  const fill = new THREE.DirectionalLight(0x00d4aa, 0.3);
  fill.position.set(-20, 10, -20);
  scene.add(fill);

  // ---- OrbitControls ----
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.minDistance = 5;
  controls.maxDistance = 120;
  controls.maxPolarAngle = Math.PI / 2.05;

  // ---- State ----
  let terrainMesh = null;
  let wireframeOn  = false;
  let flyActive    = false;
  let flyAngle     = 0;
  let animId       = null;
  const DEFAULT_CAM = { x: 0, y: 18, z: 28 };

  // ---- Animate loop ----
  function animate() {
    animId = requestAnimationFrame(animate);
    if (flyActive && terrainMesh) {
      flyAngle += 0.004;
      const r  = 32;
      const hy = 14;
      camera.position.set(
        Math.cos(flyAngle) * r,
        hy,
        Math.sin(flyAngle) * r
      );
      camera.lookAt(0, 0, 0);
    }
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // ---- Public API ----

  /**
   * Build (or rebuild) the terrain mesh from a flat heightmap array.
   * @param {number[]} heightmap  flat row-major array length rows*cols
   * @param {number}   rows
   * @param {number}   cols
   * @param {number}   maxElev   maximum elevation value (metres) for Y scale
   */
  function buildMesh(heightmap, rows, cols, maxElev) {
    // Remove old mesh
    if (terrainMesh) {
      scene.remove(terrainMesh);
      terrainMesh.geometry.dispose();
      terrainMesh.material.dispose();
      terrainMesh = null;
    }

    const planeW = 20;
    const planeH = 20 * (rows / cols);
    const segW   = cols  - 1;
    const segH   = rows  - 1;
    const YSCALE = maxElev > 0 ? (6 / maxElev) : 1;   // compress to ~6 units

    const geo = new THREE.PlaneGeometry(planeW, planeH, segW, segH);
    geo.rotateX(-Math.PI / 2);

    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const idx = row * cols + col;
      const h   = (heightmap[idx] || 0) * YSCALE;
      pos.setY(i, h);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();

    // Colour vertices by elevation (teal low -> white high)
    const colors = new Float32Array(pos.count * 3);
    const cLow  = new THREE.Color(0x004d3f);
    const cMid  = new THREE.Color(0x00d4aa);
    const cHigh = new THREE.Color(0xffffff);
    for (let i = 0; i < pos.count; i++) {
      const t = Math.min(1, pos.getY(i) / (maxElev * YSCALE || 1));
      const c = t < 0.5
        ? cLow.clone().lerp(cMid, t * 2)
        : cMid.clone().lerp(cHigh, (t - 0.5) * 2);
      colors[i * 3]     = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      wireframe: wireframeOn,
      shininess: 18,
    });

    terrainMesh = new THREE.Mesh(geo, mat);
    terrainMesh.receiveShadow = true;
    terrainMesh.castShadow    = true;
    scene.add(terrainMesh);

    // Reposition camera to see the full mesh
    camera.position.set(DEFAULT_CAM.x, DEFAULT_CAM.y, DEFAULT_CAM.z);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function startFlythrough() {
    flyActive = true;
    flyAngle  = 0;
    controls.enabled = false;
  }

  function stopFlythrough() {
    flyActive = false;
    controls.enabled = true;
  }

  function resetCamera() {
    stopFlythrough();
    camera.position.set(DEFAULT_CAM.x, DEFAULT_CAM.y, DEFAULT_CAM.z);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function toggleWireframe() {
    wireframeOn = !wireframeOn;
    if (terrainMesh) terrainMesh.material.wireframe = wireframeOn;
    return wireframeOn;
  }

  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  return { buildMesh, startFlythrough, stopFlythrough, resetCamera, toggleWireframe, resize };
}