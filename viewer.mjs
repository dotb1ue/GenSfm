import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { validateCamera, projectionRows, backprojectCorner, aspectFitRect } from './projection.mjs';

const FLIP = new THREE.Matrix4().makeScale(1, -1, -1);
const CORNERS = [[0, 0], [1, 0], [1, 1], [0, 1]];
const CAMERA_COLORS = [0x438c96, 0xbd913f, 0x8c78ad, 0xb66e89];
const IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

function matrix(rows, label = 'transform') {
  if (!Array.isArray(rows) || rows.length !== 4 || !rows.every(row => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite))) throw new Error(`Invalid ${label}`);
  if (rows[3].some((value, index) => Math.abs(value - (index === 3 ? 1 : 0)) > 1e-6)) throw new Error(`Invalid homogeneous ${label}`);
  const value = new THREE.Matrix4().set(...rows.flat());
  if (Math.abs(value.determinant()) < 1e-12) throw new Error(`Singular ${label}`);
  return value;
}

function validateManifest(manifest) {
  if (manifest.schemaVersion !== 1 || typeof manifest.mesh !== 'string') throw new Error('Unsupported interactive case manifest');
  if (manifest.gtAssistedAlignment !== false) throw new Error('The interactive viewer requires an unaligned native prediction');
  if (!Array.isArray(manifest.cameras) || manifest.cameras.length !== 4 || !Array.isArray(manifest.images) || manifest.images.length !== 4) throw new Error('Expected four recovered cameras and input images');
  if (manifest.coordinates?.intrinsics !== 'normalized-cropped' || manifest.coordinates?.extrinsics !== 'world-to-camera') throw new Error('Unsupported camera coordinate convention');
  matrix(manifest.coordinates?.dataToMesh, 'data-to-mesh transform');
  manifest.cameras.forEach(({ extrinsics, intrinsics }) => {
    validateCamera(extrinsics, intrinsics);
    matrix(extrinsics, 'camera extrinsics');
    backprojectCorner(intrinsics, 0, 0, 1);
  });
  manifest.images.forEach((image, index) => {
    if (typeof image.file !== 'string' || ![image.width, image.height].every(value => Number.isFinite(value) && value > 0) || (image.viewIndex !== undefined && image.viewIndex !== index)) throw new Error('Invalid input image metadata or camera order');
  });
  if (manifest.referenceCameras) {
    const reference = manifest.referenceCameras;
    if (!['dataset-ground-truth', 'hardware-calibrated'].includes(reference.kind) || !Array.isArray(reference.extrinsics) || reference.extrinsics.length !== 4 || !Array.isArray(reference.intrinsics) || reference.intrinsics.length !== 4) throw new Error('Invalid reference camera provenance');
    matrix(reference.worldTransform, 'reference camera world transform');
    reference.extrinsics.forEach((E, index) => {
      validateCamera(E, reference.intrinsics[index]);
      matrix(E, 'reference camera extrinsics');
      backprojectCorner(reference.intrinsics[index], 0, 0, 1);
    });
  }
}

function nativeCamera(E, K, dataToMesh) {
  const camera = new THREE.Camera();
  camera.near = 0.001;
  camera.far = 1000;
  camera.matrixAutoUpdate = false;
  // Keep the full finite-precision pose. Decomposition and reconstruction can
  // silently alter a recovered rotation and invalidate native reprojection.
  camera.matrix.copy(dataToMesh).multiply(matrix(E).invert()).multiply(FLIP);
  camera.matrixWorld.copy(camera.matrix);
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  camera.projectionMatrix.set(...projectionRows(K, camera.near, camera.far).flat());
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  return camera;
}

function cameraRig(cameras, worldTransform, reference = false) {
  const group = new THREE.Group();
  group.userData.centers = [];
  const depth = 0.23;
  cameras.forEach(({ extrinsics, intrinsics }, index) => {
    const pose = worldTransform.clone().multiply(matrix(extrinsics).invert());
    const center = new THREE.Vector3().applyMatrix4(pose);
    const rays = CORNERS.map(([u, v]) => new THREE.Vector3(...backprojectCorner(intrinsics, u, v, depth)).applyMatrix4(pose).sub(center));
    // Bound only glyph size; never move the actual camera centers.
    const glyphScale = Math.min(1, depth / Math.max(...rays.map(ray => ray.length())));
    const corners = rays.map(ray => center.clone().addScaledVector(ray, glyphScale));
    const vertices = [];
    for (let corner = 0; corner < 4; corner++) vertices.push(...center.toArray(), ...corners[corner].toArray(), ...corners[corner].toArray(), ...corners[(corner + 1) % 4].toArray());
    const top = corners[0].clone().lerp(corners[1], 0.5);
    vertices.push(...top.toArray(), ...top.clone().lerp(center, -0.16).toArray());
    const options = { color: reference ? 0x747e89 : CAMERA_COLORS[index], transparent: true, opacity: reference ? 0.72 : 0.95, depthTest: false, toneMapped: false };
    const material = reference ? new THREE.LineDashedMaterial({ ...options, dashSize: 0.025, gapSize: 0.018 }) : new THREE.LineBasicMaterial(options);
    const line = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3)), material);
    line.computeLineDistances();
    line.renderOrder = reference ? 2 : 3;
    line.userData.viewIndex = index;
    group.add(line);
    group.userData.centers.push(center);
  });
  return group;
}

function makeUnlit(object) {
  const oldMaterials = new Set(), unusedTextures = new Set(), retainedTextures = new Set();
  let meshes = 0, triangles = 0;
  object.traverse(node => {
    if (!node.isMesh) return;
    meshes++;
    triangles += (node.geometry.index?.count || node.geometry.attributes.position?.count || 0) / 3;
    const source = Array.isArray(node.material) ? node.material : [node.material];
    const materials = source.map(old => {
      if (!old) throw new Error('Mesh has no material');
      oldMaterials.add(old);
      Object.values(old).forEach(value => { if (value?.isTexture) unusedTextures.add(value); });
      if (old.map) { old.map.colorSpace = THREE.SRGBColorSpace; retainedTextures.add(old.map); }
      if (old.alphaMap) retainedTextures.add(old.alphaMap);
      // The exported prediction contains baked RGB. Preserve texture factors,
      // vertex colors and alpha without adding any lighting or tone mapping.
      return new THREE.MeshBasicMaterial({
        map: old.map || null,
        alphaMap: old.alphaMap || null,
        color: old.color?.clone() || new THREE.Color(0xffffff),
        vertexColors: Boolean(node.geometry.getAttribute('color')),
        side: THREE.DoubleSide,
        transparent: old.transparent,
        opacity: old.opacity,
        alphaTest: old.alphaTest,
        depthWrite: old.depthWrite,
        toneMapped: false,
      });
    });
    node.material = Array.isArray(node.material) ? materials : materials[0];
  });
  oldMaterials.forEach(material => material.dispose());
  unusedTextures.forEach(texture => { if (!retainedTextures.has(texture)) texture.dispose(); });
  if (!meshes || !triangles) throw new Error('Selected GLB contains no mesh triangles');
  return { meshes, triangles };
}

function disposeObjects(object, extraTextures = []) {
  const geometries = new Set(), materials = new Set(), textures = new Set(extraTextures), images = new Set();
  object?.traverse(node => {
    if (node.geometry) geometries.add(node.geometry);
    for (const material of (Array.isArray(node.material) ? node.material : [node.material])) {
      if (!material) continue;
      materials.add(material);
      Object.values(material).forEach(value => { if (value?.isTexture) textures.add(value); });
    }
  });
  geometries.forEach(geometry => geometry.dispose());
  materials.forEach(material => material.dispose());
  textures.forEach(texture => { if (texture.image) images.add(texture.image); texture.dispose(); });
  images.forEach(image => { if (typeof image.close === 'function') image.close(); else if (image instanceof HTMLImageElement) image.src = ''; });
}

async function fetchResponse(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Unable to load ${new URL(url).pathname.split('/').pop()} (${response.status})`);
  return response;
}

async function inputImage(url, metadata, signal) {
  const blob = await (await fetchResponse(url, signal)).blob();
  if (signal.aborted) throw new DOMException('Case selection changed', 'AbortError');
  const source = URL.createObjectURL(blob), image = new Image();
  try {
    await new Promise((resolve, reject) => {
      const abort = () => {
        signal.removeEventListener('abort', abort);
        image.onload = image.onerror = null;
        image.src = '';
        reject(new DOMException('Case selection changed', 'AbortError'));
      };
      const finish = callback => value => { signal.removeEventListener('abort', abort); image.onload = image.onerror = null; callback(value); };
      image.onload = finish(resolve);
      image.onerror = finish(() => reject(new Error('An input image could not be decoded')));
      signal.addEventListener('abort', abort, { once: true });
      image.src = source;
      if (signal.aborted) abort();
    });
    if (image.naturalWidth !== metadata.width || image.naturalHeight !== metadata.height) throw new Error('Input image dimensions differ from the camera manifest');
    return image;
  } catch (error) {
    image.src = '';
    throw error;
  } finally { URL.revokeObjectURL(source); }
}

function projectionErrors(manifest, cameras, dataToMesh) {
  return manifest.cameras.map(({ extrinsics, intrinsics }, index) => {
    const pose = dataToMesh.clone().multiply(matrix(extrinsics).invert());
    const camera = cameras[index], image = manifest.images[index];
    return Math.max(...CORNERS.map(([u, v]) => {
      const projected = new THREE.Vector3(...backprojectCorner(intrinsics, u, v, 0.7)).applyMatrix4(pose).applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix);
      return Math.hypot(((projected.x + 1) / 2 - u) * image.width, ((1 - projected.y) / 2 - v) * image.height);
    }));
  });
}

function boxCorners(bounds) {
  const points = [];
  for (const x of [bounds.min.x, bounds.max.x]) for (const y of [bounds.min.y, bounds.max.y]) for (const z of [bounds.min.z, bounds.max.z]) points.push(new THREE.Vector3(x, y, z));
  return points;
}

function rigVertices(rig) {
  const points = [];
  for (const line of rig.children) {
    const positions = line.geometry.getAttribute('position');
    for (let index = 0; index < positions.count; index++) points.push(new THREE.Vector3().fromBufferAttribute(positions, index));
  }
  return points;
}

export async function createViewer({ container, onState = () => {} } = {}) {
  if (!(container instanceof HTMLElement)) throw new Error('A viewer container is required');
  let disposed = false, generation = 0, controller = null, asset = null, frame = 0, renderCount = 0, contextLost = false;
  let width = 1, height = 1, imageRect = null, lastManifestUrl = null;
  let flight = null, presented = null, savedOrbit = null;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const state = { ready: false, available: true, loading: false, mode: 'orbit', viewIndex: null, transitioning: false, transitionProgress: 1, hasReferenceCameras: false, referenceLabel: null, error: null, overlayOpacity: 0.5, camerasVisible: true, referenceCamerasVisible: true, cubeVisible: true, caseId: null };
  const emit = changes => { Object.assign(state, changes); if (!disposed) onState({ ...state }); };
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
  } catch (error) {
    const fallback = document.createElement('p');
    fallback.className = 'viewer-fallback';
    fallback.setAttribute('role', 'status');
    fallback.textContent = 'The interactive 3D viewer is unavailable in this browser. Input images remain available below.';
    container.append(fallback);
    emit({ available: false, error: 'WebGL is unavailable in this browser.' });
    return { loadCase: async () => false, setView() {}, orbit() {}, setOverlay() {}, setCameras() {}, setReferenceCameras() {}, setCube() {}, inspect: () => ({ ...state }), dispose() { disposed = true; fallback.remove(); } };
  }
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.autoClear = false;
  const canvas = renderer.domElement;
  canvas.className = 'interactive-viewer-canvas';
  canvas.style.width = canvas.style.height = '100%';
  canvas.style.display = 'block';
  canvas.style.touchAction = 'none';
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', '3D reconstruction. Drag to orbit, scroll or pinch to zoom.');
  container.append(canvas);
  const scene = new THREE.Scene();
  const observer = new THREE.PerspectiveCamera(43, 1, 0.001, 1000);
  observer.position.set(2, 1.4, 2.5);
  const flightCamera = new THREE.Camera();
  flightCamera.matrixAutoUpdate = false;
  const surfaceColor = new THREE.Color(0xf7f8fb), viewColor = new THREE.Color(0xffffff), clearColor = new THREE.Color();
  const overlayScene = new THREE.Scene();
  const overlayCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2);
  overlayCamera.position.z = 1;
  const overlayMaterial = new THREE.MeshBasicMaterial({ transparent: true, opacity: state.overlayOpacity, depthTest: false, depthWrite: false, toneMapped: false });
  const overlayPlane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), overlayMaterial);
  overlayScene.add(overlayPlane);

  function requestRender() {
    if (disposed || contextLost || frame) return;
    frame = requestAnimationFrame(time => { frame = 0; render(time); });
  }

  const controls = new OrbitControls(observer, canvas);
  // With damping disabled, controls emit only while the user is interacting.
  // Flights schedule frames only until they settle; idle scenes stay idle.
  controls.enableDamping = false;
  controls.enabled = false;
  controls.screenSpacePanning = true;
  controls.listenToKeyEvents(canvas);
  controls.addEventListener('change', requestRender);

  function updateVisibility(opacity) {
    if (!asset) return;
    asset.rig.visible = opacity > 0 && state.camerasVisible;
    asset.referenceRig.visible = opacity > 0 && state.referenceCamerasVisible;
    asset.cube.visible = opacity > 0 && state.cubeVisible;
    asset.rig.children.forEach(line => { line.material.opacity = 0.95 * opacity; });
    asset.referenceRig.children.forEach(line => { line.material.opacity = 0.72 * opacity; });
    asset.cube.material.opacity = 0.45 * opacity;
  }

  function observerPoints(fitCameras, visibleOnly = false) {
    if (!asset) return [];
    const points = [...asset.objectPoints];
    if ((fitCameras && !visibleOnly) || state.cubeVisible) points.push(...asset.cubePoints);
    if (fitCameras && (!visibleOnly || state.camerasVisible)) points.push(...asset.rigPoints);
    if (fitCameras && (!visibleOnly || state.referenceCamerasVisible)) points.push(...asset.referencePoints);
    return points;
  }

  function updateObserverClipping() {
    observer.updateMatrixWorld(true);
    let minimum = Infinity, maximum = -Infinity;
    for (const point of observerPoints(true, true)) {
      const depth = -point.clone().applyMatrix4(observer.matrixWorldInverse).z;
      minimum = Math.min(minimum, depth);
      maximum = Math.max(maximum, depth);
    }
    // Tight depth bounds protect thin baked surfaces from depth fighting. A
    // glyph passing behind the observer must not push the near plane forward.
    observer.near = minimum > 0 ? Math.max(0.001, minimum * 0.8) : 0.001;
    observer.far = Math.max(observer.near + 0.05, maximum + Math.max(0.05, (maximum - minimum) * 0.1));
    observer.updateProjectionMatrix();
  }

  function settledFrame(mode = state.mode, index = state.viewIndex) {
    if (mode === 'view' && asset && index !== null) {
      const input = asset.manifest.images[index], rect = aspectFitRect(width, height, input.width, input.height);
      return { camera: asset.cameras[index], rect: { x: rect.x / width, y: rect.y / height, width: rect.width / width, height: rect.height / height }, helpers: 0, background: 1, overlayIndex: index, overlayAlpha: 1 };
    }
    updateObserverClipping();
    return { camera: observer, rect: { x: 0, y: 0, width: 1, height: 1 }, helpers: 1, background: 0, overlayIndex: null, overlayAlpha: 0 };
  }

  function copyFrame(value) {
    const camera = new THREE.Camera();
    camera.matrixAutoUpdate = false;
    camera.matrix.copy(value.camera.matrixWorld);
    camera.matrixWorld.copy(camera.matrix);
    camera.matrixWorldInverse.copy(camera.matrix).invert();
    camera.projectionMatrix.copy(value.camera.projectionMatrix);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    return { ...value, camera, rect: { ...value.rect } };
  }

  const ease = value => value * value * value * (10 + value * (-15 + 6 * value));

  function sampleFlight(time) {
    const current = flight, destination = settledFrame();
    const progress = THREE.MathUtils.clamp((time - current.start) / current.duration, 0, 1);
    state.transitionProgress = progress;
    if (progress >= 1) {
      flight = null;
      controls.enabled = state.mode === 'orbit' && !contextLost;
      emit({ transitioning: false, transitionProgress: 1 });
      return destination;
    }
    const t = ease(progress), from = current.from;
    const arc = new THREE.Quaternion().slerp(current.arc, t);
    const radius = Math.exp(THREE.MathUtils.lerp(Math.log(current.r0), Math.log(current.r1), t)) + Math.sin(Math.PI * t) ** 2 * Math.min(current.r0, current.r1) * 0.08;
    const position = current.direction.clone().applyQuaternion(arc).multiplyScalar(radius);
    // Parallel transport around the cube avoids a lookAt singularity over a
    // pole. The correction restores the actual recovered optical axis/roll.
    const rotation = arc.multiply(current.q0).multiply(new THREE.Quaternion().slerp(current.correction, t));
    const residual = new THREE.Matrix4();
    for (let i = 0; i < 16; i++) residual.elements[i] = THREE.MathUtils.lerp(current.residual0.elements[i], current.residual1.elements[i], t);
    // Retain affine residuals as well as rotation: native poses are finite
    // precision matrices, so quaternion-only interpolation would snap on arrival.
    flightCamera.matrix.compose(position, rotation, new THREE.Vector3(1, 1, 1)).multiply(residual);
    flightCamera.matrixWorld.copy(flightCamera.matrix);
    flightCamera.matrixWorldInverse.copy(flightCamera.matrix).invert();
    for (let i = 0; i < 16; i++) flightCamera.projectionMatrix.elements[i] = THREE.MathUtils.lerp(from.camera.projectionMatrix.elements[i], destination.camera.projectionMatrix.elements[i], t);
    flightCamera.projectionMatrixInverse.copy(flightCamera.projectionMatrix).invert();
    const rect = {};
    for (const key of ['x', 'y', 'width', 'height']) rect[key] = THREE.MathUtils.lerp(from.rect[key], destination.rect[key], t);
    // Images fade away before travel, then return only as the native camera
    // settles. Never carry a fixed input photograph across unrelated views.
    const leaving = 1 - ease(Math.min(1, progress / 0.16));
    const arriving = ease(Math.max(0, (progress - 0.84) / 0.16));
    return { camera: flightCamera, rect, helpers: THREE.MathUtils.lerp(from.helpers, destination.helpers, t), background: THREE.MathUtils.lerp(from.background, destination.background, t), overlayIndex: progress < 0.5 ? from.overlayIndex : destination.overlayIndex, overlayAlpha: progress < 0.5 ? from.overlayAlpha * leaving : destination.overlayAlpha * arriving };
  }

  function startFlight(from, mode, viewIndex = null) {
    flight = null;
    controls.enabled = false;
    const destination = settledFrame(mode, viewIndex);
    if (!reducedMotion.matches) {
      const p0 = new THREE.Vector3(), p1 = new THREE.Vector3(), q0 = new THREE.Quaternion(), q1 = new THREE.Quaternion(), scale = new THREE.Vector3();
      from.camera.matrixWorld.decompose(p0, q0, scale);
      destination.camera.matrixWorld.decompose(p1, q1, scale);
      q0.normalize(); q1.normalize();
      const r0 = Math.max(p0.length(), 1e-6), r1 = Math.max(p1.length(), 1e-6);
      const direction = p0.clone().divideScalar(r0), endDirection = p1.clone().divideScalar(r1);
      const arc = new THREE.Quaternion().setFromUnitVectors(direction, endDirection);
      const correction = arc.clone().multiply(q0).invert().multiply(q1);
      const residual0 = new THREE.Matrix4().compose(p0, q0, new THREE.Vector3(1, 1, 1)).invert().multiply(from.camera.matrixWorld);
      const residual1 = new THREE.Matrix4().compose(p1, q1, new THREE.Vector3(1, 1, 1)).invert().multiply(destination.camera.matrixWorld);
      flight = { from, start: performance.now(), duration: 950, direction, r0, r1, arc, q0, correction, residual0, residual1 };
    }
    controls.enabled = !flight && mode === 'orbit' && !contextLost;
    canvas.style.cursor = flight ? 'progress' : '';
    emit({ mode, viewIndex, transitioning: Boolean(flight), transitionProgress: flight ? 0 : 1 });
    requestRender();
  }

  function render(time = performance.now()) {
    if (disposed || contextLost) return;
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, width, height);
    renderer.setClearColor(0xf7f8fb, 1);
    renderer.clear(true, true, true);
    imageRect = null;
    if (asset) {
      presented = flight ? sampleFlight(time) : settledFrame();
      updateVisibility(presented.helpers);
      const rect = presented.rect;
      imageRect = { x: rect.x * width, y: rect.y * height, width: rect.width * width, height: rect.height * height };
      const bottom = height - imageRect.y - imageRect.height;
      renderer.setViewport(imageRect.x, bottom, imageRect.width, imageRect.height);
      renderer.setScissor(imageRect.x, bottom, imageRect.width, imageRect.height);
      renderer.setScissorTest(true);
      renderer.setClearColor(clearColor.copy(surfaceColor).lerp(viewColor, presented.background), 1);
      renderer.clear(true, true, true);
      renderer.render(scene, presented.camera);
      if (presented.overlayIndex !== null && presented.overlayAlpha * state.overlayOpacity > 0) {
        overlayMaterial.map = asset.inputTextures[presented.overlayIndex];
        overlayMaterial.opacity = presented.overlayAlpha * state.overlayOpacity;
        renderer.render(overlayScene, overlayCamera);
      }
      if (flight) requestRender();
      else canvas.style.cursor = '';
    }
    renderer.setScissorTest(false);
    renderCount++;
  }

  function resize() {
    if (disposed) return;
    width = Math.max(1, Math.round(container.clientWidth));
    height = Math.max(1, Math.round(container.clientHeight));
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);
    observer.aspect = width / height;
    observer.updateProjectionMatrix();
    requestRender();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  window.addEventListener('resize', resize);

  function fitObserver(fitCameras = true) {
    if (!asset) return;
    const target = new THREE.Vector3();
    const points = observerPoints(fitCameras);
    const yaw = THREE.MathUtils.degToRad(-28), pitch = THREE.MathUtils.degToRad(20);
    const direction = new THREE.Vector3(Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw));
    observer.fov = 43;
    observer.up.set(0, 1, 0);
    const right = new THREE.Vector3().crossVectors(observer.up, direction).normalize();
    const vertical = new THREE.Vector3().crossVectors(direction, right).normalize();
    const tanV = Math.tan(THREE.MathUtils.degToRad(observer.fov / 2)), tanH = tanV * observer.aspect;
    let distance = 0.01, radius = 0.1;
    // Fit actual projected corners around the unchanged unit-cube origin.
    // A 3D bounding sphere wastes most of a wide viewport for a sparse rig.
    for (const point of points) {
      const offset = point.clone().sub(target), z = offset.dot(direction);
      radius = Math.max(radius, offset.length());
      distance = Math.max(distance, z + 1.10 * Math.abs(offset.dot(right)) / tanH, z + 1.10 * Math.abs(offset.dot(vertical)) / tanV, z + 0.01);
    }
    observer.position.copy(target).addScaledVector(direction, distance + 0.01);
    controls.target.copy(target);
    controls.minDistance = Math.max(0.03, asset.bounds.getBoundingSphere(new THREE.Sphere()).radius * 0.08);
    controls.maxDistance = Math.max(10, radius * 20);
    controls.update();
    updateObserverClipping();
  }

  function rememberOrbit() {
    return { position: observer.position.clone(), quaternion: observer.quaternion.clone(), target: controls.target.clone(), up: observer.up.clone(), fov: observer.fov, minDistance: controls.minDistance, maxDistance: controls.maxDistance };
  }

  function restoreOrbit(value) {
    observer.position.copy(value.position);
    observer.quaternion.copy(value.quaternion);
    observer.up.copy(value.up);
    observer.fov = value.fov;
    controls.target.copy(value.target);
    controls.minDistance = value.minDistance;
    controls.maxDistance = value.maxDistance;
    controls.update();
    updateObserverClipping();
  }

  function orbit({ fitCameras = true, restore = false } = {}) {
    if (disposed || !asset || !state.ready) return;
    const from = copyFrame(presented || settledFrame());
    if (restore && savedOrbit) restoreOrbit(savedOrbit);
    else fitObserver(fitCameras);
    startFlight(from, 'orbit');
  }

  function setView(index) {
    if (disposed || !asset || !state.ready) return;
    if (!Number.isInteger(index) || index < 0 || index >= 4) throw new Error('Camera view index must be between 0 and 3');
    if (state.mode === 'view' && state.viewIndex === index) return;
    if (!flight && state.mode === 'orbit') savedOrbit = rememberOrbit();
    startFlight(copyFrame(presented || settledFrame()), 'view', index);
  }

  function leaveNativeForInteraction(event) {
    if (!asset || !state.ready) return;
    if (flight && state.mode === 'orbit') { event?.preventDefault(); return; }
    if (state.mode !== 'view') return;
    event?.preventDefault();
    orbit({ restore: true });
  }
  const nativeKeyboardExit = event => { if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) leaveNativeForInteraction(event); };
  canvas.addEventListener('pointerdown', leaveNativeForInteraction, true);
  canvas.addEventListener('wheel', leaveNativeForInteraction, { capture: true, passive: false });
  canvas.addEventListener('keydown', nativeKeyboardExit, true);
  const raycaster = new THREE.Raycaster();
  function frustumAt(event) {
    if (!asset || !state.ready || state.mode !== 'orbit' || state.transitioning || !state.camerasVisible) return null;
    const rect = canvas.getBoundingClientRect();
    raycaster.params.Line.threshold = Math.max(0.025, asset.bounds.getBoundingSphere(new THREE.Sphere()).radius * 0.035);
    raycaster.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1), observer);
    return raycaster.intersectObjects(asset.rig.children, false)[0] || null;
  }
  function selectFrustum(event) {
    const hit = frustumAt(event);
    if (hit) setView(hit.object.userData.viewIndex);
  }
  function hoverFrustum(event) {
    if (!state.transitioning && !event.buttons) canvas.style.cursor = frustumAt(event) ? 'pointer' : '';
  }
  const clearHover = () => { if (!state.transitioning) canvas.style.cursor = ''; };
  canvas.addEventListener('dblclick', selectFrustum);
  canvas.addEventListener('pointermove', hoverFrustum);
  canvas.addEventListener('pointerleave', clearHover);

  function loseContext(event) {
    event.preventDefault();
    contextLost = true;
    flight = null;
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    emit({ ready: false, transitioning: false, transitionProgress: 1, error: 'The 3D context was interrupted. Waiting for the browser to restore it.' });
  }
  function restoreContext() {
    contextLost = false;
    emit({ ready: Boolean(asset), error: null });
    controls.enabled = Boolean(asset) && state.mode === 'orbit';
    requestRender();
  }
  canvas.addEventListener('webglcontextlost', loseContext);
  canvas.addEventListener('webglcontextrestored', restoreContext);

  async function loadCase(manifestUrl) {
    if (disposed) return false;
    const selection = ++generation;
    controller?.abort();
    controller = new AbortController();
    const signal = controller.signal;
    lastManifestUrl = new URL(manifestUrl, document.baseURI).href;
    const url = lastManifestUrl;
    flight = null; presented = null; savedOrbit = null;
    emit({ ready: false, loading: true, error: null, mode: 'orbit', viewIndex: null, transitioning: false, transitionProgress: 1 });
    controls.enabled = false;
    requestRender();
    let candidate = null, decodedImages = [];
    try {
      const manifest = await (await fetchResponse(url, signal)).json();
      validateManifest(manifest);
      const downloads = await Promise.allSettled([
        fetchResponse(new URL(manifest.mesh, url).href, signal).then(response => response.arrayBuffer()),
        ...manifest.images.map(image => inputImage(new URL(image.file, url).href, image, signal)),
      ]);
      decodedImages = downloads.slice(1).filter(result => result.status === 'fulfilled').map(result => result.value);
      const failure = downloads.find(result => result.status === 'rejected');
      if (failure) throw failure.reason;
      if (signal.aborted || selection !== generation || disposed) throw new DOMException('Case selection changed', 'AbortError');
      const bytes = downloads[0].value;
      const gltf = await new GLTFLoader().parseAsync(bytes, new URL('.', new URL(manifest.mesh, url)).href);
      candidate = { group: new THREE.Group(), inputTextures: [], manifest, manifestUrl: url, byteLength: bytes.byteLength };
      candidate.group.add(gltf.scene);
      candidate.counts = makeUnlit(gltf.scene);
      const dataToMesh = matrix(manifest.coordinates.dataToMesh);
      candidate.cameras = manifest.cameras.map(({ extrinsics, intrinsics }) => nativeCamera(extrinsics, intrinsics, dataToMesh));
      candidate.rig = cameraRig(manifest.cameras, dataToMesh);
      const reference = manifest.referenceCameras;
      candidate.referenceRig = reference ? cameraRig(reference.extrinsics.map((E, index) => ({ extrinsics: E, intrinsics: reference.intrinsics[index] })), matrix(reference.worldTransform), true) : new THREE.Group();
      candidate.referenceRig.userData.centers ||= [];
      const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
      candidate.cube = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeometry), new THREE.LineBasicMaterial({ color: 0x9ba8af, transparent: true, opacity: 0.45, toneMapped: false }));
      boxGeometry.dispose();
      candidate.bounds = new THREE.Box3().setFromObject(gltf.scene);
      if (candidate.bounds.isEmpty() || !Number.isFinite(candidate.bounds.min.length() + candidate.bounds.max.length())) throw new Error('Selected mesh has invalid bounds');
      candidate.objectPoints = boxCorners(candidate.bounds);
      candidate.cubePoints = boxCorners(new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5)));
      candidate.rigPoints = rigVertices(candidate.rig);
      candidate.referencePoints = rigVertices(candidate.referenceRig);
      candidate.group.add(candidate.rig, candidate.referenceRig, candidate.cube);
      candidate.inputTextures = decodedImages.map(image => {
        const texture = new THREE.Texture(image);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.minFilter = texture.magFilter = THREE.LinearFilter;
        texture.generateMipmaps = false;
        texture.needsUpdate = true;
        return texture;
      });
      candidate.projectionErrors = projectionErrors(manifest, candidate.cameras, dataToMesh);
      if (Math.max(...candidate.projectionErrors) > 1e-5) throw new Error('Native camera projection failed its corner check');
      if (signal.aborted || selection !== generation || disposed) throw new DOMException('Case selection changed', 'AbortError');
      if (asset) { scene.remove(asset.group); disposeObjects(asset.group, asset.inputTextures); }
      overlayMaterial.map = null;
      asset = candidate;
      candidate = null;
      decodedImages = [];
      scene.add(asset.group);
      presented = null;
      emit({ ready: !contextLost, loading: false, caseId: manifest.id || null, hasReferenceCameras: Boolean(reference), referenceLabel: reference ? (reference.label || 'GT Cameras') : null, error: null });
      controls.enabled = !contextLost;
      fitObserver(true);
      renderer.renderLists.dispose();
      requestRender();
      return true;
    } catch (error) {
      if (candidate) disposeObjects(candidate.group, candidate.inputTextures);
      decodedImages.forEach(image => { image.src = ''; });
      if (selection !== generation || disposed || signal.aborted || error.name === 'AbortError') return false;
      if (asset) { scene.remove(asset.group); disposeObjects(asset.group, asset.inputTextures); asset = null; }
      overlayMaterial.map = null;
      emit({ ready: false, loading: false, caseId: null, hasReferenceCameras: false, referenceLabel: null, error: error.message || 'The selected reconstruction could not be loaded.' });
      renderer.renderLists.dispose();
      requestRender();
      return false;
    }
  }

  function setOverlay(opacity) {
    if (!Number.isFinite(opacity)) return;
    emit({ overlayOpacity: THREE.MathUtils.clamp(opacity, 0, 1) });
    requestRender();
  }
  function setCameras(visible) { emit({ camerasVisible: Boolean(visible) }); requestRender(); }
  function setReferenceCameras(visible) { emit({ referenceCamerasVisible: Boolean(visible) }); requestRender(); }
  function setCube(visible) { emit({ cubeVisible: Boolean(visible) }); requestRender(); }
  function cameraDetails(camera) {
    camera.updateMatrixWorld(true);
    return { matrixEncoding: 'column-major', world: camera.matrixWorld.toArray(), worldInverse: camera.matrixWorldInverse.toArray(), projection: camera.projectionMatrix.toArray() };
  }
  function inspect() {
    return {
      ...state, disposed, renderCount, pendingRender: Boolean(frame), controlsEnabled: controls.enabled, manifestUrl: lastManifestUrl,
      canvas: { width, height, pixelRatio: renderer.getPixelRatio() },
      imageRect: imageRect ? { ...imageRect } : null,
      overlayVisible: Boolean(asset && presented && presented.overlayIndex !== null && presented.overlayAlpha * state.overlayOpacity > 0),
      nativeProjectionCornerErrorPixels: asset ? Math.max(...asset.projectionErrors) : null,
      projectionCornerErrorsPixels: asset ? [...asset.projectionErrors] : [],
      cameraMatrices: asset ? asset.cameras.map(cameraDetails) : [],
      activeCamera: cameraDetails(presented?.camera || observer),
      mesh: asset ? { ...asset.counts, byteLength: asset.byteLength, declaredSha256: asset.manifest.meshSha256 || null, rootTransform: IDENTITY, gtAssistedAlignment: false } : null,
      renderer: { outputColorSpace: renderer.outputColorSpace, toneMapping: renderer.toneMapping, geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures, programs: renderer.info.programs?.length || 0 },
    };
  }
  function dispose() {
    if (disposed) return;
    disposed = true;
    flight = null; presented = null; savedOrbit = null;
    generation++;
    controller?.abort();
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    resizeObserver.disconnect();
    window.removeEventListener('resize', resize);
    canvas.removeEventListener('pointerdown', leaveNativeForInteraction, true);
    canvas.removeEventListener('wheel', leaveNativeForInteraction, true);
    canvas.removeEventListener('keydown', nativeKeyboardExit, true);
    canvas.removeEventListener('dblclick', selectFrustum);
    canvas.removeEventListener('pointermove', hoverFrustum);
    canvas.removeEventListener('pointerleave', clearHover);
    canvas.removeEventListener('webglcontextlost', loseContext);
    canvas.removeEventListener('webglcontextrestored', restoreContext);
    controls.removeEventListener('change', requestRender);
    controls.dispose();
    if (asset) disposeObjects(asset.group, asset.inputTextures);
    asset = null;
    overlayMaterial.map = null;
    disposeObjects(overlayScene);
    renderer.dispose();
    renderer.forceContextLoss();
    canvas.remove();
  }
  resize();
  emit({});
  return { loadCase, setView, orbit, setOverlay, setCameras, setReferenceCameras, setCube, dispose, inspect };
}
