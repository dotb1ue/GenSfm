// Kept in step with paper_video/renderer/projection.mjs. Normalized images
// use +u right, +v down; OpenCV cameras use +Z forward, Three.js uses -Z.
export function validateCamera(E, K) {
  const matrix = (value, rows, cols) => Array.isArray(value) && value.length === rows && value.every(row => Array.isArray(row) && row.length === cols && row.every(Number.isFinite));
  if (!matrix(E, 4, 4) || !matrix(K, 3, 3)) throw new Error('Expected finite 4x4 E and 3x3 normalized K');
  if (K[0][0] <= 0 || K[1][1] <= 0 || Math.abs(K[2][0]) > 1e-7 || Math.abs(K[2][1]) > 1e-7 || Math.abs(K[2][2] - 1) > 1e-7) throw new Error('Expected positive focal lengths and canonical homogeneous row in K');
  if (E[3].some((v, i) => Math.abs(v - (i === 3 ? 1 : 0)) > 1e-6)) throw new Error('Expected homogeneous world-to-camera E');
}

export function projectionRows(K, near = 0.01, far = 100) {
  validateCamera([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], K);
  if (!(near > 0 && far > near)) throw new Error('Invalid clipping distances');
  return [
    [2*K[0][0], -2*K[0][1], 1-2*K[0][2], 0],
    [-2*K[1][0], 2*K[1][1], 2*K[1][2]-1, 0],
    [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
    [0, 0, -1, 0],
  ];
}

export function projectCv(point, E, K) {
  const c = E.slice(0,3).map(row => row[0]*point[0]+row[1]*point[1]+row[2]*point[2]+row[3]);
  const h = K.map(row => row.reduce((sum, value, i) => sum+value*c[i], 0));
  return { u:h[0]/h[2], v:h[1]/h[2], depth:c[2] };
}

export function backprojectCorner(K, u, v, depth) {
  // Invert both skew entries rather than assuming a diagonal focal block.
  const a=K[0][0], b=K[0][1], c=K[1][0], d=K[1][1];
  const determinant=a*d-b*c;
  if (Math.abs(determinant)<1e-12) throw new Error('Singular camera intrinsics');
  const x=u-K[0][2], y=v-K[1][2];
  return [depth*(d*x-b*y)/determinant, depth*(-c*x+a*y)/determinant, depth];
}

// CSS-pixel coordinates, measured from the top left. The renderer converts the
// same rectangle to its bottom-left viewport for both native render and input.
export function aspectFitRect(width, height, imageWidth, imageHeight) {
  if (![width, height, imageWidth, imageHeight].every(value => Number.isFinite(value) && value > 0)) throw new Error('Expected positive viewport and image dimensions');
  const scale = Math.min(width / imageWidth, height / imageHeight);
  const w = imageWidth * scale, h = imageHeight * scale;
  return { x: (width - w) / 2, y: (height - h) / 2, width: w, height: h };
}
