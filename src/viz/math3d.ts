// ─────────────────────────────────────────────────────────────
// math3d.ts — minimal mat4/vec3 helpers for the camera. Zero
// deps so the viz drops cleanly into the existing build.
//
// Conventions: column-major mat4 (matches WGSL's mat4x4<f32>),
// right-handed coordinates, looking down −z by default.
// ─────────────────────────────────────────────────────────────

export type Vec3 = [number, number, number];
/** 16 floats, column-major. m[col * 4 + row]. */
export type Mat4 = Float32Array;

export function vec3(x: number, y: number, z: number): Vec3 {
  return [x, y, z];
}

export function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function length(a: Vec3): number {
  return Math.sqrt(dot(a, a));
}

export function normalize(a: Vec3): Vec3 {
  const l = length(a);
  if (l < 1e-30) return [0, 0, 0];
  return [a[0] / l, a[1] / l, a[2] / l];
}

export function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** Right-handed look-at; result lives in `out` (16 floats, column-major). */
export function mat4LookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const f = normalize(sub(target, eye));
  const s = normalize(cross(f, up));
  const u = cross(s, f);

  const m = new Float32Array(16);
  m[0] = s[0];   m[4] = s[1];   m[8]  = s[2];   m[12] = -dot(s, eye);
  m[1] = u[0];   m[5] = u[1];   m[9]  = u[2];   m[13] = -dot(u, eye);
  m[2] = -f[0];  m[6] = -f[1];  m[10] = -f[2];  m[14] = dot(f, eye);
  m[3] = 0;      m[7] = 0;      m[11] = 0;      m[15] = 1;
  return m;
}

/** Right-handed perspective. fovY in radians. */
export function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

/** C = A · B, all column-major mat4. */
export function mat4Mul(A: Mat4, B: Mat4): Mat4 {
  const C = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) {
        s += A[k * 4 + r]! * B[c * 4 + k]!;
      }
      C[c * 4 + r] = s;
    }
  }
  return C;
}
