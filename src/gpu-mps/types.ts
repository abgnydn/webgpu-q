// ─────────────────────────────────────────────────────────────
// types.ts — shared types for the GPU MPS layer.
//
// Storage convention: complex matrices are stored as
// Float32Array of length 2·rows·cols, row-major, with [re, im]
// per element. Matches the existing CPU `ComplexMatrix` layout
// in src/linalg.ts (which uses Float64Array). On the GPU the
// same layout maps to `array<vec2<f32>>`.
// ─────────────────────────────────────────────────────────────

export interface CMatF32 {
  rows: number;
  cols: number;
  /** Row-major interleaved [re, im]. Length 2 · rows · cols. */
  data: Float32Array;
}

export function cmatZeros(rows: number, cols: number): CMatF32 {
  return { rows, cols, data: new Float32Array(2 * rows * cols) };
}

export function cmatFromF64(re_im: Float64Array, rows: number, cols: number): CMatF32 {
  const expected = 2 * rows * cols;
  if (re_im.length !== expected) {
    throw new Error(`cmatFromF64: data length ${re_im.length} ≠ ${expected}`);
  }
  const f32 = new Float32Array(expected);
  for (let i = 0; i < expected; i++) f32[i] = re_im[i]!;
  return { rows, cols, data: f32 };
}

/** Random complex matrix; entries iid Gaussian. Useful for tests. */
export function cmatRandom(rows: number, cols: number, seed = 0xC0FFEE): CMatF32 {
  let s = seed >>> 0;
  const rng = (): number => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  // Box-Muller for two independent N(0,1) per draw.
  const gauss = (): [number, number] => {
    let u = rng(); if (u < 1e-300) u = 1e-300;
    const v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)];
  };
  const data = new Float32Array(2 * rows * cols);
  for (let i = 0; i < rows * cols; i++) {
    const [a, b] = gauss();
    data[i * 2] = a;
    data[i * 2 + 1] = b;
  }
  return { rows, cols, data };
}

/** CPU complex matmul (FP32, for cross-checking GPU output). */
export function cmatMul(A: CMatF32, B: CMatF32): CMatF32 {
  if (A.cols !== B.rows) {
    throw new Error(`cmatMul shape: ${A.rows}×${A.cols} · ${B.rows}×${B.cols}`);
  }
  const C = cmatZeros(A.rows, B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let k = 0; k < B.cols; k++) {
      let sr = 0, si = 0;
      for (let j = 0; j < A.cols; j++) {
        const ai = (i * A.cols + j) * 2;
        const bi = (j * B.cols + k) * 2;
        const ar = A.data[ai]!, ay = A.data[ai + 1]!;
        const br = B.data[bi]!, by = B.data[bi + 1]!;
        sr += ar * br - ay * by;
        si += ar * by + ay * br;
      }
      const ci = (i * B.cols + k) * 2;
      C.data[ci] = sr;
      C.data[ci + 1] = si;
    }
  }
  return C;
}

/** Frobenius distance ‖A − B‖_F over CMatF32. */
export function cmatFrobDistance(A: CMatF32, B: CMatF32): number {
  if (A.rows !== B.rows || A.cols !== B.cols) {
    throw new Error("cmatFrobDistance: shape mismatch");
  }
  let s = 0;
  for (let i = 0; i < A.data.length; i++) {
    const d = A.data[i]! - B.data[i]!;
    s += d * d;
  }
  return Math.sqrt(s);
}
