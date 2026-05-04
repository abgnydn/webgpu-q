// ─────────────────────────────────────────────────────────────
// Gate matrices (2×2 complex).
// Each element encoded as [real, imag]. Matrix is row-major:
//   [m00, m01, m10, m11]
// where each mXY is a Complex = readonly [number, number].
// ─────────────────────────────────────────────────────────────

export type Complex = readonly [number, number];
export type Matrix2 = readonly [Complex, Complex, Complex, Complex];

const SQRT1_2 = 1 / Math.SQRT2;
const ZERO: Complex = [0, 0];
const ONE: Complex = [1, 0];
const NEG_ONE: Complex = [-1, 0];
const I: Complex = [0, 1];
const NEG_I: Complex = [0, -1];

// ── Pauli + Clifford ─────────────────────────────────────────
export const H: Matrix2 = [
  [SQRT1_2, 0], [SQRT1_2, 0],
  [SQRT1_2, 0], [-SQRT1_2, 0],
];

export const X: Matrix2 = [ZERO, ONE, ONE, ZERO];
export const Y: Matrix2 = [ZERO, NEG_I, I, ZERO];
export const Z: Matrix2 = [ONE, ZERO, ZERO, NEG_ONE];

// S = √Z, diag(1, i)
export const S: Matrix2 = [ONE, ZERO, ZERO, I];
// S† = diag(1, -i)
export const SDG: Matrix2 = [ONE, ZERO, ZERO, NEG_I];

// T = √S, diag(1, e^{iπ/4})
export const T: Matrix2 = [ONE, ZERO, ZERO, [SQRT1_2, SQRT1_2]];
// T† = diag(1, e^{-iπ/4})
export const TDG: Matrix2 = [ONE, ZERO, ZERO, [SQRT1_2, -SQRT1_2]];

// Identity — occasionally useful for padding
export const I2: Matrix2 = [ONE, ZERO, ZERO, ONE];

// ── Rotations ────────────────────────────────────────────────
export function Rx(theta: number): Matrix2 {
  const c = Math.cos(theta / 2);
  const s = Math.sin(theta / 2);
  return [
    [c, 0], [0, -s],
    [0, -s], [c, 0],
  ];
}

export function Ry(theta: number): Matrix2 {
  const c = Math.cos(theta / 2);
  const s = Math.sin(theta / 2);
  return [
    [c, 0], [-s, 0],
    [s, 0], [c, 0],
  ];
}

export function Rz(theta: number): Matrix2 {
  const c = Math.cos(theta / 2);
  const s = Math.sin(theta / 2);
  return [
    [c, -s], ZERO,
    ZERO, [c, s],
  ];
}

// Phase gate diag(1, e^{iθ})
export function P(theta: number): Matrix2 {
  return [ONE, ZERO, ZERO, [Math.cos(theta), Math.sin(theta)]];
}

// ─────────────────────────────────────────────────────────────
// Serialize a Matrix2 into the uniform-buffer layout expected
// by single-qubit.wgsl / two-qubit.wgsl:
//   m00.re m00.im m01.re m01.im m10.re m10.im m11.re m11.im
// ─────────────────────────────────────────────────────────────
export function matrixFloats(m: Matrix2): Float32Array {
  const [m00, m01, m10, m11] = m;
  return new Float32Array([
    m00[0], m00[1],
    m01[0], m01[1],
    m10[0], m10[1],
    m11[0], m11[1],
  ]);
}
