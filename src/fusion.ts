// ─────────────────────────────────────────────────────────────
// fusion.ts — CPU helpers that mirror the fused-chain shader.
//
// Applying matrices [M_0, M_1, …, M_{k-1}] to the same qubit in
// order is algebraically equivalent to applying the single
// product matrix P = M_{k-1} · … · M_1 · M_0. This helper builds
// P so callers can:
//   • validate the GPU fused path against a one-shot product
//     (E8 correctness harness);
//   • pre-fuse statically-known chains CPU-side and emit a single
//     unfused dispatch, bypassing the shader when convenient.
//
// The 2×2 complex multiplication here is unrolled and allocates
// a fresh tuple so the returned value satisfies the readonly
// Matrix2 type without aliasing.
// ─────────────────────────────────────────────────────────────

import type { Complex, Matrix2 } from "./gates.js";

function cmul(a: Complex, b: Complex): Complex {
  return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]] as const;
}

function cadd(a: Complex, b: Complex): Complex {
  return [a[0] + b[0], a[1] + b[1]] as const;
}

/**
 * 2×2 complex matrix product C = A · B.
 * Row-major layout: `[m00, m01, m10, m11]` each as `[re, im]`.
 */
export function matmul2(A: Matrix2, B: Matrix2): Matrix2 {
  const [a00, a01, a10, a11] = A;
  const [b00, b01, b10, b11] = B;
  return [
    cadd(cmul(a00, b00), cmul(a01, b10)),
    cadd(cmul(a00, b01), cmul(a01, b11)),
    cadd(cmul(a10, b00), cmul(a11, b10)),
    cadd(cmul(a10, b01), cmul(a11, b11)),
  ] as const;
}

const IDENT: Matrix2 = [[1, 0], [0, 0], [0, 0], [1, 0]];

/**
 * Product of a gate chain in application order — applying
 * `matrices[0]` first, then `matrices[1]`, etc. is equivalent to
 * applying the single matrix
 *   P = matrices[k-1] · … · matrices[1] · matrices[0]
 * to the same qubit.
 */
export function fuseMatrices(matrices: readonly Matrix2[]): Matrix2 {
  let P: Matrix2 = IDENT;
  for (const M of matrices) P = matmul2(M, P);
  return P;
}
