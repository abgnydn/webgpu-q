// ─────────────────────────────────────────────────────────────
// two-qubit-dense.ts — 4×4 complex unitary helpers used by the
// brick-wall layer fusion path. The shipped JIT chain fusion
// only fuses gates on the SAME qubit; layer fusion folds:
//
//   single_q ⊗ single_{q+1}  →  4×4 unitary
//   then CNOT(q, q+1)         →  multiply on the left
//
// into one 4×4 matrix and one dispatch per qubit pair per layer.
//
// Storage convention matches `Matrix2`: row-major flat array of
// Complex, length 16 (i, j) with `M[i*4 + j]`. Basis is
// (s_qHi, s_qLo) so j = 2·s_qHi + s_qLo.
// ─────────────────────────────────────────────────────────────

import type { Complex, Matrix2 } from "./gates.js";

/** Row-major 4×4 complex matrix, 16 entries each `[re, im]`. */
export type Matrix4 = readonly Complex[];

function cmul(a: Complex, b: Complex): Complex {
  return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
}

function cadd(a: Complex, b: Complex): Complex {
  return [a[0] + b[0], a[1] + b[1]];
}

const ZERO: Complex = [0, 0];

/** Identity 4×4. */
export function identity4(): Matrix4 {
  const out: Complex[] = new Array(16).fill(ZERO);
  for (let i = 0; i < 4; i++) out[i * 4 + i] = [1, 0];
  return out;
}

/** Matrix product C = A · B, 4×4 complex, row-major. */
export function matmul4(A: Matrix4, B: Matrix4): Matrix4 {
  const out: Complex[] = new Array(16).fill(ZERO);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s: Complex = ZERO;
      for (let k = 0; k < 4; k++) {
        s = cadd(s, cmul(A[i * 4 + k]!, B[k * 4 + j]!));
      }
      out[i * 4 + j] = s;
    }
  }
  return out;
}

/**
 * Kronecker product (U_qHi ⊗ U_qLo) → 4×4 matrix in basis (s_qHi, s_qLo).
 * Index encoding: row r = 2·s_qHi_out + s_qLo_out, col c = 2·s_qHi_in + s_qLo_in.
 *
 *   (U_qHi ⊗ U_qLo)[r, c] = U_qHi[s_qHi_out, s_qHi_in] · U_qLo[s_qLo_out, s_qLo_in]
 */
export function kron4(Uhi: Matrix2, Ulo: Matrix2): Matrix4 {
  const out: Complex[] = new Array(16).fill(ZERO);
  for (let aHi = 0; aHi < 2; aHi++) {
    for (let aLo = 0; aLo < 2; aLo++) {
      for (let cHi = 0; cHi < 2; cHi++) {
        for (let cLo = 0; cLo < 2; cLo++) {
          const r = 2 * aHi + aLo;
          const c = 2 * cHi + cLo;
          out[r * 4 + c] = cmul(Uhi[2 * aHi + cHi]!, Ulo[2 * aLo + cLo]!);
        }
      }
    }
  }
  return out;
}

/** CNOT with control = qLo, target = qHi, in basis (s_qHi, s_qLo). */
export const CNOT_LO_CTRL: Matrix4 = [
  [1, 0], [0, 0], [0, 0], [0, 0],
  [0, 0], [0, 0], [0, 0], [1, 0],
  [0, 0], [0, 0], [1, 0], [0, 0],
  [0, 0], [1, 0], [0, 0], [0, 0],
];

/** CNOT with control = qHi, target = qLo, in basis (s_qHi, s_qLo). */
export const CNOT_HI_CTRL: Matrix4 = [
  [1, 0], [0, 0], [0, 0], [0, 0],
  [0, 0], [1, 0], [0, 0], [0, 0],
  [0, 0], [0, 0], [0, 0], [1, 0],
  [0, 0], [0, 0], [1, 0], [0, 0],
];

/**
 * Brick-wall pair fusion: combine a layer's two single-qubit gates
 * and one CNOT into a single 4×4 unitary.
 *
 * Order of operations (right-to-left): single-qubit gates apply
 * first, then CNOT — matching the canonical brick-wall builder
 * (E5/E7 etc.) where `apply(single, q)` precedes `cnot(c, t)`.
 *
 *   U_total = CNOT · (U_qHi ⊗ U_qLo)
 */
export function fuseBrickwallPair(
  Ulo: Matrix2,
  Uhi: Matrix2,
  cnotControl: "lo" | "hi" = "lo",
): Matrix4 {
  const tensor = kron4(Uhi, Ulo);
  const cnot = cnotControl === "lo" ? CNOT_LO_CTRL : CNOT_HI_CTRL;
  return matmul4(cnot, tensor);
}

/** Pack a Matrix4 into the Float32Array layout the WGSL kernel expects. */
export function matrix4Floats(M: Matrix4): Float32Array {
  if (M.length !== 16) throw new Error(`matrix4Floats: need 16 entries, got ${M.length}`);
  const out = new Float32Array(32);
  for (let i = 0; i < 16; i++) {
    out[2 * i]     = M[i]![0];
    out[2 * i + 1] = M[i]![1];
  }
  return out;
}
