// ─────────────────────────────────────────────────────────────
// three-qubit-dense.ts — 8×8 complex unitary helpers for L3
// Tier C tile fusion.
//
// Where Tier B (`two-qubit-dense.ts`) fuses a brick-wall pair —
// (single_q ⊗ single_{q+1}) · CNOT(q, q+1) — into one 4×4
// dispatch, Tier C does the same trick on a 3-qubit window and
// can fold a 5-gate "cascade" (3 singles + 2 CNOTs) or two
// brick-wall sub-layers into a single 8×8 dispatch.
//
// Storage: row-major flat array of 64 Complex entries. Index
// (r, c) at `M[r*8 + c]`. Basis is (s_qHi, s_qMid, s_qLo), so
// j = 4·s_qHi + 2·s_qMid + s_qLo.
// ─────────────────────────────────────────────────────────────

import type { Complex, Matrix2 } from "./gates.js";
import type { Matrix4 } from "./two-qubit-dense.js";
import { CNOT_LO_CTRL } from "./two-qubit-dense.js";

/** Row-major 8×8 complex matrix: 64 entries, each `[re, im]`. */
export type Matrix8 = readonly Complex[];

function cmul(a: Complex, b: Complex): Complex {
  return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
}

function cadd(a: Complex, b: Complex): Complex {
  return [a[0] + b[0], a[1] + b[1]];
}

const ZERO: Complex = [0, 0];

/** Identity 8×8. */
export function identity8(): Matrix8 {
  const out: Complex[] = new Array(64).fill(ZERO);
  for (let i = 0; i < 8; i++) out[i * 8 + i] = [1, 0];
  return out;
}

/** Matrix product C = A · B for two 8×8 row-major complex matrices. */
export function matmul8(A: Matrix8, B: Matrix8): Matrix8 {
  const out: Complex[] = new Array(64).fill(ZERO);
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      let s: Complex = ZERO;
      for (let k = 0; k < 8; k++) {
        s = cadd(s, cmul(A[i * 8 + k]!, B[k * 8 + j]!));
      }
      out[i * 8 + j] = s;
    }
  }
  return out;
}

/**
 * Triple Kronecker product: U_qHi ⊗ U_qMid ⊗ U_qLo as an 8×8 matrix
 * in basis (s_qHi, s_qMid, s_qLo) with j = 4·s_qHi + 2·s_qMid + s_qLo.
 */
export function kron8(Uhi: Matrix2, Umid: Matrix2, Ulo: Matrix2): Matrix8 {
  const out: Complex[] = new Array(64).fill(ZERO);
  for (let aHi = 0; aHi < 2; aHi++) {
    for (let aMid = 0; aMid < 2; aMid++) {
      for (let aLo = 0; aLo < 2; aLo++) {
        for (let cHi = 0; cHi < 2; cHi++) {
          for (let cMid = 0; cMid < 2; cMid++) {
            for (let cLo = 0; cLo < 2; cLo++) {
              const r = 4 * aHi + 2 * aMid + aLo;
              const c = 4 * cHi + 2 * cMid + cLo;
              const t1 = cmul(Uhi[2 * aHi + cHi]!, Umid[2 * aMid + cMid]!);
              out[r * 8 + c] = cmul(t1, Ulo[2 * aLo + cLo]!);
            }
          }
        }
      }
    }
  }
  return out;
}

/**
 * Embed a 4×4 gate acting on the (qLo, qMid) sub-pair into the 8×8
 * three-qubit space (qHi acts as identity). The 4×4 uses the same
 * "lo/hi" basis convention as Matrix4 — j_4 = 2·s_qMid + s_qLo, with
 * qMid playing the "hi-of-pair" role.
 */
export function embedLowMid(M4: Matrix4): Matrix8 {
  if (M4.length !== 16) throw new Error(`embedLowMid: need 16 entries, got ${M4.length}`);
  const out: Complex[] = new Array(64).fill(ZERO);
  for (let sHi = 0; sHi < 2; sHi++) {
    for (let outLM = 0; outLM < 4; outLM++) {
      for (let inLM = 0; inLM < 4; inLM++) {
        const r = 4 * sHi + outLM;
        const c = 4 * sHi + inLM;
        out[r * 8 + c] = M4[outLM * 4 + inLM]!;
      }
    }
  }
  return out;
}

/**
 * Embed a 4×4 gate acting on the (qMid, qHi) sub-pair into the 8×8
 * three-qubit space (qLo acts as identity). The 4×4 here uses
 * j_4 = 2·s_qHi + s_qMid, with qHi as "hi-of-pair" and qMid as
 * "lo-of-pair".
 */
export function embedMidHigh(M4: Matrix4): Matrix8 {
  if (M4.length !== 16) throw new Error(`embedMidHigh: need 16 entries, got ${M4.length}`);
  const out: Complex[] = new Array(64).fill(ZERO);
  for (let sLo = 0; sLo < 2; sLo++) {
    for (let outHM = 0; outHM < 4; outHM++) {
      for (let inHM = 0; inHM < 4; inHM++) {
        const sHiOut = outHM >>> 1;
        const sMidOut = outHM & 1;
        const sHiIn = inHM >>> 1;
        const sMidIn = inHM & 1;
        const r = 4 * sHiOut + 2 * sMidOut + sLo;
        const c = 4 * sHiIn + 2 * sMidIn + sLo;
        out[r * 8 + c] = M4[outHM * 4 + inHM]!;
      }
    }
  }
  return out;
}

/**
 * Cascade fusion: collapse the 5-op stair
 *
 *     (U_qHi ⊗ U_qMid ⊗ U_qLo)  ·  CNOT(qLo, qMid)  ·  CNOT(qMid, qHi)
 *
 * (right-to-left: singles first, then the two CNOTs) into one 8×8
 * unitary. This is the canonical "encoder ladder" pattern and the
 * cleanest demonstration of Tier C dispatch collapse: 5 ops → 1
 * 8×8 dispatch, vs 5 separate dispatches in the unfused path.
 */
export function fuseTriCascade(
  Ulo: Matrix2,
  Umid: Matrix2,
  Uhi: Matrix2,
): Matrix8 {
  const tensor = kron8(Uhi, Umid, Ulo);
  const cnotLM = embedLowMid(CNOT_LO_CTRL);   // CNOT(control=qLo, target=qMid)
  const cnotMH = embedMidHigh(CNOT_LO_CTRL);  // CNOT(control=qMid, target=qHi)
  return matmul8(cnotMH, matmul8(cnotLM, tensor));
}

/**
 * Pack a Matrix8 into the Float32Array layout the WGSL kernel expects:
 * 32 vec4<f32> entries, each holding two consecutive complex values
 * (xy = M[2k], zw = M[2k+1]). Total 128 floats = 512 bytes.
 */
export function matrix8Floats(M: Matrix8): Float32Array {
  if (M.length !== 64) throw new Error(`matrix8Floats: need 64 entries, got ${M.length}`);
  const out = new Float32Array(128);
  for (let k = 0; k < 32; k++) {
    const a = M[2 * k]!;
    const b = M[2 * k + 1]!;
    out[4 * k] = a[0];
    out[4 * k + 1] = a[1];
    out[4 * k + 2] = b[0];
    out[4 * k + 3] = b[1];
  }
  return out;
}
