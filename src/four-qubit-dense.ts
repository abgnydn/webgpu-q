// ─────────────────────────────────────────────────────────────
// four-qubit-dense.ts — 16×16 complex unitary helpers for L3
// Tier D tile fusion.
//
// Tier B (`two-qubit-dense.ts`)  : 4×4   — 3 ops per pair → 1 dispatch
// Tier C (`three-qubit-dense.ts`): 8×8   — 5 ops per triple → 1 dispatch
// Tier D (this file)             : 16×16 — 7 ops per quadruple → 1 dispatch
//
// Storage: row-major flat array of 256 Complex entries. Index
// (r, c) at `M[r*16 + c]`. Basis is (s_qHi, s_qMid2, s_qMid1,
// s_qLo), so j = 8·s_qHi + 4·s_qMid2 + 2·s_qMid1 + s_qLo.
// ─────────────────────────────────────────────────────────────

import type { Complex, Matrix2 } from "./gates.js";
import type { Matrix4 } from "./two-qubit-dense.js";
import { CNOT_LO_CTRL } from "./two-qubit-dense.js";

/** Row-major 16×16 complex matrix: 256 entries, each `[re, im]`. */
export type Matrix16 = readonly Complex[];

function cmul(a: Complex, b: Complex): Complex {
  return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
}

function cadd(a: Complex, b: Complex): Complex {
  return [a[0] + b[0], a[1] + b[1]];
}

const ZERO: Complex = [0, 0];

/** Identity 16×16. */
export function identity16(): Matrix16 {
  const out: Complex[] = new Array(256).fill(ZERO);
  for (let i = 0; i < 16; i++) out[i * 16 + i] = [1, 0];
  return out;
}

/** Matrix product C = A · B for two 16×16 row-major complex matrices. */
export function matmul16(A: Matrix16, B: Matrix16): Matrix16 {
  const out: Complex[] = new Array(256).fill(ZERO);
  for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 16; j++) {
      let s: Complex = ZERO;
      for (let k = 0; k < 16; k++) {
        s = cadd(s, cmul(A[i * 16 + k]!, B[k * 16 + j]!));
      }
      out[i * 16 + j] = s;
    }
  }
  return out;
}

/**
 * Quadruple Kronecker product: U_qHi ⊗ U_qMid2 ⊗ U_qMid1 ⊗ U_qLo as a
 * 16×16 matrix in basis (s_qHi, s_qMid2, s_qMid1, s_qLo) with
 * j = 8·s_qHi + 4·s_qMid2 + 2·s_qMid1 + s_qLo.
 */
export function kron16(
  Uhi: Matrix2,
  Umid2: Matrix2,
  Umid1: Matrix2,
  Ulo: Matrix2,
): Matrix16 {
  const out: Complex[] = new Array(256).fill(ZERO);
  for (let aHi = 0; aHi < 2; aHi++) {
    for (let aM2 = 0; aM2 < 2; aM2++) {
      for (let aM1 = 0; aM1 < 2; aM1++) {
        for (let aLo = 0; aLo < 2; aLo++) {
          for (let cHi = 0; cHi < 2; cHi++) {
            for (let cM2 = 0; cM2 < 2; cM2++) {
              for (let cM1 = 0; cM1 < 2; cM1++) {
                for (let cLo = 0; cLo < 2; cLo++) {
                  const r = 8 * aHi + 4 * aM2 + 2 * aM1 + aLo;
                  const c = 8 * cHi + 4 * cM2 + 2 * cM1 + cLo;
                  const t1 = cmul(Uhi[2 * aHi + cHi]!, Umid2[2 * aM2 + cM2]!);
                  const t2 = cmul(Umid1[2 * aM1 + cM1]!, Ulo[2 * aLo + cLo]!);
                  out[r * 16 + c] = cmul(t1, t2);
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

/**
 * Embed a 4×4 gate acting on (qLo, qMid1) into the 16×16 four-qubit space
 * (qMid2, qHi act as identity). The 4×4 uses j_4 = 2·s_qMid1 + s_qLo.
 */
export function embed4Q_LoMid1(M4: Matrix4): Matrix16 {
  if (M4.length !== 16) throw new Error(`embed4Q_LoMid1: need 16 entries, got ${M4.length}`);
  const out: Complex[] = new Array(256).fill(ZERO);
  for (let sHi = 0; sHi < 2; sHi++) {
    for (let sM2 = 0; sM2 < 2; sM2++) {
      const top = 8 * sHi + 4 * sM2;
      for (let outLM = 0; outLM < 4; outLM++) {
        for (let inLM = 0; inLM < 4; inLM++) {
          const r = top + outLM;
          const c = top + inLM;
          out[r * 16 + c] = M4[outLM * 4 + inLM]!;
        }
      }
    }
  }
  return out;
}

/**
 * Embed a 4×4 gate acting on (qMid1, qMid2) into the 16×16. qLo, qHi
 * are identity. The 4×4 uses j_4 = 2·s_qMid2 + s_qMid1.
 */
export function embed4Q_Mid1Mid2(M4: Matrix4): Matrix16 {
  if (M4.length !== 16) throw new Error(`embed4Q_Mid1Mid2: need 16 entries, got ${M4.length}`);
  const out: Complex[] = new Array(256).fill(ZERO);
  for (let sHi = 0; sHi < 2; sHi++) {
    for (let sLo = 0; sLo < 2; sLo++) {
      for (let outMM = 0; outMM < 4; outMM++) {
        for (let inMM = 0; inMM < 4; inMM++) {
          const sM2Out = outMM >>> 1;
          const sM1Out = outMM & 1;
          const sM2In = inMM >>> 1;
          const sM1In = inMM & 1;
          const r = 8 * sHi + 4 * sM2Out + 2 * sM1Out + sLo;
          const c = 8 * sHi + 4 * sM2In + 2 * sM1In + sLo;
          out[r * 16 + c] = M4[outMM * 4 + inMM]!;
        }
      }
    }
  }
  return out;
}

/**
 * Embed a 4×4 gate acting on (qMid2, qHi) into the 16×16. qLo, qMid1
 * are identity. The 4×4 uses j_4 = 2·s_qHi + s_qMid2.
 */
export function embed4Q_Mid2Hi(M4: Matrix4): Matrix16 {
  if (M4.length !== 16) throw new Error(`embed4Q_Mid2Hi: need 16 entries, got ${M4.length}`);
  const out: Complex[] = new Array(256).fill(ZERO);
  for (let sM1 = 0; sM1 < 2; sM1++) {
    for (let sLo = 0; sLo < 2; sLo++) {
      for (let outMH = 0; outMH < 4; outMH++) {
        for (let inMH = 0; inMH < 4; inMH++) {
          const sHiOut = outMH >>> 1;
          const sM2Out = outMH & 1;
          const sHiIn = inMH >>> 1;
          const sM2In = inMH & 1;
          const r = 8 * sHiOut + 4 * sM2Out + 2 * sM1 + sLo;
          const c = 8 * sHiIn + 4 * sM2In + 2 * sM1 + sLo;
          out[r * 16 + c] = M4[outMH * 4 + inMH]!;
        }
      }
    }
  }
  return out;
}

/**
 * Cascade fusion: collapse the 7-op stair
 *
 *   (U_qHi ⊗ U_qMid2 ⊗ U_qMid1 ⊗ U_qLo)  ·
 *   CNOT(qLo, qMid1)  ·  CNOT(qMid1, qMid2)  ·  CNOT(qMid2, qHi)
 *
 * (right-to-left: singles first, then three cascading CNOTs) into one
 * 16×16 unitary. The Tier D headline pattern.
 */
export function fuseQuadCascade(
  Ulo: Matrix2,
  Umid1: Matrix2,
  Umid2: Matrix2,
  Uhi: Matrix2,
): Matrix16 {
  const tensor = kron16(Uhi, Umid2, Umid1, Ulo);
  const cnotLM1 = embed4Q_LoMid1(CNOT_LO_CTRL);   // CNOT(qLo → qMid1)
  const cnotM1M2 = embed4Q_Mid1Mid2(CNOT_LO_CTRL); // CNOT(qMid1 → qMid2)
  const cnotM2H = embed4Q_Mid2Hi(CNOT_LO_CTRL);   // CNOT(qMid2 → qHi)
  // U_total = cnotM2H · cnotM1M2 · cnotLM1 · tensor
  return matmul16(cnotM2H, matmul16(cnotM1M2, matmul16(cnotLM1, tensor)));
}

/**
 * Pack a Matrix16 into the Float32Array layout the WGSL kernel expects:
 * 128 vec4<f32> entries, each holding two consecutive complex values
 * (xy = M[2k], zw = M[2k+1]). Total 512 floats = 2048 bytes.
 */
export function matrix16Floats(M: Matrix16): Float32Array {
  if (M.length !== 256) throw new Error(`matrix16Floats: need 256 entries, got ${M.length}`);
  const out = new Float32Array(512);
  for (let k = 0; k < 128; k++) {
    const a = M[2 * k]!;
    const b = M[2 * k + 1]!;
    out[4 * k] = a[0];
    out[4 * k + 1] = a[1];
    out[4 * k + 2] = b[0];
    out[4 * k + 3] = b[1];
  }
  return out;
}
