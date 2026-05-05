// ─────────────────────────────────────────────────────────────
// dmrg-env.ts — environment tensors and matrix-free H_eff apply
// for two-site DMRG on a real-symmetric MPO.
//
// MPS site convention (real-only): T_q is a real Float64Array of
// length chiL · 2 · chiR, indexed as T[a · 2 · chiR + s · chiR + b]
// = T[a, s, b]. Bra = ket (real → no conjugation).
//
// Environment L[q] is a real Float64Array of length D · chiL · chiL,
// indexed L[w · chiL² + a · chiL + a'], representing the contraction
// of MPO + MPS-bra + MPS-ket over sites 0..q-1. Right env R[q]
// has the same shape on the right side of bond q.
//
// MPO must have all imaginary parts = 0 (TFIM, or any other
// real-Hamiltonian model). The factory checks this at construction
// time so a Y-bearing MPO (Heisenberg, XXZ) fails fast rather than
// silently dropping the imaginary contribution.
// ─────────────────────────────────────────────────────────────

import { type MPO } from "./mpo.js";

/** A single MPS site as a real rank-3 tensor (chiL × 2 × chiR). */
export interface SiteTensor {
  chiL: number;
  chiR: number;
  /** Length chiL · 2 · chiR, row-major: T[a, s, b] at a·2·chiR + s·chiR + b. */
  data: Float64Array;
}

/** Drop the imaginary half of an MPO tensor; throw if any imag is non-zero. */
export interface RealMPOTensor {
  D_l: number;
  D_r: number;
  /** Length D_l · D_r · 4, real W[w_l, w_r, σ', σ] at ((w_l·D_r+w_r)·2+sp)·2+s. */
  data: Float64Array;
}

export function mpoToReal(M: MPO, tol = 1e-15): RealMPOTensor[] {
  const out: RealMPOTensor[] = [];
  for (let q = 0; q < M.tensors.length; q++) {
    const W = M.tensors[q]!;
    const len = W.D_l * W.D_r * 4;
    const r = new Float64Array(len);
    let maxImag = 0;
    for (let i = 0; i < len; i++) {
      r[i] = W.data[i * 2]!;
      const im = W.data[i * 2 + 1]!;
      if (Math.abs(im) > maxImag) maxImag = Math.abs(im);
    }
    if (maxImag > tol) {
      throw new Error(`mpoToReal: site ${q} has imaginary entries (max=${maxImag}); use a complex-Hermitian DMRG path`);
    }
    out.push({ D_l: W.D_l, D_r: W.D_r, data: r });
  }
  return out;
}

// ── Environment indexing helpers ────────────────────────────

function envIdx(_D: number, chi: number, w: number, a: number, ap: number): number {
  return (w * chi + a) * chi + ap;
}

/** Initial left environment: L[0] of shape (1, 1, 1) with value 1. */
export function leftEnvSeed(): Float64Array {
  const out = new Float64Array(1);
  out[0] = 1;
  return out;
}

/** Initial right environment: R[N] of shape (1, 1, 1) with value 1. */
export function rightEnvSeed(): Float64Array {
  const out = new Float64Array(1);
  out[0] = 1;
  return out;
}

/**
 * Build L[q+1] from L[q] (size D_l · chiL² of T_q), MPS site T_q, and MPO W_q.
 *   L[q+1][w', a_top, a_bot] = Σ_{w, b_top, b_bot, σ, σ'}
 *     L[q][w, b_top, b_bot] · T_q[b_top, σ, a_top] · W_q[w, w', σ, σ'] · T_q[b_bot, σ', a_bot]
 *
 * Note: σ on T_top, σ' on T_bot, MPO row index σ, col index σ'.
 * Conjugation collapses to identity in the real case.
 */
export function leftEnvUpdate(
  Lprev: Float64Array,
  T: SiteTensor,
  W: RealMPOTensor,
): Float64Array {
  const chiL = T.chiL;
  const chiR = T.chiR;
  const Dl = W.D_l;
  const Dr = W.D_r;
  const out = new Float64Array(Dr * chiR * chiR);
  // Loop ordering: outer wr/atop/abot, inner reduce over (w, btop, bbot, σ, σ').
  for (let wr = 0; wr < Dr; wr++) {
    for (let atop = 0; atop < chiR; atop++) {
      for (let abot = 0; abot < chiR; abot++) {
        let sum = 0;
        for (let wl = 0; wl < Dl; wl++) {
          for (let s = 0; s < 2; s++) {
            for (let sp = 0; sp < 2; sp++) {
              const wEntry = W.data[((wl * Dr + wr) * 2 + s) * 2 + sp]!;
              if (wEntry === 0) continue;
              for (let btop = 0; btop < chiL; btop++) {
                for (let bbot = 0; bbot < chiL; bbot++) {
                  const Lentry = Lprev[envIdx(Dl, chiL, wl, btop, bbot)]!;
                  if (Lentry === 0) continue;
                  const Ttop = T.data[btop * 2 * chiR + s * chiR + atop]!;
                  const Tbot = T.data[bbot * 2 * chiR + sp * chiR + abot]!;
                  sum += Lentry * Ttop * wEntry * Tbot;
                }
              }
            }
          }
        }
        out[envIdx(Dr, chiR, wr, atop, abot)] = sum;
      }
    }
  }
  return out;
}

/**
 * Build R[q-1] from R[q] (size D_r · chiR² of T_q), MPS site T_q, and MPO W_q.
 *   R[q-1][w, b_top, b_bot] = Σ_{w', a_top, a_bot, σ, σ'}
 *     R[q][w', a_top, a_bot] · T_q[b_top, σ, a_top] · W_q[w, w', σ, σ'] · T_q[b_bot, σ', a_bot]
 */
export function rightEnvUpdate(
  Rnext: Float64Array,
  T: SiteTensor,
  W: RealMPOTensor,
): Float64Array {
  const chiL = T.chiL;
  const chiR = T.chiR;
  const Dl = W.D_l;
  const Dr = W.D_r;
  const out = new Float64Array(Dl * chiL * chiL);
  for (let wl = 0; wl < Dl; wl++) {
    for (let btop = 0; btop < chiL; btop++) {
      for (let bbot = 0; bbot < chiL; bbot++) {
        let sum = 0;
        for (let wr = 0; wr < Dr; wr++) {
          for (let s = 0; s < 2; s++) {
            for (let sp = 0; sp < 2; sp++) {
              const wEntry = W.data[((wl * Dr + wr) * 2 + s) * 2 + sp]!;
              if (wEntry === 0) continue;
              for (let atop = 0; atop < chiR; atop++) {
                for (let abot = 0; abot < chiR; abot++) {
                  const Rentry = Rnext[envIdx(Dr, chiR, wr, atop, abot)]!;
                  if (Rentry === 0) continue;
                  const Ttop = T.data[btop * 2 * chiR + s * chiR + atop]!;
                  const Tbot = T.data[bbot * 2 * chiR + sp * chiR + abot]!;
                  sum += Rentry * Ttop * wEntry * Tbot;
                }
              }
            }
          }
        }
        out[envIdx(Dl, chiL, wl, btop, bbot)] = sum;
      }
    }
  }
  return out;
}

/**
 * Matrix-free H_eff apply at sites (q, q+1). Input ψ has shape
 * (chiL, 2, 2, chiR) flattened as ψ[a · 4 · chiR + sq · 2 · chiR + sq1 · chiR + b].
 *
 *   (H_eff ψ)[a, σq, σq1, b] = Σ_{w_l, w_m, w_r, a', σ'q, σ'q1, b'}
 *     L[w_l, a, a'] · W_q[w_l, w_m, σq, σ'q] · W_{q+1}[w_m, w_r, σq1, σ'q1] · R[w_r, b, b'] · ψ[a', σ'q, σ'q1, b']
 *
 * Contraction order (fast):
 *   step 1: temp1[w_l, a, σ'q, σ'q1, b'] = Σ_{a'} L[w_l, a, a'] · ψ[a', σ'q, σ'q1, b']
 *   step 2: temp2[w_m, a, σq, σ'q1, b'] = Σ_{w_l, σ'q} temp1[w_l, a, σ'q, σ'q1, b'] · W_q[w_l, w_m, σq, σ'q]
 *   step 3: temp3[w_r, a, σq, σq1, b'] = Σ_{w_m, σ'q1} temp2[w_m, a, σq, σ'q1, b'] · W_{q+1}[w_m, w_r, σq1, σ'q1]
 *   step 4: out[a, σq, σq1, b] = Σ_{w_r, b'} temp3[w_r, a, σq, σq1, b'] · R[w_r, b, b']
 */
export function makeApplyHEff(
  L: Float64Array, R: Float64Array,
  Wq: RealMPOTensor, Wq1: RealMPOTensor,
  chiL: number, chiR: number,
): (psi: Float64Array, out: Float64Array) => void {
  const Dl_q = Wq.D_l;
  const Dr_q = Wq.D_r;
  const Dl_q1 = Wq1.D_l;
  const Dr_q1 = Wq1.D_r;
  if (Dr_q !== Dl_q1) {
    throw new Error(`MPO bond mismatch: W_q.D_r=${Dr_q}, W_{q+1}.D_l=${Dl_q1}`);
  }
  const Dmid = Dr_q;
  // ψ length = chiL · 4 · chiR.
  return (psi: Float64Array, out: Float64Array): void => {
    // Step 1: temp1[w_l, a, sq', sq1', b']
    const temp1 = new Float64Array(Dl_q * chiL * 4 * chiR);
    for (let wl = 0; wl < Dl_q; wl++) {
      for (let a = 0; a < chiL; a++) {
        for (let sqp = 0; sqp < 2; sqp++) {
          for (let sq1p = 0; sq1p < 2; sq1p++) {
            for (let bp = 0; bp < chiR; bp++) {
              let s = 0;
              for (let ap = 0; ap < chiL; ap++) {
                const Lv = L[envIdx(Dl_q, chiL, wl, a, ap)]!;
                if (Lv === 0) continue;
                const psiV = psi[ap * 4 * chiR + sqp * 2 * chiR + sq1p * chiR + bp]!;
                s += Lv * psiV;
              }
              temp1[((wl * chiL + a) * 4 + sqp * 2 + sq1p) * chiR + bp] = s;
            }
          }
        }
      }
    }
    // Step 2: temp2[w_m, a, sq, sq1', b']
    const temp2 = new Float64Array(Dmid * chiL * 4 * chiR);
    for (let wm = 0; wm < Dmid; wm++) {
      for (let a = 0; a < chiL; a++) {
        for (let sq = 0; sq < 2; sq++) {
          for (let sq1p = 0; sq1p < 2; sq1p++) {
            for (let bp = 0; bp < chiR; bp++) {
              let s = 0;
              for (let wl = 0; wl < Dl_q; wl++) {
                for (let sqp = 0; sqp < 2; sqp++) {
                  const wEntry = Wq.data[((wl * Dr_q + wm) * 2 + sq) * 2 + sqp]!;
                  if (wEntry === 0) continue;
                  const t1 = temp1[((wl * chiL + a) * 4 + sqp * 2 + sq1p) * chiR + bp]!;
                  s += wEntry * t1;
                }
              }
              temp2[((wm * chiL + a) * 4 + sq * 2 + sq1p) * chiR + bp] = s;
            }
          }
        }
      }
    }
    // Step 3: temp3[w_r, a, sq, sq1, b']
    const temp3 = new Float64Array(Dr_q1 * chiL * 4 * chiR);
    for (let wr = 0; wr < Dr_q1; wr++) {
      for (let a = 0; a < chiL; a++) {
        for (let sq = 0; sq < 2; sq++) {
          for (let sq1 = 0; sq1 < 2; sq1++) {
            for (let bp = 0; bp < chiR; bp++) {
              let s = 0;
              for (let wm = 0; wm < Dmid; wm++) {
                for (let sq1p = 0; sq1p < 2; sq1p++) {
                  const wEntry = Wq1.data[((wm * Dr_q1 + wr) * 2 + sq1) * 2 + sq1p]!;
                  if (wEntry === 0) continue;
                  const t2 = temp2[((wm * chiL + a) * 4 + sq * 2 + sq1p) * chiR + bp]!;
                  s += wEntry * t2;
                }
              }
              temp3[((wr * chiL + a) * 4 + sq * 2 + sq1) * chiR + bp] = s;
            }
          }
        }
      }
    }
    // Step 4: out[a, sq, sq1, b]
    for (let a = 0; a < chiL; a++) {
      for (let sq = 0; sq < 2; sq++) {
        for (let sq1 = 0; sq1 < 2; sq1++) {
          for (let b = 0; b < chiR; b++) {
            let s = 0;
            for (let wr = 0; wr < Dr_q1; wr++) {
              for (let bp = 0; bp < chiR; bp++) {
                const Rv = R[envIdx(Dr_q1, chiR, wr, b, bp)]!;
                if (Rv === 0) continue;
                const t3 = temp3[((wr * chiL + a) * 4 + sq * 2 + sq1) * chiR + bp]!;
                s += t3 * Rv;
              }
            }
            out[a * 4 * chiR + sq * 2 * chiR + sq1 * chiR + b] = s;
          }
        }
      }
    }
  };
}
