// ─────────────────────────────────────────────────────────────
// uhf-cphf.ts — Coupled-Perturbed UHF static polarizability.
//
// Open-shell extension of `cphf.ts`. Solves the static linear-
// response equations on the combined α + β orbital-rotation space:
//
//   [(A+B)^αα  (A+B)^αβ] [X^α]   [F^α]
//   [(A+B)^βα  (A+B)^ββ] [X^β] = [F^β]
//
// with the spin-resolved orbital Hessian (singlet-static form):
//
//   (A+B)^σσ_{ai,bj} = (ε_a^σ − ε_i^σ) δ_{ai,bj}
//                      + 2 (ai|bj)_σσ − (ab|ij)_σσ − (aj|bi)_σσ
//   (A+B)^σσ'_{ai,bj} = 2 (ai|bj)_σσ'   (σ ≠ σ', no exchange)
//
// using the 3-block AO→MO ERI transform already used by UMP2 +
// UCCSD. Polarizability:
//
//   α_μν = 2 · [Σ_aiα X^μ_α_ai · F^ν_α_ai + Σ_aiβ X^μ_β_ai · F^ν_β_ai]
//
// (Factor 2 from the response-derivative convention; no spin sum
// because each spin block contributes its own independent sum.)
//
// For closed-shell UHF (n_α = n_β, identical orbitals), reduces
// exactly to the RHF CPHF result from `cphf.ts`.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { UHFResult } from "./uhf-scf.js";
import { dipole_cg, type CGShell } from "./integrals-cg.js";
import { transformERIBlock } from "./uccsd.js";

export interface UHFCPHFPolarizabilityResult {
  /** 3×3 polarizability tensor, row-major (a.u.). */
  readonly alpha: Float64Array;
  /** Isotropic polarizability ⅓·tr(α). */
  readonly isotropic: number;
}

export function uhfCphfPolarizability(
  uhf: UHFResult,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
): UHFCPHFPolarizabilityResult {
  const n = integrals.n;
  const { nAlpha, nBeta, C_alpha, C_beta, orbitalEnergiesAlpha, orbitalEnergiesBeta } = uhf;
  const nVirtA = n - nAlpha;
  const nVirtB = n - nBeta;
  if (nVirtA === 0 && nVirtB === 0) {
    throw new Error("uhfCphfPolarizability: empty virtual space (both spins)");
  }
  const dimA = nAlpha * nVirtA;
  const dimB = nBeta  * nVirtB;
  const dim = dimA + dimB;

  // ── Build the 3 MO-basis ERI blocks: (αα|αα), (αα|ββ), (ββ|ββ).
  // Notation: eri_ss'_uu' has chemist indices spanning the s, s'
  // MOs as the first pair and u, u' as the second.
  const eri_AA = transformERIBlock(integrals.eri_AO, C_alpha, C_alpha, n);
  const eri_AB = transformERIBlock(integrals.eri_AO, C_alpha, C_beta,  n);
  const eri_BB = transformERIBlock(integrals.eri_AO, C_beta,  C_beta,  n);
  // (ββ|αα) by position-swap symmetry: eri_BB_AA[p,r,q,s] = eri_AB[q,s,p,r]
  const eriBA = (p: number, r: number, q: number, s: number): number =>
    eri_AB[((q * n + s) * n + p) * n + r]!;

  // Accessors in chemist notation (pq|rs) per spin block.
  const e_AA = (p: number, q: number, r: number, s: number): number =>
    eri_AA[((p * n + q) * n + r) * n + s]!;
  const e_AB = (p: number, q: number, r: number, s: number): number =>
    eri_AB[((p * n + q) * n + r) * n + s]!;
  const e_BA = (p: number, q: number, r: number, s: number): number => eriBA(p, q, r, s);
  const e_BB = (p: number, q: number, r: number, s: number): number =>
    eri_BB[((p * n + q) * n + r) * n + s]!;

  // ── Build (A+B) on the combined (α-OV + β-OV) space.
  // Indexing convention: row k ∈ [0, dimA) is α-OV pair (i_α, a_α)
  // with i = k / nVirtA, a = k % nVirtA. Rows [dimA, dim) are β-OV.
  const ApB = new Float64Array(dim * dim);

  // αα block.
  for (let i = 0; i < nAlpha; i++) {
    for (let a = 0; a < nVirtA; a++) {
      const aMO = nAlpha + a;
      const row = (i * nVirtA + a) * dim;
      const eOrb = orbitalEnergiesAlpha[aMO]! - orbitalEnergiesAlpha[i]!;
      for (let j = 0; j < nAlpha; j++) {
        for (let b = 0; b < nVirtA; b++) {
          const bMO = nAlpha + b;
          const col = j * nVirtA + b;
          const diag = (i === j && a === b) ? eOrb : 0;
          // (A+B)^αα = ε-diag + 2(ai|bj)_αα − (ab|ij)_αα − (aj|bi)_αα
          const t1 = 2 * e_AA(i, aMO, j, bMO);
          const t2 = e_AA(aMO, bMO, i, j);
          const t3 = e_AA(aMO, j, bMO, i);
          ApB[row + col] = diag + t1 - t2 - t3;
        }
      }
    }
  }
  // αβ block (no exchange).
  for (let i = 0; i < nAlpha; i++) {
    for (let a = 0; a < nVirtA; a++) {
      const aMO = nAlpha + a;
      const row = (i * nVirtA + a) * dim;
      for (let j = 0; j < nBeta; j++) {
        for (let b = 0; b < nVirtB; b++) {
          const bMO = nBeta + b;
          const col = dimA + j * nVirtB + b;
          ApB[row + col] = 2 * e_AB(i, aMO, j, bMO);
        }
      }
    }
  }
  // βα block.
  for (let i = 0; i < nBeta; i++) {
    for (let a = 0; a < nVirtB; a++) {
      const aMO = nBeta + a;
      const row = (dimA + i * nVirtB + a) * dim;
      for (let j = 0; j < nAlpha; j++) {
        for (let b = 0; b < nVirtA; b++) {
          const bMO = nAlpha + b;
          const col = j * nVirtA + b;
          ApB[row + col] = 2 * e_BA(i, aMO, j, bMO);
        }
      }
    }
  }
  // ββ block.
  for (let i = 0; i < nBeta; i++) {
    for (let a = 0; a < nVirtB; a++) {
      const aMO = nBeta + a;
      const row = (dimA + i * nVirtB + a) * dim;
      const eOrb = orbitalEnergiesBeta[aMO]! - orbitalEnergiesBeta[i]!;
      for (let j = 0; j < nBeta; j++) {
        for (let b = 0; b < nVirtB; b++) {
          const bMO = nBeta + b;
          const col = dimA + j * nVirtB + b;
          const diag = (i === j && a === b) ? eOrb : 0;
          const t1 = 2 * e_BB(i, aMO, j, bMO);
          const t2 = e_BB(aMO, bMO, i, j);
          const t3 = e_BB(aMO, j, bMO, i);
          ApB[row + col] = diag + t1 - t2 - t3;
        }
      }
    }
  }

  // ── Dipole AO matrices (3 × n × n).
  const muAO: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(n * n), new Float64Array(n * n), new Float64Array(n * n),
  ];
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu; nu < n; nu++) {
      for (let axis = 0 as 0 | 1 | 2; axis < 3; axis++) {
        const v = dipole_cg(shells[mu]!, shells[nu]!, axis);
        muAO[axis][mu * n + nu] = v;
        muAO[axis][nu * n + mu] = v;
      }
    }
  }
  // Transform to MO + extract α-OV and β-OV blocks.
  const muOV: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(dim), new Float64Array(dim), new Float64Array(dim),
  ];
  for (let axis = 0 as 0 | 1 | 2; axis < 3; axis++) {
    // α block.
    for (let i = 0; i < nAlpha; i++) {
      for (let a = 0; a < nVirtA; a++) {
        const aMO = nAlpha + a;
        let s = 0;
        for (let mu = 0; mu < n; mu++) {
          const Ci = C_alpha[mu * n + i]!;
          if (Ci === 0) continue;
          for (let nu = 0; nu < n; nu++) {
            s += Ci * muAO[axis][mu * n + nu]! * C_alpha[nu * n + aMO]!;
          }
        }
        muOV[axis][i * nVirtA + a] = s;
      }
    }
    // β block.
    for (let i = 0; i < nBeta; i++) {
      for (let a = 0; a < nVirtB; a++) {
        const aMO = nBeta + a;
        let s = 0;
        for (let mu = 0; mu < n; mu++) {
          const Ci = C_beta[mu * n + i]!;
          if (Ci === 0) continue;
          for (let nu = 0; nu < n; nu++) {
            s += Ci * muAO[axis][mu * n + nu]! * C_beta[nu * n + aMO]!;
          }
        }
        muOV[axis][dimA + i * nVirtB + a] = s;
      }
    }
  }

  // ── Solve (A+B) X = F via Gauss-Jordan with augmented 3-RHS block. ──
  const aug = new Float64Array(dim * (dim + 3));
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) aug[i * (dim + 3) + j] = ApB[i * dim + j]!;
    aug[i * (dim + 3) + dim + 0] = muOV[0][i]!;
    aug[i * (dim + 3) + dim + 1] = muOV[1][i]!;
    aug[i * (dim + 3) + dim + 2] = muOV[2][i]!;
  }
  const stride = dim + 3;
  for (let p = 0; p < dim; p++) {
    let maxRow = p, maxVal = Math.abs(aug[p * stride + p]!);
    for (let r = p + 1; r < dim; r++) {
      const v = Math.abs(aug[r * stride + p]!);
      if (v > maxVal) { maxVal = v; maxRow = r; }
    }
    if (maxVal < 1e-14) {
      throw new Error(`uhfCphfPolarizability: (A+B) singular at pivot ${p}`);
    }
    if (maxRow !== p) {
      for (let c = 0; c < stride; c++) {
        const t = aug[p * stride + c]!;
        aug[p * stride + c] = aug[maxRow * stride + c]!;
        aug[maxRow * stride + c] = t;
      }
    }
    const pivot = aug[p * stride + p]!;
    for (let c = 0; c < stride; c++) aug[p * stride + c]! /= pivot;
    for (let r = 0; r < dim; r++) {
      if (r === p) continue;
      const f = aug[r * stride + p]!;
      if (f === 0) continue;
      for (let c = 0; c < stride; c++) aug[r * stride + c]! -= f * aug[p * stride + c]!;
    }
  }
  const X: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(dim), new Float64Array(dim), new Float64Array(dim),
  ];
  for (let i = 0; i < dim; i++) {
    X[0][i] = aug[i * stride + dim + 0]!;
    X[1][i] = aug[i * stride + dim + 1]!;
    X[2][i] = aug[i * stride + dim + 2]!;
  }

  // ── α_μν = 2 · Σ_aiσ X^μ_σ_ai · F^ν_σ_ai (factor 2 from
  //    response-derivative convention; no spin-sum factor because
  //    α + β sectors each contribute independently to the sum).
  const alpha = new Float64Array(9);
  for (let x = 0; x < 3; x++) {
    for (let y = 0; y < 3; y++) {
      const Xx = X[x]!, Fy = muOV[y]!;
      let s = 0;
      for (let k = 0; k < dim; k++) s += Xx[k]! * Fy[k]!;
      alpha[x * 3 + y] = 2 * s;
    }
  }
  // Symmetrize.
  for (let x = 0; x < 3; x++) {
    for (let y = x + 1; y < 3; y++) {
      const a_xy = alpha[x * 3 + y]!;
      const a_yx = alpha[y * 3 + x]!;
      const avg = 0.5 * (a_xy + a_yx);
      alpha[x * 3 + y] = avg;
      alpha[y * 3 + x] = avg;
    }
  }
  const isotropic = (alpha[0]! + alpha[4]! + alpha[8]!) / 3;
  return { alpha, isotropic };
}
