// ─────────────────────────────────────────────────────────────
// cphf.ts — Coupled-Perturbed Hartree-Fock (Stevens-Pitzer-
// Lipscomb 1963 / Pople-Krishnan-Schlegel-Binkley 1979).
//
// Solves the static response equations for closed-shell RHF to
// give the analytical dipole polarizability tensor in O(N^4)
// per axis (one linear solve) instead of the 6-SCF finite-field
// route (one SCF per ±direction). Order-of-magnitude faster
// when more than one polarizability axis is needed, AND avoids
// FD truncation noise.
//
// Formulation (spin-summed, RHF closed-shell, static / ω = 0):
//
//   For each Cartesian direction μ ∈ {x, y, z}:
//     1. Build occupied-virtual matrix elements F^μ_ai = ⟨i|r_μ|a⟩
//        in the MO basis.
//     2. Solve  (A + B) X^μ = F^μ  for the response vector X^μ.
//     3. α_μν = 2 · Σ_ai X^μ_ai · F^ν_ai     (factor 2 = RHF
//        double-occupation; the X solution carries the −1 from
//        the response identity U = −(A+B)^(-1) F, absorbed into
//        a +2 in the final contraction).
//
//   The orbital Hessian (A + B) for closed-shell singlet
//   response in chemist notation:
//     (A + B)_{ai,bj} = (ε_a − ε_i) δ_{ai,bj}
//                       + 4 (ai|bj)        Coulomb (spin-summed)
//                       −   (ab|ij)        exchange A-piece
//                       −   (aj|bi)        exchange B-piece
//
// Limitations:
//   - RHF only (UHF / open-shell CPHF is mirrored straight-
//     forwardly but not wired here).
//   - Static (ω = 0). Frequency-dependent CPHF needs (A − B)
//     too and is a follow-up.
//   - Dense (A + B) of size (n_occ·n_virt)² is built explicitly
//     and solved by Gauss-Jordan. For dim ≲ a few hundred (every
//     small-molecule case we care about) this is fast; bigger
//     systems would benefit from an iterative CG solver.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";
import { transformERIToMO } from "./mp2.js";
import { dipole_cg, type CGShell } from "./integrals-cg.js";

export interface CPHFPolarizabilityResult {
  /** 3×3 polarizability tensor, row-major (a.u. = e²·a₀²/E_h). */
  readonly alpha: Float64Array;
  /** Isotropic polarizability ⅓·tr(α). */
  readonly isotropic: number;
}

/**
 * Analytical static dipole polarizability via CPHF. Closed-shell
 * RHF only. Returns the 3×3 α tensor + isotropic average.
 */
export function cphfPolarizability(
  hf: HFResult,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
): CPHFPolarizabilityResult {
  const n = integrals.n;
  const nOcc = hf.nOccupied;
  const nVirt = n - nOcc;
  if (nVirt === 0) {
    throw new Error("cphfPolarizability: empty virtual space");
  }
  const dim = nOcc * nVirt;

  // ── Build the (A + B) orbital Hessian on OV space. ─────────
  const eri_MO = transformERIToMO(integrals.eri_AO, hf.C_MO, n);
  const eri = (p: number, q: number, r: number, s: number): number =>
    eri_MO[((p * n + q) * n + r) * n + s]!;
  const eps = hf.orbitalEnergies;

  const ApB = new Float64Array(dim * dim);
  for (let i = 0; i < nOcc; i++) {
    for (let a = 0; a < nVirt; a++) {
      const aMO = nOcc + a;
      const row = (i * nVirt + a) * dim;
      const eOrb = eps[aMO]! - eps[i]!;
      for (let j = 0; j < nOcc; j++) {
        for (let b = 0; b < nVirt; b++) {
          const bMO = nOcc + b;
          const col = j * nVirt + b;
          const diag = (i === j && a === b) ? eOrb : 0;
          const t1 = 4 * eri(i, aMO, j, bMO);   // 4·(ai|bj)
          const t2 = eri(aMO, bMO, i, j);       // (ab|ij)
          const t3 = eri(aMO, j, bMO, i);       // (aj|bi)
          ApB[row + col] = diag + t1 - t2 - t3;
        }
      }
    }
  }

  // ── Build dipole AO matrices (3 × n × n). ──────────────────
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

  // ── Transform to MO, extract OV block (n_occ × n_virt). ──
  const muOV: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(dim), new Float64Array(dim), new Float64Array(dim),
  ];
  for (let axis = 0 as 0 | 1 | 2; axis < 3; axis++) {
    for (let i = 0; i < nOcc; i++) {
      for (let a = 0; a < nVirt; a++) {
        const aMO = nOcc + a;
        let s = 0;
        for (let mu = 0; mu < n; mu++) {
          const Ci = hf.C_MO[mu * n + i]!;
          if (Ci === 0) continue;
          for (let nu = 0; nu < n; nu++) {
            s += Ci * muAO[axis][mu * n + nu]! * hf.C_MO[nu * n + aMO]!;
          }
        }
        muOV[axis][i * nVirt + a] = s;
      }
    }
  }

  // ── Solve (A + B) · X^μ = F^μ via Gauss-Jordan (augmented). ──
  // The Gauss-Jordan elimination handles all 3 RHS simultaneously
  // by augmenting the dim×dim coefficient matrix with a dim×3 block.
  const aug = new Float64Array(dim * (dim + 3));
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      aug[i * (dim + 3) + j] = ApB[i * dim + j]!;
    }
    aug[i * (dim + 3) + dim + 0] = muOV[0][i]!;
    aug[i * (dim + 3) + dim + 1] = muOV[1][i]!;
    aug[i * (dim + 3) + dim + 2] = muOV[2][i]!;
  }
  const stride = dim + 3;
  for (let p = 0; p < dim; p++) {
    // Partial pivot.
    let maxRow = p, maxVal = Math.abs(aug[p * stride + p]!);
    for (let r = p + 1; r < dim; r++) {
      const v = Math.abs(aug[r * stride + p]!);
      if (v > maxVal) { maxVal = v; maxRow = r; }
    }
    if (maxVal < 1e-14) {
      throw new Error(`cphfPolarizability: (A+B) singular at pivot ${p}`);
    }
    if (maxRow !== p) {
      for (let c = 0; c < stride; c++) {
        const tmp = aug[p * stride + c]!;
        aug[p * stride + c] = aug[maxRow * stride + c]!;
        aug[maxRow * stride + c] = tmp;
      }
    }
    // Scale pivot row.
    const pivot = aug[p * stride + p]!;
    for (let c = 0; c < stride; c++) aug[p * stride + c]! /= pivot;
    // Eliminate.
    for (let r = 0; r < dim; r++) {
      if (r === p) continue;
      const f = aug[r * stride + p]!;
      if (f === 0) continue;
      for (let c = 0; c < stride; c++) {
        aug[r * stride + c]! -= f * aug[p * stride + c]!;
      }
    }
  }
  // Extract solutions.
  const X: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(dim), new Float64Array(dim), new Float64Array(dim),
  ];
  for (let i = 0; i < dim; i++) {
    X[0][i] = aug[i * stride + dim + 0]!;
    X[1][i] = aug[i * stride + dim + 1]!;
    X[2][i] = aug[i * stride + dim + 2]!;
  }

  // ── α_μν = 4 · Σ_ai X^μ_ai · F^ν_ai (factor 4 for RHF: 2× spin
  //    + 2× response-derivative convention). Symmetric — averaged below.
  const alpha = new Float64Array(9);
  for (let x = 0; x < 3; x++) {
    for (let y = 0; y < 3; y++) {
      let s = 0;
      const Xx = X[x]!;
      const Fy = muOV[y]!;
      for (let k = 0; k < dim; k++) s += Xx[k]! * Fy[k]!;
      alpha[x * 3 + y] = 4 * s;
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
