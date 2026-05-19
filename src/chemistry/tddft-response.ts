// ─────────────────────────────────────────────────────────────
// tddft-response.ts — Frequency-dependent dipole polarizability
// α(ω) via TDDFT linear response. Same RPA equation as
// `tdhfPolarizability`, but with the A and B blocks built on top
// of a Kohn-Sham reference (LDA / GGA / hybrid) — i.e. the
// orbital Hessian carries the XC kernel f_xc as well as
// Hartree-Fock exchange (with mixing for hybrids).
//
//   [(A − B)·(A + B) − ω²·I] X = (A − B)·F^μ
//   α(ω)_μν = 4 · Σ_ai X^μ_ai · F^ν_ai
//
// Reuses the existing `buildTDABlocks` (now exported) from
// `tda-dft.ts` to construct A and B for any of the supported
// functionals (hf, lda-svwn, bvwn5, blyp, b3vwn5, b3lyp5). With
// `method="hf"` the result is identical to `tdhfPolarizability`.
//
// Why DFT response matters: HF systematically overestimates
// excitation energies and underestimates α(ω) (no correlation
// in either reference or response kernel). For molecular property
// work, TDDFT with a hybrid functional is the de facto standard
// — typical errors on isotropic α(0) drop from ~20% (HF) to ~5%
// (B3LYP) vs CCSD/cc-pVTZ.
//
// Singlet (default) and triplet sectors are both supported (with
// the same functional-coverage caveats as `runTDDFT`: triplet
// works for HF, LDA, BVWN5, B3VWN5; LYP-bearing functionals refuse
// triplets pending the spin-polarized LYP kernel).
//
// Scope:
//   - Closed-shell RKS reference only.
//   - Pole-aware solver: same Gauss-Jordan as TDHF; throws cleanly
//     when ω crosses a TDDFT excitation. Stay below the first pole
//     (which is typically the first TDDFT singlet of the chosen
//     functional).
//   - Static limit (ω → 0) reproduces CPKS-style analytical static
//     α including the XC kernel — would otherwise need a separate
//     KS-CPHF code path.
//
// Reference: Casida (1995); Furche & Ahlrichs JCP 117, 7433 (2002);
// Bauernschmitt & Ahlrichs CPL 256, 454 (1996) for TDDFT response.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFLike } from "./cis.js";
import type { CGShell } from "./integrals-cg.js";
import { dipole_cg } from "./integrals-cg.js";
import { buildTDABlocks, type TDAOpts } from "./tda-dft.js";

export interface TDDFTPolarizabilityResult {
  /** 3×3 polarizability tensor at ω, row-major (a.u.). */
  readonly alpha: Float64Array;
  /** Isotropic α(ω) = ⅓·tr(α). */
  readonly isotropic: number;
  /** The frequency at which α was evaluated (Hartree). */
  readonly omega: number;
}

/**
 * Analytical frequency-dependent dipole polarizability α(ω) via
 * TDDFT linear response. Closed-shell RKS only. `opts.method`
 * selects HF / LDA / GGA / hybrid; `opts.spin` defaults to singlet.
 *
 * At ω = 0 reproduces the static analytical polarizability for the
 * chosen functional (including XC kernel contribution).
 */
export function tddftPolarizability(
  integrals: MolecularIntegrals,
  hf: HFLike,
  shells: readonly CGShell[],
  opts: TDAOpts,
  omega: number,
): TDDFTPolarizabilityResult {
  // ── Build A and B from the existing TDA-DFT machinery. ─────
  const blocks = buildTDABlocks(integrals, hf, opts, /*buildB=*/true);
  const { A, B, nOcc, nVirt } = blocks;
  if (!B) {
    throw new Error("tddftPolarizability: internal — B matrix missing");
  }
  const dim = nOcc * nVirt;

  // ── A + B and A − B on the OV space. ──────────────────────
  const ApB = new Float64Array(dim * dim);
  const AmB = new Float64Array(dim * dim);
  for (let k = 0; k < dim * dim; k++) {
    ApB[k] = A[k]! + B[k]!;
    AmB[k] = A[k]! - B[k]!;
  }

  // ── Build dipole AO matrices + transform to OV MO block. ──
  const n = integrals.n;
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

  // ── M = (A − B)·(A + B) − ω² · I.  O(dim³) matrix product.
  const M = new Float64Array(dim * dim);
  for (let r = 0; r < dim; r++) {
    for (let c = 0; c < dim; c++) {
      let s = 0;
      for (let k = 0; k < dim; k++) {
        s += AmB[r * dim + k]! * ApB[k * dim + c]!;
      }
      M[r * dim + c] = s;
    }
    M[r * dim + r]! -= omega * omega;
  }
  // ── RHS_μ = (A − B) · F^μ.
  const RHS: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(dim), new Float64Array(dim), new Float64Array(dim),
  ];
  for (let axis = 0 as 0 | 1 | 2; axis < 3; axis++) {
    const F = muOV[axis]!;
    const out = RHS[axis]!;
    for (let r = 0; r < dim; r++) {
      let s = 0;
      for (let k = 0; k < dim; k++) s += AmB[r * dim + k]! * F[k]!;
      out[r] = s;
    }
  }
  // ── Augmented Gauss-Jordan (3 RHS).
  const stride = dim + 3;
  const aug = new Float64Array(dim * stride);
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) aug[i * stride + j] = M[i * dim + j]!;
    aug[i * stride + dim + 0] = RHS[0][i]!;
    aug[i * stride + dim + 1] = RHS[1][i]!;
    aug[i * stride + dim + 2] = RHS[2][i]!;
  }
  for (let p = 0; p < dim; p++) {
    let maxRow = p, maxVal = Math.abs(aug[p * stride + p]!);
    for (let r = p + 1; r < dim; r++) {
      const v = Math.abs(aug[r * stride + p]!);
      if (v > maxVal) { maxVal = v; maxRow = r; }
    }
    if (maxVal < 1e-14) {
      throw new Error(
        `tddftPolarizability: M singular at pivot ${p}, ω=${omega}. ` +
        `Possibly too close to a TDDFT pole — reduce |ω|.`,
      );
    }
    if (maxRow !== p) {
      for (let c = 0; c < stride; c++) {
        const tmp = aug[p * stride + c]!;
        aug[p * stride + c] = aug[maxRow * stride + c]!;
        aug[maxRow * stride + c] = tmp;
      }
    }
    const pivot = aug[p * stride + p]!;
    for (let c = 0; c < stride; c++) aug[p * stride + c]! /= pivot;
    for (let r = 0; r < dim; r++) {
      if (r === p) continue;
      const f = aug[r * stride + p]!;
      if (f === 0) continue;
      for (let c = 0; c < stride; c++) {
        aug[r * stride + c]! -= f * aug[p * stride + c]!;
      }
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
  // α_μν = 4·Σ X^μ · F^ν (RHF closed-shell singlet convention,
  // matching cphf.ts / tdhf.ts).
  const alpha = new Float64Array(9);
  for (let x = 0; x < 3; x++) {
    const Xx = X[x]!;
    for (let y = 0; y < 3; y++) {
      const Fy = muOV[y]!;
      let s = 0;
      for (let k = 0; k < dim; k++) s += Xx[k]! * Fy[k]!;
      alpha[x * 3 + y] = 4 * s;
    }
  }
  for (let x = 0; x < 3; x++) {
    for (let y = x + 1; y < 3; y++) {
      const avg = 0.5 * (alpha[x * 3 + y]! + alpha[y * 3 + x]!);
      alpha[x * 3 + y] = avg;
      alpha[y * 3 + x] = avg;
    }
  }
  const isotropic = (alpha[0]! + alpha[4]! + alpha[8]!) / 3;
  return { alpha, isotropic, omega };
}
