// ─────────────────────────────────────────────────────────────
// tdhf.ts — Time-Dependent Hartree-Fock (RPA) frequency-dependent
// dipole polarizability α(ω). Extension of CPHF (which gives α(0))
// to non-zero real frequencies.
//
// The RPA linear-response equations for a real Hermitian
// perturbation (dipole moment) reduce, after eliminating (X − Y),
// to the symmetric form:
//
//   [(A − B)·(A + B) − ω²·I] (X + Y) = 2·(A − B)·F^μ
//
// Defining X' := (X + Y) / 2 absorbs the factor of 2 into the same
// 4·X'·F contraction used by static CPHF:
//
//   [(A − B)·(A + B) − ω²·I] X' = (A − B)·F^μ
//   α(ω)_μν = 4 · Σ_ai X'^μ_ai · F^ν_ai
//
// At ω → 0 the system reduces to (A+B) X' = F^μ — exactly the
// static CPHF equation — and α_TDHF(0) reproduces α_CPHF(0).
//
// Pole structure: M = (A−B)(A+B) − ω² becomes singular when
// ω² equals an RPA eigenvalue ω_n². α(ω) → ∞ at those poles
// (the molecular absorption frequencies). Real arithmetic in
// the response solve breaks down in a small neighborhood of
// each pole — caller responsibility to stay clear of them
// (typically by passing ω below the first excitation).
//
// Convention: ω, α in atomic units (Hartree, e²·a₀²/E_h).
//   ω = 0.05 Ha ≈ 1.36 eV ≈ 913 nm (near-IR, sub-resonance for
//   most small molecules — well below first electronic excitation).
//
// Closed-shell RHF singlet response only. Open-shell TDHF (UHF
// reference) is a follow-up; spin-symmetry breaks the simple
// (A±B) decomposition and needs 4 spin-block matrices.
//
// Cost: O(dim³) for the (A−B)·(A+B) matrix product + O(dim³)
// per ω for the linear solve (3 Cartesians combined). For
// dim = n_occ · n_virt ≲ 200 this is milliseconds.
//
// Reference: Casida (1995), Furche & Ahlrichs JCP 117, 7433
// (2002), Stratmann-Scuseria-Frisch JCP 109, 8218 (1998).
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";
import type { CGShell } from "./integrals-cg.js";
import { buildCPHFHessians } from "./cphf.js";

export interface TDHFPolarizabilityResult {
  /** 3×3 polarizability tensor at ω, row-major (a.u.). */
  readonly alpha: Float64Array;
  /** Isotropic α(ω) = ⅓·tr(α). */
  readonly isotropic: number;
  /** The frequency at which α was evaluated (Hartree). */
  readonly omega: number;
}

/**
 * Analytical frequency-dependent dipole polarizability α(ω) via
 * the RPA / TDHF linear response equations. Closed-shell RHF only.
 *
 * For ω = 0 this returns the same result as `cphfPolarizability`
 * to numerical precision. For non-zero ω below the first
 * excitation, α(ω) grows monotonically toward the first pole.
 */
export function tdhfPolarizability(
  hf: HFResult,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
  omega: number,
): TDHFPolarizabilityResult {
  const { ApB, AmB, muOV, dim } = buildCPHFHessians(hf, integrals, shells);
  // Real ω → subtract ω² from diagonal. M = (A−B)(A+B) − ω²·I.
  const alpha = solveRPAResponse(ApB, AmB, muOV, dim, -omega * omega, omega);
  const isotropic = (alpha[0]! + alpha[4]! + alpha[8]!) / 3;
  return { alpha, isotropic, omega };
}

/**
 * Analytical dipole polarizability at IMAGINARY frequency, α(iω).
 *
 * Same RPA response equation as `tdhfPolarizability` but with
 * (iω)² = −ω² so the diagonal-shift sign flips:
 *   [(A−B)(A+B) + ω²·I] X = (A−B)·F^μ
 *
 * M is strictly positive-definite (for stable RHF reference) so
 * the solve is well-conditioned at every ω > 0 — no poles on the
 * imaginary axis. α(iω) is real, positive, and decreases monotonically
 * from α(0) at ω = 0 to 0 as ω → ∞.
 *
 * Primary use: Casimir-Polder integrand for C₆ dispersion coefficients
 * (`c6Coefficient` in `dispersion.ts`).
 */
export function tdhfPolarizabilityImag(
  hf: HFResult,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
  omega: number,
): TDHFPolarizabilityResult {
  const { ApB, AmB, muOV, dim } = buildCPHFHessians(hf, integrals, shells);
  // Imaginary ω → add ω² to diagonal. M = (A−B)(A+B) + ω²·I.
  const alpha = solveRPAResponse(ApB, AmB, muOV, dim, +omega * omega, omega);
  const isotropic = (alpha[0]! + alpha[4]! + alpha[8]!) / 3;
  return { alpha, isotropic, omega };
}

/**
 * Internal: build M = (A−B)·(A+B) + omegaSquaredTerm·I, solve the
 * 3-RHS linear system M·X = (A−B)·F, and contract the 3×3 α tensor.
 * `omegaSquaredTerm` is −ω² for real frequencies, +ω² for imaginary.
 * `omegaForDiagnostic` is the bare ω (for error messages only).
 */
function solveRPAResponse(
  ApB: Float64Array,
  AmB: Float64Array,
  muOV: readonly [Float64Array, Float64Array, Float64Array],
  dim: number,
  omegaSquaredTerm: number,
  omegaForDiagnostic: number,
): Float64Array {
  // M = (A − B) · (A + B) + omegaSquaredTerm · I.
  const M = new Float64Array(dim * dim);
  for (let r = 0; r < dim; r++) {
    for (let c = 0; c < dim; c++) {
      let s = 0;
      for (let k = 0; k < dim; k++) {
        s += AmB[r * dim + k]! * ApB[k * dim + c]!;
      }
      M[r * dim + c] = s;
    }
    M[r * dim + r]! += omegaSquaredTerm;
  }
  // RHS_μ = (A − B) · F^μ.
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
  // Augmented Gauss-Jordan (3 RHS).
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
        `tdhfResponse: M singular at pivot ${p}, ω=${omegaForDiagnostic}. ` +
        `For real ω: possibly too close to an RPA pole — reduce |ω|. ` +
        `For imaginary ω: unexpected — RHF reference may be unstable.`,
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
  // α_μν = 4 · Σ X^μ · F^ν, then symmetrize.
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
  return alpha;
}
