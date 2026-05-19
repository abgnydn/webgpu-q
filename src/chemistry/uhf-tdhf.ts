// ─────────────────────────────────────────────────────────────
// uhf-tdhf.ts — Open-shell TDHF (UHF reference) frequency-dependent
// dipole polarizability α(ω). Mirror of `tdhf.ts` on the 4-spin-block
// orbital Hessians from `buildUHFCPHFHessians`.
//
// Response equation (same form as closed-shell TDHF, but with all
// matrices on the combined (α-OV + β-OV) space):
//
//   [(A − B)·(A + B) − ω²·I] X = (A − B)·F^μ
//   α(ω)_μν = 2 · Σ_aiσ X^μ_σ_ai · F^ν_σ_ai
//
// The factor of 2 in the contraction (vs 4 for RHF) carries over
// from the static UHF CPHF — UHF sums α and β sectors independently
// without a closed-shell ×2 spin-multiplicity factor.
//
// Imaginary-frequency variant `uhfTdhfPolarizabilityImag` solves the
// same response with `+ω²·I` on the diagonal instead of `−ω²·I` — no
// poles on the imaginary axis, used by `c6CoefficientOpenShell` (TBD)
// for radical-dimer C₆ coefficients.
//
// Closed-shell limit (n_α = n_β, identical α/β orbitals):
//   - The 4-spin-block (A+B) collapses to twice the closed-shell
//     RHF (A+B) (when contracted against a spin-symmetric F).
//   - The (A−B) cross-spin block is zero and the same-spin block
//     equals the RHF (A−B).
//   - α(ω) returned here matches `tdhfPolarizability(rhf, ...)` to
//     ≤ 1e-7 per tensor element. (Same convergence as the static
//     UHF CPHF matches RHF CPHF.)
//
// Reference: Furche & Ahlrichs JCP 117, 7433 (2002) — open-shell
// linear-response TDHF in spin-resolved form.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { UHFResult } from "./uhf-scf.js";
import type { CGShell } from "./integrals-cg.js";
import { buildUHFCPHFHessians } from "./uhf-cphf.js";

export interface UHFTDHFPolarizabilityResult {
  /** 3×3 polarizability tensor at ω, row-major (a.u.). */
  readonly alpha: Float64Array;
  /** Isotropic α(ω) = ⅓·tr(α). */
  readonly isotropic: number;
  /** The frequency at which α was evaluated (Hartree). */
  readonly omega: number;
}

/**
 * Open-shell analytical α(ω) via UHF TDHF. Solves the same RPA linear
 * response as `tdhfPolarizability` but on UHF's 4-spin-block orbital
 * Hessians. At ω → 0 reproduces `uhfCphfPolarizability`. For closed-
 * shell UHF input (n_α = n_β) reduces to the RHF result.
 */
export function uhfTdhfPolarizability(
  uhf: UHFResult,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
  omega: number,
): UHFTDHFPolarizabilityResult {
  const { ApB, AmB, muOV, dim } = buildUHFCPHFHessians(uhf, integrals, shells);
  const alpha = solveUHFRPAResponse(ApB, AmB, muOV, dim, -omega * omega, omega);
  const isotropic = (alpha[0]! + alpha[4]! + alpha[8]!) / 3;
  return { alpha, isotropic, omega };
}

/**
 * Open-shell α(iω) at IMAGINARY frequency. Strictly positive-definite
 * M means no poles on the imaginary axis — α(iω) decreases monotonically
 * from α(0) at ω = 0 to 0 as ω → ∞. Primary use: open-shell Casimir-
 * Polder integrand for radical-dimer C₆ coefficients.
 */
export function uhfTdhfPolarizabilityImag(
  uhf: UHFResult,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
  omega: number,
): UHFTDHFPolarizabilityResult {
  const { ApB, AmB, muOV, dim } = buildUHFCPHFHessians(uhf, integrals, shells);
  const alpha = solveUHFRPAResponse(ApB, AmB, muOV, dim, +omega * omega, omega);
  const isotropic = (alpha[0]! + alpha[4]! + alpha[8]!) / 3;
  return { alpha, isotropic, omega };
}

/**
 * Internal: build M = (A−B)·(A+B) + omegaSquaredTerm·I, solve the
 * 3-RHS linear system M·X = (A−B)·F, contract α tensor with the
 * UHF factor of 2.
 */
function solveUHFRPAResponse(
  ApB: Float64Array,
  AmB: Float64Array,
  muOV: readonly [Float64Array, Float64Array, Float64Array],
  dim: number,
  omegaSquaredTerm: number,
  omegaForDiagnostic: number,
): Float64Array {
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
        `uhfTdhfResponse: M singular at pivot ${p}, ω=${omegaForDiagnostic}. ` +
        `For real ω: too close to an open-shell RPA pole. ` +
        `For imaginary ω: unexpected — UHF reference may be unstable.`,
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
  // α_μν = 2 · Σ X^μ · F^ν (UHF factor; α + β sectors sum independently).
  const alpha = new Float64Array(9);
  for (let x = 0; x < 3; x++) {
    const Xx = X[x]!;
    for (let y = 0; y < 3; y++) {
      const Fy = muOV[y]!;
      let s = 0;
      for (let k = 0; k < dim; k++) s += Xx[k]! * Fy[k]!;
      alpha[x * 3 + y] = 2 * s;
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
