// ─────────────────────────────────────────────────────────────
// dft-gradient.ts — analytical Restricted Kohn-Sham DFT energy
// gradient ∂E_DFT/∂R_N. Tier 2 stage 6.
//
// E_DFT = E_HF-like(P, hfMix) + E_xc[ρ]
//   E_HF-like  = Σ P·h + ½ ΣPP·(μν|λσ) − (hfMix/4) ΣPP·(μλ|νσ) + V_NN
//   E_xc       = ∫ ρ(r) · ε_xc(ρ(r), γ(r)) dr   (numerically: Σ_p w_p · ε_p · ρ_p)
//
// The gradient breaks into two pieces:
//
//   ∂E_HF-like/∂R_N = ΣP·∂h + ½ΣPP·∂(μν|λσ) − (hfMix/4)ΣPP·∂(μλ|νσ)
//                    − ΣW·∂S + ∂V_NN/∂R_N
// (Pulay 1969; same as HF gradient but with the K-coupling scaled
// by hfMix. We reuse `hfGradient` with `kFactor = hfMix`.)
//
//   ∂E_xc/∂R_N = Σ_p w_p · v_xc(ρ_p) · ∂ρ_p/∂R_N         (LDA)
//              + (∂w_p/∂R_N) · ε_p · ρ_p terms           (grid weight)
//              + 2 Σ_p w_p · v_γ · ∇ρ · ∂(∇ρ)/∂R_N       (GGA)
//
// What this module currently implements:
//   ✓ HF-like part for any hfMix (including pure DFT, hfMix = 0).
//   ✓ LDA XC contribution via ∂ρ/∂R_N (uses the existing
//     `evalBasisGradOnGrid` for ∇φ).
//   ✗ Grid weight derivative — held fixed (the "weights-fixed"
//     approximation, common in early DFT implementations). This
//     introduces a small grid-quadrature error; for the default
//     50r × 12θ × 24φ grid it's typically sub-mHa/Bohr.
//   ✗ GGA term ∂γ/∂R requires basis Hessians (∂² φ_μ / ∂r_a ∂r_b).
//     Deferred to a follow-up; for now, only `lda-svwn` is supported.
//
// Pulay's identity that makes ∂ρ/∂R_N tractable:
//   ∂φ_μ(r)/∂R_N = − ∇_r φ_μ(r) · δ(atom(μ), N)
// (a basis function "follows" its anchoring atom; moving the atom
// is equivalent to moving the evaluation point in the opposite
// direction). So:
//   ∂ρ(r_p)/∂R_N^k = − 2 · Σ_{μ on N, ν} P_μν · (∂_k φ_μ)·φ_ν
// (the factor 2 comes from the symmetric μ ↔ ν sum on P).
// ─────────────────────────────────────────────────────────────

import type { CGShell } from "./integrals-cg.js";
import type { Nucleus } from "./cg-molecular.js";
import { type FunctionalKind, hfExchangeMixOf, evalXC } from "./dft/functional.js";
import { molecularGrid, type GridOpts } from "./dft/grid.js";
import {
  evalBasisOnGrid, evalBasisGradOnGrid, evalDensityOnGrid,
  type BasisValuesOnGrid, type BasisGradientsOnGrid,
} from "./dft/density.js";
import { hfGradient } from "./hf-gradient.js";
import type { AtomSymbol } from "./atoms.js";

export interface DFTGradientInputs {
  readonly shells: readonly CGShell[];
  readonly nuclei: readonly Nucleus[];
  readonly shellAtomIdx: readonly number[];
  readonly nucleiSymbols: readonly AtomSymbol[];
  /** SCF AO density matrix (n × n). */
  readonly P: Float64Array;
  /** Energy-weighted AO density matrix (n × n). */
  readonly W: Float64Array;
  /** XC functional kind — must match what was used in the SCF. */
  readonly functional: FunctionalKind;
  /** Numerical-grid options — must match what was used in the SCF. */
  readonly grid?: GridOpts;
}

/**
 * Compute the analytical RKS-DFT energy gradient (Hartree / Bohr)
 * for every atom, returned as a (3·nAtoms) Float64Array.
 *
 * Currently supports `lda-svwn` only (the GGA functionals require
 * basis Hessians for the ∂γ/∂R term — a planned follow-up). For
 * unsupported functionals, throws with a clear error.
 */
export function dftGradient(inp: DFTGradientInputs): Float64Array {
  const { shells, nuclei, shellAtomIdx, P, W, functional } = inp;
  if (functional !== "lda-svwn") {
    throw new Error(
      `dftGradient: only 'lda-svwn' supported for now; '${functional}' needs ` +
      `the GGA ∂γ/∂R term (basis Hessians) — TODO follow-up.`,
    );
  }
  const hfMix = hfExchangeMixOf(functional);
  const n = shells.length;

  // ── HF-like part: shared with the HF gradient, scaled K. ───
  const grad = hfGradient({
    shells, nuclei, shellAtomIdx, P, W,
    kFactor: hfMix,
  });

  // ── XC LDA part: Σ_p w_p · v_ρ_p · ∂ρ_p/∂R_N. ──────────────
  // Build the same molecular grid the SCF used (deterministic
  // Becke-partitioned radial × angular product).
  const grid = molecularGrid(nuclei, inp.nucleiSymbols, inp.grid ?? {});
  const basisVals: BasisValuesOnGrid = evalBasisOnGrid(shells, grid);
  const basisGrad: BasisGradientsOnGrid = evalBasisGradOnGrid(shells, grid);
  const nGrid = grid.x.length;
  const rho = evalDensityOnGrid(P, basisVals);

  // Evaluate the functional at every grid point.
  const epsXc = new Float64Array(nGrid);
  const vRho   = new Float64Array(nGrid);
  // LDA path: γ + vGamma not used.
  evalXC(functional, rho, null, epsXc, vRho, null);

  // Per-grid-point precompute Pφ at each μ:
  //   (Pφ)_μ(r_p) = Σ_ν P_μν · φ_ν(r_p)
  // so ∂ρ_p/∂R_N^k = −2 · Σ_{μ on N} (∂_k φ_μ_p) · (Pφ)_μ(r_p).
  const phi = basisVals.phi;
  const phix = basisGrad.phix;
  const phiy = basisGrad.phiy;
  const phiz = basisGrad.phiz;
  const Pphi = new Float64Array(n);

  for (let p = 0; p < nGrid; p++) {
    const w = grid.w[p]!;
    const factor = -2 * w * vRho[p]!;        // Σ_p w_p · v_ρ_p · (−2 · ∂ρ/∂R term)
    if (factor === 0) continue;
    const off = p * n;

    // Pφ at this grid point.
    for (let mu = 0; mu < n; mu++) {
      let s = 0;
      const muRow = mu * n;
      for (let nu = 0; nu < n; nu++) {
        s += P[muRow + nu]! * phi[off + nu]!;
      }
      Pphi[mu] = s;
    }

    // Per-shell contribution; routed by atom index.
    for (let mu = 0; mu < n; mu++) {
      const aMu = shellAtomIdx[mu]!;
      const Pphi_mu = Pphi[mu]!;
      grad[aMu * 3 + 0]! += factor * phix[off + mu]! * Pphi_mu;
      grad[aMu * 3 + 1]! += factor * phiy[off + mu]! * Pphi_mu;
      grad[aMu * 3 + 2]! += factor * phiz[off + mu]! * Pphi_mu;
    }
  }

  return grad;
}
