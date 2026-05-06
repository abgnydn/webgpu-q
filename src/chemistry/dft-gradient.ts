// ─────────────────────────────────────────────────────────────
// dft-gradient.ts — analytical Restricted Kohn-Sham DFT energy
// gradient ∂E_DFT/∂R_N. Tier 2 stages 6 (LDA) + 6b (GGA / hybrid).
//
// E_DFT = E_HF-like(P, hfMix) + E_xc[ρ, γ]
//   E_HF-like  = Σ P·h + ½ ΣPP·(μν|λσ) − (hfMix/4) ΣPP·(μλ|νσ) + V_NN
//   E_xc       = ∫ ρ(r) · ε_xc(ρ(r), γ(r)) dr
//
// The gradient breaks into two pieces:
//
//   ∂E_HF-like/∂R_N = ΣP·∂h + ½ΣPP·∂(μν|λσ) − (hfMix/4)ΣPP·∂(μλ|νσ)
//                    − ΣW·∂S + ∂V_NN/∂R_N
// (Pulay 1969; same as HF gradient but with the K-coupling scaled
// by hfMix. We reuse `hfGradient` with `kFactor = hfMix`.)
//
//   ∂E_xc/∂R_N = Σ_p w_p · v_ρ(ρ_p) · ∂ρ_p/∂R_N        (always)
//              + Σ_p w_p · v_γ(ρ_p, γ_p) · ∂γ_p/∂R_N    (GGA only)
//              + (∂w_p/∂R_N) · ε_p · ρ_p terms          (NOT yet — fixed)
//
// Pulay's identity that makes ∂ρ/∂R_N tractable:
//   ∂φ_μ(r)/∂R_N = − ∇_r φ_μ(r) · δ(atom(μ), N)
// so   ∂ρ(r_p)/∂R_N^k    = −2 · Σ_{μ on N} (∂_k φ_μ_p)·(Pφ)_μ_p
// and  ∂(∇ρ)_a/∂R_N^k    = −2 · Σ_{μ on N} { (∂_k φ_μ_p)·(P·∂_a φ)_μ_p
//                                          + (∂_k ∂_a φ_μ_p)·(Pφ)_μ_p }
// hence ∂γ_p/∂R_N^k = 2·Σ_a (∇ρ)_a · ∂(∇ρ)_a/∂R_N^k.
//
// What this module implements:
//   ✓ HF-like part for any hfMix (including pure DFT, hfMix = 0).
//   ✓ LDA XC contribution via ∂ρ/∂R_N.
//   ✓ GGA XC contribution via ∂γ/∂R_N — requires the basis Hessian
//     ∂²φ_μ/∂r_a∂r_b at every grid point (`evalBasisHessianOnGrid`).
//   ✗ Grid weight derivative — held fixed (the "weights-fixed"
//     approximation). Introduces a small grid-quadrature residual
//     in translational invariance (~1e-3 Ha/Bohr Σ). Sub-mHa/Bohr
//     in component error at the default grid. Tightening is a
//     planned follow-up.
// ─────────────────────────────────────────────────────────────

import type { CGShell } from "./integrals-cg.js";
import type { Nucleus } from "./cg-molecular.js";
import { type FunctionalKind, hfExchangeMixOf, evalXC } from "./dft/functional.js";
import { molecularGrid, type GridOpts } from "./dft/grid.js";
import {
  evalBasisOnGrid, evalBasisGradOnGrid, evalBasisHessianOnGrid,
  evalDensityOnGrid, evalDensityAndGradient,
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
 * Supported functionals: `lda-svwn`, `bvwn5`, `blyp`, `b3vwn5`,
 * `b3lyp5`. The GGA / hybrid path requires the basis Hessian
 * ∂²φ_μ/∂r_a∂r_b on the grid — same O(n · nGrid) cost as the
 * existing gradient evaluator.
 */
export function dftGradient(inp: DFTGradientInputs): Float64Array {
  const { shells, nuclei, shellAtomIdx, P, W, functional } = inp;
  const hfMix = hfExchangeMixOf(functional);
  const isGGA = functional !== "lda-svwn";
  const n = shells.length;

  // ── HF-like part: shared with the HF gradient, scaled K. ───
  const grad = hfGradient({
    shells, nuclei, shellAtomIdx, P, W,
    kFactor: hfMix,
  });

  // ── XC part: build the same molecular grid the SCF used. ────
  const grid = molecularGrid(nuclei, inp.nucleiSymbols, inp.grid ?? {});
  const basisVals: BasisValuesOnGrid = evalBasisOnGrid(shells, grid);
  const basisGrad: BasisGradientsOnGrid = evalBasisGradOnGrid(shells, grid);
  const basisHess = isGGA ? evalBasisHessianOnGrid(shells, grid) : null;
  const nGrid = grid.x.length;

  // Density, gradient, γ.
  let rho: Float64Array;
  let gradX: Float64Array | null = null;
  let gradY: Float64Array | null = null;
  let gradZ: Float64Array | null = null;
  let gamma: Float64Array | null = null;
  if (isGGA) {
    const dg = evalDensityAndGradient(P, basisVals, basisGrad);
    rho = dg.rho; gradX = dg.gradX; gradY = dg.gradY; gradZ = dg.gradZ; gamma = dg.gamma;
  } else {
    rho = evalDensityOnGrid(P, basisVals);
  }

  // Evaluate the functional at every grid point.
  const epsXc = new Float64Array(nGrid);
  const vRho   = new Float64Array(nGrid);
  const vGamma = isGGA ? new Float64Array(nGrid) : null;
  evalXC(functional, rho, gamma, epsXc, vRho, vGamma);

  const phi  = basisVals.phi;
  const phix = basisGrad.phix, phiy = basisGrad.phiy, phiz = basisGrad.phiz;
  // Per-grid-point reusable buffers.
  const Pphi   = new Float64Array(n);                  // (Pφ)_μ
  const Pphi_x = isGGA ? new Float64Array(n) : null;   // (P · ∂_x φ)_μ
  const Pphi_y = isGGA ? new Float64Array(n) : null;
  const Pphi_z = isGGA ? new Float64Array(n) : null;

  for (let p = 0; p < nGrid; p++) {
    const w = grid.w[p]!;
    const vR = vRho[p]!;
    const off = p * n;

    // Build Pφ and (for GGA) the three (P·∂_a φ) at this grid point.
    if (isGGA) {
      for (let mu = 0; mu < n; mu++) {
        let s = 0, sx = 0, sy = 0, sz = 0;
        const muRow = mu * n;
        for (let nu = 0; nu < n; nu++) {
          const Pmn = P[muRow + nu]!;
          s  += Pmn * phi [off + nu]!;
          sx += Pmn * phix[off + nu]!;
          sy += Pmn * phiy[off + nu]!;
          sz += Pmn * phiz[off + nu]!;
        }
        Pphi[mu] = s;
        Pphi_x![mu] = sx; Pphi_y![mu] = sy; Pphi_z![mu] = sz;
      }
    } else {
      for (let mu = 0; mu < n; mu++) {
        let s = 0;
        const muRow = mu * n;
        for (let nu = 0; nu < n; nu++) s += P[muRow + nu]! * phi[off + nu]!;
        Pphi[mu] = s;
      }
    }

    // ── LDA contribution: −2·w·v_ρ · (∂_k φ_μ) · (Pφ)_μ. ────
    const fLda = -2 * w * vR;
    if (fLda !== 0) {
      for (let mu = 0; mu < n; mu++) {
        const aMu = shellAtomIdx[mu]!;
        const Pphi_mu = Pphi[mu]!;
        grad[aMu * 3 + 0]! += fLda * phix[off + mu]! * Pphi_mu;
        grad[aMu * 3 + 1]! += fLda * phiy[off + mu]! * Pphi_mu;
        grad[aMu * 3 + 2]! += fLda * phiz[off + mu]! * Pphi_mu;
      }
    }

    // ── GGA contribution: ∂γ/∂R^k = 2·∇ρ · ∂(∇ρ)/∂R^k.
    //    ∂(∇ρ)_a/∂R^k = −2·Σ_{μ on N} { (∂_k φ_μ)·(P·∂_a φ)_μ
    //                                  + (∂_k ∂_a φ_μ)·(Pφ)_μ }
    //    Per-grid-point grad contribution is then
    //      −4·w·v_γ · Σ_{μ on N} { (∂_k φ_μ)·(∇ρ · Pφ_d)_μ
    //                            + (Pφ)_μ·(∇ρ · ∂_k ∇φ_μ) }
    //    where (∇ρ · Pφ_d)_μ = ρ_x·Pφ_x_μ + ρ_y·Pφ_y_μ + ρ_z·Pφ_z_μ
    //    is shared across k, and the Hessian-row sum varies by k.
    if (isGGA && basisHess) {
      const vG = vGamma![p]!;
      const rhox = gradX![p]!, rhoy = gradY![p]!, rhoz = gradZ![p]!;
      const fGga = -4 * w * vG;
      if (fGga !== 0) {
        const hxx = basisHess.phixx, hyy = basisHess.phiyy, hzz = basisHess.phizz;
        const hxy = basisHess.phixy, hxz = basisHess.phixz, hyz = basisHess.phiyz;
        for (let mu = 0; mu < n; mu++) {
          const aMu = shellAtomIdx[mu]!;
          const Pphi_mu = Pphi[mu]!;
          const Pphi_x_mu = Pphi_x![mu]!;
          const Pphi_y_mu = Pphi_y![mu]!;
          const Pphi_z_mu = Pphi_z![mu]!;
          const dot_a_d = rhox * Pphi_x_mu + rhoy * Pphi_y_mu + rhoz * Pphi_z_mu;
          // Hessian-row sums (Σ_a (∇ρ)_a · ∂_k ∂_a φ_μ) per k.
          const hSumX = rhox * hxx[off + mu]! + rhoy * hxy[off + mu]! + rhoz * hxz[off + mu]!;
          const hSumY = rhox * hxy[off + mu]! + rhoy * hyy[off + mu]! + rhoz * hyz[off + mu]!;
          const hSumZ = rhox * hxz[off + mu]! + rhoy * hyz[off + mu]! + rhoz * hzz[off + mu]!;
          grad[aMu * 3 + 0]! += fGga * (phix[off + mu]! * dot_a_d + Pphi_mu * hSumX);
          grad[aMu * 3 + 1]! += fGga * (phiy[off + mu]! * dot_a_d + Pphi_mu * hSumY);
          grad[aMu * 3 + 2]! += fGga * (phiz[off + mu]! * dot_a_d + Pphi_mu * hSumZ);
        }
      }
    }
  }

  return grad;
}
