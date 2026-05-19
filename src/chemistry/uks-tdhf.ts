// ─────────────────────────────────────────────────────────────
// uks-tdhf.ts — UKS-TDDFT frequency-dependent dipole polarizability
// α(ω) for open-shell DFT references. LSDA only this stage.
//
// Closes the {RHF, UHF, RKS, UKS} × {static, dynamic α(ω), α(iω), C₆}
// matrix.
//
// Response equation (same form as tdhf and uhf-tdhf):
//
//   [(A−B)·(A+B) − ω²·I] X = (A−B)·F^μ
//   α(ω)_μν = 2 · Σ_aiσ X^μ_σ_ai · F^ν_σ_ai
//
// For UKS-LSDA the (A−B) blocks simplify dramatically:
//   (A−B)^σσ_{ai,bj}  = (ε_a^σ − ε_i^σ) δ_{ai,bj}     ← only orbital gap
//   (A−B)^σσ'_{ai,bj} = 0                              ← cross-spin block zero
//
// because:
//   - Coulomb 2(ai|bj) is in both A and B → cancels in A−B
//   - LSDA kernel f^στ is symmetric under (σ,τ) → also cancels in A−B
//   - HF exchange would survive in A−B but hfMix=0 for pure LSDA
//
// This makes (A−B) diagonal — invertible and well-conditioned for
// any ω that isn't on a TDDFT pole.
//
// At ω = 0, the response equation reduces to (A+B)·X = F (same as
// `uksCphfPolarizability`), with the diagonal (A−B) cancelling out
// of the equation entirely. Verified by closed-shell limit and the
// static-CPHF cross-check.
//
// Imaginary-frequency variant `uksTddftPolarizabilityImag` handles
// the imaginary axis (+ω²·I on diagonal); no poles, well-conditioned
// for all ω > 0. Useful for radical-dimer C₆ via Casimir-Polder.
//
// Scope (this stage):
//   - LSDA only. Hybrid + GGA refuse pending spin-polarized GGA kernel.
//   - Same closed-shell convergence and consistency checks as the
//     other dynamic-α modules.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { UKSResult } from "./dft/uks-scf.js";
import type { CGShell } from "./integrals-cg.js";
import type { AtomSymbol } from "./atoms.js";
import type { FunctionalKind } from "./dft/functional.js";
import { dipole_cg } from "./integrals-cg.js";
import { molecularGrid, type GridOpts } from "./dft/grid.js";
import { evalBasisOnGrid, evalDensityOnGrid } from "./dft/density.js";
import { evalXCKernelLSDA_spin } from "./dft/functional-spin-kernel.js";
import { transformERIBlock } from "./uccsd.js";

export interface UKSTDHFPolarizabilityResult {
  readonly alpha: Float64Array;
  readonly isotropic: number;
  readonly omega: number;
}

export interface UKSTDHFOpts {
  readonly functional?: FunctionalKind;
  readonly nucleiSymbols?: readonly AtomSymbol[];
  readonly grid?: GridOpts;
}

export function uksTddftPolarizability(
  uks: UKSResult,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
  omega: number,
  opts: UKSTDHFOpts = {},
): UKSTDHFPolarizabilityResult {
  return solveUKSResponse(uks, integrals, shells, omega, -omega * omega, opts);
}

export function uksTddftPolarizabilityImag(
  uks: UKSResult,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
  omega: number,
  opts: UKSTDHFOpts = {},
): UKSTDHFPolarizabilityResult {
  return solveUKSResponse(uks, integrals, shells, omega, +omega * omega, opts);
}

function solveUKSResponse(
  uks: UKSResult,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
  omegaForDiagnostic: number,
  omegaSquaredTerm: number,
  opts: UKSTDHFOpts,
): UKSTDHFPolarizabilityResult {
  const functional = opts.functional ?? "lda-svwn";
  if (functional !== "lda-svwn") {
    throw new Error(
      `uksTddftPolarizability: only "lda-svwn" supported this stage (got "${functional}").`,
    );
  }
  if (!opts.nucleiSymbols) {
    throw new Error("uksTddftPolarizability: opts.nucleiSymbols required.");
  }

  const n = integrals.n;
  const { nAlpha, nBeta, C_alpha, C_beta, orbitalEnergiesAlpha, orbitalEnergiesBeta } = uks;
  const nVirtA = n - nAlpha;
  const nVirtB = n - nBeta;
  if (nVirtA === 0 && nVirtB === 0) {
    throw new Error("uksTddftPolarizability: empty virtual space");
  }
  const dimA = nAlpha * nVirtA;
  const dimB = nBeta  * nVirtB;
  const dim = dimA + dimB;

  // ── Build (A+B) (Coulomb + LSDA kernel) — same as uks-cphf. ──
  const eri_AA = transformERIBlock(integrals.eri_AO, C_alpha, C_alpha, n);
  const eri_AB = transformERIBlock(integrals.eri_AO, C_alpha, C_beta,  n);
  const eri_BB = transformERIBlock(integrals.eri_AO, C_beta,  C_beta,  n);
  const eriBA = (p: number, r: number, q: number, s: number): number =>
    eri_AB[((q * n + s) * n + p) * n + r]!;
  const e_AA = (p: number, q: number, r: number, s: number) => eri_AA[((p * n + q) * n + r) * n + s]!;
  const e_AB = (p: number, q: number, r: number, s: number) => eri_AB[((p * n + q) * n + r) * n + s]!;
  const e_BA = (p: number, q: number, r: number, s: number) => eriBA(p, q, r, s);
  const e_BB = (p: number, q: number, r: number, s: number) => eri_BB[((p * n + q) * n + r) * n + s]!;

  const ApB = new Float64Array(dim * dim);
  // αα Coulomb + ε-diag.
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
          ApB[row + col] = diag + 2 * e_AA(i, aMO, j, bMO);
        }
      }
    }
  }
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
          ApB[row + col] = diag + 2 * e_BB(i, aMO, j, bMO);
        }
      }
    }
  }

  // ── XC kernel into (A+B). Factor 2 (orbital-rotation convention). ──
  const grid = molecularGrid(integrals.nuclei, opts.nucleiSymbols, opts.grid ?? {});
  const basis = evalBasisOnGrid(integrals.shells, grid);
  const nGrid = grid.x.length;
  const rhoUp = evalDensityOnGrid(uks.D_alpha, basis);
  const rhoDn = evalDensityOnGrid(uks.D_beta,  basis);
  const { fUU, fUD, fDD } = evalXCKernelLSDA_spin(rhoUp, rhoDn);

  const phi = basis.phi;
  const phiAlpha = new Float64Array(nGrid * n);
  const phiBeta  = new Float64Array(nGrid * n);
  for (let g = 0; g < nGrid; g++) {
    const off = g * n;
    for (let p = 0; p < n; p++) {
      let sA = 0, sB = 0;
      for (let mu = 0; mu < n; mu++) {
        const phimu = phi[off + mu]!;
        sA += phimu * C_alpha[mu * n + p]!;
        sB += phimu * C_beta[mu * n + p]!;
      }
      phiAlpha[off + p] = sA;
      phiBeta[off + p]  = sB;
    }
  }
  const psiA = new Float64Array(dimA);
  const psiB = new Float64Array(dimB);
  for (let g = 0; g < nGrid; g++) {
    const w = grid.w[g]!;
    const wfUU = 2 * w * fUU[g]!;
    const wfUD = 2 * w * fUD[g]!;
    const wfDD = 2 * w * fDD[g]!;
    if (wfUU === 0 && wfUD === 0 && wfDD === 0) continue;
    const off = g * n;
    for (let i = 0; i < nAlpha; i++) {
      const phii = phiAlpha[off + i]!;
      for (let a = 0; a < nVirtA; a++) {
        psiA[i * nVirtA + a] = phii * phiAlpha[off + nAlpha + a]!;
      }
    }
    for (let i = 0; i < nBeta; i++) {
      const phii = phiBeta[off + i]!;
      for (let a = 0; a < nVirtB; a++) {
        psiB[i * nVirtB + a] = phii * phiBeta[off + nBeta + a]!;
      }
    }
    for (let ia = 0; ia < dimA; ia++) {
      const ψα = psiA[ia]!;
      if (ψα === 0) continue;
      const cα = wfUU * ψα;
      const rowα = ia * dim;
      for (let jb = 0; jb < dimA; jb++) {
        ApB[rowα + jb]! += cα * psiA[jb]!;
      }
      const cαβ = wfUD * ψα;
      for (let jb = 0; jb < dimB; jb++) {
        ApB[rowα + dimA + jb]! += cαβ * psiB[jb]!;
      }
    }
    for (let ia = 0; ia < dimB; ia++) {
      const ψβ = psiB[ia]!;
      if (ψβ === 0) continue;
      const cβα = wfUD * ψβ;
      const rowβ = (dimA + ia) * dim;
      for (let jb = 0; jb < dimA; jb++) {
        ApB[rowβ + jb]! += cβα * psiA[jb]!;
      }
      const cβ = wfDD * ψβ;
      for (let jb = 0; jb < dimB; jb++) {
        ApB[rowβ + dimA + jb]! += cβ * psiB[jb]!;
      }
    }
  }

  // ── Build (A−B). LSDA: Coulomb cancels, no HF exchange (hfMix=0),
  // LSDA kernel is symmetric so it also cancels in A−B. Only the
  // orbital-energy diagonal survives.
  const AmB = new Float64Array(dim * dim);
  for (let i = 0; i < nAlpha; i++) {
    for (let a = 0; a < nVirtA; a++) {
      const k = i * nVirtA + a;
      AmB[k * dim + k] = orbitalEnergiesAlpha[nAlpha + a]! - orbitalEnergiesAlpha[i]!;
    }
  }
  for (let i = 0; i < nBeta; i++) {
    for (let a = 0; a < nVirtB; a++) {
      const k = dimA + i * nVirtB + a;
      AmB[k * dim + k] = orbitalEnergiesBeta[nBeta + a]! - orbitalEnergiesBeta[i]!;
    }
  }

  // ── Dipole OV (both spins). ───────────────────────────────
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

  // ── M = (A−B)·(A+B) + omegaSquaredTerm·I. ──────────────────
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
  // RHS_μ = (A−B)·F^μ. For LSDA AmB is diagonal, so this is just
  // ε_ai · F^μ_ai element-wise.
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

  // ── Gauss-Jordan (3 RHS). ──────────────────────────────────
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
        `uksTddftPolarizability: M singular at pivot ${p}, ω=${omegaForDiagnostic}. ` +
        `For real ω: too close to a TDDFT pole.`,
      );
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
  const alpha = new Float64Array(9);
  for (let x = 0; x < 3; x++) {
    for (let y = 0; y < 3; y++) {
      const Xx = X[x]!, Fy = muOV[y]!;
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
  const isotropic = (alpha[0]! + alpha[4]! + alpha[8]!) / 3;
  return { alpha, isotropic, omega: omegaForDiagnostic };
}
