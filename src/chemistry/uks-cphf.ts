// ─────────────────────────────────────────────────────────────
// uks-cphf.ts — Coupled-Perturbed Kohn-Sham static polarizability
// for open-shell UKS-DFT references. LSDA-SVWN5 only this stage;
// GGA / hybrid follow-ups need spin-polarized GGA kernel + scaled
// HF exchange in the orbital Hessian.
//
// Mathematical structure: same 4-spin-block (A+B) as `uhf-cphf.ts`
// (HF reference), but for KS-DFT replace the same-spin HF exchange
// with the spin-resolved LSDA XC kernel f^σσ' (cross-spin block
// inherits a kernel piece too):
//
//   (A+B)^σσ_{ai,bj} = (ε_a^σ − ε_i^σ) δ_{ai,bj}
//                      + 2 (ai|bj)_σσ
//                      + ⟨ai | f^σσ | bj⟩_grid
//
//   (A+B)^σσ'_{ai,bj} = 2 (ai|bj)_σσ'
//                      + ⟨ai | f^σσ' | bj⟩_grid
//
//   ⟨ai | f | bj⟩_grid = ∫ φ_i^σ(r)·φ_a^σ(r) · f(r) · φ_j^σ'(r)·φ_b^σ'(r) dr
//
// For LSDA (no GGA gradient pieces), this is the textbook UKS-CPHF
// static-response equation. Solve (A+B) X = F for each Cartesian
// dipole F^μ; α_μν = 2·Σ_σ Σ_{ai} X^μ_σ_ai · F^ν_σ_ai.
//
// Closed-shell collapse: at n_α = n_β with identical orbitals,
// f^αα = f^ββ = f^αβ = f_xc(ρ) (closed-shell LSDA second derivative).
// The 4-spin-block (A+B) then reduces to twice the closed-shell
// RKS-CPHF (A+B) when applied to a symmetric F = (F_α, F_β) = (F, F).
// The static-α value matches RKS-CPKS to numerical precision —
// validated in the test.
//
// Scope (this stage):
//   - LSDA only (functional = "lda-svwn"). Hybrid / GGA refuse.
//   - Closed- and open-shell UKS references.
//   - Per-Cartesian Gauss-Jordan solve (3 RHS combined).
//   - α tensor symmetrized, isotropic average reported.
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

export interface UKSCPHFPolarizabilityResult {
  readonly alpha: Float64Array;
  readonly isotropic: number;
}

export interface UKSCPHFOpts {
  readonly functional?: FunctionalKind;       // currently must be "lda-svwn"
  readonly nucleiSymbols?: readonly AtomSymbol[];
  readonly grid?: GridOpts;
}

export function uksCphfPolarizability(
  uks: UKSResult,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
  opts: UKSCPHFOpts = {},
): UKSCPHFPolarizabilityResult {
  const functional = opts.functional ?? "lda-svwn";
  if (functional !== "lda-svwn") {
    throw new Error(
      `uksCphfPolarizability: only "lda-svwn" supported this stage (got "${functional}"). ` +
      `Hybrid + GGA kernel paths are queued.`,
    );
  }
  if (!opts.nucleiSymbols) {
    throw new Error("uksCphfPolarizability: opts.nucleiSymbols required to rebuild XC grid.");
  }

  const n = integrals.n;
  const { nAlpha, nBeta, C_alpha, C_beta, orbitalEnergiesAlpha, orbitalEnergiesBeta } = uks;
  const nVirtA = n - nAlpha;
  const nVirtB = n - nBeta;
  if (nVirtA === 0 && nVirtB === 0) {
    throw new Error("uksCphfPolarizability: empty virtual space (both spins)");
  }
  const dimA = nAlpha * nVirtA;
  const dimB = nBeta  * nVirtB;
  const dim = dimA + dimB;

  // ── Build (A+B) Hartree (Coulomb-only J part) + ε-diagonal. ──
  // For LSDA hfMix=0, there's no HF exchange contribution. Only
  // 2(ai|bj) Coulomb + XC-kernel pieces.
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
  // αα block: ε-diag + 2(ai|bj)_αα (Coulomb only; no exchange for LDA hfMix=0)
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
  // αβ block: 2(ai|bj)_αβ
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
  // βα block: 2(ai|bj)_βα
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
  // ββ block: ε-diag + 2(ai|bj)_ββ
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

  // ── Add LSDA XC kernel contribution to (A+B). ──────────────
  // Evaluate ρ_α, ρ_β on grid; compute fUU, fUD, fDD; integrate
  // ⟨ia|f|jb⟩ = ∫ ψ_ia^σ(r) · f(r) · ψ_jb^σ'(r) dr.
  const grid = molecularGrid(integrals.nuclei, opts.nucleiSymbols, opts.grid ?? {});
  const basis = evalBasisOnGrid(integrals.shells, grid);
  const nGrid = grid.x.length;
  const rhoUp = evalDensityOnGrid(uks.D_alpha, basis);
  const rhoDn = evalDensityOnGrid(uks.D_beta,  basis);
  const { fUU, fUD, fDD } = evalXCKernelLSDA_spin(rhoUp, rhoDn);

  // Evaluate α-MO and β-MO orbitals at every grid point.
  // φ_p^α(g) = Σ_μ C_alpha[μ,p] · φ_μ(g).
  const phi = basis.phi;        // basis.phi[g*n + μ]
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

  // Build OV-products ψ_ia^σ(g) = φ_i^σ(g) · φ_a^σ(g) and accumulate
  // the kernel contribution into (A+B).
  // For each grid point, precompute weighted products once and contract.
  const psiA = new Float64Array(dimA);     // scratch: ψ_ia^α at one grid point
  const psiB = new Float64Array(dimB);     // ψ_ia^β at one grid point
  // Factor 2 on the XC kernel piece comes from the same convention
  // that puts 2·(ai|bj) on the Coulomb piece in UHF/UKS — the
  // orbital-rotation density-response factor √2 squares to 2 for
  // each second-derivative term in the orbital Hessian.
  for (let g = 0; g < nGrid; g++) {
    const w = grid.w[g]!;
    const wfUU = 2 * w * fUU[g]!;
    const wfUD = 2 * w * fUD[g]!;
    const wfDD = 2 * w * fDD[g]!;
    if (wfUU === 0 && wfUD === 0 && wfDD === 0) continue;
    const off = g * n;
    // ψ_ia^α(g)
    for (let i = 0; i < nAlpha; i++) {
      const phii = phiAlpha[off + i]!;
      if (phii === 0) {
        for (let a = 0; a < nVirtA; a++) psiA[i * nVirtA + a] = 0;
        continue;
      }
      for (let a = 0; a < nVirtA; a++) {
        psiA[i * nVirtA + a] = phii * phiAlpha[off + nAlpha + a]!;
      }
    }
    // ψ_ia^β(g)
    for (let i = 0; i < nBeta; i++) {
      const phii = phiBeta[off + i]!;
      if (phii === 0) {
        for (let a = 0; a < nVirtB; a++) psiB[i * nVirtB + a] = 0;
        continue;
      }
      for (let a = 0; a < nVirtB; a++) {
        psiB[i * nVirtB + a] = phii * phiBeta[off + nBeta + a]!;
      }
    }
    // Accumulate: ApB^αα += wfUU · psiA[ia] · psiA[jb]; symmetric in (ia, jb).
    for (let ia = 0; ia < dimA; ia++) {
      const ψα = psiA[ia]!;
      if (ψα === 0) continue;
      // αα block
      const cα = wfUU * ψα;
      const rowαα = ia * dim;
      for (let jb = 0; jb < dimA; jb++) {
        ApB[rowαα + jb]! += cα * psiA[jb]!;
      }
      // αβ block
      const cαβ = wfUD * ψα;
      const rowαβ = ia * dim;        // (ia is α index, jb will be β-offset)
      for (let jb = 0; jb < dimB; jb++) {
        ApB[rowαβ + dimA + jb]! += cαβ * psiB[jb]!;
      }
    }
    for (let ia = 0; ia < dimB; ia++) {
      const ψβ = psiB[ia]!;
      if (ψβ === 0) continue;
      // βα block (symmetric to αβ)
      const cβα = wfUD * ψβ;
      const rowβα = (dimA + ia) * dim;
      for (let jb = 0; jb < dimA; jb++) {
        ApB[rowβα + jb]! += cβα * psiA[jb]!;
      }
      // ββ block
      const cβ = wfDD * ψβ;
      const rowββ = (dimA + ia) * dim;
      for (let jb = 0; jb < dimB; jb++) {
        ApB[rowββ + dimA + jb]! += cβ * psiB[jb]!;
      }
    }
  }

  // ── Dipole OV elements for both spins. ─────────────────────
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
    // α-OV block
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
    // β-OV block
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

  // ── Solve (A+B) X = F via augmented Gauss-Jordan (3 RHS). ───
  const stride = dim + 3;
  const aug = new Float64Array(dim * stride);
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) aug[i * stride + j] = ApB[i * dim + j]!;
    aug[i * stride + dim + 0] = muOV[0][i]!;
    aug[i * stride + dim + 1] = muOV[1][i]!;
    aug[i * stride + dim + 2] = muOV[2][i]!;
  }
  for (let p = 0; p < dim; p++) {
    let maxRow = p, maxVal = Math.abs(aug[p * stride + p]!);
    for (let r = p + 1; r < dim; r++) {
      const v = Math.abs(aug[r * stride + p]!);
      if (v > maxVal) { maxVal = v; maxRow = r; }
    }
    if (maxVal < 1e-14) {
      throw new Error(`uksCphfPolarizability: (A+B) singular at pivot ${p}`);
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

  // ── α_μν = 2 · Σ_aiσ X^μ_σ_ai · F^ν_σ_ai (UKS factor 2). ────
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
  return { alpha, isotropic };
}
