// ─────────────────────────────────────────────────────────────
// uks-scf.ts — Unrestricted Kohn-Sham DFT SCF (open-shell).
//
// Closes the {RHF, UHF, RKS, UKS} 4-cell SCF matrix. Mirror of
// `rks-scf.ts` on spin-resolved densities D_α, D_β with separate
// α/β Fock matrices built from the spin-polarized XC kernel.
//
//   Build per-spin density:
//     D_σ = C_σ · C_σ^T                  (no factor of 2)
//     D_total = D_α + D_β
//
//   Spin-polarized energy decomposition:
//     E = Σ_σ Σ D_σ h + ½ Σ D_total J(D_total)
//       + E_xc[ρ_α, ρ_β, γ_αα, γ_αβ, γ_ββ]
//       − ¼ hfMix [Σ D_α K(D_α) + Σ D_β K(D_β)] + V_nn
//
//   Per-spin Fock:
//     F_σ = h + J(D_total) + V_xc^σ  − ½ hfMix · K(D_σ)
//
// For closed-shell input (n_α = n_β, no symmetry breaking) the UKS
// converged density and energy collapse to RKS to numerical precision
// — same "open-shell-as-closed-shell" sanity check that uhf-scf.ts
// uses against rhf-scf.
//
// Scope (this stage):
//   - LDA-SVWN5 only. GGA (BVWN5/BLYP/B3VWN5/B3LYP5) needs a
//     spin-polarized buildVxcGGA helper (per-spin v_γαα + cross-spin
//     v_γαβ terms with both ∇ρ_α and ∇ρ_β); follow-up.
//   - Hybrid functionals would also need spin-resolved K builds; the
//     scaffolding is here but only LDA (hfMix=0) is wired.
//   - DIIS on stacked (α + β) error vector — same pattern as uhf-scf.
//   - Optional virtual-orbital level shift on per-spin Fock.
//   - Symmetry-breaking perturbation on initial α-Fock (default
//     0.01 Ha) to encourage open-shell convergence.
// ─────────────────────────────────────────────────────────────

import { type MolecularIntegrals } from "../cg-molecular.js";
import { eigsymmetric } from "../../manybody/dense-eig.js";
import { molecularGrid, type GridOpts, type MolecularGrid } from "./grid.js";
import {
  evalBasisOnGrid, evalDensityOnGrid,
  type BasisValuesOnGrid,
} from "./density.js";
import { evalXC_LSDA } from "./functional-spin.js";
import { hfExchangeMixOf, type FunctionalKind } from "./functional.js";
import { buildJ, buildK, buildVxcLDA } from "./rks-scf.js";
import type { AtomSymbol } from "../atoms.js";

export interface UKSResult {
  readonly energy: number;
  readonly electronicEnergy: number;
  readonly exchangeCorrelationEnergy: number;
  /** α-orbital MO coefficients (n × n). */
  readonly C_alpha: Float64Array;
  readonly C_beta: Float64Array;
  readonly orbitalEnergiesAlpha: Float64Array;
  readonly orbitalEnergiesBeta: Float64Array;
  readonly D_alpha: Float64Array;
  readonly D_beta: Float64Array;
  /** Spin density D_α − D_β. */
  readonly spinDensity: Float64Array;
  readonly nAlpha: number;
  readonly nBeta: number;
  /** UKS ⟨S²⟩ via the same Mayer formula used in UHF:
   *  ⟨S²⟩ = S(S+1)_exact + (nβ − Σ_{ij}|⟨α_i|β_j⟩|²). */
  readonly s2: number;
  readonly s2Exact: number;
  readonly iter: number;
  readonly converged: boolean;
  readonly history: readonly number[];
  readonly nGridPoints: number;
}

export interface UKSOpts {
  /** XC functional. Only "lda-svwn" supported in this stage. */
  readonly functional?: FunctionalKind;
  readonly maxIter?: number;
  readonly energyTol?: number;
  readonly residualTol?: number;
  readonly useDIIS?: boolean;
  readonly diisHistory?: number;
  readonly damping?: number;
  readonly grid?: GridOpts;
  /** Initial α-Fock diagonal perturbation to break spin symmetry.
   *  Default 0.01 Ha. Set to 0 for closed-shell-like inputs. */
  readonly symmetryBreaking?: number;
  readonly levelShift?: number;
}

export function runUKSDFT(
  integrals: MolecularIntegrals,
  nAlpha: number,
  nBeta: number,
  nucleiSymbols: readonly AtomSymbol[],
  opts: UKSOpts = {},
): UKSResult {
  if (nAlpha < 0 || nBeta < 0) {
    throw new Error(`runUKSDFT: invalid electron counts (nα=${nAlpha}, nβ=${nBeta})`);
  }
  const n = integrals.n;
  if (nAlpha > n || nBeta > n) {
    throw new Error(`runUKSDFT: too many electrons (nα=${nAlpha}, nβ=${nBeta}, n=${n})`);
  }
  const functional: FunctionalKind = opts.functional ?? "lda-svwn";
  if (functional !== "lda-svwn") {
    throw new Error(
      `runUKSDFT: only "lda-svwn" supported in this stage (got "${functional}"). ` +
      `GGA UKS (BVWN5/BLYP/B3VWN5/B3LYP5) is a queued follow-up.`,
    );
  }
  const hfMix = hfExchangeMixOf(functional);     // 0 for pure LDA
  const maxIter = opts.maxIter ?? 100;
  const eTol = opts.energyTol ?? 1e-7;
  const rTol = opts.residualTol ?? 1e-5;
  const useDIIS = opts.useDIIS ?? true;
  const diisMaxHistory = opts.diisHistory ?? 8;
  const damping = opts.damping ?? 0.5;
  const levelShift = opts.levelShift ?? 0;
  const symBreak = opts.symmetryBreaking ?? 0.01;

  const { S_AO, h_AO, eri_AO, X, Vnn } = integrals;

  // ── Molecular grid + basis values. ─────────────────────────
  const grid: MolecularGrid = molecularGrid(integrals.nuclei, nucleiSymbols, opts.grid ?? {});
  const basis: BasisValuesOnGrid = evalBasisOnGrid(integrals.shells, grid);
  const nGrid = grid.x.length;

  // ── Initial α / β guess from core h with α-spin perturbation. ──
  const hPrime = transformSymmetric(h_AO, X, n);
  const hPrimeAlpha = new Float64Array(hPrime);
  if (symBreak > 0) {
    for (let i = 0; i < n; i++) hPrimeAlpha[i * n + i]! -= symBreak;
  }
  const initAlpha = solveFock(hPrimeAlpha, X, n);
  const initBeta  = solveFock(hPrime,     X, n);
  let C_alpha = initAlpha.C_MO;
  let C_beta  = initBeta.C_MO;
  let epsAlpha = initAlpha.eps;
  let epsBeta  = initBeta.eps;
  let cPrimeAlphaPrev = initAlpha.cPrime;
  let cPrimeBetaPrev  = initBeta.cPrime;
  let D_alpha = densityFromCSpin(C_alpha, nAlpha, n);
  let D_beta  = densityFromCSpin(C_beta,  nBeta,  n);

  let E_old = Infinity;
  let E_xc_last = 0;
  const history: number[] = [];
  let converged = false;
  let iter = 0;

  const diisFa: Float64Array[] = [];
  const diisFb: Float64Array[] = [];
  const diisEa: Float64Array[] = [];
  const diisEb: Float64Array[] = [];

  const epsXc = new Float64Array(nGrid);
  const vRhoUp = new Float64Array(nGrid);
  const vRhoDn = new Float64Array(nGrid);

  for (iter = 1; iter <= maxIter; iter++) {
    const D_total = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) D_total[i] = D_alpha[i]! + D_beta[i]!;

    const J = buildJ(D_total, eri_AO, n);
    const Ka = hfMix > 0 ? buildK(D_alpha, eri_AO, n) : null;
    const Kb = hfMix > 0 ? buildK(D_beta,  eri_AO, n) : null;

    // Spin-resolved density on grid.
    const rhoUp = evalDensityOnGrid(D_alpha, basis);
    const rhoDn = evalDensityOnGrid(D_beta,  basis);

    // ── Spin-polarized LDA-SVWN5 functional evaluation. ───────
    evalXC_LSDA(rhoUp, rhoDn, epsXc, vRhoUp, vRhoDn);

    // Spin-resolved V_xc AO matrices via the same LDA builder.
    const VxcA = buildVxcLDA(vRhoUp, basis, grid.w);
    const VxcB = buildVxcLDA(vRhoDn, basis, grid.w);

    // ── Fock_σ = h + J + V_xc^σ − ½·hfMix·K_σ ─────────────────
    const F_alpha = new Float64Array(n * n);
    const F_beta  = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) {
      F_alpha[i] = h_AO[i]! + J[i]! + VxcA[i]!;
      F_beta[i]  = h_AO[i]! + J[i]! + VxcB[i]!;
    }
    if (Ka && Kb && hfMix > 0) {
      for (let i = 0; i < n * n; i++) {
        F_alpha[i]! -= 0.5 * hfMix * Ka[i]!;
        F_beta[i]!  -= 0.5 * hfMix * Kb[i]!;
      }
    }

    // ── Energy ────────────────────────────────────────────────
    let EoneA = 0, EoneB = 0, EjT = 0, Exc = 0, EkA = 0, EkB = 0;
    for (let i = 0; i < n * n; i++) {
      EoneA += D_alpha[i]! * h_AO[i]!;
      EoneB += D_beta[i]!  * h_AO[i]!;
      EjT   += D_total[i]! * J[i]!;
      if (Ka) EkA += D_alpha[i]! * Ka[i]!;
      if (Kb) EkB += D_beta[i]!  * Kb[i]!;
    }
    for (let p = 0; p < nGrid; p++) Exc += grid.w[p]! * epsXc[p]! * (rhoUp[p]! + rhoDn[p]!);
    const Ehfx = -0.5 * hfMix * (EkA + EkB);    // hybrid HF-exchange total
    const E = EoneA + EoneB + 0.5 * EjT + Exc + Ehfx + Vnn;
    history.push(E);
    E_xc_last = Exc + Ehfx;

    // ── DIIS on stacked (α + β) error vector. ─────────────────
    let Fa_use: Float64Array = F_alpha;
    let Fb_use: Float64Array = F_beta;
    let errMax = 0;
    if (useDIIS) {
      const eA = buildDIISError(F_alpha, D_alpha, S_AO, X, n);
      const eB = buildDIISError(F_beta,  D_beta,  S_AO, X, n);
      for (let i = 0; i < n * n; i++) {
        const a = Math.abs(eA[i]!); if (a > errMax) errMax = a;
        const b = Math.abs(eB[i]!); if (b > errMax) errMax = b;
      }
      diisFa.push(F_alpha); diisFb.push(F_beta);
      diisEa.push(eA);      diisEb.push(eB);
      if (diisFa.length > diisMaxHistory) {
        diisFa.shift(); diisFb.shift();
        diisEa.shift(); diisEb.shift();
      }
      if (diisFa.length >= 2) {
        const c = solveDIISCoeffs(diisEa, diisEb, n);
        if (c !== null) {
          const Fa_ext = new Float64Array(n * n);
          const Fb_ext = new Float64Array(n * n);
          for (let k = 0; k < diisFa.length; k++) {
            const ck = c[k]!;
            const Fak = diisFa[k]!, Fbk = diisFb[k]!;
            for (let i = 0; i < n * n; i++) {
              Fa_ext[i]! += ck * Fak[i]!;
              Fb_ext[i]! += ck * Fbk[i]!;
            }
          }
          Fa_use = Fa_ext;
          Fb_use = Fb_ext;
        }
      }
    }

    // ── Diagonalize → new C_σ, eps_σ. ─────────────────────────
    const FPrimeA = transformSymmetric(Fa_use, X, n);
    const FPrimeB = transformSymmetric(Fb_use, X, n);

    if (levelShift > 0) {
      applyLevelShift(FPrimeA, cPrimeAlphaPrev, nAlpha, n, levelShift);
      applyLevelShift(FPrimeB, cPrimeBetaPrev,  nBeta,  n, levelShift);
    }

    const solA = solveFock(FPrimeA, X, n);
    const solB = solveFock(FPrimeB, X, n);
    C_alpha = solA.C_MO;
    C_beta  = solB.C_MO;
    epsAlpha = solA.eps;
    epsBeta  = solB.eps;
    cPrimeAlphaPrev = solA.cPrime;
    cPrimeBetaPrev  = solB.cPrime;

    const D_alpha_new = densityFromCSpin(C_alpha, nAlpha, n);
    const D_beta_new  = densityFromCSpin(C_beta,  nBeta,  n);

    let dNorm = 0;
    for (let i = 0; i < n * n; i++) {
      const da = D_alpha_new[i]! - D_alpha[i]!;
      const db = D_beta_new[i]!  - D_beta[i]!;
      dNorm += da * da + db * db;
    }
    dNorm = Math.sqrt(dNorm);

    if (useDIIS) {
      D_alpha = D_alpha_new;
      D_beta  = D_beta_new;
    } else {
      for (let i = 0; i < n * n; i++) {
        D_alpha[i] = damping * D_alpha_new[i]! + (1 - damping) * D_alpha[i]!;
        D_beta[i]  = damping * D_beta_new[i]!  + (1 - damping) * D_beta[i]!;
      }
    }

    const residOk = useDIIS ? errMax < rTol : dNorm < rTol;
    if (Math.abs(E - E_old) < eTol && residOk && iter >= 2) {
      converged = true;
      E_old = E;
      break;
    }
    E_old = E;
  }

  // Strip level-shift from reported virtual eigenvalues.
  if (levelShift > 0) {
    const epsAOut = new Float64Array(epsAlpha);
    const epsBOut = new Float64Array(epsBeta);
    for (let i = nAlpha; i < n; i++) epsAOut[i]! -= levelShift;
    for (let i = nBeta;  i < n; i++) epsBOut[i]! -= levelShift;
    epsAlpha = epsAOut;
    epsBeta  = epsBOut;
  }

  // ── ⟨S²⟩ Mayer formula (matches UHF). ─────────────────────
  // ⟨S²⟩ = S_exact(S_exact + 1) + (nβ − Σ_ij |⟨α_i|β_j⟩|²)
  // where overlap matrix between α and β occupied: M_ij = Σ_μν C_α,μi S_μν C_β,νj.
  const S = 0.5 * (nAlpha - nBeta);
  const s2Exact = S * (S + 1);
  let mayerOverlap = 0;
  for (let i = 0; i < nAlpha; i++) {
    for (let j = 0; j < nBeta; j++) {
      let m = 0;
      for (let mu = 0; mu < n; mu++) {
        const Caμi = C_alpha[mu * n + i]!;
        if (Caμi === 0) continue;
        let sum_nu = 0;
        for (let nu = 0; nu < n; nu++) {
          sum_nu += S_AO[mu * n + nu]! * C_beta[nu * n + j]!;
        }
        m += Caμi * sum_nu;
      }
      mayerOverlap += m * m;
    }
  }
  const s2 = s2Exact + (nBeta - mayerOverlap);

  // Spin density.
  const spinDensity = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) spinDensity[i] = D_alpha[i]! - D_beta[i]!;

  return {
    energy: E_old,
    electronicEnergy: E_old - Vnn,
    exchangeCorrelationEnergy: E_xc_last,
    C_alpha, C_beta,
    orbitalEnergiesAlpha: epsAlpha,
    orbitalEnergiesBeta:  epsBeta,
    D_alpha, D_beta,
    spinDensity,
    nAlpha, nBeta,
    s2, s2Exact,
    iter, converged,
    history,
    nGridPoints: nGrid,
  };
}

// ─────────────────────────────────────────────────────────────
// Local helpers (mirror of the rks-scf private ones).
// ─────────────────────────────────────────────────────────────

function transformSymmetric(M: Float64Array, X: Float64Array, n: number): Float64Array {
  const tmp = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += X[k * n + i]! * M[k * n + j]!;
      tmp[i * n + j] = s;
    }
  }
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += tmp[i * n + k]! * X[k * n + j]!;
      out[i * n + j] = s;
    }
  }
  return out;
}

function solveFock(FPrime: Float64Array, X: Float64Array, n: number): {
  C_MO: Float64Array;
  cPrime: Float64Array;
  eps: Float64Array;
} {
  const eig = eigsymmetric(FPrime, n);
  const C_MO = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += X[r * n + k]! * eig.vectors[c * n + k]!;
      C_MO[r * n + c] = s;
    }
  }
  return { C_MO, cPrime: eig.vectors, eps: eig.values };
}

/** Unrestricted density: D_σ[μν] = Σ_{i=0}^{n_σ-1} C_σ[μ,i] C_σ[ν,i]. NO factor of 2. */
function densityFromCSpin(C: Float64Array, nOcc: number, n: number): Float64Array {
  const D = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let i = 0; i < nOcc; i++) s += C[mu * n + i]! * C[nu * n + i]!;
      D[mu * n + nu] = s;
    }
  }
  return D;
}

function applyLevelShift(
  FPrime: Float64Array,
  cPrimePrev: Float64Array,
  nOcc: number,
  n: number,
  shift: number,
): void {
  for (let p = nOcc; p < n; p++) {
    for (let i = 0; i < n; i++) {
      const cpi = cPrimePrev[p * n + i]!;
      if (cpi === 0) continue;
      const shifted_cpi = shift * cpi;
      for (let j = 0; j < n; j++) {
        FPrime[i * n + j]! += shifted_cpi * cPrimePrev[p * n + j]!;
      }
    }
  }
}

function buildDIISError(F: Float64Array, D: Float64Array, S: Float64Array, X: Float64Array, n: number): Float64Array {
  const FD  = matmul(F, D, n);
  const FDS = matmul(FD, S, n);
  const SD  = matmul(S, D, n);
  const SDF = matmul(SD, F, n);
  const eAO = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) eAO[i] = FDS[i]! - SDF[i]!;
  return transformGeneral(eAO, X, n);
}

function transformGeneral(M: Float64Array, X: Float64Array, n: number): Float64Array {
  const tmp = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += X[k * n + i]! * M[k * n + j]!;
      tmp[i * n + j] = s;
    }
  }
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += tmp[i * n + k]! * X[k * n + j]!;
      out[i * n + j] = s;
    }
  }
  return out;
}

function matmul(A: Float64Array, B: Float64Array, n: number): Float64Array {
  const C = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += A[i * n + k]! * B[k * n + j]!;
      C[i * n + j] = s;
    }
  }
  return C;
}

/**
 * Stacked-spin DIIS: minimize ‖Σ c_k [e^α_k; e^β_k]‖² subject to Σ c_k = 1.
 * Returns coefficients or null on singular system. Mirrors uhf-scf's pattern.
 */
function solveDIISCoeffs(
  eA: readonly Float64Array[],
  eB: readonly Float64Array[],
  n: number,
): Float64Array | null {
  const k = eA.length;
  // Build (k+1) × (k+1) Lagrange-multiplier system.
  const dim = k + 1;
  const M = new Float64Array(dim * dim);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let mu = 0; mu < n * n; mu++) {
        s += eA[i]![mu]! * eA[j]![mu]! + eB[i]![mu]! * eB[j]![mu]!;
      }
      M[i * dim + j] = s;
    }
    M[i * dim + k] = -1;
    M[k * dim + i] = -1;
  }
  M[k * dim + k] = 0;
  const rhs = new Float64Array(dim);
  rhs[k] = -1;
  // Gauss-Jordan solve in place.
  for (let p = 0; p < dim; p++) {
    let maxRow = p, maxVal = Math.abs(M[p * dim + p]!);
    for (let r = p + 1; r < dim; r++) {
      const v = Math.abs(M[r * dim + p]!);
      if (v > maxVal) { maxVal = v; maxRow = r; }
    }
    if (maxVal < 1e-14) return null;
    if (maxRow !== p) {
      for (let c = 0; c < dim; c++) {
        const t = M[p * dim + c]!; M[p * dim + c] = M[maxRow * dim + c]!; M[maxRow * dim + c] = t;
      }
      const t = rhs[p]!; rhs[p] = rhs[maxRow]!; rhs[maxRow] = t;
    }
    const piv = M[p * dim + p]!;
    for (let c = 0; c < dim; c++) M[p * dim + c]! /= piv;
    rhs[p]! /= piv;
    for (let r = 0; r < dim; r++) {
      if (r === p) continue;
      const f = M[r * dim + p]!;
      if (f === 0) continue;
      for (let c = 0; c < dim; c++) M[r * dim + c]! -= f * M[p * dim + c]!;
      rhs[r]! -= f * rhs[p]!;
    }
  }
  const coeffs = new Float64Array(k);
  for (let i = 0; i < k; i++) coeffs[i] = rhs[i]!;
  return coeffs;
}
