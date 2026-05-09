// ─────────────────────────────────────────────────────────────
// uhf-scf.ts — Unrestricted Hartree-Fock SCF.
// Tier 2 stage 21.
//
// UHF allows independent spin-up (α) and spin-down (β) orbital
// sets, each filled per the requested electron counts (nα, nβ).
// The α and β Fock matrices share Coulomb but have independent
// exchange:
//
//   J(D_total)[μν]      = Σ_λσ (D_α + D_β)_λσ · (μν|λσ)
//   K_σ[μν]              = Σ_λσ (D_σ)_λσ · (μλ|νσ)
//   F_σ[μν]              = h_AO[μν] + J(D_total)[μν] − K_σ[μν]
//
// Per-spin density (no factor 2 — that's the closed-shell RHF
// convention only):
//   (D_σ)_μν = Σ_i^{n_σ} C_σ_μi · C_σ_νi
//
// UHF energy:
//   E_elec = ½ · Σ_{μν} [(D_α + D_β)_μν · h_μν
//                       + (D_α)_μν · (F_α)_μν
//                       + (D_β)_μν · (F_β)_μν]
//   E_total = E_elec + V_NN
//
// ⟨S²⟩ for the UHF determinant — useful for spin-contamination
// diagnostics:
//   ⟨S²⟩ = ⟨Ŝ²⟩_exact + (n_β − n_doubly-occupied-overlap)
//        = S(S+1) + n_β − Σ_{ij}^{α-occ, β-occ} |⟨α_i|β_j⟩|²
// where S = (n_α − n_β)/2. The deviation from S(S+1) measures
// how much spin contamination (mixing of higher spin states) the
// UHF wavefunction carries.
//
// DIIS: stack α and β error vectors into a single combined error
// vector of length 2·n² so the standard Pulay extrapolation just
// works.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import { eigsymmetric } from "../manybody/dense-eig.js";

export interface UHFOpts {
  readonly maxIter?: number;
  readonly energyTol?: number;
  readonly densityTol?: number;
  readonly useDIIS?: boolean;
  readonly diisHistory?: number;
  readonly damping?: number;
  /** Optional symmetry-breaking strength applied to the initial
   *  α-Fock diagonal to encourage open-shell convergence. Default
   *  0.01 Ha. Set to 0 for closed-shell-like inputs. */
  readonly symmetryBreaking?: number;
}

export interface UHFResult {
  readonly energy: number;
  readonly electronicEnergy: number;
  /** α-orbital MO coefficients (n × n, AO basis, sorted ascending). */
  readonly C_alpha: Float64Array;
  /** β-orbital MO coefficients. */
  readonly C_beta: Float64Array;
  readonly orbitalEnergiesAlpha: Float64Array;
  readonly orbitalEnergiesBeta: Float64Array;
  /** Per-spin AO density matrices. */
  readonly D_alpha: Float64Array;
  readonly D_beta: Float64Array;
  /** Spin density matrix D_α − D_β. */
  readonly spinDensity: Float64Array;
  readonly nAlpha: number;
  readonly nBeta: number;
  /** ⟨S²⟩ expectation for the UHF determinant. Exact value is
   *  S(S+1) where S = (nα − nβ)/2 — deviations measure spin
   *  contamination. */
  readonly s2: number;
  /** Exact S(S+1) for comparison. */
  readonly s2Exact: number;
  readonly iter: number;
  readonly converged: boolean;
  readonly history: readonly number[];
}

/**
 * Run open-shell UHF SCF on the given molecular integrals with
 * `nAlpha` α and `nBeta` β electrons (must sum to total electrons,
 * neutral or ionic). For nα = nβ this collapses to RHF; for
 * nα = nβ + 1 you get a doublet, etc.
 */
export function runUHFSCF(
  integrals: MolecularIntegrals,
  nAlpha: number,
  nBeta: number,
  opts: UHFOpts = {},
): UHFResult {
  if (nAlpha < 0 || nBeta < 0) {
    throw new Error(`runUHFSCF: invalid electron counts (nα=${nAlpha}, nβ=${nBeta})`);
  }
  const n = integrals.n;
  if (nAlpha > n || nBeta > n) {
    throw new Error(`runUHFSCF: too many electrons (nα=${nAlpha}, nβ=${nBeta}, n=${n})`);
  }
  const maxIter = opts.maxIter ?? 200;
  const eTol = opts.energyTol ?? 1e-8;
  const dTol = opts.densityTol ?? 1e-6;
  const useDIIS = opts.useDIIS ?? true;
  const diisMaxHistory = opts.diisHistory ?? 8;
  const damping = opts.damping ?? 0.5;
  const symBreak = opts.symmetryBreaking ?? 0.01;

  const { S_AO, h_AO, eri_AO, X, Vnn } = integrals;

  // ── Initial guess: diagonalize core h. To break α/β symmetry
  // for radicals, optionally tilt the α-Fock diagonal by ±symBreak
  // on the would-be SOMO. Simplest: shift the highest occupied α
  // diagonal slightly down so that, after diagonalization, the
  // α HOMO sits below the β HOMO. ─────────────────────────────
  const hPrime = transformSymmetric(h_AO, X, n);
  const hPrimeAlpha = symBreak > 0 ? perturbDiagonal(hPrime, n, nAlpha - 1, -symBreak)
                                   : hPrime;
  const guessAlpha = solveFock(hPrimeAlpha, X, n);
  const guessBeta  = solveFock(hPrime, X, n);
  let C_alpha = guessAlpha.C_MO;
  let C_beta  = guessBeta.C_MO;
  let epsAlpha = guessAlpha.eps;
  let epsBeta  = guessBeta.eps;
  let D_alpha = densityFromCSpin(C_alpha, nAlpha, n);
  let D_beta  = densityFromCSpin(C_beta,  nBeta,  n);

  const history: number[] = [];
  let E_old = Infinity;
  let converged = false;
  let iter = 0;

  // Combined α+β DIIS subspace — error stacks to length 2·n².
  const diisF: Float64Array[] = [];   // stacked [F_α; F_β], 2·n² each
  const diisE: Float64Array[] = [];

  for (iter = 1; iter <= maxIter; iter++) {
    // Total density for J build.
    const D_total = new Float64Array(n * n);
    for (let k = 0; k < n * n; k++) D_total[k] = D_alpha[k]! + D_beta[k]!;

    // Build the spin-Fock matrices.
    const J_total = buildJ(D_total, eri_AO, n);
    const K_alpha = buildK(D_alpha, eri_AO, n);
    const K_beta  = buildK(D_beta,  eri_AO, n);
    const F_alpha = new Float64Array(n * n);
    const F_beta  = new Float64Array(n * n);
    for (let k = 0; k < n * n; k++) {
      F_alpha[k] = h_AO[k]! + J_total[k]! - K_alpha[k]!;
      F_beta[k]  = h_AO[k]! + J_total[k]! - K_beta[k]!;
    }

    // UHF energy.
    let Eelec = 0;
    for (let k = 0; k < n * n; k++) {
      Eelec += 0.5 * (
          D_total[k]! * h_AO[k]!
        + D_alpha[k]! * F_alpha[k]!
        + D_beta[k]!  * F_beta[k]!
      );
    }
    const E = Eelec + Vnn;
    history.push(E);

    let F_alpha_use: Float64Array = F_alpha;
    let F_beta_use:  Float64Array = F_beta;
    let errMax = 0;

    if (useDIIS) {
      const e_alpha = buildDIISError(F_alpha, D_alpha, S_AO, X, n);
      const e_beta  = buildDIISError(F_beta,  D_beta,  S_AO, X, n);
      const eStacked = new Float64Array(2 * n * n);
      eStacked.set(e_alpha, 0);
      eStacked.set(e_beta,  n * n);
      for (let k = 0; k < eStacked.length; k++) {
        const a = Math.abs(eStacked[k]!);
        if (a > errMax) errMax = a;
      }
      const fStacked = new Float64Array(2 * n * n);
      fStacked.set(F_alpha, 0);
      fStacked.set(F_beta,  n * n);
      diisF.push(fStacked);
      diisE.push(eStacked);
      if (diisF.length > diisMaxHistory) {
        diisF.shift();
        diisE.shift();
      }
      if (diisF.length >= 2) {
        const c = solveDIISCoeffs(diisE);
        if (c !== null) {
          const F_ext = new Float64Array(2 * n * n);
          for (let k = 0; k < diisF.length; k++) {
            const ck = c[k]!;
            const Fk = diisF[k]!;
            for (let m = 0; m < F_ext.length; m++) F_ext[m]! += ck * Fk[m]!;
          }
          F_alpha_use = F_ext.subarray(0, n * n) as Float64Array;
          F_beta_use  = F_ext.subarray(n * n, 2 * n * n) as Float64Array;
        }
      }
    }

    // Diagonalize each spin Fock to get new orbitals.
    const FaPrime = transformSymmetric(F_alpha_use, X, n);
    const FbPrime = transformSymmetric(F_beta_use,  X, n);
    const solA = solveFock(FaPrime, X, n);
    const solB = solveFock(FbPrime, X, n);
    C_alpha = solA.C_MO; epsAlpha = solA.eps;
    C_beta  = solB.C_MO; epsBeta  = solB.eps;

    const D_alpha_new = densityFromCSpin(C_alpha, nAlpha, n);
    const D_beta_new  = densityFromCSpin(C_beta,  nBeta,  n);
    let dNorm = 0;
    for (let k = 0; k < n * n; k++) {
      const d = (D_alpha_new[k]! - D_alpha[k]!) + (D_beta_new[k]! - D_beta[k]!);
      dNorm += d * d;
    }
    dNorm = Math.sqrt(dNorm);

    if (useDIIS) {
      D_alpha = D_alpha_new; D_beta = D_beta_new;
    } else {
      for (let k = 0; k < n * n; k++) {
        D_alpha[k] = damping * D_alpha_new[k]! + (1 - damping) * D_alpha[k]!;
        D_beta[k]  = damping * D_beta_new[k]!  + (1 - damping) * D_beta[k]!;
      }
    }

    const residOk = useDIIS ? errMax < dTol : dNorm < dTol;
    if (Math.abs(E - E_old) < eTol && residOk) {
      converged = true;
      E_old = E;
      break;
    }
    E_old = E;
  }

  // ── ⟨S²⟩ for the converged determinant. ────────────────────
  // Standard formula: ⟨S²⟩ = S(S+1) + n_β − Σ_{ij occupied} |⟨α_i|S|β_j⟩|²
  // where ⟨α_i | S | β_j⟩ = Σ_μν C_α_μi · S_AO_μν · C_β_νj.
  const S = (nAlpha - nBeta) / 2;
  const s2Exact = S * (S + 1);
  let overlapSq = 0;
  // Compute α^T · S_AO · β only on occupied blocks.
  for (let i = 0; i < nAlpha; i++) {
    for (let j = 0; j < nBeta; j++) {
      let s = 0;
      for (let mu = 0; mu < n; mu++) {
        const Ci = C_alpha[mu * n + i]!;
        if (Ci === 0) continue;
        for (let nu = 0; nu < n; nu++) {
          s += Ci * S_AO[mu * n + nu]! * C_beta[nu * n + j]!;
        }
      }
      overlapSq += s * s;
    }
  }
  const s2 = s2Exact + nBeta - overlapSq;

  // Spin density.
  const spinDensity = new Float64Array(n * n);
  for (let k = 0; k < n * n; k++) spinDensity[k] = D_alpha[k]! - D_beta[k]!;

  return {
    energy: E_old,
    electronicEnergy: E_old - Vnn,
    C_alpha, C_beta,
    orbitalEnergiesAlpha: epsAlpha,
    orbitalEnergiesBeta:  epsBeta,
    D_alpha, D_beta, spinDensity,
    nAlpha, nBeta,
    s2, s2Exact,
    iter, converged, history,
  };
}

// ── Helpers ─────────────────────────────────────────────────

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
  return { C_MO, eps: eig.values };
}

/** Spin-resolved density: D_σ[μν] = Σ_i^{n_σ} C[μ,i]·C[ν,i] (NO factor 2). */
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

/** Coulomb J build: J[μν] = Σ_λσ D[λσ]·(μν|λσ). */
function buildJ(D: Float64Array, eri: Float64Array, n: number): Float64Array {
  const J = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          s += D[la * n + si]! * eri[((mu * n + nu) * n + la) * n + si]!;
        }
      }
      J[mu * n + nu] = s;
    }
  }
  return J;
}

/** Exchange K build: K[μν] = Σ_λσ D[λσ]·(μλ|νσ). */
function buildK(D: Float64Array, eri: Float64Array, n: number): Float64Array {
  const K = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          s += D[la * n + si]! * eri[((mu * n + la) * n + nu) * n + si]!;
        }
      }
      K[mu * n + nu] = s;
    }
  }
  return K;
}

/** Pulay DIIS error vector e_σ = X^T (F_σ D_σ S − S D_σ F_σ) X
 *  in the orthogonal basis. Vanishes when the Fock and density of
 *  spin σ commute. */
function buildDIISError(
  F: Float64Array, D: Float64Array, S: Float64Array, X: Float64Array, n: number,
): Float64Array {
  // FDS − SDF.
  const FD = matmul(F, D, n);
  const FDS = matmul(FD, S, n);
  const SD = matmul(S, D, n);
  const SDF = matmul(SD, F, n);
  const comm = new Float64Array(n * n);
  for (let k = 0; k < n * n; k++) comm[k] = FDS[k]! - SDF[k]!;
  return transformSymmetric(comm, X, n);
}

function matmul(A: Float64Array, B: Float64Array, n: number): Float64Array {
  const C = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const aik = A[i * n + k]!;
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) {
        C[i * n + j]! += aik * B[k * n + j]!;
      }
    }
  }
  return C;
}

/** Solve the Pulay DIIS system for the linear-extrapolation
 *  coefficients c_i (i = 1..M) such that Σ c_i = 1 and
 *  Σ c_i c_j ⟨e_i, e_j⟩ is minimized. Returns null if the system
 *  is singular (caller should skip extrapolation that step). */
function solveDIISCoeffs(errors: readonly Float64Array[]): Float64Array | null {
  const M = errors.length;
  // Build (M+1)×(M+1) system: B[i,j] = ⟨e_i, e_j⟩ + Lagrange row/col for Σc=1.
  const B = new Float64Array((M + 1) * (M + 1));
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < M; j++) {
      let s = 0;
      const ei = errors[i]!, ej = errors[j]!;
      for (let k = 0; k < ei.length; k++) s += ei[k]! * ej[k]!;
      B[i * (M + 1) + j] = s;
    }
    B[i * (M + 1) + M] = -1;
    B[M * (M + 1) + i] = -1;
  }
  B[M * (M + 1) + M] = 0;
  const rhs = new Float64Array(M + 1);
  rhs[M] = -1;
  // Gaussian elimination on the (M+1) × (M+1) system.
  const A = new Float64Array(B);
  const b = new Float64Array(rhs);
  const dim = M + 1;
  for (let i = 0; i < dim; i++) {
    let pivot = A[i * dim + i]!;
    if (Math.abs(pivot) < 1e-14) {
      // Try partial pivoting.
      let bestRow = i;
      for (let r = i + 1; r < dim; r++) {
        if (Math.abs(A[r * dim + i]!) > Math.abs(pivot)) {
          pivot = A[r * dim + i]!;
          bestRow = r;
        }
      }
      if (Math.abs(pivot) < 1e-14) return null;
      // Swap rows i and bestRow.
      for (let c = 0; c < dim; c++) {
        const tmp = A[i * dim + c]!;
        A[i * dim + c] = A[bestRow * dim + c]!;
        A[bestRow * dim + c] = tmp;
      }
      const tmp = b[i]!;
      b[i] = b[bestRow]!;
      b[bestRow] = tmp;
    }
    for (let r = i + 1; r < dim; r++) {
      const f = A[r * dim + i]! / A[i * dim + i]!;
      for (let c = i; c < dim; c++) A[r * dim + c]! -= f * A[i * dim + c]!;
      b[r]! -= f * b[i]!;
    }
  }
  // Back-substitution.
  const c = new Float64Array(M);
  const xFull = new Float64Array(dim);
  for (let i = dim - 1; i >= 0; i--) {
    let s = b[i]!;
    for (let j = i + 1; j < dim; j++) s -= A[i * dim + j]! * xFull[j]!;
    xFull[i] = s / A[i * dim + i]!;
  }
  for (let i = 0; i < M; i++) c[i] = xFull[i]!;
  return c;
}

/** Apply a small perturbation to a single diagonal element of M.
 *  Used for symmetry-breaking the α initial guess. Returns a new array. */
function perturbDiagonal(M: Float64Array, n: number, idx: number, delta: number): Float64Array {
  const out = new Float64Array(M);
  if (idx >= 0 && idx < n) out[idx * n + idx]! += delta;
  return out;
}
