// ─────────────────────────────────────────────────────────────
// hf-scf.ts — closed-shell Hartree-Fock self-consistent field
// (RHF). Phase D: the bridge from minimal-basis FCI to bigger
// basis sets and post-HF methods (MP2, CCSD coming in Phase E).
//
// What HF gives you:
//   • E_HF: the lowest-energy single Slater determinant of the
//     molecule. Includes electron-electron Coulomb + exchange
//     mean-field, but NO correlation beyond Pauli exclusion.
//   • C: MO coefficient matrix (AO → MO transform). The columns
//     are the molecular orbitals, ordered by energy.
//   • ε: orbital energies.
//   • D: AO-basis density matrix (Σ_occupied 2 C_i C_i^T).
//
// Why it matters:
//   • HF is the WORK-HORSE single-reference method for chemistry.
//     For most molecules near equilibrium, HF + a small post-HF
//     correction (MP2 or CCSD) reaches chemical accuracy at
//     polynomial cost — vs FCI's exponential cost.
//   • cc-pVDZ on water has 24 spatial orbitals → FCI is infeasible
//     (2^48 Hilbert space) but HF takes a few seconds.
//   • Active-space FCI: do FCI within a 6-orbital active space
//     around HF, freeze the rest as core. Standard approach for
//     drug-sized molecules.
//
// Algorithm (Roothaan-Hall iteration):
//   1. Compute S, h, ERI in AO basis (provided).
//   2. X = S^{-1/2} (orthogonalizing transform).
//   3. Initial guess: diagonalize core h → initial C, D.
//   4. Loop:
//      a. G[μν] = Σ_{λσ} D[λσ] · ((μν|λσ) - ½ (μλ|νσ))
//      b. F = h + G  (Fock matrix)
//      c. F' = X^T F X
//      d. ε, C' = eigh(F')
//      e. C = X C'
//      f. D_new[μν] = 2 Σ_i^occ C[μi] C[νi]
//      g. Damping: D ← α D_new + (1-α) D_old  (α = 0.5 default)
//      h. E_HF = ½ Σ D · (h + F) + Vnn
//      i. Check |E_new − E_old| < tol AND |D_new − D_old|_F < tol
//
// Without damping RHF can oscillate or diverge for some basis
// sets. Damping at α = 0.5 is robust for our minimal-basis
// molecules; DIIS would converge faster but is heavier and not
// needed at this scale.
// ─────────────────────────────────────────────────────────────

import { type MolecularIntegrals } from "./cg-molecular.js";
import { eigsymmetric } from "../manybody/dense-eig.js";

export interface HFResult {
  /** Total HF energy (Hartree) including nuclear repulsion. */
  readonly energy: number;
  /** Electronic energy = energy − Vnn. */
  readonly electronicEnergy: number;
  /** AO → MO coefficient matrix, n × n row-major. Column i = MO i. */
  readonly C_MO: Float64Array;
  /** Orbital energies, length n, ascending. */
  readonly orbitalEnergies: Float64Array;
  /** AO-basis density matrix, n × n row-major. */
  readonly D: Float64Array;
  /** Number of doubly occupied orbitals = nElectrons / 2. */
  readonly nOccupied: number;
  /** Number of SCF iterations performed. */
  readonly iter: number;
  /** True if both energy and density tolerances were met. */
  readonly converged: boolean;
  /** Per-iteration energy trace (for debugging / plotting). */
  readonly history: readonly number[];
}

export interface HFOpts {
  /** Hard cap on SCF iterations. Default 100. */
  readonly maxIter?: number;
  /** Energy tolerance for convergence. Default 1e-8 Ha. */
  readonly energyTol?: number;
  /** Density-matrix Frobenius-norm tolerance. Default 1e-6. */
  readonly densityTol?: number;
  /**
   * Damping factor α ∈ (0, 1]. D_new ← α · D_new + (1 − α) · D_old.
   * α = 1 means no damping; α = 0.5 is the default robust value.
   */
  readonly damping?: number;
}

/**
 * Run closed-shell Hartree-Fock SCF on the given molecular
 * integrals with `nElectrons` electrons (must be even).
 */
export function runRHFSCF(
  integrals: MolecularIntegrals,
  nElectrons: number,
  opts: HFOpts = {},
): HFResult {
  if (nElectrons % 2 !== 0) {
    throw new Error(`runRHFSCF: open-shell systems require UHF (got nElectrons=${nElectrons}, odd)`);
  }
  const nOcc = nElectrons / 2;
  const n = integrals.n;
  if (nOcc > n) {
    throw new Error(`runRHFSCF: ${nElectrons} electrons in ${n} spatial orbitals — too many`);
  }
  const maxIter = opts.maxIter ?? 100;
  const eTol = opts.energyTol ?? 1e-8;
  const dTol = opts.densityTol ?? 1e-6;
  const damping = opts.damping ?? 0.5;

  const { S_AO, h_AO, eri_AO, X, Vnn } = integrals;
  void S_AO;  // referenced via X = S^{-1/2}

  // ── Initial guess: diagonalize core h to get starting C ──
  // Build h' = X^T h X, diagonalize, back-transform.
  const hPrime = transformSymmetric(h_AO, X, n);
  let { C_MO, eps } = solveFock(hPrime, X, n);
  const D = densityFromC(C_MO, nOcc, n);
  let E_old = Infinity;
  const history: number[] = [];
  let converged = false;
  let iter = 0;

  for (iter = 1; iter <= maxIter; iter++) {
    // ── Build Fock F = h + G(D) ─────────────────────────────
    const G = buildG(D, eri_AO, n);
    const F = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) F[i] = h_AO[i]! + G[i]!;

    // ── HF energy: E = ½ Σ D · (h + F) + Vnn ───────────────
    let Eelec = 0;
    for (let i = 0; i < n * n; i++) Eelec += 0.5 * D[i]! * (h_AO[i]! + F[i]!);
    const E = Eelec + Vnn;
    history.push(E);

    // ── F' = X^T F X, diagonalize → new C, eps ─────────────
    const FPrime = transformSymmetric(F, X, n);
    const sol = solveFock(FPrime, X, n);
    C_MO = sol.C_MO;
    eps = sol.eps;

    // ── New density with damping ───────────────────────────
    const D_new = densityFromC(C_MO, nOcc, n);
    let dNorm = 0;
    for (let i = 0; i < n * n; i++) {
      const d = D_new[i]! - D[i]!;
      dNorm += d * d;
    }
    dNorm = Math.sqrt(dNorm);
    for (let i = 0; i < n * n; i++) {
      D[i] = damping * D_new[i]! + (1 - damping) * D[i]!;
    }

    if (Math.abs(E - E_old) < eTol && dNorm < dTol) {
      converged = true;
      E_old = E;
      break;
    }
    E_old = E;
  }

  return {
    energy: E_old,
    electronicEnergy: E_old - Vnn,
    C_MO,
    orbitalEnergies: eps,
    D,
    nOccupied: nOcc,
    iter,
    converged,
    history,
  };
}

// ── Helpers ─────────────────────────────────────────────────

/** M' = X^T M X for a symmetric M, n × n. Both X and M row-major. */
function transformSymmetric(M: Float64Array, X: Float64Array, n: number): Float64Array {
  // Compute (X^T M) first, then multiply by X.
  const tmp = new Float64Array(n * n);
  // tmp[i, j] = Σ_k X[k, i] M[k, j]   (X^T M)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += X[k * n + i]! * M[k * n + j]!;
      tmp[i * n + j] = s;
    }
  }
  const out = new Float64Array(n * n);
  // out[i, j] = Σ_k tmp[i, k] X[k, j]
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += tmp[i * n + k]! * X[k * n + j]!;
      out[i * n + j] = s;
    }
  }
  return out;
}

/**
 * Solve F' C' = ε C' (eigh in orthogonal basis), then back-
 * transform C = X C'. Returns the AO-basis MO coefficients
 * sorted by energy ascending.
 *
 * dense-eig.ts returns column-major eigenvectors (vectors[col*N + row]),
 * so we read them out into row-major here.
 */
function solveFock(FPrime: Float64Array, X: Float64Array, n: number): {
  C_MO: Float64Array;
  eps: Float64Array;
} {
  const eig = eigsymmetric(FPrime, n);
  // C' is column-major in eig.vectors. Build C = X C' as row-major.
  const C_MO = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        // X[r, k] · C'[k, c] = X[r, k] · eig.vectors[c * n + k]
        s += X[r * n + k]! * eig.vectors[c * n + k]!;
      }
      C_MO[r * n + c] = s;
    }
  }
  return { C_MO, eps: eig.values };
}

/** D[μ, ν] = 2 Σ_{i=0}^{nOcc-1} C[μ, i] C[ν, i]. */
function densityFromC(C: Float64Array, nOcc: number, n: number): Float64Array {
  const D = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let i = 0; i < nOcc; i++) s += C[mu * n + i]! * C[nu * n + i]!;
      D[mu * n + nu] = 2 * s;
    }
  }
  return D;
}

/**
 * Build the two-electron G matrix:
 *   G[μ, ν] = Σ_{λ, σ} D[λ, σ] · ( (μν|λσ) − ½ (μλ|νσ) )
 *
 * Uses the chemist's notation eri_AO[((μ·n+ν)·n+λ)·n+σ] = (μν|λσ).
 * O(n⁴) per call — fine for n ≤ ~30.
 */
function buildG(D: Float64Array, eri_AO: Float64Array, n: number): Float64Array {
  const G = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          const Dls = D[la * n + si]!;
          if (Dls === 0) continue;
          const J = eri_AO[((mu * n + nu) * n + la) * n + si]!;
          const K = eri_AO[((mu * n + la) * n + nu) * n + si]!;
          s += Dls * (J - 0.5 * K);
        }
      }
      G[mu * n + nu] = s;
    }
  }
  return G;
}
