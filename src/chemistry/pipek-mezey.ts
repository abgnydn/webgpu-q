// ─────────────────────────────────────────────────────────────
// pipek-mezey.ts — Pipek-Mezey orbital localization (1989).
//
// Alternative to Foster-Boys (`foster-boys.ts`). Both produce
// "chemical" orbitals (lone pairs, bonds, cores) but PM uses
// Mulliken atomic populations as the localization metric rather
// than Boys' dipole-centroid criterion:
//
//   L_PM(U) = Σ_i Σ_A (Q^i_A)²
//
// where Q^i_A = Mulliken population of orbital i on atom A.
//
// Compared to Foster-Boys:
//   • PM separates σ and π bonds (Boys often mixes them into
//     "banana bond" hybrids).
//   • PM is invariant to rotation/translation of the molecule
//     (Mulliken populations don't depend on the origin choice).
//   • Boys gives smaller orbital tails; PM gives clearer
//     atom-localized character.
//
// Algorithm: 2×2 Jacobi sweep — identical structure to Foster-
// Boys. For pair (i, j) at iteration step:
//
//   A_ij = ¼·Σ_A (P^A_ii − P^A_jj)²  −  Σ_A (P^A_ij)²
//   B_ij = Σ_A (P^A_ii − P^A_jj) · P^A_ij
//   θ_opt = ¼·atan2(B, A)
//
// where P^A_ij is the symmetric atom-A Mulliken matrix element
// between MOs i and j:
//
//   P^A_ij = ½ Σ_{μ∈A} [C_μi · (S·C)_μj + C_μj · (S·C)_μi]
//
// References:
//   Pipek & Mezey, J. Chem. Phys. 90, 4916 (1989).
//   Lehtola & Jónsson, J. Chem. Theory Comput. 9, 5365 (2013)
//   for the modern proof that PM converges + comparison vs Boys.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";

export interface PipekMezeyResult {
  /** Localization rotation matrix U (n_occ × n_occ). C_LMO = C_MO[:, :nOcc] · U. */
  readonly U: Float64Array;
  /** Per-sweep value of the PM objective L. Monotonically non-decreasing. */
  readonly L_history: readonly number[];
  readonly nSweeps: number;
  readonly converged: boolean;
}

export interface PipekMezeyOpts {
  readonly tol?: number;          // default 1e-8
  readonly maxSweeps?: number;    // default 100
}

export function pipekMezey(
  C_MO: Float64Array,
  integrals: MolecularIntegrals,
  shellAtomIdx: readonly number[],
  nOccupied: number,
  opts: PipekMezeyOpts = {},
): PipekMezeyResult {
  const n = integrals.n;
  const tol = opts.tol ?? 1e-8;
  const maxSweeps = opts.maxSweeps ?? 100;
  if (nOccupied < 1 || nOccupied > n) {
    throw new Error(`pipekMezey: nOccupied (${nOccupied}) out of range [1, ${n}]`);
  }
  if (shellAtomIdx.length !== n) {
    throw new Error(
      `pipekMezey: shellAtomIdx length ${shellAtomIdx.length} ≠ n ${n}`,
    );
  }
  const nAtoms = integrals.nuclei.length;
  const S = integrals.S_AO;

  // ── Precompute SC = S · C (only need the occupied block columns). ──
  // SC[μ, p] = Σ_ν S[μ, ν] · C_MO[ν, p],  p ∈ [0, nOccupied).
  const SC = new Float64Array(n * nOccupied);
  for (let mu = 0; mu < n; mu++) {
    for (let p = 0; p < nOccupied; p++) {
      let s = 0;
      for (let nu = 0; nu < n; nu++) s += S[mu * n + nu]! * C_MO[nu * n + p]!;
      SC[mu * nOccupied + p] = s;
    }
  }

  // ── Build P^A[i, j] tensor: nAtoms × nOcc × nOcc. ─────────
  // P^A_ij = ½ Σ_{μ∈A} [C_μi · (SC)_μj + C_μj · (SC)_μi].
  const P = new Float64Array(nAtoms * nOccupied * nOccupied);
  for (let mu = 0; mu < n; mu++) {
    const A = shellAtomIdx[mu]!;
    const APaa = A * nOccupied * nOccupied;
    for (let i = 0; i < nOccupied; i++) {
      const C_mu_i = C_MO[mu * n + i]!;
      const SC_mu_i = SC[mu * nOccupied + i]!;
      for (let j = 0; j < nOccupied; j++) {
        const C_mu_j = C_MO[mu * n + j]!;
        const SC_mu_j = SC[mu * nOccupied + j]!;
        P[APaa + i * nOccupied + j]! += 0.5 * (C_mu_i * SC_mu_j + C_mu_j * SC_mu_i);
      }
    }
  }

  // U = identity (nOcc × nOcc).
  const U = new Float64Array(nOccupied * nOccupied);
  for (let i = 0; i < nOccupied; i++) U[i * nOccupied + i] = 1;

  const L_history: number[] = [];
  let prevL = Number.NEGATIVE_INFINITY;
  let converged = false;
  let sweep = 0;
  for (sweep = 1; sweep <= maxSweeps; sweep++) {
    for (let i = 0; i < nOccupied; i++) {
      for (let j = i + 1; j < nOccupied; j++) {
        // ── Compute A_ij, B_ij. ──────────────────────────────
        let A_ij = 0;
        let B_ij = 0;
        for (let At = 0; At < nAtoms; At++) {
          const off = At * nOccupied * nOccupied;
          const Pii = P[off + i * nOccupied + i]!;
          const Pjj = P[off + j * nOccupied + j]!;
          const Pij = P[off + i * nOccupied + j]!;
          const diff = Pii - Pjj;
          A_ij += 0.25 * diff * diff - Pij * Pij;
          B_ij += diff * Pij;
        }
        const norm = Math.sqrt(A_ij * A_ij + B_ij * B_ij);
        if (norm < tol) continue;

        const theta = 0.25 * Math.atan2(B_ij, A_ij);
        const c = Math.cos(theta);
        const s = Math.sin(theta);
        if (Math.abs(s) < 1e-14) continue;

        // ── Update P^A for all atoms: rotate rows and columns i, j. ──
        for (let At = 0; At < nAtoms; At++) {
          const off = At * nOccupied * nOccupied;
          // Columns first.
          for (let k = 0; k < nOccupied; k++) {
            const Pki = P[off + k * nOccupied + i]!;
            const Pkj = P[off + k * nOccupied + j]!;
            P[off + k * nOccupied + i] =  c * Pki + s * Pkj;
            P[off + k * nOccupied + j] = -s * Pki + c * Pkj;
          }
          // Rows.
          for (let k = 0; k < nOccupied; k++) {
            const Pik = P[off + i * nOccupied + k]!;
            const Pjk = P[off + j * nOccupied + k]!;
            P[off + i * nOccupied + k] =  c * Pik + s * Pjk;
            P[off + j * nOccupied + k] = -s * Pik + c * Pjk;
          }
        }
        // ── Update U. ─────────────────────────────────────────
        for (let k = 0; k < nOccupied; k++) {
          const Uki = U[k * nOccupied + i]!;
          const Ukj = U[k * nOccupied + j]!;
          U[k * nOccupied + i] =  c * Uki + s * Ukj;
          U[k * nOccupied + j] = -s * Uki + c * Ukj;
        }
      }
    }
    // ── Compute total L = Σ_i Σ_A (P^A_ii)². ───────────────
    let L = 0;
    for (let i = 0; i < nOccupied; i++) {
      for (let At = 0; At < nAtoms; At++) {
        const Pii = P[At * nOccupied * nOccupied + i * nOccupied + i]!;
        L += Pii * Pii;
      }
    }
    L_history.push(L);
    if (Math.abs(L - prevL) < tol) {
      converged = true;
      break;
    }
    prevL = L;
  }

  return { U, L_history, nSweeps: sweep, converged };
}
