// ─────────────────────────────────────────────────────────────
// foster-boys.ts — Foster-Boys orbital localization (Boys 1960).
//
// Transforms a set of canonical MOs into a set of "localized" MOs
// (Boys orbitals) where each orbital is spatially compact and
// chemically interpretable: lone pairs, σ-bonds, π-bonds, core
// orbitals all become individually identifiable.
//
// Objective: maximize the "self-extent" functional
//
//   L(U) = Σ_i  | ⟨ψ_i | r | ψ_i⟩ |²       (centroid-squared sum)
//
// which equivalently MINIMIZES the spread Σ_i ⟨ψ_i | r² | ψ_i⟩
// (Edmiston-Ruedenberg observation: Foster-Boys is the
// minimum-orbital-spread criterion in disguise).
//
// Algorithm: 2×2 Jacobi sweep over occupied-orbital pairs (i, j).
// For each pair, the pairwise change under rotation by θ is
//
//   ΔL_pair = A · (cos 4θ − 1)  +  B · sin 4θ
//   A_ij = ¼ Σ_a (μ_ii^a − μ_jj^a)²  −  Σ_a (μ_ij^a)²
//   B_ij = Σ_a (μ_ii^a − μ_jj^a) · μ_ij^a
//
//   Maximum at 4θ = atan2(B, A) ⇒ θ = ¼·atan2(B, A)
//   (gives the rotation that monotonically increases L).
//
// Iterate sweeps until total ΔL < tolerance. Typically 5–15 sweeps
// for small molecules; the convergence is linear in the number of
// occupied orbitals.
//
// API:
//   `fosterBoys(C_MO, dipoleMO, nOccupied, opts?)` → { U, L_history }
//   where U is the localization rotation matrix (n_occ × n_occ),
//   applied to C_MO[:, :nOcc] to give localized MOs.
//
// Reference: Foster & Boys, Rev. Mod. Phys. 32, 300 (1960).
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { CGShell } from "./integrals-cg.js";
import { dipole_cg } from "./integrals-cg.js";

export interface FosterBoysResult {
  /** Localization rotation matrix U (n_occ × n_occ). Localized MOs:
   *  C_LMO = C_MO[:, :nOcc] · U. */
  readonly U: Float64Array;
  /** Per-sweep value of the Foster-Boys objective L. Monotonically
   *  non-decreasing under Jacobi sweeps. */
  readonly L_history: readonly number[];
  /** Number of sweeps performed. */
  readonly nSweeps: number;
  /** True if ΔL between final two sweeps < tolerance. */
  readonly converged: boolean;
}

export interface FosterBoysOpts {
  /** Convergence tolerance on ΔL per sweep. Default 1e-8. */
  readonly tol?: number;
  /** Max number of sweeps. Default 100. */
  readonly maxSweeps?: number;
}

/**
 * Run Foster-Boys localization on the occupied block of an SCF result.
 *
 * @param C_MO         AO → MO coefficient matrix (n × n, row μ, col p).
 *                     Only the first `nOccupied` columns are localized;
 *                     the rest are left untouched in the returned U
 *                     (identity block).
 * @param integrals    For the dipole AO matrix elements.
 * @param shells       CG shell list for `dipole_cg`.
 * @param nOccupied    Number of doubly-occupied MOs (or alpha-occupied
 *                     for UHF/UKS — caller chooses).
 */
export function fosterBoys(
  C_MO: Float64Array,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
  nOccupied: number,
  opts: FosterBoysOpts = {},
): FosterBoysResult {
  const n = integrals.n;
  const tol = opts.tol ?? 1e-8;
  const maxSweeps = opts.maxSweeps ?? 100;
  if (nOccupied < 1 || nOccupied > n) {
    throw new Error(`fosterBoys: nOccupied (${nOccupied}) out of range [1, ${n}]`);
  }
  // ── Build dipole MO matrix elements μ_ij^a for axes a ∈ {x,y,z}
  //    on the occupied block (nOcc × nOcc per axis). ────────────
  // μ_AO^a = AO-basis dipole; μ_MO[i,j]^a = (C^T μ_AO C)[i,j].
  const muAO_x = new Float64Array(n * n);
  const muAO_y = new Float64Array(n * n);
  const muAO_z = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu; nu < n; nu++) {
      const vx = dipole_cg(shells[mu]!, shells[nu]!, 0);
      const vy = dipole_cg(shells[mu]!, shells[nu]!, 1);
      const vz = dipole_cg(shells[mu]!, shells[nu]!, 2);
      muAO_x[mu * n + nu] = vx; muAO_x[nu * n + mu] = vx;
      muAO_y[mu * n + nu] = vy; muAO_y[nu * n + mu] = vy;
      muAO_z[mu * n + nu] = vz; muAO_z[nu * n + mu] = vz;
    }
  }
  // muMO[axis][i*nOcc + j] = Σ_μν C_μi · μ_AO[μν] · C_νj.
  // Start from C_MO occupied block; we'll update muMO in place as
  // we rotate orbitals during the Jacobi sweep.
  const muMO_x = transformOccBlock(C_MO, muAO_x, n, nOccupied);
  const muMO_y = transformOccBlock(C_MO, muAO_y, n, nOccupied);
  const muMO_z = transformOccBlock(C_MO, muAO_z, n, nOccupied);

  // U = identity (nOcc × nOcc).
  const U = new Float64Array(nOccupied * nOccupied);
  for (let i = 0; i < nOccupied; i++) U[i * nOccupied + i] = 1;

  const L_history: number[] = [];
  let prevL = Number.NEGATIVE_INFINITY;
  let converged = false;
  let sweep = 0;
  for (sweep = 1; sweep <= maxSweeps; sweep++) {
    // One Jacobi sweep: visit every pair (i < j).
    for (let i = 0; i < nOccupied; i++) {
      for (let j = i + 1; j < nOccupied; j++) {
        // ── Compute A_ij and B_ij for this pair. ─────────────
        // A = ¼·Σ_a (μ_ii − μ_jj)²  −  Σ_a μ_ij²
        // B = Σ_a (μ_ii − μ_jj) · μ_ij
        let A_ij = 0;
        let B_ij = 0;
        for (const muMO of [muMO_x, muMO_y, muMO_z]) {
          const mii = muMO[i * nOccupied + i]!;
          const mjj = muMO[j * nOccupied + j]!;
          const mij = muMO[i * nOccupied + j]!;
          const diff = mii - mjj;
          A_ij += 0.25 * diff * diff - mij * mij;
          B_ij += diff * mij;
        }
        const norm = Math.sqrt(A_ij * A_ij + B_ij * B_ij);
        if (norm < tol) continue;        // no benefit from rotation

        // ── Optimal rotation angle. ──────────────────────────
        // ΔL(θ) = A cos(4θ) + B sin(4θ) = √(A²+B²)·cos(4θ − φ)
        // where φ = atan2(B, A). Max at 4θ = φ → θ = ¼·atan2(B,A).
        const theta = 0.25 * Math.atan2(B_ij, A_ij);
        const c = Math.cos(theta);
        const s = Math.sin(theta);
        if (Math.abs(s) < 1e-14) continue;     // numerically zero rotation

        // ── Apply rotation to the three muMO matrices. ───────
        rotatePair(muMO_x, nOccupied, i, j, c, s);
        rotatePair(muMO_y, nOccupied, i, j, c, s);
        rotatePair(muMO_z, nOccupied, i, j, c, s);
        // ── Apply rotation to U (right-multiply by Givens). ──
        for (let k = 0; k < nOccupied; k++) {
          const Uki = U[k * nOccupied + i]!;
          const Ukj = U[k * nOccupied + j]!;
          U[k * nOccupied + i] = c * Uki + s * Ukj;
          U[k * nOccupied + j] = -s * Uki + c * Ukj;
        }
      }
    }
    // ── Compute total L = Σ_i Σ_axis (μ_ii^a)². ────────────────
    let L = 0;
    for (let i = 0; i < nOccupied; i++) {
      const xi = muMO_x[i * nOccupied + i]!;
      const yi = muMO_y[i * nOccupied + i]!;
      const zi = muMO_z[i * nOccupied + i]!;
      L += xi * xi + yi * yi + zi * zi;
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

// ─────────────────────────────────────────────────────────────

function transformOccBlock(
  C_MO: Float64Array,
  muAO: Float64Array,
  n: number,
  nOccupied: number,
): Float64Array {
  // muMO[i, j] = Σ_μν C[μ, i] · muAO[μ, ν] · C[ν, j],  i,j ∈ [0, nOccupied).
  const out = new Float64Array(nOccupied * nOccupied);
  const tmp = new Float64Array(n * nOccupied);
  // tmp[ν, j] = Σ_λ muAO[ν, λ] · C[λ, j]
  for (let nu = 0; nu < n; nu++) {
    for (let j = 0; j < nOccupied; j++) {
      let s = 0;
      for (let la = 0; la < n; la++) s += muAO[nu * n + la]! * C_MO[la * n + j]!;
      tmp[nu * nOccupied + j] = s;
    }
  }
  for (let i = 0; i < nOccupied; i++) {
    for (let j = 0; j < nOccupied; j++) {
      let s = 0;
      for (let nu = 0; nu < n; nu++) s += C_MO[nu * n + i]! * tmp[nu * nOccupied + j]!;
      out[i * nOccupied + j] = s;
    }
  }
  return out;
}

/** Apply the symmetric 2×2 rotation to a matrix in place:
 *    M' = R^T · M · R   where R is the Givens rotation in the (i, j) plane.
 *  This updates both rows i, j and columns i, j of M. */
function rotatePair(M: Float64Array, n: number, i: number, j: number, c: number, s: number): void {
  // Columns first: M[:, i] and M[:, j].
  for (let k = 0; k < n; k++) {
    const Mki = M[k * n + i]!;
    const Mkj = M[k * n + j]!;
    M[k * n + i] = c * Mki + s * Mkj;
    M[k * n + j] = -s * Mki + c * Mkj;
  }
  // Then rows: M[i, :] and M[j, :].
  for (let k = 0; k < n; k++) {
    const Mik = M[i * n + k]!;
    const Mjk = M[j * n + k]!;
    M[i * n + k] = c * Mik + s * Mjk;
    M[j * n + k] = -s * Mik + c * Mjk;
  }
}
