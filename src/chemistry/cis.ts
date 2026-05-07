// ─────────────────────────────────────────────────────────────
// cis.ts — Configuration Interaction Singles (CIS) on top of an
// RHF reference. Tier 2 stage 8: first cut at excited states.
//
// Closed-shell singlet/triplet excitation energies via direct
// dense diagonalization of the CIS Hamiltonian on the singles
// excitation manifold (occupied → virtual):
//
//   A_{ia, jb}^singlet = (ε_a − ε_i) δ_ij δ_ab + 2·(ia|jb) − (ij|ab)
//   A_{ia, jb}^triplet = (ε_a − ε_i) δ_ij δ_ab            − (ij|ab)
//
// (Bartlett-Stanton 1989 / Foresman et al. 1992. Chemist
// notation: (pq|rs) = ∫∫ φ_p(1)φ_q(1) r_12⁻¹ φ_r(2)φ_s(2).)
//
// Excitation energies are the eigenvalues of A; the corresponding
// eigenvectors give the linear-response amplitudes:
//   |Ψ_excited⟩ = Σ_{ia} c_{ia} a^†_a a_i |Ψ_HF⟩.
//
// Limitations / scope notes:
//   • Dense eigsymm — fine for n_occ · n_virt up to a few hundred
//     dimensions (STO-3G H₂O: 5·2=10; cc-pVDZ H₂O: 5·19=95). For
//     large systems we'd want a Davidson iterative solver.
//   • Single-reference RHF only. UHF / open-shell CIS would need
//     a separate alpha/beta block decomposition.
//   • TDA/RPA distinction: this is the Tamm-Dancoff Approximation
//     (only A is diagonalized). Full RPA / TDDFT diagonalizes the
//     2×2 (A, B) eigenvalue problem instead — modest follow-up.
//   • TDA-DFT (using KS orbitals + hybrid functional mix) is a
//     trivial extension once we plumb the XC kernel f_xc; deferred.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import { transformERIToMO } from "./mp2.js";
import { eigsymmetric } from "../manybody/dense-eig.js";

export interface HFLike {
  /** AO → MO coefficients (row-major, column i = MO i). */
  readonly C_MO: Float64Array;
  /** Orbital energies (ascending). */
  readonly orbitalEnergies: Float64Array;
  /** Number of doubly-occupied spatial MOs. */
  readonly nOccupied: number;
  /** AO-basis density matrix. Required by TDA-DFT (XC kernel
   *  evaluation needs ρ on the grid). CIS itself doesn't use it. */
  readonly D: Float64Array;
}

export interface CISOpts {
  /** Number of lowest excitation roots to keep. Default: all. */
  readonly nRoots?: number;
  /** Spin sector (default both). Singlet only is the common case. */
  readonly spin?: "singlet" | "triplet" | "both";
}

export interface CISStateBlock {
  /** Excitation energies (Hartree, ascending). */
  readonly energies: Float64Array;
  /** Amplitudes c_{ia} for each root, row-major as
   *  amps[root · (nOcc · nVirt) + i · nVirt + a_idx], where
   *  a_idx is the virtual-orbital index in [0, nVirt). */
  readonly amplitudes: Float64Array;
}

export interface CISResult {
  /** Number of occupied / virtual spatial orbitals (closed-shell RHF). */
  readonly nOccupied: number;
  readonly nVirtual: number;
  /** Singlet sector. Empty energies array if spin = "triplet". */
  readonly singlet: CISStateBlock;
  /** Triplet sector. Empty energies array if spin = "singlet". */
  readonly triplet: CISStateBlock;
}

/**
 * Run CIS on an RHF reference. Builds the MO-basis ERI tensor
 * once (O(n⁵)), then assembles and diagonalizes the singlet and
 * triplet A matrices in the (occ → virt) singles manifold.
 *
 * Cost: AO → MO transform is O(n⁵), CIS matrix is
 * O(n_occ² · n_virt²) to build and (n_occ · n_virt)³ to diagonalize.
 * For STO-3G H₂O this is sub-millisecond.
 */
export function runCIS(
  integrals: MolecularIntegrals,
  hf: HFLike,
  opts: CISOpts = {},
): CISResult {
  const n = integrals.n;
  const nOcc = hf.nOccupied;
  const nVirt = n - nOcc;
  if (nVirt <= 0) {
    throw new Error(`runCIS: no virtual orbitals (n=${n}, nOcc=${nOcc})`);
  }
  const dim = nOcc * nVirt;
  const nRoots = Math.min(opts.nRoots ?? dim, dim);
  const spin = opts.spin ?? "both";

  // ── MO ERI tensor (pq|rs). ──────────────────────────────────
  const eri_MO = transformERIToMO(integrals.eri_AO, hf.C_MO, n);

  // Convenience index helper for chemist-notation ERI access.
  // (pq|rs) is at ((p·n + q)·n + r)·n + s in row-major n^4.
  const eri = (p: number, q: number, r: number, s: number): number =>
    eri_MO[((p * n + q) * n + r) * n + s]!;

  // Block builder: A_ia,jb = (ε_a − ε_i) δ_ij δ_ab + sCoul·(ia|jb) − (ij|ab)
  // where sCoul = 2 for singlet, 0 for triplet.
  const buildA = (sCoul: number): Float64Array => {
    const A = new Float64Array(dim * dim);
    const eps = hf.orbitalEnergies;
    for (let i = 0; i < nOcc; i++) {
      for (let a = 0; a < nVirt; a++) {
        const ai = nOcc + a;                 // global MO index of virtual a
        const row = (i * nVirt + a) * dim;
        const eOrb = eps[ai]! - eps[i]!;
        for (let j = 0; j < nOcc; j++) {
          for (let b = 0; b < nVirt; b++) {
            const bj = nOcc + b;
            const col = j * nVirt + b;
            const diag = (i === j && a === b) ? eOrb : 0;
            // (ia|jb) = chemist notation, MO indices (i, ai, j, bj)
            // (ij|ab) = chemist notation, MO indices (i, j, ai, bj)
            const coul = sCoul !== 0 ? sCoul * eri(i, ai, j, bj) : 0;
            const exch = eri(i, j, ai, bj);
            A[row + col] = diag + coul - exch;
          }
        }
      }
    }
    return A;
  };

  const diagAndExtract = (A: Float64Array): CISStateBlock => {
    const eig = eigsymmetric(A, dim);
    const energies = new Float64Array(nRoots);
    const amplitudes = new Float64Array(nRoots * dim);
    // eig.vectors is column-major: vectors[col·dim + row] is the
    // row-th component of the col-th eigenvector. Eigenvalues are
    // ascending, so columns 0..nRoots-1 give the lowest excitations.
    for (let r = 0; r < nRoots; r++) {
      energies[r] = eig.values[r]!;
      for (let k = 0; k < dim; k++) {
        amplitudes[r * dim + k] = eig.vectors[r * dim + k]!;
      }
    }
    return { energies, amplitudes };
  };

  const empty: CISStateBlock = { energies: new Float64Array(0), amplitudes: new Float64Array(0) };
  const singlet = (spin === "singlet" || spin === "both") ? diagAndExtract(buildA(2)) : empty;
  const triplet = (spin === "triplet" || spin === "both") ? diagAndExtract(buildA(0)) : empty;

  return { nOccupied: nOcc, nVirtual: nVirt, singlet, triplet };
}
