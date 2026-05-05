// ─────────────────────────────────────────────────────────────
// sector.ts — particle-number sector projection for second-
// quantized molecular Hamiltonians.
//
// The full second-quantized H built by the *-builder.ts modules
// is block-diagonal in N̂ (since [H, N̂] = 0): each particle-
// number sector is an independent eigenproblem. The "physical"
// ground state for a neutral molecule lives in N = N_electrons,
// which is just one block — usually orders of magnitude smaller
// than the full 2^(2·n_orbitals) space, since only C(2·n, N_e)
// occupation strings have the right particle count.
//
// Splitting this out (was inline in lih-builder.ts) so beh2-
// builder, h2-builder, and any future molecular builder share
// one implementation.
// ─────────────────────────────────────────────────────────────

import { eigsymmetric } from "../manybody/dense-eig.js";

/**
 * Lowest eigenstate restricted to a single particle-number sector.
 *
 * Returns { energy, psi } with `psi` expanded back to the full
 * 2^nQubits-dim statevector (zero amplitude outside the sector) so
 * downstream code (VQE oracles, observable expectations) can use
 * the same shape as for a non-projected eigenproblem.
 */
export function lowestInParticleSector(
  H: Float64Array,
  nQubits: number,
  particleNumber: number,
): { energy: number; psi: Float64Array } {
  const dim = 1 << nQubits;
  // Enumerate basis states with the requested popcount.
  const sectorIdx: number[] = [];
  for (let i = 0; i < dim; i++) {
    let n = 0;
    for (let q = 0; q < nQubits; q++) if ((i >>> q) & 1) n++;
    if (n === particleNumber) sectorIdx.push(i);
  }
  const k = sectorIdx.length;
  if (k === 0) {
    throw new Error(
      `lowestInParticleSector: no basis states with N=${particleNumber} in ${nQubits}-qubit space`,
    );
  }
  // Project H to the sector: Hsec[a, b] = H[sectorIdx[a], sectorIdx[b]].
  const Hsec = new Float64Array(k * k);
  for (let a = 0; a < k; a++) {
    const ia = sectorIdx[a]!;
    for (let b = 0; b < k; b++) {
      Hsec[a * k + b] = H[ia * dim + sectorIdx[b]!]!;
    }
  }
  const eig = eigsymmetric(Hsec, k);
  // dense-eig returns column-major eigenvectors: vectors[col*N + row].
  // Column 0 (smallest-eigenvalue eigenvector) lives at vectors[0..k-1].
  const psi = new Float64Array(dim);
  for (let a = 0; a < k; a++) {
    psi[sectorIdx[a]!] = eig.vectors[a]!;
  }
  return { energy: eig.values[0]!, psi };
}
