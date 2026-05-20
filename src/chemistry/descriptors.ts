// ─────────────────────────────────────────────────────────────
// descriptors.ts — molecular fingerprint / descriptor utilities
// for machine-learning applications and similarity comparisons.
//
// Implements the Coulomb matrix (Rupp et al., PRL 108, 058301
// (2012)) — the canonical ML featurization of molecules:
//
//   M_ij = 0.5 · Z_i^2.4         (i = j; "self-interaction" diagonal)
//        = Z_i · Z_j / R_ij        (i ≠ j; pairwise Coulomb)
//
// To make the descriptor permutation-invariant (a chemistry-
// agnostic ML feature shouldn't depend on atom labeling), the
// matrix is sorted by row L2 norm in descending order. This
// "sorted Coulomb matrix" is the standard feature for atomization-
// energy regression and related tasks.
//
// Output: flat upper-triangular Float64Array (length N(N+1)/2)
// — the compact ML-feature form — plus the full N×N matrix for
// inspection.
//
// Cost: O(n_atoms²). Microseconds for typical molecules.
// ─────────────────────────────────────────────────────────────

import type { Atom } from "./atoms.js";
import { Z_FOR } from "./atoms.js";

const ANGSTROM_TO_BOHR = 1.8897261245650618;

export interface CoulombMatrixResult {
  /** Full Coulomb matrix (N × N row-major). */
  readonly matrix: Float64Array;
  /** Permutation-invariant upper-triangular vector (length N(N+1)/2),
   *  derived from the row-norm-sorted matrix. The canonical ML feature. */
  readonly sortedUpperTriangle: Float64Array;
  /** Number of atoms (matrix size). */
  readonly nAtoms: number;
}

export interface CoulombMatrixOpts {
  /** Use atomic units (Bohr) instead of Å for distances. Default true. */
  readonly atomicUnits?: boolean;
}

/**
 * Build the Coulomb matrix for a molecule. Returns both the raw
 * matrix and the row-norm-sorted upper-triangular flattening
 * (the ML-ready permutation-invariant feature vector).
 */
export function coulombMatrix(
  atoms: readonly Atom[],
  opts: CoulombMatrixOpts = {},
): CoulombMatrixResult {
  const atomicUnits = opts.atomicUnits ?? true;
  const n = atoms.length;
  const M = new Float64Array(n * n);
  const scale = atomicUnits ? ANGSTROM_TO_BOHR : 1;

  for (let i = 0; i < n; i++) {
    const Zi = atoms[i]!.ghost ? 0 : Z_FOR[atoms[i]!.symbol];
    M[i * n + i] = 0.5 * Math.pow(Zi, 2.4);
    for (let j = i + 1; j < n; j++) {
      const Zj = atoms[j]!.ghost ? 0 : Z_FOR[atoms[j]!.symbol];
      const dx = (atoms[i]!.pos[0] - atoms[j]!.pos[0]) * scale;
      const dy = (atoms[i]!.pos[1] - atoms[j]!.pos[1]) * scale;
      const dz = (atoms[i]!.pos[2] - atoms[j]!.pos[2]) * scale;
      const R = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const value = R > 1e-8 ? Zi * Zj / R : 0;
      M[i * n + j] = value;
      M[j * n + i] = value;
    }
  }

  // Row L2 norms.
  const rowNorms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += M[i * n + j]! * M[i * n + j]!;
    rowNorms[i] = Math.sqrt(s);
  }
  // Sort indices by row norm descending.
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => rowNorms[b]! - rowNorms[a]!);

  // Build sorted upper-triangle.
  const sortedUpperTriangle = new Float64Array((n * (n + 1)) / 2);
  let k = 0;
  for (let ii = 0; ii < n; ii++) {
    const i = order[ii]!;
    for (let jj = ii; jj < n; jj++) {
      const j = order[jj]!;
      sortedUpperTriangle[k++] = M[i * n + j]!;
    }
  }
  return { matrix: M, sortedUpperTriangle, nAtoms: n };
}

/**
 * Euclidean distance between two Coulomb-matrix descriptors. Useful
 * for similarity / k-NN searches in molecular ML applications.
 * Pads the shorter descriptor with zeros to handle different molecule
 * sizes.
 */
export function descriptorDistance(a: Float64Array, b: Float64Array): number {
  const n = Math.max(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const ai = i < a.length ? a[i]! : 0;
    const bi = i < b.length ? b[i]! : 0;
    s += (ai - bi) * (ai - bi);
  }
  return Math.sqrt(s);
}
