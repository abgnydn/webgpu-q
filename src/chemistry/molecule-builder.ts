// ─────────────────────────────────────────────────────────────
// molecule-builder.ts — one-shot molecule → sector-projected H.
//
// Usage:
//   const data = buildMoleculeFCI([
//     { symbol: "O", pos: [0, 0, 0] },
//     { symbol: "H", pos: [0.7596, 0, 0.5870] },
//     { symbol: "H", pos: [-0.7596, 0, 0.5870] },
//   ]);
//   // data.sector.H is the dense 1001×1001 N=10 sector matrix
//   // data.fci(...) returns the ground-state energy via Lanczos.
//
// All Phase C v3+ molecules go through this — h2o-builder.ts and
// ch4-builder.ts are thin wrappers that supply the geometry.
// ─────────────────────────────────────────────────────────────

import { type Atom, moleculeToShellsNuclei } from "./atoms.js";
import { computeMolecularIntegrals, type MolecularIntegrals } from "./cg-molecular.js";
import { buildSectorH, type SectorH } from "./sector-builder.js";
import { buildSparseSectorH, sparseMatvec, type SparseSectorH } from "./sector-matvec.js";
import { lanczosGroundState, LANCZOS_TOL } from "../manybody/lanczos.js";

export interface MoleculeData {
  readonly atoms: readonly Atom[];
  readonly nElectrons: number;
  readonly nSpatial: number;
  readonly nQubits: number;
  readonly integrals: MolecularIntegrals;
  /** Storage strategy used for Hsec (chosen automatically). */
  readonly storage: "dense" | "sparse";
  /** Dense H, present only when storage = "dense". */
  readonly sector?: SectorH;
  /** Sparse CSR H, present only when storage = "sparse". */
  readonly sparseSector?: SparseSectorH;
  /** FCI ground-state energy via matrix-free Lanczos in the N-electron sector. */
  fci(opts?: { maxIter?: number; tol?: number }): { energy: number; iter: number };
}

export interface BuildMoleculeOpts {
  /**
   * Force a particular Hsec storage. By default we pick "dense" when
   * the sector dim is below ~2000 (Jacobi reference + simpler) and
   * "sparse" when it's larger (avoids O(k²) memory + much faster
   * Lanczos due to ~98% sparsity).
   */
  storage?: "dense" | "sparse" | "auto";
}

const SPARSE_THRESHOLD_K = 2000;

/**
 * Build the second-quantized Hamiltonian for a molecule (specified
 * by a list of atoms) projected to its physical particle-number sector
 * (= number of electrons in the neutral molecule). Returns a handle
 * that exposes the sector matrix and a lazy `fci()` method that runs
 * matrix-free Lanczos.
 *
 * Auto-storage policy: dense Hsec when the sector dim ≤ 2000 (cheap
 * ~30 MB, lets us cross-check with Jacobi); sparse CSR Hsec when
 * larger (CH₄'s 43758-dim sector is 15 GB dense vs 240 MB sparse,
 * and matvec is 100× faster on sparse).
 */
export function buildMoleculeFCI(atoms: readonly Atom[], opts: BuildMoleculeOpts = {}): MoleculeData {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const nSpatial = shells.length;
  // Predict sector dim from C(2 * nSpatial, nElectrons) without enumerating.
  const k_predict = combinations(2 * nSpatial, nElectrons);
  const storage: "dense" | "sparse" = opts.storage === "auto" || opts.storage === undefined
    ? (k_predict <= SPARSE_THRESHOLD_K ? "dense" : "sparse")
    : opts.storage;

  if (storage === "dense") {
    const sector = buildSectorH(nSpatial, integrals.h_OAO, integrals.eri_OAO, integrals.Vnn, nElectrons);
    const fci = (o: { maxIter?: number; tol?: number } = {}): { energy: number; iter: number } => {
      const { H, k } = sector;
      const matvec = (x: Float64Array, out: Float64Array) => {
        for (let i = 0; i < k; i++) {
          let s = 0;
          const row = i * k;
          for (let j = 0; j < k; j++) s += H[row + j]! * x[j]!;
          out[i] = s;
        }
      };
      const x0 = new Float64Array(k);
      for (let i = 0; i < k; i++) x0[i] = 1;
      const r = lanczosGroundState(matvec, x0, { maxIter: o.maxIter ?? 120, tol: o.tol ?? LANCZOS_TOL });
      return { energy: r.energy, iter: r.iter };
    };
    return { atoms, nElectrons, nSpatial, nQubits: 2 * nSpatial, integrals, storage, sector, fci };
  }

  // sparse path
  const sparseSector = buildSparseSectorH(nSpatial, integrals.h_OAO, integrals.eri_OAO, integrals.Vnn, nElectrons);
  const fci = (o: { maxIter?: number; tol?: number } = {}): { energy: number; iter: number } => {
    const matvec = (x: Float64Array, out: Float64Array) => sparseMatvec(sparseSector, x, out);
    const x0 = new Float64Array(sparseSector.k);
    for (let i = 0; i < sparseSector.k; i++) x0[i] = 1;
    const r = lanczosGroundState(matvec, x0, { maxIter: o.maxIter ?? 200, tol: o.tol ?? LANCZOS_TOL });
    return { energy: r.energy, iter: r.iter };
  };
  return { atoms, nElectrons, nSpatial, nQubits: 2 * nSpatial, integrals, storage, sparseSector, fci };
}

function combinations(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  k = Math.min(k, n - k);
  let r = 1;
  for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1);
  return Math.round(r);
}
