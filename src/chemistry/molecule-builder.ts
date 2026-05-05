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
import { lanczosGroundState } from "../manybody/lanczos.js";

export interface MoleculeData {
  readonly atoms: readonly Atom[];
  readonly nElectrons: number;
  readonly nSpatial: number;
  readonly nQubits: number;
  readonly integrals: MolecularIntegrals;
  readonly sector: SectorH;
  /** FCI ground-state energy via matrix-free Lanczos in the N-electron sector. */
  fci(opts?: { maxIter?: number; tol?: number }): { energy: number; iter: number };
}

/**
 * Build the second-quantized Hamiltonian for a molecule (specified
 * by a list of atoms) projected to its physical particle-number sector
 * (= number of electrons in the neutral molecule). Returns a handle
 * that exposes the sector matrix and a lazy `fci()` method that runs
 * matrix-free Lanczos.
 */
export function buildMoleculeFCI(atoms: readonly Atom[]): MoleculeData {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const nSpatial = shells.length;
  const sector = buildSectorH(nSpatial, integrals.h_OAO, integrals.eri_OAO, integrals.Vnn, nElectrons);

  const fci = (opts: { maxIter?: number; tol?: number } = {}): { energy: number; iter: number } => {
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
    const r = lanczosGroundState(matvec, x0, {
      maxIter: opts.maxIter ?? 120,
      tol: opts.tol ?? 1e-10,
    });
    return { energy: r.energy, iter: r.iter };
  };

  return { atoms, nElectrons, nSpatial, nQubits: 2 * nSpatial, integrals, sector, fci };
}
