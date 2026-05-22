// Real-chemistry swarm kernel — Step 3 of Phase D.
//
// Each tile is a *complete molecule specification* (atoms + method +
// basis). The worker tab runs the full SCF + (post-HF if requested)
// pipeline locally and returns the energy. This avoids shipping ERI
// tensors across the wire: only the molecule definition (a few hundred
// bytes) goes per tile, the heavy compute lives in the worker tab.
//
// Use cases this makes feasible across a browser-tab swarm:
//   - Geometry / bond-length scans (one tile per geometry)
//   - Multi-functional surveys (one tile per functional)
//   - Reaction-coordinate energies
//   - Batch property evaluation
//
// The kernel is registered as "chem-energy" so any /swarm.html tab
// (or future molecule.html with the swarm wired in) can act as worker.

import { moleculeToShellsNuclei, type Atom, type BasisName } from "../chemistry/atoms.js";
import { computeMolecularIntegrals } from "../chemistry/cg-molecular.js";
import { runRHFSCF } from "../chemistry/hf-scf.js";

export interface ChemEnergyTile {
  readonly label: string;
  readonly atoms: readonly Atom[];
  /** Currently only "hf"/STO-3G supported across the swarm; the kernel
   *  is method-agnostic at the protocol layer though. */
  readonly method: "hf";
  readonly basis: BasisName;
}

export interface ChemEnergyResult {
  readonly label: string;
  readonly energy: number;
  readonly nElectrons: number;
  readonly nBasisFunctions: number;
  readonly iter: number;
  readonly converged: boolean;
  readonly durationMs: number;
}

export const CHEM_ENERGY_KIND = "chem-energy" as const;

export function runChemEnergyTile(tile: ChemEnergyTile): ChemEnergyResult {
  const t0 = performance.now();
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(tile.atoms, tile.basis);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons, {
    useDIIS: true, energyTol: 1e-9, densityTol: 1e-7, maxIter: 200,
  });
  return {
    label: tile.label,
    energy: hf.energy,
    nElectrons,
    nBasisFunctions: integrals.n,
    iter: hf.iter,
    converged: hf.converged,
    durationMs: performance.now() - t0,
  };
}

/** Generate a 1D bond-length scan for a diatomic; returns N tiles each
 *  with the bond length set to a different value in [rMin, rMax]. */
export function bondScanTiles(opts: {
  readonly atomA: Atom["symbol"];
  readonly atomB: Atom["symbol"];
  readonly basis: BasisName;
  readonly rMin: number;
  readonly rMax: number;
  readonly nPoints: number;
}): ChemEnergyTile[] {
  const tiles: ChemEnergyTile[] = [];
  const step = (opts.rMax - opts.rMin) / Math.max(1, opts.nPoints - 1);
  for (let i = 0; i < opts.nPoints; i++) {
    const r = opts.rMin + i * step;
    tiles.push({
      label: `${opts.atomA}-${opts.atomB} r=${r.toFixed(3)}Å`,
      atoms: [
        { symbol: opts.atomA, pos: [0, 0, 0] },
        { symbol: opts.atomB, pos: [0, 0, r] },
      ],
      method: "hf",
      basis: opts.basis,
    });
  }
  return tiles;
}
