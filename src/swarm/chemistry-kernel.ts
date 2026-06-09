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
import { runRHFAuto } from "../chemistry/rhf-auto.js";

export interface ChemEnergyTile {
  readonly label: string;
  readonly atoms: readonly Atom[];
  /** Currently only "hf" across the swarm; the kernel is method-agnostic at the
   *  protocol layer. The DF/GPU path is chosen per-tile by runRHFAuto. */
  readonly method: "hf";
  readonly basis: BasisName;
  /** Force the integral path: "exact" | "df". Default: auto by tile size. */
  readonly force?: "exact" | "df";
  /** Use the hybrid GPU/WASM DF build on this worker (chemistry-grade, GPU does
   *  the s/p/d-aux integral bulk). Default false → exact/WASM by size. */
  readonly fast?: boolean;
}

export interface ChemEnergyResult {
  readonly label: string;
  readonly energy: number;
  readonly nElectrons: number;
  readonly nBasisFunctions: number;
  readonly iter: number;
  readonly converged: boolean;
  readonly durationMs: number;
  /** How THIS worker produced the number — the swarm is heterogeneous, so each
   *  tile records its own method/engine/precision (e.g. one tab exact, another
   *  gpu+wasm DF). */
  readonly engine: "wasm" | "gpu" | "gpu+wasm";
  readonly dfMethod: "exact-eri" | "density-fitting";
}

export const CHEM_ENERGY_KIND = "chem-energy" as const;

export async function runChemEnergyTile(tile: ChemEnergyTile): Promise<ChemEnergyResult> {
  const t0 = performance.now();
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(tile.atoms, tile.basis);
  // runRHFAuto: exact ERI for small tiles, DF (hybrid GPU when fast) for large —
  // each worker auto-picks, so a swarm batch mixes engines tile-by-tile.
  const res = await runRHFAuto(shells, nuclei, nElectrons, {
    force: tile.force, fast: tile.fast,
    hf: { useDIIS: true, energyTol: 1e-9, densityTol: 1e-7, maxIter: 200 },
  });
  return {
    label: tile.label,
    energy: res.hf.energy,
    nElectrons,
    nBasisFunctions: shells.length,
    iter: res.hf.iter,
    converged: res.hf.converged,
    durationMs: performance.now() - t0,
    engine: res.provenance.engine,
    dfMethod: res.provenance.method,
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
