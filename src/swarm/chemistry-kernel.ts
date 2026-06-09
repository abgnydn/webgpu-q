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

import { moleculeToShellsNuclei, type Atom, type AtomSymbol, type BasisName } from "../chemistry/atoms.js";
import { runRHFAuto, runRKSAuto, runUHFAuto, runUKSAuto } from "../chemistry/rhf-auto.js";
import type { FunctionalKind } from "../chemistry/dft/functional.js";

export interface ChemEnergyTile {
  readonly label: string;
  readonly atoms: readonly Atom[];
  /** "hf" (Hartree–Fock) or "dft" (Kohn–Sham). Default "hf". */
  readonly method?: "hf" | "dft";
  /** DFT functional (method="dft"). Default "lda-svwn". */
  readonly functional?: FunctionalKind;
  /** Open-shell: explicit α/β electron counts → the unrestricted path (UHF/UKS).
   *  Omit for closed-shell (RHF/RKS). Radicals, doublets, O₂ etc. set this. */
  readonly open?: { readonly nAlpha: number; readonly nBeta: number };
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
  /** Which SCF flavor ran this tile. */
  readonly scf: "rhf" | "uhf" | "rks" | "uks";
}

export const CHEM_ENERGY_KIND = "chem-energy" as const;

export async function runChemEnergyTile(tile: ChemEnergyTile): Promise<ChemEnergyResult> {
  const t0 = performance.now();
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(tile.atoms, tile.basis);
  const symbols = tile.atoms.map((a) => a.symbol) as AtomSymbol[];
  const o = { force: tile.force, fast: tile.fast };
  const isDFT = tile.method === "dft";
  const open = tile.open;

  // Each worker auto-picks exact / DF / hybrid-GPU by size, and the right SCF
  // flavor by closed/open shell — so a swarm batch is heterogeneous tile-by-tile.
  let energy: number, iter: number, converged: boolean, scf: ChemEnergyResult["scf"];
  let prov: { engine: ChemEnergyResult["engine"]; method: ChemEnergyResult["dfMethod"] };
  if (isDFT && open) {
    const r = await runUKSAuto(shells, nuclei, open.nAlpha, open.nBeta, symbols, { ...o, functional: tile.functional });
    ({ energy, iter, converged } = r.uks); prov = r.provenance; scf = "uks";
  } else if (isDFT) {
    const r = await runRKSAuto(shells, nuclei, nElectrons, symbols, { ...o, functional: tile.functional });
    ({ energy, iter, converged } = r.rks); prov = r.provenance; scf = "rks";
  } else if (open) {
    const r = await runUHFAuto(shells, nuclei, open.nAlpha, open.nBeta, o);
    ({ energy, iter, converged } = r.uhf); prov = r.provenance; scf = "uhf";
  } else {
    const r = await runRHFAuto(shells, nuclei, nElectrons, { ...o, hf: { useDIIS: true, energyTol: 1e-9, densityTol: 1e-7, maxIter: 200 } });
    ({ energy, iter, converged } = r.hf); prov = r.provenance; scf = "rhf";
  }

  return {
    label: tile.label, energy, nElectrons, nBasisFunctions: shells.length, iter, converged,
    durationMs: performance.now() - t0, engine: prov.engine, dfMethod: prov.method, scf,
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

/** Open-shell diatomic bond scan — one tile per bond length, each run on the
 *  unrestricted path (UHF/UKS) with the given α/β counts. For radical curves
 *  (OH•, CN•, …) distributed across the swarm. force/fast pick the DF/GPU path. */
export function radicalBondScanTiles(opts: {
  readonly atomA: Atom["symbol"];
  readonly atomB: Atom["symbol"];
  readonly basis: BasisName;
  readonly rMin: number;
  readonly rMax: number;
  readonly nPoints: number;
  readonly open: { readonly nAlpha: number; readonly nBeta: number };
  readonly method?: "hf" | "dft";
  readonly functional?: FunctionalKind;
  readonly force?: "exact" | "df";
  readonly fast?: boolean;
}): ChemEnergyTile[] {
  const tiles: ChemEnergyTile[] = [];
  const step = (opts.rMax - opts.rMin) / Math.max(1, opts.nPoints - 1);
  for (let i = 0; i < opts.nPoints; i++) {
    const r = opts.rMin + i * step;
    tiles.push({
      label: `${opts.atomA}-${opts.atomB}• r=${r.toFixed(3)}Å`,
      atoms: [
        { symbol: opts.atomA, pos: [0, 0, 0] },
        { symbol: opts.atomB, pos: [0, 0, r] },
      ],
      method: opts.method ?? "hf",
      functional: opts.functional,
      open: opts.open,
      basis: opts.basis,
      force: opts.force,
      fast: opts.fast,
    });
  }
  return tiles;
}
