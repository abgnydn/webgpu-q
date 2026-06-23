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

import { moleculeToShellsNuclei, defaultFrozenCore, type Atom, type AtomSymbol, type BasisName } from "../chemistry/atoms.js";
import { runRHFAuto, runRKSAuto, runUHFAuto, runUKSAuto } from "../chemistry/rhf-auto.js";
import { mp2EnergyDF } from "../chemistry/mp2.js";
import { generateAutoAux, buildAuxBasisDFStreaming } from "../chemistry/df-aux.js";
import type { FunctionalKind } from "../chemistry/dft/functional.js";
import { computeMolecularIntegrals } from "../chemistry/cg-molecular.js";
import { runRHFSCF } from "../chemistry/hf-scf.js";
import { runCCSD } from "../chemistry/ccsd.js";
import { runCCSDT } from "../chemistry/ccsd-t.js";

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
  /** HOMO–LUMO gap in eV — the screening property (relates to color, reactivity,
   *  stability). Closed-shell: LUMO−HOMO; open-shell: from the α channel. */
  readonly homoLumoGapEv: number;
}

const HARTREE_TO_EV = 27.211386245988;
/** LUMO−HOMO in eV from an ascending orbital-energy list + occupied count. */
function gapEv(orbE: Float64Array, nOcc: number): number {
  if (nOcc <= 0 || nOcc >= orbE.length) return NaN; // no virtual (or no occ) → undefined
  return (orbE[nOcc]! - orbE[nOcc - 1]!) * HARTREE_TO_EV;
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
  let energy: number, iter: number, converged: boolean, scf: ChemEnergyResult["scf"], gap: number;
  let prov: { engine: ChemEnergyResult["engine"]; method: ChemEnergyResult["dfMethod"] };
  if (isDFT && open) {
    const r = await runUKSAuto(shells, nuclei, open.nAlpha, open.nBeta, symbols, { ...o, functional: tile.functional });
    ({ energy, iter, converged } = r.uks); prov = r.provenance; scf = "uks";
    gap = gapEv(r.uks.orbitalEnergiesAlpha, r.uks.nAlpha);
  } else if (isDFT) {
    const r = await runRKSAuto(shells, nuclei, nElectrons, symbols, { ...o, functional: tile.functional });
    ({ energy, iter, converged } = r.rks); prov = r.provenance; scf = "rks";
    gap = gapEv(r.rks.orbitalEnergies, r.rks.nOccupied);
  } else if (open) {
    const r = await runUHFAuto(shells, nuclei, open.nAlpha, open.nBeta, o);
    ({ energy, iter, converged } = r.uhf); prov = r.provenance; scf = "uhf";
    gap = gapEv(r.uhf.orbitalEnergiesAlpha, r.uhf.nAlpha);
  } else {
    const r = await runRHFAuto(shells, nuclei, nElectrons, { ...o, hf: { useDIIS: true, energyTol: 1e-9, densityTol: 1e-7, maxIter: 200 } });
    ({ energy, iter, converged } = r.hf); prov = r.provenance; scf = "rhf";
    gap = gapEv(r.hf.orbitalEnergies, r.hf.nOccupied);
  }

  return {
    label: tile.label, energy, nElectrons, nBasisFunctions: shells.length, iter, converged,
    durationMs: performance.now() - t0, engine: prov.engine, dfMethod: prov.method, scf,
    homoLumoGapEv: gap,
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Distributed DF-MP2 — the swarm's first collaborative *correlation* reduction.
//
// chem-energy distributes N independent molecules (no-talk, embarrassingly
// parallel). Distributed J/K already builds one molecule's Fock collaboratively.
// This adds the post-HF axis: ONE molecule's MP2 correlation energy. The DF-MP2
// sum E_corr = Σ_i Σ_j Σ_ab … partitions exactly over the OUTER occupied index i
// (each i carries its full inner j/a/b sum — a disjoint, complete cover, no i≤j
// symmetry factor), so each worker owns a contiguous i-slice and returns one
// scalar partial. The master sums the scalars → E_corr, total = E_HF + E_corr.
//
// What actually distributes, stated honestly (a scientific-critic pass forced
// this precision, 2026-06-15):
//   • DISTRIBUTED: the MP2 contraction, O(nocc²·nvirt²·nAux). This is the
//     asymptotically dominant term (an extra nocc·nvirt over the half-transform),
//     so for LARGE molecules a wall-time crossover is expected — the slice runs
//     1/k of it. NOT YET MEASURED at scale; predicted, not claimed.
//   • REDUNDANT on every worker: the full SCF, the DF B-tensor build, and the
//     entire occ×virt×nAux B_ov half-transform. At small sizes these match or
//     exceed the contraction, so small-molecule runs are a CORRECTNESS + comm-
//     pattern demo (partition-sum == single-machine to ~1e-12), not a speedup.
//   • MEMORY does NOT partition by outer-i alone: the inner j-loop reads B_ov[j]
//     for ALL occupied j, so every worker still holds the full B_ov. A genuine
//     per-worker memory partition needs 2D (i-block × j-block) tiling so a worker
//     holds only B_ov rows for i-block ∪ j-block — the next step if the goal is
//     "an MP2 too large for one tab's heap." (Aux-P partitioning can't help: the
//     energy is nonlinear in B — (ia|jb)=Σ_P B·B — so partial-P sums don't reduce.)
//
// Comm is free here — molecule spec in (~hundreds of bytes), one f64 out, one
// round — precisely BECAUSE each worker redundantly rebuilds the deterministic
// reference instead of shipping it. Comm-optimality is bought with compute
// redundancy; the distributed axis is the contraction, not the bytes.

export interface MP2SliceTile {
  readonly label: string;
  readonly atoms: readonly Atom[];
  readonly basis: BasisName;
  /** Outer occupied-index slice this worker owns: [iLo, iHi). */
  readonly iLo: number;
  readonly iHi: number;
  /** Frozen-core count (lowest occupied MOs excluded). Default: per-atom 1s. */
  readonly nFrozenCore?: number;
  readonly force?: "exact" | "df";
  readonly fast?: boolean;
}

export interface MP2SliceResult {
  readonly label: string;
  /** This slice's contribution to E_corr (Hartree). */
  readonly partial: number;
  readonly iLo: number;
  readonly iHi: number;
  /** HF total energy — identical on every worker (deterministic SCF); the
   *  reducer reads it from any one slice to form E_total = E_HF + Σ partials. */
  readonly eHF: number;
  readonly nOccupied: number;
  readonly nBasisFunctions: number;
  readonly durationMs: number;
  readonly engine: "wasm" | "gpu" | "gpu+wasm";
}

export const MP2_SLICE_KIND = "mp2-slice" as const;

export async function runMP2SliceTile(tile: MP2SliceTile): Promise<MP2SliceResult> {
  const t0 = performance.now();
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(tile.atoms, tile.basis);
  const n = shells.length;

  // Same deterministic reference on every worker → no need to ship C/ε.
  const r = await runRHFAuto(shells, nuclei, nElectrons, { force: tile.force, fast: tile.fast });
  const { C_MO, orbitalEnergies, nOccupied } = r.hf;

  // DF B-tensor for the (ia|jb) reconstruction (RI-MP2; never builds the 4-index ERI).
  const aux = generateAutoAux(shells, 1);
  const df = await buildAuxBasisDFStreaming(shells, aux);

  const nFrozen = tile.nFrozenCore ?? defaultFrozenCore(tile.atoms);
  const partial = mp2EnergyDF(C_MO, orbitalEnergies, nOccupied, n, nFrozen, df, [tile.iLo, tile.iHi]);

  return {
    label: tile.label, partial, iLo: tile.iLo, iHi: tile.iHi,
    eHF: r.hf.energy, nOccupied, nBasisFunctions: n,
    durationMs: performance.now() - t0, engine: r.provenance.engine,
  };
}

/** Partition one molecule's active-occupied range into `nSlices` tiles for a
 *  distributed DF-MP2. Contiguous, near-equal i-ranges over [nFrozen, nOcc). */
export function mp2SliceTiles(opts: {
  readonly label: string;
  readonly atoms: readonly Atom[];
  readonly basis: BasisName;
  readonly nSlices: number;
  readonly nFrozenCore?: number;
  readonly force?: "exact" | "df";
  readonly fast?: boolean;
}): MP2SliceTile[] {
  const { nElectrons } = moleculeToShellsNuclei(opts.atoms, opts.basis);
  const nOcc = nElectrons / 2;                       // closed-shell
  const nFrozen = opts.nFrozenCore ?? defaultFrozenCore(opts.atoms);
  const active = Math.max(0, nOcc - nFrozen);
  const k = Math.max(1, Math.min(opts.nSlices, active || 1));
  const tiles: MP2SliceTile[] = [];
  for (let s = 0; s < k; s++) {
    const iLo = nFrozen + Math.floor((s * active) / k);
    const iHi = nFrozen + Math.floor(((s + 1) * active) / k);
    if (iHi <= iLo) continue;                        // empty slice (more slices than rows)
    tiles.push({
      label: `${opts.label} i∈[${iLo},${iHi})`,
      atoms: opts.atoms, basis: opts.basis, iLo, iHi,
      nFrozenCore: nFrozen, force: opts.force, fast: opts.fast,
    });
  }
  return tiles;
}

/** Reduce distributed MP2 slices into the molecule's total energy. Sums the
 *  scalar partials (== E_corr) and adds the (shared) HF energy.
 *
 *  The whole no-ship-tensor design assumes every worker built the IDENTICAL
 *  C/ε reference (deterministic SCF), so each partial is over the same MOs. If
 *  two workers diverged (e.g. one ran exact-ERI HF, another DF HF, or different
 *  engines rounded differently), their partials reference different orbitals and
 *  the sum is silently wrong. Guard against it: the per-worker E_HF must agree.
 *  This is cheap insurance for the assumption the correctness rests on. */
export function reduceMP2Slices(slices: readonly MP2SliceResult[]): {
  readonly correlationEnergy: number;
  readonly hfEnergy: number;
  readonly totalEnergy: number;
} {
  if (slices.length === 0) throw new Error("reduceMP2Slices: no slices");
  const hfEnergy = slices[0]!.eHF;
  let E_corr = 0;
  for (const s of slices) {
    if (Math.abs(s.eHF - hfEnergy) > 1e-7) {
      throw new Error(
        `reduceMP2Slices: HF references disagree across workers ` +
        `(${s.eHF.toFixed(8)} vs ${hfEnergy.toFixed(8)}) — slices were computed ` +
        `against different references (heterogeneous engine/path?); the sum is invalid.`,
      );
    }
    E_corr += s.partial;
  }
  return { correlationEnergy: E_corr, hfEnergy, totalEnergy: hfEnergy + E_corr };
}

// ─────────────────────────────────────────────────────────────────────────
// Federated CCSD(T) — distribute the (T) triples across the swarm.
//
// The distributed-MP2 honest negative (above) pinned single-molecule speedup
// near 1× because the redundant SCF+DF setup (S) dwarfs MP2's O(N⁵) splittable
// grind (C): speedup = (S+C)/(S+C/k) ≈ 1 while S ≫ C. The (T) correction is a
// better target — O(N⁷), NON-iterative, embarrassingly parallel over the outer
// occupied spin-orbital i. But MEASURED honestly (H₂O cc-pVDZ, frozen-core,
// single-threaded): CCSD=53 s, (T)=40 s → S(SCF+CCSD)=53 s vs C((T))=40 s,
// C/S=0.76, predicted 2-tab speedup 1.28× (vs MP2's 1.10×). So distributing (T)
// helps MORE than MP2 but does NOT yet dominate — the redundant **CCSD**, not
// SCF/DF, is now the bottleneck (my "(T)≫setup" prior was wrong). Since (T) is
// O(N⁷) and CCSD O(N⁶), C/S grows ~linearly with N, so the clean crossover
// (C≫S → ~2×) is at molecules LARGER than H₂O cc-pVDZ; winning at accessible
// sizes also needs distributing (or sharing) the CCSD. See
// experiments/results/2026-06-23/federated-ccsdt-regime/.
//
// Each worker redundantly rebuilds the IDENTICAL deterministic RHF+CCSD reference
// (no shipped amplitudes), then runs only its i-slice of the (T) sum.
//
// E_(T) = Σ_i (independent outer sum), so slicing i into disjoint ranges and
// summing partials == single-shot to float noise (tests/chemistry/ccsdt-slice).
// Exact 4-index ERI path (the (T) core needs ⟨pq||rs⟩), so this is a
// medium-molecule kernel; the size ceiling is the ERI, not the swarm.

export const CCSDT_SLICE_KIND = "ccsdt-slice" as const;

export interface CCSDTSliceTile {
  readonly label: string;
  readonly atoms: readonly Atom[];
  readonly basis: BasisName;
  /** Outer occupied SPIN-ORBITAL slice this worker owns: [iLo, iHi).
   *  Closed-shell NOCC (spin-orbitals) = nElectrons. */
  readonly iLo: number;
  readonly iHi: number;
  /** Frozen-core spatial count (lowest occupied MOs excluded). Default: per-atom 1s. */
  readonly nFrozenCore?: number;
}

export interface CCSDTSliceResult {
  readonly label: string;
  /** This slice's contribution to the (T) correction E_(T) (Hartree). */
  readonly partial: number;
  readonly iLo: number;
  readonly iHi: number;
  /** Shared deterministic reference — the converged CCSD total energy. Must agree
   *  across workers (else they correlate different orbitals and the sum is wrong);
   *  the reducer guards on it. Folds in E_HF + CCSD correlation. */
  readonly eCCSD: number;
  readonly nOccupiedSO: number;
  readonly durationMs: number;
}

export async function runCCSDTSliceTile(tile: CCSDTSliceTile): Promise<CCSDTSliceResult> {
  const t0 = performance.now();
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(tile.atoms, tile.basis);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  // Deterministic reference: every worker rebuilds the IDENTICAL RHF + CCSD so no
  // amplitudes need shipping; each then computes only its (T) i-slice.
  const hf = runRHFSCF(integrals, nElectrons, { maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
  const ccsd = runCCSD(hf, integrals, { tol: 1e-9, maxIter: 200 });
  const nFrozen = tile.nFrozenCore ?? defaultFrozenCore(tile.atoms);
  const t = runCCSDT(ccsd, hf, integrals, { nFrozenCore: nFrozen, iRange: [tile.iLo, tile.iHi] });
  return {
    label: tile.label, partial: t.tripleCorrection, iLo: tile.iLo, iHi: tile.iHi,
    eCCSD: ccsd.totalEnergy, nOccupiedSO: 2 * hf.nOccupied,
    durationMs: performance.now() - t0,
  };
}

/** Partition one molecule's active occupied spin-orbital range into `nSlices`
 *  near-equal contiguous i-slices over [2·nFrozen, NOCC) for a distributed (T). */
export function ccsdtSliceTiles(opts: {
  readonly label: string;
  readonly atoms: readonly Atom[];
  readonly basis: BasisName;
  readonly nSlices: number;
  readonly nFrozenCore?: number;
}): CCSDTSliceTile[] {
  const { nElectrons } = moleculeToShellsNuclei(opts.atoms, opts.basis);
  const NOCC = nElectrons;                          // closed-shell: NOCC (spin) = nElectrons
  const nFrozen = opts.nFrozenCore ?? defaultFrozenCore(opts.atoms);
  const loSO = 2 * nFrozen;                          // RHF interleaved → frozen SOs [0, 2·nFrozen)
  const active = Math.max(0, NOCC - loSO);
  const k = Math.max(1, Math.min(opts.nSlices, active || 1));
  const tiles: CCSDTSliceTile[] = [];
  for (let s = 0; s < k; s++) {
    const iLo = loSO + Math.floor((s * active) / k);
    const iHi = loSO + Math.floor(((s + 1) * active) / k);
    if (iHi <= iLo) continue;                        // empty slice (more slices than rows)
    tiles.push({
      label: `${opts.label} i∈[${iLo},${iHi})`,
      atoms: opts.atoms, basis: opts.basis, iLo, iHi, nFrozenCore: nFrozen,
    });
  }
  return tiles;
}

/** Reduce distributed (T) slices into the molecule's CCSD(T) energy. Sums the
 *  scalar (T) partials (== E_(T)) and adds the shared CCSD energy. Guards the
 *  deterministic-reference assumption: every worker's CCSD energy must agree, or
 *  the slices correlate different orbitals and the (T) sum is silently wrong. */
export function reduceCCSDTSlices(slices: readonly CCSDTSliceResult[]): {
  readonly tripleCorrection: number;
  readonly ccsdEnergy: number;
  readonly totalEnergy: number;
} {
  if (slices.length === 0) throw new Error("reduceCCSDTSlices: no slices");
  const ccsdEnergy = slices[0]!.eCCSD;
  let E_T = 0;
  for (const s of slices) {
    if (Math.abs(s.eCCSD - ccsdEnergy) > 1e-7) {
      throw new Error(
        `reduceCCSDTSlices: CCSD references disagree across workers ` +
        `(${s.eCCSD.toFixed(8)} vs ${ccsdEnergy.toFixed(8)}) — slices were computed ` +
        `against different references; the (T) sum is invalid.`,
      );
    }
    E_T += s.partial;
  }
  return { tripleCorrection: E_T, ccsdEnergy, totalEnergy: ccsdEnergy + E_T };
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
