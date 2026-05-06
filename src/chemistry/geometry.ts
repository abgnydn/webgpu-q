// ─────────────────────────────────────────────────────────────
// geometry.ts — molecular geometry optimization on the HF or
// DFT energy surface. Tier 2 stages 1, 5, 6: BFGS minimization
// of E(R) with analytical gradients via Pulay 1969 (HF — see
// `hf-gradient.ts`) or RKS-DFT/LDA (see `dft-gradient.ts`).
//
// Two gradient paths:
//   • `useAnalyticGrad: true` (DEFAULT) — analytical Pulay
//     gradients. Each gradient costs ~5–10× one energy build.
//   • `useAnalyticGrad: false` — central FD. Kept for cross-
//     checks and as a fallback if the analytical path is
//     ever suspected of bugs.
//
// `method` selects HF or one of the DFT functionals. Default
// is "hf"; the analytical-DFT path currently supports only
// "lda-svwn" — GGA / hybrid functionals will throw a clear
// error from `dftGradient` until the GGA ∂γ/∂R term lands.
//
// Scale of FD step (FD path only): positions are in Å, energy
// in Hartrees with ~10 digits of relative precision (DIIS to
// 1e-10 Ha). Optimal central-FD step h satisfies h² ≈ ε|E|/g,
// so h ≈ √(1e-10 · 100 / 1) ≈ 1e-4 Å. Default 1e-3 Å is generous.
// ─────────────────────────────────────────────────────────────

import { computeMolecularIntegrals, type IntegralOpts } from "./cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type BasisName } from "./atoms.js";
import { runRHFSCF, type HFOpts } from "./hf-scf.js";
import { lbfgs, type LBFGSOptions, type LBFGSResult } from "./optimizer.js";
import { hfGradient, buildEnergyWeightedDensity } from "./hf-gradient.js";
import { runRKSDFT, type RKSOpts } from "./dft/rks-scf.js";
import { dftGradient } from "./dft-gradient.js";
import type { FunctionalKind } from "./dft/functional.js";

const ANGSTROM_TO_BOHR = 1.8897261339;

export type EnergyMethod = "hf" | FunctionalKind;

export interface GeometryOptOpts {
  /** Basis set (default sto-3g). */
  readonly basis?: BasisName;
  /** Spherical-d transform on the integrals? Default false. */
  readonly spherical?: boolean;
  /** Energy method — "hf" (default) or any RKS-DFT functional kind.
   *  The analytical-gradient path supports "hf" and "lda-svwn"; GGA
   *  and hybrid functionals will throw from `dftGradient` until
   *  the GGA ∂γ/∂R term lands. Set `useAnalyticGrad: false` to
   *  optimize on a GGA / hybrid surface via FD gradients in the
   *  meantime. */
  readonly method?: EnergyMethod;
  /** HF SCF options forwarded to runRHFSCF (used when method is "hf"). */
  readonly hf?: HFOpts;
  /** RKS-DFT SCF options (used when method is a DFT functional). */
  readonly dft?: RKSOpts;
  /** L-BFGS options. fdStep default overridden to 1e-3 Å. */
  readonly lbfgs?: LBFGSOptions;
  /** Use analytical gradient (default true). False falls back to FD. */
  readonly useAnalyticGrad?: boolean;
}

export interface GeometryOptResult {
  /** Optimized atoms (positions in Å). */
  readonly atoms: readonly Atom[];
  /** Final energy (Hartrees) — HF or DFT depending on `method`. */
  readonly energy: number;
  /** L-BFGS termination + history. */
  readonly optimizer: LBFGSResult;
  /** Number of energy evaluations performed (each FD grad = 6·N_atoms). */
  readonly nEvaluations: number;
  /** Wall-clock seconds. */
  readonly seconds: number;
}

/**
 * Minimize E_HF over atomic positions starting from `atoms`.
 *
 * Returns the optimized geometry, energy, and BFGS termination info.
 * The structure of `atoms` is preserved (order, symbols); only the
 * `pos` of each atom changes.
 */
export function optimizeGeometry(
  atoms: readonly Atom[],
  opts: GeometryOptOpts = {},
): GeometryOptResult {
  const tStart = performance.now();
  const basis = opts.basis ?? "sto-3g";
  const method: EnergyMethod = opts.method ?? "hf";
  const integralOpts: IntegralOpts = { spherical: opts.spherical ?? false };
  const hfOpts: HFOpts = {
    useDIIS: true,
    energyTol: 1e-10,
    densityTol: 1e-8,
    maxIter: 200,
    ...opts.hf,
  };
  const dftOpts: RKSOpts = {
    functional: method === "hf" ? "lda-svwn" : method,
    energyTol: 1e-10,
    residualTol: 1e-7,
    maxIter: 200,
    ...opts.dft,
  };

  let nEvals = 0;
  const energyAt = (x: Float64Array): number => {
    nEvals++;
    const moved = vectorToAtoms(x, atoms);
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(moved, basis);
    const integrals = computeMolecularIntegrals(shells, nuclei, integralOpts);
    if (method === "hf") {
      return runRHFSCF(integrals, nElectrons, hfOpts).energy;
    }
    const symbols = moved.map((a) => a.symbol);
    return runRKSDFT(integrals, nElectrons, symbols, dftOpts).energy;
  };

  const useAnalytic = opts.useAnalyticGrad ?? true;
  // Analytical gradient: integrate hf-gradient.ts (HF) or
  // dft-gradient.ts (DFT). Atoms are in Å, gradient returns in
  // Ha/Bohr; convert to Ha/Å for the optimizer.
  const gradientAt = useAnalytic ? (x: Float64Array): Float64Array => {
    nEvals++;
    const moved = vectorToAtoms(x, atoms);
    const { shells, nuclei, nElectrons, shellAtomIdx } =
      moleculeToShellsNuclei(moved, basis);
    const integrals = computeMolecularIntegrals(shells, nuclei, integralOpts);
    let gBohr: Float64Array;
    if (method === "hf") {
      const hf = runRHFSCF(integrals, nElectrons, hfOpts);
      const W = buildEnergyWeightedDensity(
        hf.C_MO, hf.orbitalEnergies, hf.nOccupied, integrals.n,
      );
      gBohr = hfGradient({ shells, nuclei, shellAtomIdx, P: hf.D, W });
    } else {
      const symbols = moved.map((a) => a.symbol);
      const dft = runRKSDFT(integrals, nElectrons, symbols, dftOpts);
      const W = buildEnergyWeightedDensity(
        dft.C_MO, dft.orbitalEnergies, dft.nOccupied, integrals.n,
      );
      gBohr = dftGradient({
        shells, nuclei, shellAtomIdx, nucleiSymbols: symbols,
        P: dft.D, W, functional: method,
        grid: opts.dft?.grid,
      });
    }
    const gAng = new Float64Array(gBohr.length);
    for (let i = 0; i < gBohr.length; i++) gAng[i] = gBohr[i]! * ANGSTROM_TO_BOHR;
    return gAng;
  } : undefined;

  const x0 = atomsToVector(atoms);
  const lbfgsOpts: LBFGSOptions & { gradient?: typeof gradientAt } = {
    maxIter: 100,
    fTol: 1e-8,
    gTol: 1e-4,        // 0.0001 Ha/Å — tight but achievable with 1e-3 FD step
    historySize: 8,
    fdStep: 1e-3,      // Å scale (only used when useAnalyticGrad = false)
    initialStep: 0.5,  // half-Å starter is conservative; line search adapts
    ...opts.lbfgs,
  };
  if (gradientAt) lbfgsOpts.gradient = gradientAt;
  const result = lbfgs(energyAt, x0, lbfgsOpts);
  const finalAtoms = vectorToAtoms(result.bestX, atoms);
  return {
    atoms: finalAtoms,
    energy: result.bestF,
    optimizer: result,
    nEvaluations: nEvals,
    seconds: (performance.now() - tStart) / 1000,
  };
}

// ── coordinate ↔ flat vector helpers ────────────────────────

export function atomsToVector(atoms: readonly Atom[]): Float64Array {
  const x = new Float64Array(atoms.length * 3);
  for (let i = 0; i < atoms.length; i++) {
    x[i * 3 + 0] = atoms[i]!.pos[0];
    x[i * 3 + 1] = atoms[i]!.pos[1];
    x[i * 3 + 2] = atoms[i]!.pos[2];
  }
  return x;
}

export function vectorToAtoms(x: Float64Array, template: readonly Atom[]): Atom[] {
  return template.map((a, i) => ({
    symbol: a.symbol,
    pos: [x[i * 3 + 0]!, x[i * 3 + 1]!, x[i * 3 + 2]!] as [number, number, number],
  }));
}

// ── geometric helpers for tests / interrogation ─────────────

/** Distance between atoms i and j in Ångströms. */
export function bondLength(atoms: readonly Atom[], i: number, j: number): number {
  const a = atoms[i]!.pos;
  const b = atoms[j]!.pos;
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Angle (in degrees) at atom j formed by i—j—k. */
export function bondAngle(atoms: readonly Atom[], i: number, j: number, k: number): number {
  const ax = atoms[i]!.pos[0] - atoms[j]!.pos[0];
  const ay = atoms[i]!.pos[1] - atoms[j]!.pos[1];
  const az = atoms[i]!.pos[2] - atoms[j]!.pos[2];
  const bx = atoms[k]!.pos[0] - atoms[j]!.pos[0];
  const by = atoms[k]!.pos[1] - atoms[j]!.pos[1];
  const bz = atoms[k]!.pos[2] - atoms[j]!.pos[2];
  const dot = ax * bx + ay * by + az * bz;
  const na = Math.sqrt(ax * ax + ay * ay + az * az);
  const nb = Math.sqrt(bx * bx + by * by + bz * bz);
  return Math.acos(dot / (na * nb)) * 180 / Math.PI;
}
