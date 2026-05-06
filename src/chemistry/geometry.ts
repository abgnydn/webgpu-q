// ─────────────────────────────────────────────────────────────
// geometry.ts — molecular geometry optimization on the HF
// energy surface. Tier 2 stage 1: BFGS minimization of E_HF(R).
// Tier 2 stage 5: switch to analytical gradients via Pulay 1969
// + integral derivatives (`hf-gradient.ts`).
//
// Two gradient paths:
//   • `useAnalyticGrad: true` (DEFAULT) — analytical Pulay
//     gradients. Each gradient costs ~10× one HF energy. So
//     a step costs (1 energy + 1 gradient) ≈ 11×. The FD path
//     would cost (1 energy + 6·N_atoms energies) ≈ 7–22× for
//     N_atoms = 1–4 — analytical is faster from N_atoms ≥ 2.
//   • `useAnalyticGrad: false` — central FD. Kept for cross-
//     checks and as a fallback if the analytical path is
//     ever suspected of bugs.
//
// Scale of FD step (FD path only): positions are in Å, HF energy
// in Hartrees with ~10 digits of relative precision (DIIS to
// 1e-10 Ha). Optimal central-FD step h satisfies h² ≈ ε|E|/g, so
// h ≈ √(1e-10 · 100 / 1) ≈ 1e-4 Å. Default 1e-3 Å is generous.
// ─────────────────────────────────────────────────────────────

import { computeMolecularIntegrals, type IntegralOpts } from "./cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type BasisName } from "./atoms.js";
import { runRHFSCF, type HFOpts } from "./hf-scf.js";
import { lbfgs, type LBFGSOptions, type LBFGSResult } from "./optimizer.js";
import { hfGradient, buildEnergyWeightedDensity } from "./hf-gradient.js";

const ANGSTROM_TO_BOHR = 1.8897261339;

export interface GeometryOptOpts {
  /** Basis set (default sto-3g). */
  readonly basis?: BasisName;
  /** Spherical-d transform on the integrals? Default false. */
  readonly spherical?: boolean;
  /** HF SCF options forwarded to runRHFSCF. */
  readonly hf?: HFOpts;
  /** L-BFGS options. fdStep default overridden to 1e-3 Å. */
  readonly lbfgs?: LBFGSOptions;
  /** Use analytical HF gradient (default true). False falls back to FD. */
  readonly useAnalyticGrad?: boolean;
}

export interface GeometryOptResult {
  /** Optimized atoms (positions in Å). */
  readonly atoms: readonly Atom[];
  /** Final HF energy (Hartrees). */
  readonly energy: number;
  /** L-BFGS termination + history. */
  readonly optimizer: LBFGSResult;
  /** Number of HF energy evaluations performed (each FD grad = 6·N_atoms). */
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
  const integralOpts: IntegralOpts = { spherical: opts.spherical ?? false };
  const hfOpts: HFOpts = {
    useDIIS: true,
    energyTol: 1e-10,
    densityTol: 1e-8,
    maxIter: 200,
    ...opts.hf,
  };

  let nEvals = 0;
  const energyAt = (x: Float64Array): number => {
    nEvals++;
    const moved = vectorToAtoms(x, atoms);
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(moved, basis);
    const integrals = computeMolecularIntegrals(shells, nuclei, integralOpts);
    const hf = runRHFSCF(integrals, nElectrons, hfOpts);
    return hf.energy;
  };

  const useAnalytic = opts.useAnalyticGrad ?? true;
  // Analytical gradient: integrate hf-gradient.ts. Atoms are in Å,
  // gradient comes back in Ha/Bohr; convert to Ha/Å for the optimizer
  // (matches the FD path's units).
  const gradientAt = useAnalytic ? (x: Float64Array): Float64Array => {
    nEvals++;
    const moved = vectorToAtoms(x, atoms);
    const { shells, nuclei, nElectrons, shellAtomIdx } =
      moleculeToShellsNuclei(moved, basis);
    const integrals = computeMolecularIntegrals(shells, nuclei, integralOpts);
    const hf = runRHFSCF(integrals, nElectrons, hfOpts);
    const W = buildEnergyWeightedDensity(
      hf.C_MO, hf.orbitalEnergies, hf.nOccupied, integrals.n,
    );
    const gBohr = hfGradient({ shells, nuclei, shellAtomIdx, P: hf.D, W });
    // Ha/Bohr → Ha/Å: dE/d(R_Å) = dE/d(R_Bohr) · (dR_Bohr/dR_Å) = dE/d(R_Bohr) · ANG_TO_BOHR.
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
