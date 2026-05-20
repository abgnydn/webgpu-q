// ─────────────────────────────────────────────────────────────
// geometry.ts — molecular geometry optimization on the HF or
// DFT energy surface. Tier 2 stages 1, 5, 6, 6b: BFGS
// minimization of E(R) with analytical gradients via Pulay 1969
// (HF — see `hf-gradient.ts`) or RKS-DFT (LDA + GGA + hybrid —
// see `dft-gradient.ts`).
//
// Two gradient paths:
//   • `useAnalyticGrad: true` (DEFAULT) — analytical Pulay
//     gradients. Each gradient costs ~5–10× one energy build.
//   • `useAnalyticGrad: false` — central FD. Kept for cross-
//     checks and as a fallback if the analytical path is
//     ever suspected of bugs.
//
// `method` selects HF or any RKS-DFT functional kind. The
// analytical-gradient path supports the FULL ladder: HF, LDA,
// BVWN5, BLYP, B3VWN5, B3LYP5 — the DFT path uses the basis-
// Hessian-aware ∂γ/∂R term landed in stage 6b.
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
   *  Analytical gradients are wired for the full HF + 5-functional
   *  ladder (HF, LDA, BVWN5, BLYP, B3VWN5, B3LYP5). */
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
      gBohr = hfGradient({ shells, nuclei, shellAtomIdx, P: hf.D, W, sphericalT: integrals.sphericalT });
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
        sphericalT: integrals.sphericalT,
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

/**
 * Dihedral (torsion) angle i—j—k—l in degrees.
 * Standard sign convention: positive for clockwise rotation looking
 * from j → k. Returns ∈ [−180, +180]. Returns NaN on degenerate
 * geometries (collinear consecutive triplets).
 */
export function dihedralAngle(
  atoms: readonly Atom[], i: number, j: number, k: number, l: number,
): number {
  const a = atoms[i]!; const b = atoms[j]!; const c = atoms[k]!; const d = atoms[l]!;
  const b1x = b.pos[0]-a.pos[0], b1y = b.pos[1]-a.pos[1], b1z = b.pos[2]-a.pos[2];
  const b2x = c.pos[0]-b.pos[0], b2y = c.pos[1]-b.pos[1], b2z = c.pos[2]-b.pos[2];
  const b3x = d.pos[0]-c.pos[0], b3y = d.pos[1]-c.pos[1], b3z = d.pos[2]-c.pos[2];
  const n1x = b1y*b2z - b1z*b2y, n1y = b1z*b2x - b1x*b2z, n1z = b1x*b2y - b1y*b2x;
  const n2x = b2y*b3z - b2z*b3y, n2y = b2z*b3x - b2x*b3z, n2z = b2x*b3y - b2y*b3x;
  const n1n = Math.sqrt(n1x*n1x + n1y*n1y + n1z*n1z);
  const n2n = Math.sqrt(n2x*n2x + n2y*n2y + n2z*n2z);
  if (n1n < 1e-12 || n2n < 1e-12) return Number.NaN;
  const cosAng = (n1x*n2x + n1y*n2y + n1z*n2z) / (n1n * n2n);
  const b2n = Math.sqrt(b2x*b2x + b2y*b2y + b2z*b2z);
  const mx = (n1y*b2z - n1z*b2y) / b2n;
  const my = (n1z*b2x - n1x*b2z) / b2n;
  const mz = (n1x*b2y - n1y*b2x) / b2n;
  const sinAng = (mx*n2x + my*n2y + mz*n2z) / n2n;
  return Math.atan2(sinAng, cosAng) * 180 / Math.PI;
}

/** Covalent radii in Å (Pyykkö & Atsumi 2009). Tuned for ordinary
 *  single-bond detection in organic chemistry. */
const COVALENT_RADIUS_ANGSTROM: Readonly<Record<string, number>> = {
  H:  0.32, He: 0.46,
  Li: 1.33, Be: 1.02,
  C:  0.75, N:  0.71, O:  0.63, F:  0.64,
};

export interface Bond {
  readonly i: number;
  readonly j: number;
  /** Distance in Å. */
  readonly length: number;
}

/**
 * Detect bonds by pairwise distances using covalent-radii cutoff:
 *   d_ij < (R_cov_i + R_cov_j) · scale + tolerance
 *
 * Defaults (`scale = 1.2`, `tolerance = 0.4`) handle single, double,
 * triple bonds and weakly-bonded fragments. Returns unique pairs
 * (i < j), sorted by i then j.
 */
export function findBonds(
  atoms: readonly Atom[],
  opts: { readonly scale?: number; readonly tolerance?: number } = {},
): Bond[] {
  const scale = opts.scale ?? 1.2;
  const tol = opts.tolerance ?? 0.4;
  const bonds: Bond[] = [];
  for (let i = 0; i < atoms.length; i++) {
    const ri = COVALENT_RADIUS_ANGSTROM[atoms[i]!.symbol] ?? 1.0;
    for (let j = i + 1; j < atoms.length; j++) {
      const rj = COVALENT_RADIUS_ANGSTROM[atoms[j]!.symbol] ?? 1.0;
      const cutoff = (ri + rj) * scale + tol;
      const length = bondLength(atoms, i, j);
      if (length < cutoff) bonds.push({ i, j, length });
    }
  }
  return bonds;
}
