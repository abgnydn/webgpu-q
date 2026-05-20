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

/** Atomic masses in atomic mass units (amu). Most-abundant isotope. */
const ATOMIC_MASS: Readonly<Record<string, number>> = {
  H:  1.00782503207, He: 4.002603254,
  Li: 7.0160034366,  Be: 9.012183065,
  C: 12.0,            N: 14.0030740048,
  O: 15.99491461956,  F: 18.998403163,
};

/** Total mass of a molecule in amu. */
export function totalMass(atoms: readonly Atom[]): number {
  let m = 0;
  for (const a of atoms) m += ATOMIC_MASS[a.symbol] ?? 0;
  return m;
}

/**
 * Center of mass of a molecule in Å (using same units as atom.pos).
 */
export function centerOfMass(atoms: readonly Atom[]): [number, number, number] {
  let mx = 0, my = 0, mz = 0, mtot = 0;
  for (const a of atoms) {
    const m = ATOMIC_MASS[a.symbol] ?? 0;
    mx += m * a.pos[0];
    my += m * a.pos[1];
    mz += m * a.pos[2];
    mtot += m;
  }
  if (mtot < 1e-12) throw new Error("centerOfMass: zero total mass");
  return [mx / mtot, my / mtot, mz / mtot];
}

/**
 * Principal moments of inertia (amu·Å²), sorted I_a ≤ I_b ≤ I_c.
 * Returns the three principal axes as 3-vectors (columns of the
 * eigenvector matrix of the inertia tensor in the COM frame).
 */
export function principalMomentsOfInertia(atoms: readonly Atom[]): {
  readonly Ia: number;
  readonly Ib: number;
  readonly Ic: number;
  readonly axes: readonly [
    readonly [number, number, number],
    readonly [number, number, number],
    readonly [number, number, number],
  ];
} {
  const [cx, cy, cz] = centerOfMass(atoms);
  // Inertia tensor in COM frame, amu·Å².
  let Ixx = 0, Iyy = 0, Izz = 0, Ixy = 0, Ixz = 0, Iyz = 0;
  for (const a of atoms) {
    const m = ATOMIC_MASS[a.symbol] ?? 0;
    const x = a.pos[0] - cx, y = a.pos[1] - cy, z = a.pos[2] - cz;
    Ixx += m * (y * y + z * z);
    Iyy += m * (x * x + z * z);
    Izz += m * (x * x + y * y);
    Ixy -= m * x * y;
    Ixz -= m * x * z;
    Iyz -= m * y * z;
  }
  // 3×3 symmetric eigendecomposition via analytic Jacobi.
  const { eigenvalues, eigenvectors } = symmetric3x3Eig(
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz,
  );
  // Sort ascending.
  const idx = [0, 1, 2].sort((a, b) => eigenvalues[a]! - eigenvalues[b]!);
  return {
    Ia: eigenvalues[idx[0]!]!,
    Ib: eigenvalues[idx[1]!]!,
    Ic: eigenvalues[idx[2]!]!,
    axes: [
      eigenvectors[idx[0]!] as readonly [number, number, number],
      eigenvectors[idx[1]!] as readonly [number, number, number],
      eigenvectors[idx[2]!] as readonly [number, number, number],
    ],
  };
}

/**
 * Rotational constants A ≥ B ≥ C, in cm⁻¹ and GHz. Computed from
 * principal moments of inertia: B = h / (8π²·I·c) in cm⁻¹.
 *
 * Returns NaN for an axis whose moment of inertia is zero (linear
 * molecule along that axis — the rotational constant diverges).
 */
export function rotationalConstants(atoms: readonly Atom[]): {
  readonly A_cm1: number; readonly B_cm1: number; readonly C_cm1: number;
  readonly A_GHz: number; readonly B_GHz: number; readonly C_GHz: number;
} {
  const { Ia, Ib, Ic } = principalMomentsOfInertia(atoms);
  // h / (8π² c) in cm⁻¹ · amu · Å² units:
  //   B [cm⁻¹] = 16.857629205 / I [amu·Å²]   (NIST 2018)
  const FACTOR_CM1 = 16.857629205;
  const TO_GHZ = 29.9792458;   // 1 cm⁻¹ = 29.979 GHz
  const A = Ia > 1e-12 ? FACTOR_CM1 / Ia : Number.NaN;
  const B = Ib > 1e-12 ? FACTOR_CM1 / Ib : Number.NaN;
  const C = Ic > 1e-12 ? FACTOR_CM1 / Ic : Number.NaN;
  // A ≥ B ≥ C (smallest I gives largest rotational constant).
  return {
    A_cm1: A, B_cm1: B, C_cm1: C,
    A_GHz: A * TO_GHZ, B_GHz: B * TO_GHZ, C_GHz: C * TO_GHZ,
  };
}

/** Analytic 3×3 symmetric eigendecomposition via 6 Jacobi sweeps.
 *  Returns eigenvalues + column eigenvectors. */
function symmetric3x3Eig(
  xx: number, yy: number, zz: number,
  xy: number, xz: number, yz: number,
): { eigenvalues: [number, number, number]; eigenvectors: [readonly number[], readonly number[], readonly number[]] } {
  // Pack into 3×3 row-major.
  const A = [
    [xx, xy, xz],
    [xy, yy, yz],
    [xz, yz, zz],
  ];
  // Eigenvectors as columns of V, initialized to identity.
  const V = [[1,0,0],[0,1,0],[0,0,1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    // Find largest off-diagonal.
    let p = 0, q = 1, maxAbs = Math.abs(A[0]![1]!);
    if (Math.abs(A[0]![2]!) > maxAbs) { p = 0; q = 2; maxAbs = Math.abs(A[0]![2]!); }
    if (Math.abs(A[1]![2]!) > maxAbs) { p = 1; q = 2; maxAbs = Math.abs(A[1]![2]!); }
    if (maxAbs < 1e-14) break;
    const App = A[p]![p]!, Aqq = A[q]![q]!, Apq = A[p]![q]!;
    const tau = (Aqq - App) / (2 * Apq);
    const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;
    A[p]![p] = App - t * Apq;
    A[q]![q] = Aqq + t * Apq;
    A[p]![q] = 0; A[q]![p] = 0;
    for (let r = 0; r < 3; r++) {
      if (r === p || r === q) continue;
      const Arp = A[r]![p]!, Arq = A[r]![q]!;
      A[r]![p] = c * Arp - s * Arq;
      A[p]![r] = A[r]![p]!;
      A[r]![q] = s * Arp + c * Arq;
      A[q]![r] = A[r]![q]!;
    }
    for (let r = 0; r < 3; r++) {
      const Vrp = V[r]![p]!, Vrq = V[r]![q]!;
      V[r]![p] = c * Vrp - s * Vrq;
      V[r]![q] = s * Vrp + c * Vrq;
    }
  }
  return {
    eigenvalues: [A[0]![0]!, A[1]![1]!, A[2]![2]!],
    eigenvectors: [
      [V[0]![0]!, V[1]![0]!, V[2]![0]!] as const,
      [V[0]![1]!, V[1]![1]!, V[2]![1]!] as const,
      [V[0]![2]!, V[1]![2]!, V[2]![2]!] as const,
    ],
  };
}

/**
 * Root-mean-square deviation between two geometries WITHOUT alignment.
 *   RMSD = sqrt[ (1/N) · Σ_i |r1_i − r2_i|² ]
 *
 * Both arrays must have the same length and atom ordering. Returns
 * RMSD in Å.
 */
export function rmsd(atoms1: readonly Atom[], atoms2: readonly Atom[]): number {
  if (atoms1.length !== atoms2.length) {
    throw new Error(`rmsd: length mismatch ${atoms1.length} vs ${atoms2.length}`);
  }
  let sum = 0;
  for (let i = 0; i < atoms1.length; i++) {
    const a = atoms1[i]!.pos;
    const b = atoms2[i]!.pos;
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    sum += dx * dx + dy * dy + dz * dz;
  }
  return Math.sqrt(sum / atoms1.length);
}

/**
 * RMSD between two geometries AFTER optimal Kabsch alignment
 * (translation + rotation, no chirality flip). Both arrays must
 * have the same length and atom ordering. Returns:
 *   - rmsd (Å) after alignment
 *   - aligned: atoms2 after optimal rotation onto atoms1
 *
 * Kabsch algorithm: SVD of cross-covariance matrix gives optimal
 * rotation. Det check ensures we don't flip chirality.
 */
export function rmsdAligned(
  atoms1: readonly Atom[],
  atoms2: readonly Atom[],
): {
  readonly rmsd: number;
  readonly aligned: Atom[];
} {
  if (atoms1.length !== atoms2.length) {
    throw new Error(`rmsdAligned: length mismatch ${atoms1.length} vs ${atoms2.length}`);
  }
  const n = atoms1.length;
  // Step 1: center both geometries on their respective centroids.
  let c1x = 0, c1y = 0, c1z = 0;
  let c2x = 0, c2y = 0, c2z = 0;
  for (let i = 0; i < n; i++) {
    c1x += atoms1[i]!.pos[0]; c1y += atoms1[i]!.pos[1]; c1z += atoms1[i]!.pos[2];
    c2x += atoms2[i]!.pos[0]; c2y += atoms2[i]!.pos[1]; c2z += atoms2[i]!.pos[2];
  }
  c1x /= n; c1y /= n; c1z /= n;
  c2x /= n; c2y /= n; c2z /= n;
  // Step 2: build the cross-covariance matrix H = P2^T · P1 where P_i is centered.
  // Hij = Σ_k (atoms2[k] - c2)_i · (atoms1[k] - c1)_j   (3x3 row-major)
  const H = new Float64Array(9);
  for (let k = 0; k < n; k++) {
    const p2 = [atoms2[k]!.pos[0] - c2x, atoms2[k]!.pos[1] - c2y, atoms2[k]!.pos[2] - c2z];
    const p1 = [atoms1[k]!.pos[0] - c1x, atoms1[k]!.pos[1] - c1y, atoms1[k]!.pos[2] - c1z];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        H[i * 3 + j]! += p2[i]! * p1[j]!;
      }
    }
  }
  // Step 3: SVD H = U·Σ·V^T via the eigendecomposition of H^T·H (V) and
  // H·H^T (U). Then R = V·U^T.
  // For 3×3, this is feasible analytically. We'll do it by computing
  // V from the eigenvectors of H^T·H, then U = H·V·Σ⁻¹ on the
  // nonzero singular values.
  const HtH = matmul3T(H, H);   // H^T · H
  const eig = symmetric3x3Eig(HtH[0]!, HtH[4]!, HtH[8]!, HtH[1]!, HtH[2]!, HtH[5]!);
  // Singular values (sorted descending).
  const sigma2 = [eig.eigenvalues[0]!, eig.eigenvalues[1]!, eig.eigenvalues[2]!];
  const idx = [0, 1, 2].sort((a, b) => sigma2[b]! - sigma2[a]!);
  const V_cols = idx.map((i) => eig.eigenvectors[i]);
  // U_cols[i] = H · V_cols[i] / σ_i (skipping σ=0).
  const sigma = sigma2.map((s) => Math.sqrt(Math.max(s, 0)));
  const sigSorted = idx.map((i) => sigma[i]!);
  const U_cols: number[][] = [];
  for (let k = 0; k < 3; k++) {
    const v = V_cols[k]!;
    const s = sigSorted[k]!;
    if (s > 1e-10) {
      U_cols.push([
        H[0]! * v[0]! + H[1]! * v[1]! + H[2]! * v[2]!,
        H[3]! * v[0]! + H[4]! * v[1]! + H[5]! * v[2]!,
        H[6]! * v[0]! + H[7]! * v[1]! + H[8]! * v[2]!,
      ].map((x) => x / s));
    } else {
      // Degenerate singular value; orthonormalize against earlier U cols.
      // Use cross product to get a perpendicular vector.
      const u0 = U_cols[0] ?? [1, 0, 0];
      const u1 = U_cols[1] ?? [0, 1, 0];
      U_cols.push([
        u0[1]! * u1[2]! - u0[2]! * u1[1]!,
        u0[2]! * u1[0]! - u0[0]! * u1[2]!,
        u0[0]! * u1[1]! - u0[1]! * u1[0]!,
      ]);
    }
  }
  // Det check: R = V · diag(1, 1, det(U·V^T)) · U^T to avoid chirality flip.
  // Compute det(U) and det(V).
  const detU = det3(U_cols[0]!, U_cols[1]!, U_cols[2]!);
  const detV = det3(V_cols[0]! as number[], V_cols[1]! as number[], V_cols[2]! as number[]);
  const sign = detU * detV;
  // R = V · diag(1, 1, sign) · U^T
  const R = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) {
        const lambda = k === 2 ? sign : 1;
        s += V_cols[k]![i]! * lambda * U_cols[k]![j]!;
      }
      R[i * 3 + j] = s;
    }
  }
  // Apply: aligned = R · (atoms2 − c2) + c1.
  const aligned: Atom[] = atoms2.map((a) => {
    const x = a.pos[0] - c2x, y = a.pos[1] - c2y, z = a.pos[2] - c2z;
    return {
      ...a,
      pos: [
        R[0]! * x + R[1]! * y + R[2]! * z + c1x,
        R[3]! * x + R[4]! * y + R[5]! * z + c1y,
        R[6]! * x + R[7]! * y + R[8]! * z + c1z,
      ] as [number, number, number],
    };
  });
  return { rmsd: rmsd(atoms1, aligned), aligned };
}

function matmul3T(A: Float64Array, B: Float64Array): Float64Array {
  // Returns A^T · B (row-major).
  const out = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += A[k * 3 + i]! * B[k * 3 + j]!;
      out[i * 3 + j] = s;
    }
  }
  return out;
}

function det3(a: readonly number[], b: readonly number[], c: readonly number[]): number {
  return a[0]! * (b[1]! * c[2]! - b[2]! * c[1]!)
       - a[1]! * (b[0]! * c[2]! - b[2]! * c[0]!)
       + a[2]! * (b[0]! * c[1]! - b[1]! * c[0]!);
}

/**
 * Translate all atoms by a vector (in Å). Returns a new array;
 * original atoms unchanged.
 */
export function translateMolecule(
  atoms: readonly Atom[],
  vector: readonly [number, number, number],
): Atom[] {
  return atoms.map((a) => ({
    ...a,
    pos: [
      a.pos[0] + vector[0],
      a.pos[1] + vector[1],
      a.pos[2] + vector[2],
    ] as [number, number, number],
  }));
}

/**
 * Rotate all atoms by a 3×3 rotation matrix (row-major). Returns a
 * new array. Matrix should be orthogonal for a pure rotation (det=+1);
 * negative determinant would flip chirality.
 */
export function rotateMolecule(
  atoms: readonly Atom[],
  matrix: readonly number[],   // row-major 9-element 3×3
): Atom[] {
  if (matrix.length !== 9) {
    throw new Error(`rotateMolecule: matrix must be 9 elements (row-major 3×3), got ${matrix.length}`);
  }
  return atoms.map((a) => {
    const x = a.pos[0], y = a.pos[1], z = a.pos[2];
    return {
      ...a,
      pos: [
        matrix[0]! * x + matrix[1]! * y + matrix[2]! * z,
        matrix[3]! * x + matrix[4]! * y + matrix[5]! * z,
        matrix[6]! * x + matrix[7]! * y + matrix[8]! * z,
      ] as [number, number, number],
    };
  });
}

/**
 * Standard orientation: translate so center of mass is at origin,
 * then rotate so principal axes align with x, y, z (I_a along x,
 * I_b along y, I_c along z). Same convention as Gaussian's
 * "Standard orientation". Useful for comparing geometries.
 */
export function standardOrientation(atoms: readonly Atom[]): Atom[] {
  const com = centerOfMass(atoms);
  const centered = translateMolecule(atoms, [-com[0], -com[1], -com[2]]);
  const { axes } = principalMomentsOfInertia(centered);
  // Build rotation matrix R whose ROWS are the principal axes.
  // Then for each atom: r' = R · r maps from old coords to new where
  // axis I_a (col 0 of inertia eigenvectors) becomes the x-axis.
  const matrix = [
    axes[0]![0], axes[0]![1], axes[0]![2],
    axes[1]![0], axes[1]![1], axes[1]![2],
    axes[2]![0], axes[2]![1], axes[2]![2],
  ];
  return rotateMolecule(centered, matrix);
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
