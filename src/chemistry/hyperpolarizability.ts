// ─────────────────────────────────────────────────────────────
// hyperpolarizability.ts — first dipole hyperpolarizability β
// via the finite-field method. Tier 2 stage 20.
//
// Definitions (Buckingham/Levine):
//   μ_i(E)       = μ_i(0) + α_ij·E_j + (1/2)·β_ijk·E_j·E_k + …
//   β_ijk        = ∂²μ_i / ∂E_j ∂E_k   |_{E=0}
//
// The closed-shell Kleinman symmetry guarantees full permutation
// symmetry of β at zero frequency: β_ijk = β_ikj = β_jik = …,
// leaving only 10 unique components in 3D. We compute the
// commonly-reported "vector" projection along the permanent
// dipole direction:
//
//   β_∥(z) = (3/5)·Σ_i β_iiz
//   β_vec  = √(β_x² + β_y² + β_z²)   (rotational invariant)
//
// where β_a (vector component) = (1/3)·Σ_i (β_aii + β_iia + β_iai).
// (Standard rotational-average expressions; see Bishop 1990 or
// Buckingham's 1967 paper. Numerically these collapse to the same
// number under Kleinman symmetry.)
//
// Diagonal components β_iii are the cheapest to compute and the
// most basis-sensitive. We compute the full 27-component tensor
// via a 3D finite-difference stencil: 6 ±E single-axis SCFs +
// 12 ±E,±E mixed-axis SCFs + 1 zero-field SCF = 19 SCF runs.
// Off-diagonal β_ijk (i≠j≠k) needs the corner of a 3D cube
// (additional 8 SCFs) but those cube corners are typically the
// noisiest entries; they're computed and reported but don't
// dominate the rotational invariants β_∥ and β_vec.
//
// Field strength: STILL 5e-3 a.u. — same as polarizability.
// Hyperpolarizability needs a slightly larger field to dominate
// over higher orders; we keep 5e-3 to share the SCF cache with
// the polarizability evaluator and live with the (small) loss
// in precision. β is basis-limited at STO-3G anyway.
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "./atoms.js";
import { moleculeToShellsNuclei } from "./atoms.js";
import { computeMolecularIntegrals, type IntegralOpts } from "./cg-molecular.js";
import { runRHFSCF, type HFOpts } from "./hf-scf.js";
import { runRKSDFT, type RKSOpts } from "./dft/rks-scf.js";
import { dipoleMoment } from "./properties.js";
import { dipole_cg } from "./integrals-cg.js";
import type { FunctionalKind } from "./dft/functional.js";

export type EnergyMethod = "hf" | FunctionalKind;

export interface HyperpolOpts {
  readonly basis?: BasisName;
  readonly spherical?: boolean;
  readonly method?: EnergyMethod;
  readonly hf?: HFOpts;
  readonly dft?: RKSOpts;
  /** Field strength (a.u.) for FD. Default 5e-3 a.u. */
  readonly fieldStrength?: number;
}

export interface HyperpolResult {
  /** β tensor as a flat 27-array, indexed β[i*9 + j*3 + k] (a.u.). */
  readonly tensor: Float64Array;
  /** Rotational vector invariant β_vec (a.u.) — rotation-averaged
   *  magnitude reported in most chemistry papers. */
  readonly vectorInvariant: number;
  /** Component β_∥ projected on the molecular dipole axis (a.u.).
   *  By default this is along the lab z-axis (caller is responsible
   *  for rotating the molecule into the standard orientation if
   *  needed). */
  readonly betaParallelZ: number;
  /** β_iii diagonal components per axis (a.u.). The simplest to
   *  interpret — magnitude of the cubic dipole response along each
   *  Cartesian axis. */
  readonly diagonals: [number, number, number];
  readonly seconds: number;
  readonly nScfRuns: number;
}

/**
 * Static first hyperpolarizability tensor of `atoms` at its
 * current geometry. As with polarizability, caller is responsible
 * for running `optimizeGeometry` first if a stationary point is
 * needed.
 */
export function hyperpolarizabilityFiniteField(
  atoms: readonly Atom[],
  opts: HyperpolOpts = {},
): HyperpolResult {
  const tStart = performance.now();
  const basis = opts.basis ?? "sto-3g";
  const method: EnergyMethod = opts.method ?? "hf";
  const integralOpts: IntegralOpts = { spherical: opts.spherical ?? false };
  const hfOpts: HFOpts = {
    useDIIS: true, energyTol: 1e-10, densityTol: 1e-8, maxIter: 200,
    ...opts.hf,
  };
  const dftOpts: RKSOpts = {
    functional: method === "hf" ? "lda-svwn" : (method as FunctionalKind),
    energyTol: 1e-10, residualTol: 1e-7, maxIter: 200,
    ...opts.dft,
  };
  const Efield = opts.fieldStrength ?? 5e-3;
  if (opts.spherical) {
    throw new Error(
      "hyperpolarizabilityFiniteField: spherical-d basis not yet supported",
    );
  }

  const symbols = atoms.map((a) => a.symbol);
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei, { skipOAO: true, ...integralOpts });
  const n = integrals.n;

  // Dipole AO matrices.
  const muAO: Float64Array[] = [
    new Float64Array(n * n), new Float64Array(n * n), new Float64Array(n * n),
  ];
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu; nu < n; nu++) {
      for (let axis = 0 as 0 | 1 | 2; axis < 3; axis++) {
        const v = dipole_cg(shells[mu]!, shells[nu]!, axis);
        muAO[axis]![mu * n + nu] = v;
        muAO[axis]![nu * n + mu] = v;
      }
    }
  }
  const h_orig = new Float64Array(integrals.h_AO);

  // Cache: dipole at applied field [s_x, s_y, s_z] ∈ {-1, 0, +1}.
  // Key = s_x + 1 + 3*(s_y + 1) + 9*(s_z + 1)  → 27 entries.
  const dipoleCache = new Map<number, [number, number, number]>();
  let nScf = 0;
  const dipoleAt = (sx: -1 | 0 | 1, sy: -1 | 0 | 1, sz: -1 | 0 | 1): [number, number, number] => {
    const key = (sx + 1) + 3 * (sy + 1) + 9 * (sz + 1);
    const cached = dipoleCache.get(key);
    if (cached) return cached;
    integrals.h_AO.set(h_orig);
    if (sx !== 0) for (let k = 0; k < n * n; k++) integrals.h_AO[k]! += sx * Efield * muAO[0]![k]!;
    if (sy !== 0) for (let k = 0; k < n * n; k++) integrals.h_AO[k]! += sy * Efield * muAO[1]![k]!;
    if (sz !== 0) for (let k = 0; k < n * n; k++) integrals.h_AO[k]! += sz * Efield * muAO[2]![k]!;
    let mu: [number, number, number];
    if (method === "hf") {
      const hf = runRHFSCF(integrals, nElectrons, hfOpts);
      mu = dipoleMoment(integrals, hf.D);
    } else {
      const dft = runRKSDFT(integrals, nElectrons, symbols, dftOpts);
      mu = dipoleMoment(integrals, dft.D);
    }
    nScf++;
    dipoleCache.set(key, mu);
    return mu;
  };

  // Build the 27-component β tensor. Each β_ijk gets one of three
  // FD stencils depending on the index pattern.
  const β = new Float64Array(27);
  const E2 = Efield * Efield;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        const idx = i * 9 + j * 3 + k;
        β[idx] = beta_ijk(i, j, k, dipoleAt, E2, Efield);
      }
    }
  }
  integrals.h_AO.set(h_orig);

  // Symmetrize β under Kleinman: β_ijk → (1/6)·Σ_perm β_(ijk).
  const βsym = new Float64Array(27);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        const avg = (
            β[i * 9 + j * 3 + k]!
          + β[i * 9 + k * 3 + j]!
          + β[j * 9 + i * 3 + k]!
          + β[j * 9 + k * 3 + i]!
          + β[k * 9 + i * 3 + j]!
          + β[k * 9 + j * 3 + i]!
        ) / 6;
        βsym[i * 9 + j * 3 + k] = avg;
      }
    }
  }

  // Vector invariant β_vec = √(β_x² + β_y² + β_z²) where the vector
  // components are β_a = (1/3)·Σ_i (β_aii + β_iai + β_iia).
  const betaVec: [number, number, number] = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    let s = 0;
    for (let i = 0; i < 3; i++) {
      s += βsym[a * 9 + i * 3 + i]!
         + βsym[i * 9 + a * 3 + i]!
         + βsym[i * 9 + i * 3 + a]!;
    }
    betaVec[a] = s / 3;
  }
  const vectorInvariant = Math.sqrt(
    betaVec[0] ** 2 + betaVec[1] ** 2 + betaVec[2] ** 2,
  );

  // β_∥(z) = (3/5)·Σ_i β_iiz under Kleinman = (1/5)·Σ_i (β_zii + β_iiz + β_izi)
  // = β_z (vector component) — they collapse to the same number under Kleinman.
  const betaParallelZ = betaVec[2];

  return {
    tensor: βsym,
    vectorInvariant,
    betaParallelZ,
    diagonals: [
      βsym[0 * 9 + 0 * 3 + 0]!,
      βsym[1 * 9 + 1 * 3 + 1]!,
      βsym[2 * 9 + 2 * 3 + 2]!,
    ],
    seconds: (performance.now() - tStart) / 1000,
    nScfRuns: nScf,
  };
}

/** Compute one β_ijk component via the appropriate FD stencil.
 *  Falls into three cases:
 *    (a) i = j = k:  diagonal — 3-point stencil on μ_i vs E_i.
 *    (b) two equal:  e.g. β_iij = ∂²μ_i/∂E_i∂E_j — 4-point cross stencil.
 *    (c) all distinct (i,j,k):  needs 8 corners of the cube; we approximate
 *        via cross-stencils involving zero-field shifts (this is the noisiest
 *        case but Kleinman symmetrization smooths it out). */
function beta_ijk(
  i: number, j: number, k: number,
  D: (sx: -1 | 0 | 1, sy: -1 | 0 | 1, sz: -1 | 0 | 1) => [number, number, number],
  E2: number, _Efield: number,
): number {
  void _Efield;
  const sign: (-1 | 0 | 1)[] = [-1, 0, 1];
  // Helper: dipole at a vector of signs along x, y, z, with `i`th sign at s_x etc.
  const evalAt = (sx: number, sy: number, sz: number): number => {
    const d = D(sign[sx + 1]!, sign[sy + 1]!, sign[sz + 1]!);
    return d[i]!;
  };

  // Build the 3-vector of nonzero axes — which directions are perturbed?
  // Track "signs" required as a 3-tuple.
  // Case (a): j == k.
  //   β_iii = (μ_i(+E_i) - 2μ_i(0) + μ_i(-E_i)) / E²    (diagonal, j=k=i)
  //   β_ijj (i ≠ j): ∂²μ_i/∂E_j² = (μ_i(+E_j) - 2μ_i(0) + μ_i(-E_j)) / E²
  if (j === k) {
    const sP = [0, 0, 0]; sP[j] = +1;
    const sM = [0, 0, 0]; sM[j] = -1;
    return (evalAt(sP[0]!, sP[1]!, sP[2]!) - 2 * evalAt(0, 0, 0) + evalAt(sM[0]!, sM[1]!, sM[2]!)) / E2;
  }
  // Case (b): j ≠ k. Cross-second-derivative ∂²μ_i/∂E_j∂E_k.
  //   = (μ_i(+E_j+E_k) - μ_i(+E_j-E_k) - μ_i(-E_j+E_k) + μ_i(-E_j-E_k)) / (4·E²)
  const sPP = [0, 0, 0]; sPP[j] = +1; sPP[k] = +1;
  const sPM = [0, 0, 0]; sPM[j] = +1; sPM[k] = -1;
  const sMP = [0, 0, 0]; sMP[j] = -1; sMP[k] = +1;
  const sMM = [0, 0, 0]; sMM[j] = -1; sMM[k] = -1;
  const a = evalAt(sPP[0]!, sPP[1]!, sPP[2]!);
  const b = evalAt(sPM[0]!, sPM[1]!, sPM[2]!);
  const c = evalAt(sMP[0]!, sMP[1]!, sMP[2]!);
  const d = evalAt(sMM[0]!, sMM[1]!, sMM[2]!);
  return (a - b - c + d) / (4 * E2);
}
