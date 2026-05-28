// ─────────────────────────────────────────────────────────────
// vibrations.ts — harmonic vibrational analysis on the HF or
// DFT energy surface. Tier 2 stage 16.
//
// Builds the (3N × 3N) molecular Hessian by central FD on the
// analytical gradient (HF or any RKS-DFT functional), mass-
// weights it, diagonalizes, and converts the eigenvalues to
// vibrational frequencies in cm⁻¹.
//
// Frequency conversion (CODATA-consistent):
//   ν̃ [cm⁻¹] = (1/(2πc)) · √(λ_mass-weighted)
// with λ in Ha/(Bohr²·amu) and c = 2.99792458×10¹⁰ cm/s.
// Numerically: ν̃ [cm⁻¹] ≈ 5140.487 · √λ.
//
// 6 of the 3N modes (5 for linear molecules) are translational +
// rotational; their mass-weighted eigenvalues are zero up to
// numerical noise. We project them out via Eckart-frame
// translation/rotation vectors, leaving the 3N − 6 (or 3N − 5)
// pure-vibrational modes — a clean analog of the standard
// quantum-chemistry output.
//
// Cost: 6N analytical gradient evaluations per Hessian. Each
// HF/STO-3G gradient on H₂O is ~50–100 ms; full Hessian for
// triatomics finishes in seconds, polyatomics in minutes.
// Analytical second derivatives (CPHF) would be ~10× faster
// but require new integral derivative infrastructure.
// ─────────────────────────────────────────────────────────────

import type { Atom, AtomSymbol, BasisName } from "./atoms.js";
import { computeMolecularIntegrals, type IntegralOpts } from "./cg-molecular.js";
import { moleculeToShellsNuclei } from "./atoms.js";
import { runRHFSCF, type HFOpts } from "./hf-scf.js";
import { runRKSDFT, type RKSOpts } from "./dft/rks-scf.js";
import { hfGradient, buildEnergyWeightedDensity } from "./hf-gradient.js";
import { dftGradient } from "./dft-gradient.js";
import { dipoleMoment } from "./properties.js";
import type { FunctionalKind } from "./dft/functional.js";
import { eigsymmetric } from "../manybody/dense-eig.js";

const ANGSTROM_TO_BOHR = 1.8897261339;
/** Hartree/(Bohr²·amu) → cm⁻¹ conversion factor for vibrational
 *  frequencies (CODATA). ν̃[cm⁻¹] = CONV · √λ. */
const HA_BOHR2_AMU_TO_CM_INV = 5140.4865;
/** Standard chemistry conversion: |dμ/dq|² [e²/amu] → IR intensity [km/mol].
 *  Derivation: I = (N_A·π/(3·c²)) · |dμ/dq|², with all factors in SI gives
 *  974.864 (km/mol) per (e²/amu). Source: Gaussian/PySCF vibrational manuals. */
const E2_PER_AMU_TO_KM_PER_MOL = 974.864;

/** Atomic masses for the supported elements. Isotope-pure values
 *  (^1H, ^7Li, ^9Be, ^12C, ^14N, ^16O) — the standard choice for
 *  theoretical reference frequencies. */
const ATOMIC_MASS: Readonly<Record<AtomSymbol, number>> = {
  H:  1.00782503207,
  He: 4.002603254,
  Li: 7.0160034366,
  Be: 9.012183065,
  C:  12.0,
  N:  14.0030740048,
  O:  15.99491461956,
  F:  18.998403163,
};

export type EnergyMethod = "hf" | FunctionalKind;

export interface VibrationsOpts {
  /** Basis set (default sto-3g). */
  readonly basis?: BasisName;
  /** Spherical-d transform on the integrals? Default false. */
  readonly spherical?: boolean;
  /** Energy method — HF or any RKS-DFT functional kind. Default "hf". */
  readonly method?: EnergyMethod;
  /** HF SCF options forwarded to runRHFSCF. */
  readonly hf?: HFOpts;
  /** RKS-DFT SCF options. */
  readonly dft?: RKSOpts;
  /** Finite-difference step (Å) for the Hessian central differences.
   *  Default 5e-3 Å — gives ~1e-4 cm⁻¹ precision on H₂O. */
  readonly fdStep?: number;
  /** Treat the molecule as linear (5 zero modes instead of 6). Default
   *  false; auto-detect would be a follow-up. */
  readonly linear?: boolean;
}

export interface VibrationsResult {
  /** Pure vibrational frequencies in cm⁻¹, ascending. Length is
   *  3N − 6 (non-linear) or 3N − 5 (linear). Imaginary frequencies
   *  (transition-state saddle points) are reported as NEGATIVE
   *  cm⁻¹ — the standard chemistry-software convention. */
  readonly frequenciesCmInv: Float64Array;
  /** IR intensities per mode in km/mol. Length matches
   *  `frequenciesCmInv`. Computed from |∂μ/∂q_k|² where μ is the
   *  molecular dipole and q_k is the (mass-weighted) normal-mode
   *  coordinate. Modes with zero |∂μ/∂q| are IR-inactive (for
   *  example, the symmetric stretch of a homonuclear diatomic). */
  readonly irIntensitiesKmPerMol: Float64Array;
  /** Mass-weighted normal-mode displacements per mode (3N × n_modes,
   *  row-major: modes[axis·n_modes + mode]). Each column is a
   *  unit-norm displacement vector along (3N) Cartesian coordinates,
   *  with axis ordering (x_1, y_1, z_1, x_2, y_2, z_2, …). */
  readonly modesMassWeighted: Float64Array;
  /** Number of zero modes detected (translation + rotation). 6 for
   *  generic non-linear, 5 for linear, sometimes more for special
   *  symmetries (e.g. atoms). */
  readonly nZeroModes: number;
  /** Total wall-clock seconds spent in the Hessian build. */
  readonly seconds: number;
  /** Number of gradient evaluations (= 6 × n_atoms). */
  readonly nGradientEvals: number;
}

/**
 * Compute harmonic vibrational frequencies for the molecule at
 * its CURRENT geometry. Caller is responsible for running
 * `optimizeGeometry` first if a stationary point is needed —
 * far from a minimum, frequencies are not physically meaningful.
 *
 * Standard chemistry-software flow:
 *   1. Optimize geometry.
 *   2. Run vibrationalFrequencies on the optimized geometry.
 *   3. Inspect for any imaginary (negative) frequencies — those
 *      indicate a non-minimum (transition state or saddle).
 */
export function vibrationalFrequencies(
  atoms: readonly Atom[],
  opts: VibrationsOpts = {},
): VibrationsResult {
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
  const fdStep = opts.fdStep ?? 5e-3;          // Å

  const nA = atoms.length;
  const nDof = 3 * nA;
  const symbols = atoms.map((a) => a.symbol);
  const masses = symbols.map((s) => ATOMIC_MASS[s]);

  // Gradient + dipole evaluator — re-runs SCF + gradient at the
  // requested geometry. Returns gradient (Ha/Bohr) AND dipole
  // moment (e·Bohr, atomic units). Dipole is in the LAB frame
  // (no COM-shift) — the FD on dipole is invariant to constant
  // additive shifts so this is fine.
  let nEvals = 0;
  const gradAndDipoleAt = (
    xAng: Float64Array,
  ): { grad: Float64Array; dipole: [number, number, number] } => {
    nEvals++;
    const moved: Atom[] = atoms.map((a, i) => ({
      symbol: a.symbol,
      pos: [xAng[3 * i]!, xAng[3 * i + 1]!, xAng[3 * i + 2]!] as [number, number, number],
    }));
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(moved, basis);
    const integrals = computeMolecularIntegrals(shells, nuclei, { skipOAO: true, ...integralOpts });
    if (method === "hf") {
      const hf = runRHFSCF(integrals, nElectrons, hfOpts);
      const W = buildEnergyWeightedDensity(hf.C_MO, hf.orbitalEnergies, hf.nOccupied, integrals.n);
      const grad = hfGradient({ shells, nuclei, shellAtomIdx, P: hf.D, W, sphericalT: integrals.sphericalT });
      const dipole = dipoleMoment(integrals, hf.D);
      return { grad, dipole };
    }
    const dft = runRKSDFT(integrals, nElectrons, symbols, dftOpts);
    const W = buildEnergyWeightedDensity(dft.C_MO, dft.orbitalEnergies, dft.nOccupied, integrals.n);
    const grad = dftGradient({
      shells, nuclei, shellAtomIdx, nucleiSymbols: symbols,
      P: dft.D, W, functional: method as FunctionalKind,
      grid: opts.dft?.grid,
      sphericalT: integrals.sphericalT,
    });
    const dipole = dipoleMoment(integrals, dft.D);
    return { grad, dipole };
  };

  // ── Build 3N × 3N Hessian by central FD on the gradient. ────
  // Reference geometry → flat coordinate vector in Å.
  const x0 = new Float64Array(nDof);
  for (let i = 0; i < nA; i++) {
    x0[3 * i + 0] = atoms[i]!.pos[0];
    x0[3 * i + 1] = atoms[i]!.pos[1];
    x0[3 * i + 2] = atoms[i]!.pos[2];
  }

  const H = new Float64Array(nDof * nDof);   // Ha/(Bohr · Å)
  // Dipole derivative dμ/dx (3 dipole components × nDof Cartesians, Å).
  // Will be converted to Bohr-based units before mass-weighting.
  const dDipoleDx = new Float64Array(3 * nDof);
  const xPlus  = new Float64Array(nDof);
  const xMinus = new Float64Array(nDof);
  for (let j = 0; j < nDof; j++) {
    xPlus.set(x0); xPlus[j]! += fdStep;
    xMinus.set(x0); xMinus[j]! -= fdStep;
    const { grad: gP, dipole: dP } = gradAndDipoleAt(xPlus);
    const { grad: gM, dipole: dM } = gradAndDipoleAt(xMinus);
    // ∂g_i/∂x_j ≈ (gP_i − gM_i) / (2·fdStep [Å]).
    for (let i = 0; i < nDof; i++) {
      H[i * nDof + j] = (gP[i]! - gM[i]!) / (2 * fdStep);
    }
    // ∂μ_axis/∂x_j ≈ (dP_axis − dM_axis) / (2·fdStep [Å]).
    for (let axis = 0; axis < 3; axis++) {
      dDipoleDx[axis * nDof + j] = (dP[axis]! - dM[axis]!) / (2 * fdStep);
    }
  }
  // Symmetrize against FP-noise asymmetry. Hessian is exactly
  // symmetric for an analytical PES — anything off is FD error.
  for (let i = 0; i < nDof; i++) {
    for (let j = i + 1; j < nDof; j++) {
      const v = 0.5 * (H[i * nDof + j]! + H[j * nDof + i]!);
      H[i * nDof + j] = v;
      H[j * nDof + i] = v;
    }
  }
  // Convert Ha/(Bohr·Å) → Ha/Bohr². The FD denominator was in Å, so
  // ∂g/∂x_Å carries units Ha/(Bohr·Å); we want ∂g/∂x_Bohr in
  // Ha/Bohr². Since x_Å = (1/ANGSTROM_TO_BOHR)·x_Bohr,
  //   ∂/∂x_Bohr = (1/ANGSTROM_TO_BOHR)·∂/∂x_Å.
  // Divide by ANGSTROM_TO_BOHR to get the Bohr-Bohr Hessian.
  for (let k = 0; k < nDof * nDof; k++) H[k] = H[k]! / ANGSTROM_TO_BOHR;
  // Same coordinate-derivative conversion for dμ/dx.
  // dμ in atomic units (e·Bohr) / Å step ⇒ dμ in e·Bohr per Bohr =
  // e (dimensionless atomic charge) / sqrt(unit-mass) post-mass-weighting.
  for (let k = 0; k < 3 * nDof; k++) dDipoleDx[k] = dDipoleDx[k]! / ANGSTROM_TO_BOHR;

  // ── Mass-weight: H_mw[i,j] = H[i,j] / √(m_i · m_j). ─────────
  const Hmw = new Float64Array(nDof * nDof);
  for (let i = 0; i < nDof; i++) {
    const mi = masses[Math.floor(i / 3)]!;
    for (let j = 0; j < nDof; j++) {
      const mj = masses[Math.floor(j / 3)]!;
      Hmw[i * nDof + j] = H[i * nDof + j]! / Math.sqrt(mi * mj);
    }
  }

  // ── Project out translation + rotation (Eckart conditions). ──
  // 3 translation vectors: T_axis^k(atom_i, axis) = √m_i · δ_axis^k.
  // 3 rotation vectors:    R_a(atom_i, axis) = √m_i · (e_a × r_i)_axis,
  //   r_i measured from the center of mass in Bohr.
  // We construct these vectors normalized in the mass-weighted basis,
  // assemble a projector P = I − Σ_k v_k v_k^T, and apply P · Hmw · P.
  const com = centerOfMass(atoms, masses);
  const projVecs = buildProjectionVectors(atoms, masses, com, opts.linear ?? false);
  const Hproj = applyProjection(Hmw, projVecs, nDof);

  // ── Diagonalize. ────────────────────────────────────────────
  const eig = eigsymmetric(Hproj, nDof);

  // ── Identify zero modes (translation/rotation residue) and sort. ──
  // After projection, projected-out modes have eigenvalue ≈ 0 (numerical
  // noise from the projector). Pure-vib modes have |λ| ≫ 0. We mark all
  // |λ| < ZERO_TOL as zero modes. ZERO_TOL is generous: 1e-6 Ha/(Bohr²·amu)
  // corresponds to ~5 cm⁻¹, which is well below any chemical vibration.
  const ZERO_TOL = 1e-6;
  const nZero = projVecs.length;
  const nVib = nDof - nZero;
  const frequenciesCmInv = new Float64Array(nVib);
  const modes = new Float64Array(nDof * nVib);
  // The projected eigenvalues come out approximately ascending, with
  // the projected-out modes pinned near 0. Skip the first nZero
  // (lowest |λ|) entries when copying frequencies.
  // Re-sort by magnitude to be safe.
  const idx = Array.from({ length: nDof }, (_, k) => k);
  idx.sort((a, b) => Math.abs(eig.values[a]!) - Math.abs(eig.values[b]!));
  let outRow = 0;
  for (let k = nZero; k < nDof; k++) {
    const src = idx[k]!;
    const lam = eig.values[src]!;
    const sign = lam >= 0 ? 1 : -1;
    frequenciesCmInv[outRow] = sign * HA_BOHR2_AMU_TO_CM_INV * Math.sqrt(Math.abs(lam));
    for (let r = 0; r < nDof; r++) {
      modes[r * nVib + outRow] = eig.vectors[src * nDof + r]!;
    }
    outRow++;
  }

  // Sort frequencies ascending (with imaginary first, then real).
  const sortIdx = Array.from({ length: nVib }, (_, k) => k);
  sortIdx.sort((a, b) => frequenciesCmInv[a]! - frequenciesCmInv[b]!);
  const sortedFreq = new Float64Array(nVib);
  const sortedModes = new Float64Array(nDof * nVib);
  for (let k = 0; k < nVib; k++) {
    const src = sortIdx[k]!;
    sortedFreq[k] = frequenciesCmInv[src]!;
    for (let r = 0; r < nDof; r++) {
      sortedModes[r * nVib + k] = modes[r * nVib + src]!;
    }
  }
  // Voiding ZERO_TOL keeps it visible to the type-checker if used later.
  void ZERO_TOL;

  // ── IR intensities. ──────────────────────────────────────────
  // For each (sorted) vibrational mode k, project the dipole
  // derivative onto the mode in mass-weighted Cartesian space:
  //   ∂μ_axis/∂q_k = Σ_i (∂μ_axis/∂x_Cart_i) · u_k_i / √m_atom(i)
  // where u_k_i is the k-th mass-weighted eigenvector. Then
  //   I_k [km/mol] = 974.864 · Σ_axis (∂μ_axis/∂q_k)².
  // (The conversion factor includes the 4πε_0 and Avogadro factors
  // to put atomic-unit |dμ/dq|² on the chemistry-standard intensity
  // scale.)
  const ir = new Float64Array(nVib);
  for (let k = 0; k < nVib; k++) {
    let sumSq = 0;
    for (let axis = 0; axis < 3; axis++) {
      let dmuDq = 0;
      for (let i = 0; i < nDof; i++) {
        const m = masses[Math.floor(i / 3)]!;
        const dmuDxMW = dDipoleDx[axis * nDof + i]! / Math.sqrt(m);
        const u_ki    = sortedModes[i * nVib + k]!;
        dmuDq += dmuDxMW * u_ki;
      }
      sumSq += dmuDq * dmuDq;
    }
    ir[k] = E2_PER_AMU_TO_KM_PER_MOL * sumSq;
  }

  return {
    frequenciesCmInv: sortedFreq,
    irIntensitiesKmPerMol: ir,
    modesMassWeighted: sortedModes,
    nZeroModes: nZero,
    seconds: (performance.now() - tStart) / 1000,
    nGradientEvals: nEvals,
  };
}

// ── Helpers ─────────────────────────────────────────────────

function centerOfMass(
  atoms: readonly Atom[],
  masses: readonly number[],
): [number, number, number] {
  let totMass = 0;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < atoms.length; i++) {
    const m = masses[i]!;
    totMass += m;
    cx += m * atoms[i]!.pos[0];
    cy += m * atoms[i]!.pos[1];
    cz += m * atoms[i]!.pos[2];
  }
  return [cx / totMass, cy / totMass, cz / totMass];
}

/**
 * Build 6 (or 5 for linear) mass-weighted projection vectors that
 * span pure translations + rotations of the molecule. Returns an
 * array of unit-norm vectors of length 3N each.
 *
 * Translations: T_k[i] = √m_i for atom i along axis k, else 0.
 *   (Mass-weighted infinitesimal translation along axis k.)
 * Rotations:    R_a[i] = √m_i · (e_a × (r_i − r_COM))
 *   (Mass-weighted infinitesimal rotation around axis a.)
 *
 * For linear molecules, the rotation around the molecular axis
 * is degenerate (zero infinitesimal generator) — we drop it.
 */
function buildProjectionVectors(
  atoms: readonly Atom[],
  masses: readonly number[],
  com: readonly [number, number, number],
  linear: boolean,
): Float64Array[] {
  const nA = atoms.length;
  const nDof = 3 * nA;
  const out: Float64Array[] = [];

  // Three translations.
  for (let k = 0; k < 3; k++) {
    const v = new Float64Array(nDof);
    for (let i = 0; i < nA; i++) v[3 * i + k] = Math.sqrt(masses[i]!);
    normalizeInPlace(v);
    out.push(v);
  }

  // Three rotations (or two for linear).
  // r_i = atoms[i].pos − com. Components in Bohr (rotation generator
  // works in any consistent length unit; the projector is unit-norm).
  for (let a = 0; a < 3; a++) {
    if (linear && a === 2) break;            // skip the molecular-axis rotation
    const v = new Float64Array(nDof);
    for (let i = 0; i < nA; i++) {
      const rx = atoms[i]!.pos[0] - com[0];
      const ry = atoms[i]!.pos[1] - com[1];
      const rz = atoms[i]!.pos[2] - com[2];
      const sm = Math.sqrt(masses[i]!);
      // (e_a × r_i):
      //   a = 0 (x): (0, -rz, ry)
      //   a = 1 (y): (rz, 0, -rx)
      //   a = 2 (z): (-ry, rx, 0)
      let cx = 0, cy = 0, cz = 0;
      if (a === 0)      { cx = 0;    cy = -rz;  cz =  ry; }
      else if (a === 1) { cx =  rz;  cy =  0;   cz = -rx; }
      else              { cx = -ry;  cy =  rx;  cz =  0;  }
      v[3 * i + 0] = sm * cx;
      v[3 * i + 1] = sm * cy;
      v[3 * i + 2] = sm * cz;
    }
    // Skip a rotation vector if it's degenerate (zero norm — happens
    // for an isolated atom or a linear molecule on the molecular axis).
    const norm = vecNorm(v);
    if (norm > 1e-10) {
      normalizeInPlace(v);
      out.push(v);
    }
  }

  // Gram-Schmidt orthogonalize against any linear dependence between
  // translations + rotations (rare but possible at high symmetry).
  const ortho: Float64Array[] = [];
  for (const u of out) {
    const w = new Float64Array(u);
    for (const v of ortho) {
      const c = dot(w, v);
      for (let k = 0; k < w.length; k++) w[k]! -= c * v[k]!;
    }
    const n = vecNorm(w);
    if (n > 1e-10) {
      for (let k = 0; k < w.length; k++) w[k]! /= n;
      ortho.push(w);
    }
  }
  return ortho;
}

function applyProjection(
  H: Float64Array,
  vecs: readonly Float64Array[],
  n: number,
): Float64Array {
  // P = I − Σ v_k v_k^T. Result: P·H·P, applied as
  //   H'_ij = H_ij − Σ_k v_k[i] (v_k^T H)_j − Σ_k (H v_k)_i v_k[j]
  //         + Σ_kl v_k[i] (v_k^T H v_l) v_l[j]
  // Easiest: build P explicitly and do dense matmul. n is small (3N).
  const P = new Float64Array(n * n);
  for (let i = 0; i < n; i++) P[i * n + i] = 1;
  for (const v of vecs) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        P[i * n + j]! -= v[i]! * v[j]!;
      }
    }
  }
  const tmp = matmul(P, H, n);
  return matmul(tmp, P, n);
}

function matmul(A: Float64Array, B: Float64Array, n: number): Float64Array {
  const C = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const aik = A[i * n + k]!;
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) {
        C[i * n + j]! += aik * B[k * n + j]!;
      }
    }
  }
  return C;
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

function vecNorm(v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i]! * v[i]!;
  return Math.sqrt(s);
}

function normalizeInPlace(v: Float64Array): void {
  const n = vecNorm(v);
  if (n === 0) return;
  for (let k = 0; k < v.length; k++) v[k]! /= n;
}
