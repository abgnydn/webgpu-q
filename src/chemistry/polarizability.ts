// ─────────────────────────────────────────────────────────────
// polarizability.ts — static dipole polarizability via the
// finite-field method. Tier 2 stage 17.
//
// Static dipole polarizability tensor:
//   α_ij = − ∂²E/∂E_i∂E_j
//        = −(∂μ_i/∂E_j)             (Hellmann-Feynman, variational)
//
// Method: perturb the core Hamiltonian
//   h_AO_perturbed[μν] = h_AO[μν] − E_axis · ⟨χ_μ | r_axis | χ_ν⟩
// then run SCF and compute the dipole moment. Central FD on the
// dipole vs. small applied fields ±E along each axis gives α_ij.
//
// 6 SCF runs per polarizability (one per ±axis × ±field).
// Default field strength E = 5e-3 a.u. — small enough that
// non-linear (hyperpolarizability) contamination stays below
// quadrature error, large enough that FD precision dominates.
// FD error scales as h² for central differences; with h = 5e-3
// the truncation error is ~1e-5 of α, well below STO-3G basis
// error (~10–20% on properties).
//
// Reports both the full 3×3 tensor and:
//   • α_iso  = (α_xx + α_yy + α_zz)/3                  (mean)
//   • Δα     = √[(α_xx−α_yy)² + (α_yy−α_zz)² + (α_zz−α_xx)²) / 2
//                 + 3·(α_xy² + α_yz² + α_xz²)]          (anisotropy)
// (The anisotropy formula matches the standard chemistry
// definition; isotropic + anisotropic together fully characterize
// the field-induced polarization for a closed-shell molecule.)
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

export interface PolarizabilityOpts {
  /** Basis set (default sto-3g). */
  readonly basis?: BasisName;
  /** Spherical-d transform on integrals? Default false. */
  readonly spherical?: boolean;
  /** Energy method — HF or any RKS-DFT functional kind. Default "hf". */
  readonly method?: EnergyMethod;
  /** HF SCF options. */
  readonly hf?: HFOpts;
  /** RKS-DFT SCF options. */
  readonly dft?: RKSOpts;
  /** Field strength magnitude (a.u.) for the central FD step.
   *  Default 5e-3 a.u. — balances FD truncation vs. nonlinear
   *  hyperpolarizability contamination. */
  readonly fieldStrength?: number;
}

export interface PolarizabilityResult {
  /** Full 3×3 polarizability tensor, row-major (a.u. = e²·a₀²/E_h).
   *  Multiply by 0.14818 to convert to Å³ if desired. */
  readonly tensor: Float64Array;
  /** Isotropic average α_iso = tr(α)/3 (a.u.). */
  readonly isotropic: number;
  /** Anisotropy Δα (a.u.). Zero for isotropic systems (atoms,
   *  spherical-symmetric). */
  readonly anisotropy: number;
  /** Three principal components α_a ≤ α_b ≤ α_c (a.u.) — eigenvalues
   *  of the symmetric tensor. */
  readonly principalComponents: Float64Array;
  /** Wall-clock seconds. */
  readonly seconds: number;
  /** Number of SCF runs (= 6, for ±x, ±y, ±z). */
  readonly nScfRuns: number;
}

/** Convert a.u. of polarizability to Å³. (1 a.u. = 0.148184711 Å³.) */
export const AU_TO_ANGSTROM3 = 0.148184711;

/**
 * Compute the static dipole polarizability tensor of `atoms` at
 * its current geometry (no geometry optimization is performed —
 * caller is responsible for that if a stationary point is needed).
 */
export function polarizabilityFiniteField(
  atoms: readonly Atom[],
  opts: PolarizabilityOpts = {},
): PolarizabilityResult {
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

  // Build integrals once at the unperturbed geometry. We perturb
  // h_AO in place per ±E direction below.
  const symbols = atoms.map((a) => a.symbol);
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei, integralOpts);
  const n = integrals.n;
  // Build the dipole AO integral matrices (3 × n × n) — same machinery
  // used by `dipoleMoment` and the oscillator-strength code.
  // For the spherical-d case, dipole_cg returns the raw Cartesian
  // dipole; transforming the perturbation to spherical isn't needed
  // because we're only modifying h_AO_cart in-place below, and the
  // spherical-d transform applied to S_AO etc. lives downstream.
  // (Polarizability with `spherical: true` would require transforming
  // the dipole AO matrix too; we skip that for now.)
  if (opts.spherical) {
    throw new Error(
      "polarizabilityFiniteField: spherical-d basis not yet supported. " +
      "Use spherical: false.",
    );
  }
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

  // Stash original h_AO so we can restore between runs.
  const h_orig = new Float64Array(integrals.h_AO);

  // Run SCF with field E·ê_axis applied. The perturbation is folded
  // into h_AO directly: h_eff = h + E·μ_AO.
  // Sign: an electron has charge −e, so the dipole operator is
  // μ̂_e = −r̂. The dipole-field interaction is H_int = −μ̂·E, giving
  // H_int = +r̂·E for the electron — i.e. h_AO acquires +E·μ_AO
  // matrix elements (μ_AO[μν] = ⟨χ_μ | r̂ | χ_ν⟩, the position-
  // operator integrals from `dipole_cg`).
  const dipoleAtField = (axis: number, sign: number): [number, number, number] => {
    // Reset h_AO, then perturb.
    integrals.h_AO.set(h_orig);
    const E = sign * Efield;
    for (let k = 0; k < n * n; k++) {
      integrals.h_AO[k]! += E * muAO[axis]![k]!;
    }
    if (method === "hf") {
      const hf = runRHFSCF(integrals, nElectrons, hfOpts);
      return dipoleMoment(integrals, hf.D);
    }
    const dft = runRKSDFT(integrals, nElectrons, symbols, dftOpts);
    return dipoleMoment(integrals, dft.D);
  };

  const tensor = new Float64Array(9);            // row-major 3×3
  for (let j = 0; j < 3; j++) {
    const muP = dipoleAtField(j, +1);
    const muM = dipoleAtField(j, -1);
    for (let i = 0; i < 3; i++) {
      // α_ij = (∂μ_i/∂E_j)|_{E=0}.   Sign: by the standard convention
      // (Buckingham), α_ij is POSITIVE for closed-shell ground states
      // (induced dipole aligns with applied field). The FD here is
      //   α_ij ≈ [μ_i(+E_j) − μ_i(−E_j)] / (2 E_j)
      // which gives the textbook positive α.
      tensor[i * 3 + j] = (muP[i]! - muM[i]!) / (2 * Efield);
    }
  }
  // Restore h_AO so caller's `integrals` object is unchanged.
  integrals.h_AO.set(h_orig);

  // Symmetrize (FD noise can introduce a small asymmetry).
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      const v = 0.5 * (tensor[i * 3 + j]! + tensor[j * 3 + i]!);
      tensor[i * 3 + j] = v;
      tensor[j * 3 + i] = v;
    }
  }

  // Isotropic + anisotropy + principal components.
  const isotropic = (tensor[0]! + tensor[4]! + tensor[8]!) / 3;
  const dxx = tensor[0]!, dyy = tensor[4]!, dzz = tensor[8]!;
  const dxy = tensor[1]!, dxz = tensor[2]!, dyz = tensor[5]!;
  const anisotropy = Math.sqrt(
    0.5 * ((dxx - dyy) ** 2 + (dyy - dzz) ** 2 + (dzz - dxx) ** 2)
    + 3 * (dxy * dxy + dxz * dxz + dyz * dyz),
  );

  // Principal components: eigenvalues of the symmetric 3×3 tensor.
  // Closed-form via Cardano on the characteristic polynomial.
  const principalComponents = symmetric3x3Eigenvalues(tensor);

  return {
    tensor, isotropic, anisotropy, principalComponents,
    seconds: (performance.now() - tStart) / 1000,
    nScfRuns: 6,
  };
}

/** Closed-form eigenvalues of a 3×3 symmetric matrix (row-major).
 *  Returns ascending. Uses the trigonometric Cardano method. */
function symmetric3x3Eigenvalues(M: Float64Array): Float64Array {
  // Shift to traceless: A = M − (tr/3)·I, then eigenvalues = q + 2√(p)·cos(θ + 2πk/3) for k=0,1,2.
  const tr = (M[0]! + M[4]! + M[8]!) / 3;
  const a = M[0]! - tr, b = M[4]! - tr, c = M[8]! - tr;
  const ab = M[1]!,    ac = M[2]!,    bc = M[5]!;
  const p2 = a * a + b * b + c * c + 2 * (ab * ab + ac * ac + bc * bc);
  const p  = Math.sqrt(p2 / 6);
  if (p < 1e-14) {
    // Already diagonal at trace/3 — eigenvalues are tr/3 ± 0 (degenerate).
    return new Float64Array([tr, tr, tr]);
  }
  // det(A)/2 = q (in the standard parametrization).
  const detA =
      a * (b * c - bc * bc)
    - ab * (ab * c - bc * ac)
    + ac * (ab * bc - b * ac);
  const r = detA / (2 * p * p * p);
  const phi = Math.acos(Math.max(-1, Math.min(1, r))) / 3;
  const e0 = tr + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  const e2 = tr + 2 * p * Math.cos(phi);
  const e1 = 3 * tr - e0 - e2;       // tr unchanged
  const eigs = [e0, e1, e2].sort((x, y) => x - y);
  return new Float64Array(eigs);
}
