// ─────────────────────────────────────────────────────────────
// raman.ts — Placzek Raman activity from polarizability derivatives.
// Tier 2 stage 18.
//
// Raman scattering intensity for vibrational mode k is governed by
// the polarizability derivative ∂α/∂q_k along the (mass-weighted)
// normal coordinate. Placzek's polarizability theory gives the
// "Raman activity" S_k as the sum of two rotation-averaged
// invariants of the polarizability-derivative tensor:
//
//   S_k = 45·ā_k²  +  7·γ_k²              (Å⁴/amu)
//
// where
//   ā_k    = (1/3)·tr(∂α/∂q_k)            (mean polarizability deriv)
//   γ_k²   = (1/2)·[(α'_xx − α'_yy)²
//                 + (α'_yy − α'_zz)²
//                 + (α'_zz − α'_xx)²]
//             + 3·(α'_xy² + α'_yz² + α'_xz²)
//   α'_ij  = ∂α_ij/∂q_k                   (6 independent components)
//
// (Standard Placzek formulas; see Long, "The Raman Effect", 2002.)
//
// Implementation:
//   • Build vibrational modes (frequencies + mass-weighted modes)
//     by reusing `vibrationalFrequencies`.
//   • Build polarizability derivative ∂α_ij/∂R_k (3×3×3N) by
//     central FD on `polarizabilityFiniteField` at ±h displaced
//     geometries: 6N polarizability evals × 6 SCF each = 36N SCF
//     runs (in addition to the vibrational analysis cost).
//   • Project ∂α_ij/∂R_k → ∂α_ij/∂q_k by mass-weighted contraction
//     with each mode vector.
//   • Compute ā_k, γ_k², S_k per mode.
//
// Cost: H₂O default grid finishes in ~30–60 s (vibrations + 108
// extra polarizability SCFs). Scales as 36·n_atoms × SCF, so
// triatomics are fine, polyatomics get expensive — analytical
// CPHF would be ~10× faster but requires new infrastructure.
//
// Output: frequencies + IR intensities + Raman activities, the
// complete vibrational spectroscopy bundle for a given molecule.
// ─────────────────────────────────────────────────────────────

import type { Atom, AtomSymbol } from "./atoms.js";
import {
  vibrationalFrequencies, type VibrationsOpts, type VibrationsResult,
} from "./vibrations.js";
import {
  polarizabilityFiniteField, type PolarizabilityOpts,
} from "./polarizability.js";
import type { FunctionalKind } from "./dft/functional.js";

const ANGSTROM_TO_BOHR = 1.8897261339;

const ATOMIC_MASS: Readonly<Record<AtomSymbol, number>> = {
  H:  1.00782503207,
  He: 4.002603254,
  Li: 7.0160034366,
  Be: 9.012183065,
  B:  11.00930536,
  C:  12.0,
  N:  14.0030740048,
  O:  15.99491461956,
  F:  18.998403163,
  Ne: 19.9924401762,
};

export type EnergyMethod = "hf" | FunctionalKind;

export interface RamanOpts extends VibrationsOpts {
  /** Field strength for polarizability finite-field (a.u.). Default 5e-3. */
  readonly fieldStrength?: number;
  /** Geometry displacement step (Å) for ∂α/∂R FD. Default 1e-2 Å —
   *  larger than the Hessian step (5e-3) because polarizability is
   *  a 2nd-order property and FD-on-FD precision degrades faster. */
  readonly polDisplacement?: number;
}

export interface RamanResult {
  /** Frequencies in cm⁻¹ — pulled through from the vibrational
   *  analysis. Length is 3N − 6 (or 3N − 5 for linear). */
  readonly frequenciesCmInv: Float64Array;
  /** IR intensities (km/mol) — pulled through unchanged. */
  readonly irIntensitiesKmPerMol: Float64Array;
  /** Raman activities S_k (Å⁴/amu). Per Placzek polarizability
   *  theory, this is the rotation-averaged differential cross
   *  section in the standard chemistry units. Symmetry-forbidden
   *  modes have S_k = 0; no fundamental can be both IR-forbidden
   *  AND Raman-forbidden EXCEPT in centrosymmetric molecules
   *  (rule of mutual exclusion). */
  readonly ramanActivitiesA4PerAmu: Float64Array;
  /** Mean polarizability derivative ā_k = tr(∂α/∂q_k)/3 per mode
   *  (a.u.). Useful for inspecting which modes drive isotropic
   *  Raman scattering. */
  readonly meanPolDerivative: Float64Array;
  /** Anisotropic polarizability derivative γ_k² per mode (a.u.). */
  readonly anisotropyPolDerivativeSq: Float64Array;
  /** Wall-clock seconds for the full computation (vibrations +
   *  polarizability derivatives). */
  readonly seconds: number;
  /** Total SCF runs across the entire pipeline. */
  readonly nScfRuns: number;
}

/**
 * Compute the full vibrational spectrum (frequencies + IR + Raman)
 * for the molecule at its CURRENT geometry. Caller is responsible
 * for running `optimizeGeometry` first if a stationary point is
 * needed — at non-stationary geometries, the Hessian and Raman
 * intensities are not physically meaningful.
 */
export function ramanSpectrum(
  atoms: readonly Atom[],
  opts: RamanOpts = {},
): RamanResult {
  const tStart = performance.now();
  // ── 1. Vibrational analysis (frequencies + IR + modes). ─────
  // Reuses 6N gradient evals + dipole tracking from vibrations.ts.
  const vib: VibrationsResult = vibrationalFrequencies(atoms, opts);
  const nA = atoms.length;
  const nDof = 3 * nA;
  const nVib = vib.frequenciesCmInv.length;

  // ── 2. Polarizability derivative ∂α_ij/∂R_k via FD. ─────────
  // 6N polarizability evals (each = 6 SCF runs internally).
  const polOpts: PolarizabilityOpts = {
    basis: opts.basis ?? "sto-3g",
    spherical: opts.spherical ?? false,
    method: opts.method ?? "hf",
    hf: opts.hf,
    dft: opts.dft,
    fieldStrength: opts.fieldStrength ?? 5e-3,
  };
  const dStep = opts.polDisplacement ?? 1e-2;          // Å
  const dAlphaDR = new Float64Array(9 * nDof);          // (i,j,k) → 3*3*nDof
  const xPlus  = new Float64Array(nDof);
  const xMinus = new Float64Array(nDof);
  // Reference geometry → flat coordinate vector in Å.
  for (let i = 0; i < nA; i++) {
    const p = atoms[i]!.pos;
    xPlus[3 * i + 0] = p[0]; xPlus[3 * i + 1] = p[1]; xPlus[3 * i + 2] = p[2];
    xMinus[3 * i + 0] = p[0]; xMinus[3 * i + 1] = p[1]; xMinus[3 * i + 2] = p[2];
  }
  let nScf = 6 * nDof;                                  // vibrations (1 SCF per ±step + dipole = 2N each... actually 6N total)
  for (let k = 0; k < nDof; k++) {
    xPlus[k]! += dStep;
    xMinus[k]! -= dStep;
    const movedP: Atom[] = atoms.map((a, i) => ({
      symbol: a.symbol,
      pos: [xPlus[3 * i]!, xPlus[3 * i + 1]!, xPlus[3 * i + 2]!] as [number, number, number],
    }));
    const movedM: Atom[] = atoms.map((a, i) => ({
      symbol: a.symbol,
      pos: [xMinus[3 * i]!, xMinus[3 * i + 1]!, xMinus[3 * i + 2]!] as [number, number, number],
    }));
    const polP = polarizabilityFiniteField(movedP, polOpts);
    const polM = polarizabilityFiniteField(movedM, polOpts);
    nScf += polP.nScfRuns + polM.nScfRuns;
    // ∂α_ij/∂R_k ≈ (α_ij(+h) − α_ij(−h)) / (2·h_in_Å). Convert
    // Å → Bohr by dividing by ANGSTROM_TO_BOHR (same as Hessian).
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const ij = i * 3 + j;
        dAlphaDR[ij * nDof + k] = (polP.tensor[ij]! - polM.tensor[ij]!)
                                 / (2 * dStep * ANGSTROM_TO_BOHR);
      }
    }
    xPlus[k]!  -= dStep;       // restore for next iteration
    xMinus[k]! += dStep;
  }

  // ── 3. Project onto modes: ∂α/∂q_k. ─────────────────────────
  // ∂α_ij/∂q_k = Σ_a (∂α_ij/∂R_a) · u_k_a / √m_atom(a)
  // where u_k_a is the a-th component of the k-th mass-weighted mode.
  const masses = atoms.map((a) => ATOMIC_MASS[a.symbol]);
  const modes  = vib.modesMassWeighted;                 // (nDof × nVib)
  const dAlphaDq = new Float64Array(9 * nVib);          // (i,j,k_mode)
  for (let kMode = 0; kMode < nVib; kMode++) {
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const ij = i * 3 + j;
        let s = 0;
        for (let a = 0; a < nDof; a++) {
          const ma = masses[Math.floor(a / 3)]!;
          s += dAlphaDR[ij * nDof + a]! * modes[a * nVib + kMode]! / Math.sqrt(ma);
        }
        dAlphaDq[ij * nVib + kMode] = s;
      }
    }
  }

  // ── 4. Per-mode invariants + Raman activity. ─────────────────
  const meanDeriv = new Float64Array(nVib);
  const anisoSq   = new Float64Array(nVib);
  const ramanS    = new Float64Array(nVib);
  for (let k = 0; k < nVib; k++) {
    const axx = dAlphaDq[(0 * 3 + 0) * nVib + k]!;
    const ayy = dAlphaDq[(1 * 3 + 1) * nVib + k]!;
    const azz = dAlphaDq[(2 * 3 + 2) * nVib + k]!;
    const axy = dAlphaDq[(0 * 3 + 1) * nVib + k]!;
    const axz = dAlphaDq[(0 * 3 + 2) * nVib + k]!;
    const ayz = dAlphaDq[(1 * 3 + 2) * nVib + k]!;
    const aMean = (axx + ayy + azz) / 3;
    const gammaSq =
        0.5 * ((axx - ayy) ** 2 + (ayy - azz) ** 2 + (azz - axx) ** 2)
      + 3   * (axy * axy + axz * axz + ayz * ayz);
    meanDeriv[k] = aMean;
    anisoSq[k]   = gammaSq;
    ramanS[k]    = 45 * aMean * aMean + 7 * gammaSq;
  }

  return {
    frequenciesCmInv:        vib.frequenciesCmInv,
    irIntensitiesKmPerMol:   vib.irIntensitiesKmPerMol,
    ramanActivitiesA4PerAmu: ramanS,
    meanPolDerivative:       meanDeriv,
    anisotropyPolDerivativeSq: anisoSq,
    seconds:                 (performance.now() - tStart) / 1000,
    nScfRuns:                nScf,
  };
}
