// ─────────────────────────────────────────────────────────────
// thermochemistry.ts — ideal-gas statistical-thermodynamic
// corrections on top of an electronic energy + harmonic
// vibrational spectrum. Tier 2 stage 19.
//
// Standard ideal-gas thermochemistry: combines translation,
// rotation, and harmonic-oscillator vibration partition functions
// with the electronic energy E_e to produce:
//
//   ZPE       = (1/2) · Σ_k ℏω_k                           (Hartree)
//   E_thermal = ZPE + ∫₀^T (C_V) dT                         (Hartree)
//   H(T)      = E_e + E_thermal + RT                        (Hartree)  [PV = NkT for ideal gas]
//   S(T)      = S_trans + S_rot + S_vib + S_elec            (Hartree/K)
//   G(T)      = H(T) − T·S(T)                               (Hartree)
//
// The partition-function pieces are textbook:
//
//   Translation (Sackur-Tetrode at standard pressure 1 atm):
//     q_trans/V = (2πmkT/h²)^(3/2)
//     S_trans   = R · [ln(q_trans · kT/p) + 5/2]
//
//   Rotation (rigid-rotor, asymmetric/linear top):
//     Asymmetric:  q_rot = (1/σ)·√π · √(8π²kT/h²)³ · √(I_A·I_B·I_C)
//     Linear:      q_rot = (1/σ) · 8π²·I·kT/h²
//     S_rot       = R · [ln(q_rot) + n_rot/2]
//                  where n_rot = 3 (asymmetric) or 2 (linear)
//
//   Vibration (harmonic oscillator, ZPE separated):
//     For each mode k with x_k = ℏω_k/kT:
//       U_vib_k = R·T · x_k · [1/(exp(x_k)−1) + 1/2]
//       S_vib_k = R · [x_k/(exp(x_k)−1) − ln(1 − exp(−x_k))]
//     Sum over modes (skip zero/imaginary modes).
//
//   Electronic (singlet ground state at default):
//     S_elec = R · ln(g_elec) — defaults to 0 (singlet, g=1).
//
// All energies in Hartree. Temperature in Kelvin. Pressure in atm
// (default 1 atm). Symmetry number σ defaults to 1; user can
// override for symmetric tops (H₂O has σ=2, CH₄ has σ=12, etc.).
//
// References:
// - McQuarrie, Statistical Mechanics (1976), ch. 8.
// - Ochterski, "Thermochemistry in Gaussian" (Gaussian whitepaper).
// ─────────────────────────────────────────────────────────────

import type { Atom, AtomSymbol } from "./atoms.js";

// ── Physical constants (CODATA 2018) ─────────────────────────
/** Hartree → Joule. */
const HA_TO_J          = 4.3597447222071e-18;
/** Bohr → meter. */
const BOHR_TO_M        = 5.29177210903e-11;
/** Atomic mass unit → kg. */
const AMU_TO_KG        = 1.66053906660e-27;
/** Boltzmann constant in Hartree/K. */
const KB_HA_PER_K      = 3.166811563e-6;
/** Gas constant R (= N_A · k_B) in Hartree/K. (Same as k_B per
 *  particle in atomic units; we use R here for the per-mole feel.) */
const R_HA_PER_K       = KB_HA_PER_K;
/** Speed of light in vacuum (cm/s) — for cm⁻¹ ↔ Hartree conversion. */
const C_CM_PER_S       = 2.99792458e10;
/** Hartree → cm⁻¹: 1 Ha = 219474.631 cm⁻¹. */
const HA_TO_CM_INV     = 219474.6313632;
/** kcal/mol → Hartree. */
const KCAL_PER_MOL_TO_HA = 1 / 627.5094740631;
/** cal/(mol·K) → Hartree/K (for reporting entropies). */
const CAL_PER_MOL_K_TO_HA_PER_K = KCAL_PER_MOL_TO_HA / 1000;
/** Standard pressure 1 atm in Pascal. */
const ATM_TO_PA        = 101325;
/** Planck constant h in J·s. */
const H_PLANCK         = 6.62607015e-34;
/** Avogadro's number. */
const N_AVOGADRO       = 6.02214076e23;

const ATOMIC_MASS: Readonly<Record<AtomSymbol, number>> = {
  H:  1.00782503207,
  Li: 7.0160034366,
  Be: 9.012183065,
  C:  12.0,
  N:  14.0030740048,
  O:  15.99491461956,
};

export interface ThermochemistryOpts {
  /** Temperature in Kelvin. Default 298.15 (standard conditions). */
  readonly temperatureK?: number;
  /** Pressure in atm. Default 1.0 (standard conditions). */
  readonly pressureAtm?: number;
  /** Rotational symmetry number σ.
   *    H₂O (C₂ᵥ) σ=2, CH₄ (T_d) σ=12, NH₃ (C₃ᵥ) σ=3,
   *    H₂ / O₂ / N₂ (D∞h homonuclear) σ=2, HF / HCl (C∞v) σ=1.
   *  Default 1 (asymmetric top). */
  readonly symmetryNumber?: number;
  /** Electronic ground-state degeneracy. Default 1 (singlet). */
  readonly electronicDegeneracy?: number;
  /** Treat the molecule as linear (rotational n=2 instead of n=3).
   *  Default false. Only matters for diatomics + linear polyatomics. */
  readonly linear?: boolean;
  /** Skip imaginary-frequency modes in the vibrational sum.
   *  Default true — imaginary modes correspond to transition states,
   *  not stable thermal degrees of freedom. */
  readonly skipImaginaryModes?: boolean;
}

export interface ThermochemistryResult {
  readonly temperatureK: number;
  readonly pressureAtm: number;
  /** Zero-point vibrational energy (Hartree). */
  readonly zpe: number;
  /** Thermal correction to internal energy: U(T) − E_elec, including
   *  ZPE + thermal vibrational + thermal rotational + thermal
   *  translational. Hartree. */
  readonly thermalU: number;
  /** Enthalpy correction: H(T) − E_elec = U_thermal + RT. Hartree. */
  readonly thermalH: number;
  /** Total entropy S(T) at the requested T,P (Hartree/K). Multiply by
   *  T to convert into a Hartree-units energy contribution. */
  readonly entropy: number;
  /** Gibbs free-energy correction: G(T) − E_elec. Hartree. */
  readonly gibbsCorrection: number;
  /** Per-component breakdown (all Hartree where dimensions match). */
  readonly breakdown: {
    readonly translationalU: number;
    readonly rotationalU: number;
    readonly vibrationalU: number;
    readonly translationalS: number;
    readonly rotationalS: number;
    readonly vibrationalS: number;
    readonly electronicS: number;
  };
}

/**
 * Compute ideal-gas thermodynamic corrections from an electronic
 * structure at a given geometry plus its harmonic vibrational
 * frequencies. The `frequenciesCmInv` argument is what
 * `vibrationalFrequencies(atoms).frequenciesCmInv` returns.
 */
export function thermochemistry(
  atoms: readonly Atom[],
  frequenciesCmInv: Float64Array,
  opts: ThermochemistryOpts = {},
): ThermochemistryResult {
  const T  = opts.temperatureK ?? 298.15;
  const P  = (opts.pressureAtm ?? 1.0) * ATM_TO_PA;       // Pa
  const sigma = opts.symmetryNumber ?? 1;
  const gElec = opts.electronicDegeneracy ?? 1;
  const linear = opts.linear ?? false;
  const skipImag = opts.skipImaginaryModes ?? true;

  if (T <= 0) throw new Error("thermochemistry: temperatureK must be positive");

  const kT = KB_HA_PER_K * T;             // Hartree
  const kT_J = kT * HA_TO_J;              // Joules
  const RT = R_HA_PER_K * T;

  // ── Translational. ──────────────────────────────────────────
  // Total mass M (amu → kg). Sackur-Tetrode at the requested (T, P).
  let totalMassAmu = 0;
  for (const a of atoms) totalMassAmu += ATOMIC_MASS[a.symbol];
  const M_kg = totalMassAmu * AMU_TO_KG;
  // q_trans/V = (2πmkT/h²)^(3/2)
  const qTransPerVolume_perM3 = Math.pow(
    (2 * Math.PI * M_kg * kT_J) / (H_PLANCK * H_PLANCK),
    1.5,
  );
  // V per molecule at (T, P): V = kT/p (per molecule, ideal gas).
  const volumePerMolecule_m3 = kT_J / P;
  const qTrans = qTransPerVolume_perM3 * volumePerMolecule_m3;
  // U_trans = (3/2) RT.   S_trans = R · [ln(q_trans) + 5/2].
  const U_trans = 1.5 * RT;
  const S_trans = R_HA_PER_K * (Math.log(qTrans) + 2.5);

  // ── Rotational. ─────────────────────────────────────────────
  // Build the inertia tensor in the COM frame. Atoms are in Å
  // (the standard `Atom.pos` convention); convert to Bohr first
  // so masses-in-amu × distances-in-Bohr lands in atomic units.
  const com = centerOfMass(atoms);
  const inertia = inertiaTensor(atoms, com);              // amu·Bohr²
  // Convert amu·Bohr² → kg·m².
  const I_SI = (v: number): number => v * AMU_TO_KG * BOHR_TO_M * BOHR_TO_M;

  let U_rot: number, S_rot: number;
  if (linear) {
    // For a linear molecule, only one moment of inertia matters
    // (the perpendicular one). The COM-frame inertia tensor has
    // eigenvalues (I_⊥, I_⊥, 0); pick the largest two and use
    // the average (they'd be exactly equal in floating-point if the
    // molecule were perfectly linear). Skip the zero-eigenvalue axis.
    const eigs = symmetric3x3Eigenvalues(inertia).map(I_SI);
    eigs.sort((a, b) => a - b);                            // smallest = ~0
    const Iperp = 0.5 * (eigs[1]! + eigs[2]!);
    if (Iperp <= 0) {
      throw new Error("thermochemistry: linear=true but inertia tensor has no nonzero perpendicular eigenvalue");
    }
    const qRot = (8 * Math.PI * Math.PI * Iperp * kT_J) / (sigma * H_PLANCK * H_PLANCK);
    U_rot = RT;                                           // linear ⇒ 2/2 RT = RT
    S_rot = R_HA_PER_K * (Math.log(qRot) + 1);
  } else {
    const eigs = symmetric3x3Eigenvalues(inertia).map(I_SI);
    if (eigs.some((e) => e <= 0)) {
      throw new Error("thermochemistry: nonlinear molecule has a non-positive principal moment of inertia");
    }
    const qRot = (Math.sqrt(Math.PI) / sigma)
               * Math.pow(8 * Math.PI * Math.PI * kT_J / (H_PLANCK * H_PLANCK), 1.5)
               * Math.sqrt(eigs[0]! * eigs[1]! * eigs[2]!);
    U_rot = 1.5 * RT;
    S_rot = R_HA_PER_K * (Math.log(qRot) + 1.5);
  }

  // ── Vibrational. ─────────────────────────────────────────────
  // Per-mode contributions. Skip imaginary (negative cm⁻¹) modes
  // when `skipImaginaryModes` is true (default).
  let U_vib = 0;
  let S_vib = 0;
  let zpe   = 0;
  for (let k = 0; k < frequenciesCmInv.length; k++) {
    const nu = frequenciesCmInv[k]!;
    if (skipImag && nu <= 0) continue;
    if (nu <= 0) continue;                     // safety: imaginary skipped above
    // ℏω in Hartree.
    const omega_Ha = nu / HA_TO_CM_INV;
    zpe += 0.5 * omega_Ha;
    const x = omega_Ha / kT;
    if (x > 700) {
      // exp(x) overflows; the high-frequency limit gives U_vib_k
      // = ZPE_k (vibrational ground-state-only) and S_vib_k = 0.
      // ZPE is already counted; nothing to add.
      continue;
    }
    const expNeg = Math.exp(-x);
    const denom  = 1 - expNeg;
    // Thermal vibrational internal energy (excludes ZPE — ZPE summed
    // separately above for clarity in the output breakdown).
    U_vib += RT * x * expNeg / denom;
    // Vibrational entropy.
    S_vib += R_HA_PER_K * (x * expNeg / denom - Math.log(denom));
  }

  // ── Electronic. ─────────────────────────────────────────────
  const S_elec = R_HA_PER_K * Math.log(gElec);

  // ── Aggregate. ───────────────────────────────────────────────
  const thermalU = zpe + U_vib + U_rot + U_trans;
  const thermalH = thermalU + RT;          // PV = RT for ideal gas
  const entropy = S_trans + S_rot + S_vib + S_elec;
  const gibbsCorrection = thermalH - T * entropy;

  return {
    temperatureK: T,
    pressureAtm: opts.pressureAtm ?? 1.0,
    zpe,
    thermalU,
    thermalH,
    entropy,
    gibbsCorrection,
    breakdown: {
      translationalU: U_trans,
      rotationalU:    U_rot,
      vibrationalU:   U_vib,
      translationalS: S_trans,
      rotationalS:    S_rot,
      vibrationalS:   S_vib,
      electronicS:    S_elec,
    },
  };
}

// ── Helpers ─────────────────────────────────────────────────

function centerOfMass(atoms: readonly Atom[]): [number, number, number] {
  let M = 0, cx = 0, cy = 0, cz = 0;
  for (const a of atoms) {
    const m = ATOMIC_MASS[a.symbol];
    M += m; cx += m * a.pos[0]; cy += m * a.pos[1]; cz += m * a.pos[2];
  }
  return [cx / M, cy / M, cz / M];
}

/** Inertia tensor I_ij = Σ_a m_a · (δ_ij·|r_a|² − r_a_i·r_a_j) in amu·Bohr².
 *  r_a is measured from the center of mass; atom positions are in Å so we
 *  convert to Bohr inside the loop. */
function inertiaTensor(
  atoms: readonly Atom[],
  com: readonly [number, number, number],
): Float64Array {
  const A_TO_B = 1.8897261339;
  const I = new Float64Array(9);
  for (const a of atoms) {
    const m = ATOMIC_MASS[a.symbol];
    const x = (a.pos[0] - com[0]) * A_TO_B;
    const y = (a.pos[1] - com[1]) * A_TO_B;
    const z = (a.pos[2] - com[2]) * A_TO_B;
    const r2 = x * x + y * y + z * z;
    I[0]! += m * (r2 - x * x);          // I_xx
    I[4]! += m * (r2 - y * y);          // I_yy
    I[8]! += m * (r2 - z * z);          // I_zz
    I[1]! -= m * x * y;                 // I_xy
    I[2]! -= m * x * z;
    I[5]! -= m * y * z;
  }
  // Symmetric.
  I[3] = I[1]!; I[6] = I[2]!; I[7] = I[5]!;
  return I;
}

/** Closed-form eigenvalues of a 3×3 symmetric matrix (row-major).
 *  Trigonometric Cardano. Returns ascending-sorted Float64Array. */
function symmetric3x3Eigenvalues(M: Float64Array): number[] {
  const tr = (M[0]! + M[4]! + M[8]!) / 3;
  const a = M[0]! - tr, b = M[4]! - tr, c = M[8]! - tr;
  const ab = M[1]!, ac = M[2]!, bc = M[5]!;
  const p2 = a * a + b * b + c * c + 2 * (ab * ab + ac * ac + bc * bc);
  const p  = Math.sqrt(p2 / 6);
  if (p < 1e-14) return [tr, tr, tr];
  const detA =
      a * (b * c - bc * bc)
    - ab * (ab * c - bc * ac)
    + ac * (ab * bc - b * ac);
  const r = detA / (2 * p * p * p);
  const phi = Math.acos(Math.max(-1, Math.min(1, r))) / 3;
  const e0 = tr + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  const e2 = tr + 2 * p * Math.cos(phi);
  const e1 = 3 * tr - e0 - e2;
  return [e0, e1, e2].sort((x, y) => x - y);
}

// ── Convenience converters for human-readable output. ─────────

/** Convert Hartree → kcal/mol. */
export const HA_TO_KCAL_PER_MOL = 1 / KCAL_PER_MOL_TO_HA;
/** Convert Hartree/K → cal/(mol·K). */
export const HA_PER_K_TO_CAL_PER_MOL_K = 1 / CAL_PER_MOL_K_TO_HA_PER_K;
/** Re-export the speed-of-light constant for downstream code. */
export const SPEED_OF_LIGHT_CM_PER_S = C_CM_PER_S;
/** Re-export Avogadro for downstream code. */
export const AVOGADRO = N_AVOGADRO;
