// ─────────────────────────────────────────────────────────────
// cross-sections.ts — electron-impact cross sections via the
// Bethe-Born high-energy approximation. Tier 2 stage 14 / E17.
//
// At impact energies T well above the excitation threshold, the
// total inelastic cross section for charged-particle scattering
// off an atom or molecule has the Bethe asymptotic form
// (Bethe 1930, Inokuti 1971):
//
//   σ_inel(T) = (4π a₀² / T) · Σ_n (f_n / ω_n) · ln(α · T / ω_n)
//
// in atomic units (T, ω_n in Hartree; result in a₀²). Here:
//   • f_n is the dipole oscillator strength to discrete state n
//     (or differential f(E) for the continuum).
//   • ω_n is the excitation energy.
//   • α is the Bethe surface fitting constant; α = 4 is the
//     standard Inokuti-Itikawa choice.
//
// Splitting by ionization potential I_p:
//   σ_exc(T)  =  (sum over states with ω_n < I_p)
//   σ_ion(T)  =  (sum over states with ω_n > I_p)
//   σ_total = σ_exc + σ_ion.
//
// Honest scope notes:
//   • For finite-basis quantum chemistry (CIS / TDA / TDDFT), the
//     "continuum" above the ionization threshold is only sampled
//     by however many discrete-state proxies the basis can hold.
//     STO-3G H₂O has 10 single excitations total — most below I_p.
//     cc-pVDZ has 95 — the higher-energy ones approximate continuum
//     states but undershoot the true continuum sum.
//   • The high-energy logarithmic Bethe form is asymptotic. At
//     T ~ ω_n the Born approximation breaks down; results below
//     ~ 100 eV should not be trusted at the ±15 % level.
//   • For a converged comparison to Geant4-DNA or Itikawa-Mason
//     reference tables, we'd need a continuum representation
//     (Stieltjes imaging, SAC-CI in a continuum basis, or photo-
//     ionization cross sections from a B-spline / DVR basis) —
//     all genuinely substantial. The result here is the honest
//     "what the Bethe-Born formula gives from our TDDFT spectrum"
//     answer, not a converged σ.
// ─────────────────────────────────────────────────────────────

/** 1 Hartree in eV. */
export const HARTREE_TO_EV = 27.21138625;
/** 1 a₀² in cm². */
export const BOHR2_TO_CM2 = 2.8002852e-17;

export interface BetheBornInputs {
  /** Excitation energies ω_n (Hartree). Length nStates. */
  readonly excitationEnergiesHa: Float64Array;
  /** Dipole oscillator strengths f_n. Length nStates. */
  readonly oscillatorStrengths: Float64Array;
  /** Ionization-potential threshold (Hartree). States with
   *  ω_n < I_p are counted as bound excitations; ω_n > I_p as
   *  ionization-channel proxies. */
  readonly ionizationPotentialHa: number;
}

export interface BetheBornCrossSections {
  /** Total electron-impact inelastic cross section (a₀²). */
  readonly sigmaTotal: number;
  /** Discrete-excitation contribution (states with ω < I_p). */
  readonly sigmaExc: number;
  /** Ionization-channel contribution (states with ω > I_p). */
  readonly sigmaIon: number;
  /** Number of states that contributed to σ_exc and σ_ion. */
  readonly nExc: number;
  readonly nIon: number;
}

/**
 * Bethe-Born high-energy inelastic cross section for an electron
 * projectile of kinetic energy `impactEnergyHa` (Hartree). Returns
 * separate σ_exc and σ_ion partitioned by the ionization threshold.
 *
 * Skips states that contribute zero (f_n = 0 or ω_n ≤ 0) and
 * states where the Bethe logarithm `ln(α·T/ω_n)` would be
 * negative (the formula breaks down in that regime — sub-Bethe
 * energies).
 *
 * Result in atomic units of area (a₀² = (5.29×10⁻⁹ cm)²).
 * Multiply by `BOHR2_TO_CM2` for cm².
 */
export function betheBornCrossSection(
  inp: BetheBornInputs,
  impactEnergyHa: number,
  alpha: number = 4,
): BetheBornCrossSections {
  if (impactEnergyHa <= 0) {
    throw new Error(`betheBornCrossSection: impactEnergyHa must be > 0 (got ${impactEnergyHa})`);
  }
  const T = impactEnergyHa;
  const Ip = inp.ionizationPotentialHa;
  const omega = inp.excitationEnergiesHa;
  const f = inp.oscillatorStrengths;
  if (omega.length !== f.length) {
    throw new Error(
      `betheBornCrossSection: ω length ${omega.length} ≠ f length ${f.length}`,
    );
  }

  let sigmaExc = 0, sigmaIon = 0;
  let nExc = 0, nIon = 0;
  const prefactor = 4 * Math.PI / T;
  for (let n = 0; n < omega.length; n++) {
    const w = omega[n]!;
    const fn = f[n]!;
    if (w <= 0 || fn <= 0) continue;
    // Bethe log: ln(α T / ω). Skip when T < ω/α — formula non-physical.
    const logArg = (alpha * T) / w;
    if (logArg <= 1) continue;
    const contrib = prefactor * (fn / w) * Math.log(logArg);
    if (w < Ip) { sigmaExc += contrib; nExc++; }
    else        { sigmaIon += contrib; nIon++; }
  }
  return {
    sigmaTotal: sigmaExc + sigmaIon,
    sigmaExc,
    sigmaIon,
    nExc,
    nIon,
  };
}
