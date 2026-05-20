// ─────────────────────────────────────────────────────────────
// trk-sum-rule.ts — Thomas-Reiche-Kuhn (TRK) sum rule check for
// oscillator strengths from TDHF / TDDFT / CIS / EOM-CCSD.
//
// The TRK sum rule states that for a complete basis,
//
//   Σ_n f_n = N_electrons
//
// where f_n is the oscillator strength of the nth electronic
// excitation. This is a rigorous bound on the integrated optical
// absorption spectrum — it's why UV-vis spectroscopy can determine
// "how many electrons are there" (Beer-Lambert × TRK).
//
// In a finite basis the sum is less than N_e (some oscillator
// strength sits in the unrepresented continuum / Rydberg states).
// Typical recoveries:
//   STO-3G:  20–40% of N_e
//   cc-pVDZ: 50–75%
//   aug-cc-pVDZ: 75–90% (the diffuse functions catch more of the
//     low-energy continuum)
//   aug-cc-pVTZ + Rydberg augmentation: 90–99%
//
// Sum > N_e by more than ~1% is a red flag: it indicates either
// a sign error in oscillator strengths, double-counting of
// degenerate states, or a bug in the dipole-transition matrix
// elements.
//
// API:
//   `trkSumRule(oscillatorStrengths, nElectrons)` → { sum, expected,
//     fraction, status: "ok"|"under-bound"|"over-bound" }
// ─────────────────────────────────────────────────────────────

export interface TRKResult {
  /** Σ_n f_n. */
  readonly sum: number;
  /** N_electrons (the rigorous upper bound at complete basis). */
  readonly expected: number;
  /** sum / expected. < 1 in finite basis; > 1 is a bug. */
  readonly fraction: number;
  /** "ok" if 0.2 ≤ fraction ≤ 1.01;
   *  "under-bound" if fraction < 0.2 (suspicious — too few states);
   *  "over-bound" if fraction > 1.01 (rigorously impossible — bug). */
  readonly status: "ok" | "under-bound" | "over-bound";
}

export function trkSumRule(
  oscillatorStrengths: Float64Array | readonly number[],
  nElectrons: number,
): TRKResult {
  if (nElectrons <= 0) {
    throw new Error(`trkSumRule: nElectrons must be positive, got ${nElectrons}`);
  }
  let sum = 0;
  for (let n = 0; n < oscillatorStrengths.length; n++) {
    sum += oscillatorStrengths[n]!;
  }
  const fraction = sum / nElectrons;
  let status: TRKResult["status"];
  if (fraction > 1.01) status = "over-bound";
  else if (fraction < 0.2) status = "under-bound";
  else status = "ok";
  return { sum, expected: nElectrons, fraction, status };
}
