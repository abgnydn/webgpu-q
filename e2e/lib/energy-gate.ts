// ─────────────────────────────────────────────────────────────
// energy-gate.ts — honest energy assertions for the swarm / capstone specs.
//
// WHY THIS EXISTS. Several e2e specs used to bound the SCF energy with a
// hand-picked window hundreds of Ha wide. `swarm-hf-anthracene-ccpvdz` asserted
// only `-1500 < E < -100` while its own comment recorded that the run lands at
// ≈ -880 Ha against a literature anthracene HF/cc-pVDZ value of ≈ -537 Ha — so a
// 343 Ha error was certified green on every nightly. That is the most dangerous
// failure mode in the repo (LIMITATIONS.md §3): the SCF reports
// `converged: true` and returns a finite, plausible-looking number.
//
// The replacement is a size-extensive SANITY BAND — derived rather than picked,
// and named so no reader can mistake it for validation. It is a wrong-basin /
// broken-integral detector, NOT a chemical-accuracy claim (1 kcal/mol = 1.594
// mHa; this band is four orders of magnitude looser than that, by design).
// Where an external reference value actually exists, assert against the
// reference instead of against this band.
// ─────────────────────────────────────────────────────────────

import { expect } from "@playwright/test";

/**
 * Floor — the most negative energy per carbon a real hydrocarbon RHF result can
 * reach in the bases this repo ships. Anchors, all from our own ladder:
 * benzene/cc-pVDZ ≈ -38.45 Ha/C (the most negative anywhere here),
 * naphthalene/cc-pVDZ ≈ -38.34, acene STO-3G runs ≈ -37.9. -40.0 Ha/C sits ~4%
 * below the most negative genuine value — outside the spread of real
 * hydrocarbons, and nowhere near a wrong-basin collapse (the anthracene
 * cc-pVDZ wrong basin is -62.9 Ha/C, i.e. 57% past this floor).
 *
 * This is the bound that matters: variational collapse into a non-physical
 * orbital occupation always goes DOWN.
 */
export const E_PER_CARBON_FLOOR = -40.0;

/**
 * Ceiling — deliberately slack. Real values are ≈ -37.8 Ha/C, so -20.0 Ha/C
 * still permits a 47% error upward. The slack is intentional: several acene
 * specs report the last iterate of an SCF that did NOT converge (hexacene and
 * heptacene are past the multireference wall), and the last iterate of an
 * oscillating SCF is a diagnostic, not a value — tightening it would buy a
 * latent flake rather than a real check. It exists only to reject
 * order-of-magnitude nonsense (the previous bounds let C₃₄H₂₀ report -101 Ha).
 */
export const E_PER_CARBON_CEILING = -20.0;

/**
 * Assert an RHF hydrocarbon energy is physically plausible for its size.
 *
 * This is a SANITY band, not a validated energy. Passing it means "no
 * wrong-basin collapse, no broken integrals, right order of magnitude" — it
 * does NOT mean the number is correct to any chemical standard.
 */
export function assertHydrocarbonEnergySane(energy: number, nCarbon: number): void {
  expect(Number.isFinite(energy), `E = ${energy} is not finite`).toBe(true);
  const perC = energy / nCarbon;
  const band = `E/C = ${perC.toFixed(2)} Ha over ${nCarbon} C (E = ${energy.toFixed(3)} Ha); `
    + `hydrocarbon RHF sanity band is [${E_PER_CARBON_FLOOR}, ${E_PER_CARBON_CEILING}] Ha/C`;
  expect(perC, `${band} — TOO NEGATIVE: this is the wrong-basin signature (LIMITATIONS.md §3)`)
    .toBeGreaterThan(E_PER_CARBON_FLOOR);
  expect(perC, `${band} — not negative enough to be a bound hydrocarbon`)
    .toBeLessThan(E_PER_CARBON_CEILING);
}
