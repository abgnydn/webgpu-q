// Shared SN2 reaction-coordinate geometry.
//
// Imported by BOTH the labs worker and tests/chemistry/elements/sn2-profile.test.ts
// so the page and the validation can never describe different molecules —
// the exact failure mode that let ccpvdz-firstrow.test.ts validate an NH₃
// built at 88° while its comment claimed 106.7°.
//
// Identity reaction Cl⁻ + CH₃Cl → ClCH₃ + Cl⁻ along a LINEAR INTERPOLATION,
// not an optimized reaction path:
//   s ∈ [−1, 1];  r₁ = r_mid + s·d (C···Cl_a),  r₂ = r_mid − s·d (C···Cl_b)
//   umbrella angle φ(s) = 90° + s·18°, so the methyl group flips through
//   planar — the Walden inversion.
import type { Atom } from "../chemistry/atoms.js";

export const R_CH = 1.087;
export const R_SHORT = 1.785;
export const R_LONG = 3.20;
const R_MID = (R_SHORT + R_LONG) / 2;
const D = (R_LONG - R_SHORT) / 2;

export const HARTREE_TO_KCAL = 627.5095;

/**
 * Barrier along THIS interpolated path at RHF/cc-pVDZ, kcal/mol, measured
 * relative to s = −1.
 *
 * The labs page quotes this in prose and sn2-profile.test.ts pins it to
 * ±1.0 kcal/mol against a live computation, so both read this constant
 * rather than each carrying a copy. It was previously hardcoded in the
 * page's interpretation text while the worker computed STO-3G only — the
 * page asserted a number nothing on the page had calculated.
 */
export const SN2_CCPVDZ_BARRIER_KCAL = 14.6;

/**
 * Literature gas-phase barrier for Cl⁻ + CH₃Cl, kcal/mol, RELATIVE TO THE
 * ION–DIPOLE COMPLEX.
 *
 * This is a round approximate value, not a citation, and NOT an RHF number —
 * the experimental/high-level estimates it stands for are correlated results.
 * So our RHF/cc-pVDZ 14.6 landing near it is not a validation of anything;
 * treat the closeness as coincidence. What the labs page uses it for is the
 * order-of-magnitude contrast against STO-3G's 33, which is the real teaching
 * point. Anyone tightening this into a claim needs a real reference first.
 *
 * The qualifier is load-bearing, not decoration: measured from separated
 * reactants the barrier is near zero, because forming the complex releases
 * more than the barrier costs. sn2Point(-1) places the incoming chloride at
 * R_LONG = 3.20 Å — inside the ion–dipole well — so s = −1 is comparable to
 * this number and not to the separated-reactant one. Dropping the qualifier
 * invites a reader who knows the gas-phase value to conclude the lab is
 * broken.
 */
export const SN2_LITERATURE_BARRIER_KCAL = 13;

export interface SN2Point {
  readonly atoms: Atom[];
  /**
   * C···Cl distance to the LEAVING chloride, Å. Runs 1.785 → 3.20 as s goes
   * −1 → +1: it starts bonded and departs. (These two labels were swapped in
   * an earlier revision — r1 was documented as the incoming one, which is the
   * opposite of what the arithmetic does.)
   */
  readonly r1: number;
  /** C···Cl distance to the INCOMING chloride, Å. Runs 3.20 → 1.785. */
  readonly r2: number;
  /** H–C–z umbrella angle in degrees; 90° is planar (the transition state). */
  readonly phiDeg: number;
}

export function sn2Point(s: number): SN2Point {
  const r1 = R_MID + s * D;
  const r2 = R_MID - s * D;
  const phiDeg = 90 + s * 18;
  const phi = (phiDeg * Math.PI) / 180;
  const atoms: Atom[] = [
    { symbol: "C", pos: [0, 0, 0] },
    { symbol: "Cl", pos: [0, 0, -r1] },
    { symbol: "Cl", pos: [0, 0, r2] },
  ];
  for (let k = 0; k < 3; k++) {
    const a = (120 * k * Math.PI) / 180;
    atoms.push({
      symbol: "H",
      pos: [
        R_CH * Math.sin(phi) * Math.cos(a),
        R_CH * Math.sin(phi) * Math.sin(a),
        R_CH * Math.cos(phi),
      ],
    });
  }
  return { atoms, r1, r2, phiDeg };
}

/** Neutral atoms give 43 electrons; the reacting system carries charge −1. */
export const SN2_EXTRA_ELECTRONS = 1;
