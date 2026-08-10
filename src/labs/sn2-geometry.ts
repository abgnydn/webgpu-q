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

export interface SN2Point {
  readonly atoms: Atom[];
  /** C···Cl distance to the incoming chloride, Å. */
  readonly r1: number;
  /** C···Cl distance to the leaving chloride, Å. */
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
