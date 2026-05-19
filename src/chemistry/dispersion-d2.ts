// ─────────────────────────────────────────────────────────────
// dispersion-d2.ts — Grimme 2006 D2 atomic-pairwise dispersion
// correction. Adds the leading vdW dispersion energy that DFT (LDA,
// GGA, hybrid) systematically misses.
//
// Formula (Grimme JCC 27, 1787 (2006)):
//
//   E_disp = − s6 · Σ_{A<B} C6^AB / R_AB^6 · f_damp(R_AB)
//
//   C6^AB = √(C6^A · C6^B)               (Grimme combining rule)
//   R_R^AB = R_R^A + R_R^B               (sum of vdW radii)
//   f_damp(R) = 1 / (1 + exp(−d · (R/R_R^AB − 1)))   (Fermi damping)
//   d = 20.0                              (universal steepness)
//
// `s6` is functional-specific:
//   BLYP, BVWN5 → 1.20
//   BP86 → 1.05
//   B3LYP, B3LYP5, B3VWN5 → 1.05
//   PBE → 0.75
//   HF → 1.20
//   default (unknown / LDA-SVWN) → 1.05 with a warning in comments
//
// Cost: O(n_atoms²). For molecular systems (≤ 100 atoms) this is
// microseconds — far cheaper than the SCF it augments. Adds a flat
// energy correction to E_total; gradients are a mechanical extension
// (∂E_disp/∂R_A = analytic chain-rule on R_AB and f_damp).
//
// References:
//   - Grimme, JCC 27, 1787 (2006) — original D2 paper, Table I for
//     C6 / R_R values.
//   - For D3 and D4: future work; D2 is the simplest accurate baseline.
// ─────────────────────────────────────────────────────────────

import type { Atom, AtomSymbol } from "./atoms.js";
import type { FunctionalKind } from "./dft/functional.js";

const ANGSTROM_TO_BOHR = 1.8897261245650618;
const J_NM6_PER_MOL_TO_HARTREE_BOHR6 = 17.59;        // conversion factor (see header)

/** Grimme 2006 Table I: per-element C6 in J·nm^6·mol^-1. */
const C6_J_NM6_PER_MOL: Readonly<Record<AtomSymbol, number>> = {
  H:  0.14,
  He: 0.08,
  Li: 1.61,
  Be: 1.61,
  C:  1.75,
  N:  1.23,
  O:  0.70,
  F:  0.75,
};

/** Grimme 2006 Table I: per-element R_R van-der-Waals radius in Å. */
const R_R_ANGSTROM: Readonly<Record<AtomSymbol, number>> = {
  H:  1.001,
  He: 1.012,
  Li: 0.825,
  Be: 1.408,
  C:  1.452,
  N:  1.397,
  O:  1.342,
  F:  1.287,
};

/** Per-element C6 in Hartree·Bohr^6 (atomic units). */
const C6_AU: Readonly<Record<AtomSymbol, number>> = {
  H:  C6_J_NM6_PER_MOL.H  * J_NM6_PER_MOL_TO_HARTREE_BOHR6,
  He: C6_J_NM6_PER_MOL.He * J_NM6_PER_MOL_TO_HARTREE_BOHR6,
  Li: C6_J_NM6_PER_MOL.Li * J_NM6_PER_MOL_TO_HARTREE_BOHR6,
  Be: C6_J_NM6_PER_MOL.Be * J_NM6_PER_MOL_TO_HARTREE_BOHR6,
  C:  C6_J_NM6_PER_MOL.C  * J_NM6_PER_MOL_TO_HARTREE_BOHR6,
  N:  C6_J_NM6_PER_MOL.N  * J_NM6_PER_MOL_TO_HARTREE_BOHR6,
  O:  C6_J_NM6_PER_MOL.O  * J_NM6_PER_MOL_TO_HARTREE_BOHR6,
  F:  C6_J_NM6_PER_MOL.F  * J_NM6_PER_MOL_TO_HARTREE_BOHR6,
};

/** Per-element R_R in Bohr. */
const R_R_BOHR: Readonly<Record<AtomSymbol, number>> = {
  H:  R_R_ANGSTROM.H  * ANGSTROM_TO_BOHR,
  He: R_R_ANGSTROM.He * ANGSTROM_TO_BOHR,
  Li: R_R_ANGSTROM.Li * ANGSTROM_TO_BOHR,
  Be: R_R_ANGSTROM.Be * ANGSTROM_TO_BOHR,
  C:  R_R_ANGSTROM.C  * ANGSTROM_TO_BOHR,
  N:  R_R_ANGSTROM.N  * ANGSTROM_TO_BOHR,
  O:  R_R_ANGSTROM.O  * ANGSTROM_TO_BOHR,
  F:  R_R_ANGSTROM.F  * ANGSTROM_TO_BOHR,
};

/** Damping steepness — universal in D2. */
const D2_DAMPING_D = 20.0;

/** Functional-specific s6 scaling factors (Grimme 2006 Table II). */
const D2_S6: Partial<Record<FunctionalKind, number>> = {
  "lda-svwn": 1.05,    // Grimme didn't tabulate LDA; B3LYP default is reasonable
  "bvwn5": 1.20,        // BVWN5 ≈ BLYP s6 (both pure GGA with B88 exchange)
  "blyp":  1.20,
  "b3vwn5": 1.05,       // B3-hybrid family
  "b3lyp5": 1.05,
};

export interface DispersionD2Opts {
  /** Functional that picked the s6 scaling (see Grimme 2006 Table II). */
  readonly functional?: FunctionalKind;
  /** Explicit s6 override. Takes precedence over `functional`. Default 1.0. */
  readonly s6?: number;
  /** Damping steepness. Default 20.0 (Grimme universal). */
  readonly d?: number;
}

export interface DispersionD2Result {
  /** Total D2 dispersion energy in Hartree (always ≤ 0 for attractive). */
  readonly energy: number;
  /** Pairwise breakdown: pairs[i] = {A_idx, B_idx, distanceBohr, contribution}. */
  readonly pairs: ReadonlyArray<{
    readonly indexA: number;
    readonly indexB: number;
    readonly distanceBohr: number;
    readonly contributionHartree: number;
  }>;
  /** s6 actually used. */
  readonly s6: number;
}

export function dispersionD2(
  atoms: readonly Atom[],
  opts: DispersionD2Opts = {},
): DispersionD2Result {
  const s6 = opts.s6 ?? (opts.functional ? (D2_S6[opts.functional] ?? 1.0) : 1.0);
  const d = opts.d ?? D2_DAMPING_D;

  let energy = 0;
  const pairs: Array<{
    indexA: number;
    indexB: number;
    distanceBohr: number;
    contributionHartree: number;
  }> = [];

  for (let A = 0; A < atoms.length; A++) {
    const aA = atoms[A]!;
    if (aA.ghost) continue;
    const ZA = aA.symbol;
    const C6A = C6_AU[ZA];
    const RRA = R_R_BOHR[ZA];
    for (let B = A + 1; B < atoms.length; B++) {
      const aB = atoms[B]!;
      if (aB.ghost) continue;
      const ZB = aB.symbol;
      const C6B = C6_AU[ZB];
      const RRB = R_R_BOHR[ZB];

      // Distance in bohr.
      const dx = (aA.pos[0] - aB.pos[0]) * ANGSTROM_TO_BOHR;
      const dy = (aA.pos[1] - aB.pos[1]) * ANGSTROM_TO_BOHR;
      const dz = (aA.pos[2] - aB.pos[2]) * ANGSTROM_TO_BOHR;
      const R = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (R < 1e-8) continue;     // coincident atoms — skip

      const C6_AB = Math.sqrt(C6A * C6B);
      const RR_AB = RRA + RRB;
      const expArg = -d * (R / RR_AB - 1);
      const fDamp = 1 / (1 + Math.exp(expArg));
      const E_AB = -s6 * C6_AB * fDamp / Math.pow(R, 6);
      energy += E_AB;
      pairs.push({
        indexA: A, indexB: B,
        distanceBohr: R,
        contributionHartree: E_AB,
      });
    }
  }

  return { energy, pairs, s6 };
}
