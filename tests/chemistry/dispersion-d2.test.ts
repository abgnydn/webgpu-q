// D2 dispersion correction — Grimme 2006 atomic-pairwise.
//
// Pass bars:
//   1. E_disp is negative (attractive) for any non-bonded pair of atoms.
//   2. Distance scaling: E_disp ∝ -1/R^6 at large R (damping → 1).
//   3. Damping cuts off at short R (otherwise -∞ would diverge).
//   4. Functional-specific s6: B3LYP5 < BLYP (B3 is smaller scaling).
//   5. Pairwise breakdown sums to total energy.
//   6. Ghost atoms contribute zero.

import { describe, expect, test } from "vitest";
import { dispersionD2, dispersionD2Gradient, dispersionCorrectedEnergy } from "../../src/chemistry/dispersion-d2.js";
import type { Atom } from "../../src/chemistry/atoms.js";

describe("Grimme D2 dispersion correction", () => {
  test("Two He atoms at 3 Å: E_disp negative, sub-mHa", () => {
    const atoms: Atom[] = [
      { symbol: "He", pos: [0, 0, 0] },
      { symbol: "He", pos: [0, 0, 3.0] },
    ];
    const r = dispersionD2(atoms, { functional: "b3lyp5" });
    expect(r.energy).toBeLessThan(0);
    expect(r.energy).toBeGreaterThan(-0.01);     // sub-mHa for He-He at 3 Å
    expect(r.pairs.length).toBe(1);
    expect(r.pairs[0]!.contributionHartree).toBeLessThan(0);
  });

  test("H₂...H₂ at varying separation: monotonically less negative at larger R", () => {
    const make = (R: number): Atom[] => ([
      { symbol: "H", pos: [0, 0, 0]    },
      { symbol: "H", pos: [0, 0, 0.74] },
      { symbol: "H", pos: [0, 0, R]    },
      { symbol: "H", pos: [0, 0, R + 0.74] },
    ]);
    const e3 = dispersionD2(make(3.0), { functional: "b3lyp5" }).energy;
    const e5 = dispersionD2(make(5.0), { functional: "b3lyp5" }).energy;
    const e10 = dispersionD2(make(10.0), { functional: "b3lyp5" }).energy;
    expect(e3).toBeLessThan(e5);   // more negative at smaller R
    expect(e5).toBeLessThan(e10);  // monotonic
    // At large R, scaling ~ -1/R^6 → very small.
    expect(Math.abs(e10)).toBeLessThan(1e-5);
  });

  test("Damping at short R: 1/R^6 doesn't blow up at R = 1 Å", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 1.0] },
    ];
    const r = dispersionD2(atoms, { functional: "b3lyp5" });
    expect(Number.isFinite(r.energy)).toBe(true);
    // Damping pulls energy way down from raw -1/R^6 ~ -1 at unit distance.
    expect(Math.abs(r.energy)).toBeLessThan(0.01);
  });

  test("Functional scaling: B3LYP5 (s6=1.05) gives smaller |E_disp| than BLYP (s6=1.20)", () => {
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [0, 0, 4.0] },
    ];
    const eB3 = dispersionD2(atoms, { functional: "b3lyp5" }).energy;
    const eBLYP = dispersionD2(atoms, { functional: "blyp" }).energy;
    expect(eBLYP).toBeLessThan(eB3);    // BLYP more negative
    // Ratio should match s6 ratio = 1.20 / 1.05.
    expect(Math.abs(eBLYP / eB3 - 1.20 / 1.05)).toBeLessThan(1e-8);
  });

  test("Pairwise sum equals total", () => {
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [1.0, 0, 0] },
      { symbol: "H", pos: [0, 1.0, 0] },
      { symbol: "H", pos: [0, 0, 1.0] },
      { symbol: "H", pos: [-1.0, 0, 0] },
    ];
    const r = dispersionD2(atoms, { functional: "blyp" });
    let pairSum = 0;
    for (const p of r.pairs) pairSum += p.contributionHartree;
    expect(Math.abs(pairSum - r.energy)).toBeLessThan(1e-15);
    // C-H + H-H pairs: 4 C-H + 6 H-H = 10 pairs.
    expect(r.pairs.length).toBe(10);
  });

  test("Ghost atoms contribute zero", () => {
    const real: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [0, 0, 4.0] },
    ];
    const withGhost: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [0, 0, 4.0] },
      { symbol: "H", pos: [10, 0, 0], ghost: true },
    ];
    const e1 = dispersionD2(real, { functional: "b3lyp5" }).energy;
    const e2 = dispersionD2(withGhost, { functional: "b3lyp5" }).energy;
    expect(Math.abs(e1 - e2)).toBeLessThan(1e-15);
  });

  test("Default s6 = 1.0 when no functional specified", () => {
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [0, 0, 4.0] },
    ];
    const r = dispersionD2(atoms);
    expect(r.s6).toBe(1.0);
  });

  test("Explicit s6 override beats functional", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 3.0] },
    ];
    const r = dispersionD2(atoms, { functional: "blyp", s6: 2.0 });
    expect(r.s6).toBe(2.0);   // override wins
  });

  test("Analytical gradient matches central finite-difference (He-He at 3 Å)", () => {
    const baseAtoms: Atom[] = [
      { symbol: "He", pos: [0, 0, 0] },
      { symbol: "He", pos: [0, 0, 3.0] },
    ];
    const opts = { functional: "b3lyp5" as const };
    const grad = dispersionD2Gradient(baseAtoms, opts);

    // Central FD: ∂E/∂R_A^x ≈ (E(R+h) − E(R−h)) / (2h).
    // Step in Å (Atom.pos convention).
    const h = 1e-4;
    for (let A = 0; A < 2; A++) {
      for (let axis = 0; axis < 3; axis++) {
        const plusAtoms = baseAtoms.map((a, i) => {
          if (i !== A) return a;
          const pos = [a.pos[0], a.pos[1], a.pos[2]] as [number, number, number];
          pos[axis]! += h;
          return { ...a, pos };
        });
        const minusAtoms = baseAtoms.map((a, i) => {
          if (i !== A) return a;
          const pos = [a.pos[0], a.pos[1], a.pos[2]] as [number, number, number];
          pos[axis]! -= h;
          return { ...a, pos };
        });
        const ePlus  = dispersionD2(plusAtoms,  opts).energy;
        const eMinus = dispersionD2(minusAtoms, opts).energy;
        // FD is wrt Å (input pos), but grad is wrt bohr. Convert:
        // dE/dR_Å = dE/dR_bohr · (Bohr/Å)  → multiply grad by 1.8897 to compare in Å.
        const ANG2BOHR = 1.8897261245650618;
        const fdHa_per_A = (ePlus - eMinus) / (2 * h);
        const fdHa_per_Bohr = fdHa_per_A / ANG2BOHR;
        const analytic = grad[A * 3 + axis]!;
        expect(Math.abs(analytic - fdHa_per_Bohr)).toBeLessThan(1e-6);
      }
    }
  });

  test("dispersionCorrectedEnergy adds D2 to a synthetic SCF energy", () => {
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [0, 0, 4.0] },
    ];
    const syntheticSCF = -75.50;
    const r = dispersionCorrectedEnergy(syntheticSCF, atoms, { functional: "b3lyp5" });
    const d2 = dispersionD2(atoms, { functional: "b3lyp5" });
    expect(r.scfEnergy).toBe(syntheticSCF);
    expect(r.dispersionEnergy).toBeCloseTo(d2.energy, 14);
    expect(r.totalEnergy).toBeCloseTo(syntheticSCF + d2.energy, 14);
    expect(r.totalEnergy).toBeLessThan(syntheticSCF);   // dispersion stabilizes
    expect(r.s6).toBe(1.05);
  });

  test("Gradient translational invariance: Σ_A ∇_A E = 0", () => {
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [1.0, 0, 0] },
      { symbol: "H", pos: [0, 1.0, 0] },
      { symbol: "H", pos: [0, 0, 1.0] },
      { symbol: "H", pos: [-1.0, 0, 0] },
    ];
    const grad = dispersionD2Gradient(atoms, { functional: "blyp" });
    let sumX = 0, sumY = 0, sumZ = 0;
    for (let A = 0; A < atoms.length; A++) {
      sumX += grad[A * 3 + 0]!;
      sumY += grad[A * 3 + 1]!;
      sumZ += grad[A * 3 + 2]!;
    }
    // Sum of gradients should be zero (translational invariance of E_disp).
    expect(Math.abs(sumX)).toBeLessThan(1e-14);
    expect(Math.abs(sumY)).toBeLessThan(1e-14);
    expect(Math.abs(sumZ)).toBeLessThan(1e-14);
  });
});
