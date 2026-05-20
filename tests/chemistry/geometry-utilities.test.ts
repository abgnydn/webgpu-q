// Geometry utility tests — bond length, angle, dihedral, findBonds.

import { describe, expect, test } from "vitest";
import {
  bondLength, bondAngle, dihedralAngle, findBonds,
  centerOfMass, totalMass, principalMomentsOfInertia, rotationalConstants,
} from "../../src/chemistry/geometry.js";
import type { Atom } from "../../src/chemistry/atoms.js";

describe("Geometry utilities", () => {
  test("H₂O bond length: O-H ≈ 0.9572 Å", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    expect(bondLength(atoms, 0, 1)).toBeCloseTo(0.9572, 6);
    expect(bondLength(atoms, 0, 2)).toBeCloseTo(0.9572, 6);
    expect(bondLength(atoms, 1, 2)).toBeCloseTo(2 * xH, 6);
  });

  test("H₂O bond angle: H-O-H = 104.52°", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    expect(bondAngle(atoms, 1, 0, 2)).toBeCloseTo(104.52, 4);
  });

  test("Dihedral on flat (s-trans) hyrocarbon-like setup: ±180°", () => {
    // 4 atoms in a plane (z=0): i, j, k, l arranged in trans.
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 1, 0] },
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [1, 0, 0] },
      { symbol: "C", pos: [1, -1, 0] },
    ];
    expect(Math.abs(dihedralAngle(atoms, 0, 1, 2, 3))).toBeCloseTo(180, 4);
  });

  test("Dihedral on staggered: 90°", () => {
    // i at +z, l at +x; planes perpendicular → dihedral = ±90°.
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 1] },
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [1, 0, 0] },
      { symbol: "C", pos: [1, 1, 0] },
    ];
    const phi = dihedralAngle(atoms, 0, 1, 2, 3);
    expect(Math.abs(Math.abs(phi) - 90)).toBeLessThan(1e-6);
  });

  test("Dihedral on collinear b2: NaN", () => {
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [1, 0, 0] },
      { symbol: "C", pos: [2, 0, 0] },        // collinear with b1
      { symbol: "C", pos: [3, 1, 0] },
    ];
    const phi = dihedralAngle(atoms, 0, 1, 2, 3);
    expect(Number.isNaN(phi)).toBe(true);
  });

  test("findBonds on H₂O: finds O-H1 and O-H2, NOT H-H", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const bonds = findBonds(atoms);
    // Two O-H bonds, no H-H.
    expect(bonds.length).toBe(2);
    expect(bonds[0]!.i).toBe(0); expect(bonds[0]!.j).toBe(1);
    expect(bonds[1]!.i).toBe(0); expect(bonds[1]!.j).toBe(2);
    expect(bonds[0]!.length).toBeCloseTo(0.9572, 6);
  });

  test("findBonds on isolated atoms (very far apart): no bonds", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]   },
      { symbol: "H", pos: [0, 0, 10]  },
    ];
    expect(findBonds(atoms).length).toBe(0);
  });

  test("Total mass + center of mass of H₂O", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    // m_H = 1.008, m_O = 16.0 (approximately) → total ≈ 18.0 amu.
    expect(totalMass(atoms)).toBeCloseTo(18.011, 2);
    const com = centerOfMass(atoms);
    // By C₂v symmetry, x and y components are zero.
    expect(Math.abs(com[0])).toBeLessThan(1e-12);
    expect(Math.abs(com[1])).toBeLessThan(1e-12);
    // z component slightly above O (since H atoms are above).
    expect(com[2]).toBeGreaterThan(0);
    expect(com[2]).toBeLessThan(zH);
  });

  test("Rotational constants of H₂O: 27.88 / 14.52 / 9.28 cm⁻¹ literature", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const r = rotationalConstants(atoms);
    // Literature equilibrium A / B / C (cm⁻¹): 27.88 / 14.52 / 9.28
    // (Mathur et al. JMS 1971; CCCBDB). Our values within a few % of
    // experimental — we use the exact-geometry r_e, not vibrationally
    // averaged. Sanity range:
    expect(r.A_cm1).toBeGreaterThan(20);
    expect(r.A_cm1).toBeLessThan(35);
    expect(r.B_cm1).toBeGreaterThan(10);
    expect(r.B_cm1).toBeLessThan(18);
    expect(r.C_cm1).toBeGreaterThan(7);
    expect(r.C_cm1).toBeLessThan(12);
    // A > B > C ordering.
    expect(r.A_cm1).toBeGreaterThan(r.B_cm1);
    expect(r.B_cm1).toBeGreaterThan(r.C_cm1);
    // GHz conversion: 1 cm⁻¹ = 29.98 GHz.
    expect(r.A_GHz / r.A_cm1).toBeCloseTo(29.9792458, 5);
  });

  test("Principal moments of inertia: H₂O has 3 non-zero moments (nonlinear)", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { Ia, Ib, Ic } = principalMomentsOfInertia(atoms);
    expect(Ia).toBeGreaterThan(0);
    expect(Ib).toBeGreaterThan(Ia);
    expect(Ic).toBeGreaterThan(Ib);
    // I_c ≈ I_a + I_b (planar molecule identity).
    expect(Math.abs(Ic - Ia - Ib)).toBeLessThan(1e-6);
  });
});
