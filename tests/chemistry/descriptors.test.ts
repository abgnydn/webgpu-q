// Coulomb matrix descriptor tests.

import { describe, expect, test } from "vitest";
import { coulombMatrix, descriptorDistance } from "../../src/chemistry/descriptors.js";
import { h2o, h2, ch4 } from "../../src/chemistry/molecules.js";
import type { Atom } from "../../src/chemistry/atoms.js";

describe("Coulomb matrix descriptor", () => {
  test("H₂: 2×2 matrix with correct diagonal + off-diagonal", () => {
    const r = coulombMatrix(h2);
    expect(r.nAtoms).toBe(2);
    expect(r.matrix.length).toBe(4);
    // Diagonal: 0.5 · 1^2.4 = 0.5 (both H)
    expect(r.matrix[0]!).toBeCloseTo(0.5, 6);
    expect(r.matrix[3]!).toBeCloseTo(0.5, 6);
    // Off-diagonal: 1·1 / R_bohr where R = 0.7414 Å = 1.4011 Bohr
    expect(r.matrix[1]!).toBeCloseTo(1 / (0.7414 * 1.88972612), 5);
    // Symmetric.
    expect(r.matrix[1]!).toBe(r.matrix[2]!);
  });

  test("Sorted upper-triangle: length = N(N+1)/2", () => {
    const r1 = coulombMatrix(h2o);
    expect(r1.sortedUpperTriangle.length).toBe(6);   // 3 × 4 / 2
    const r2 = coulombMatrix(ch4);
    expect(r2.sortedUpperTriangle.length).toBe(15);   // 5 × 6 / 2
  });

  test("Identical molecules → identical descriptors", () => {
    const r1 = coulombMatrix(h2o);
    const r2 = coulombMatrix(h2o);
    expect(descriptorDistance(r1.sortedUpperTriangle, r2.sortedUpperTriangle)).toBe(0);
  });

  test("Different molecules → non-zero descriptor distance", () => {
    const a = coulombMatrix(h2o).sortedUpperTriangle;
    const b = coulombMatrix(ch4).sortedUpperTriangle;
    expect(descriptorDistance(a, b)).toBeGreaterThan(0.1);
  });

  test("Permutation invariance: same molecule with reordered atoms → same descriptor", () => {
    // Build H₂O with atoms in different order.
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const ordered: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const shuffled: Atom[] = [
      { symbol: "H", pos: [-xH, 0, zH] },     // H2 first
      { symbol: "H", pos: [ xH, 0, zH] },     // H1 second
      { symbol: "O", pos: [0, 0, 0] },         // O last
    ];
    const a = coulombMatrix(ordered).sortedUpperTriangle;
    const b = coulombMatrix(shuffled).sortedUpperTriangle;
    // Sorted-row-norm matrix is permutation-invariant.
    let maxDiff = 0;
    for (let i = 0; i < a.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(a[i]! - b[i]!));
    }
    expect(maxDiff).toBeLessThan(1e-10);
  });

  test("Ghost atoms have Z = 0 → don't contribute", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 1.0] },
      { symbol: "O", pos: [5, 0, 0], ghost: true },
    ];
    const r = coulombMatrix(atoms);
    // O ghost has diagonal 0.5·0^2.4 = 0; off-diagonals Z_O·Z_H / R = 0.
    expect(r.matrix[2 * 3 + 2]).toBe(0);    // O diagonal
    expect(r.matrix[0 * 3 + 2]).toBe(0);    // H-Oghost cross
  });
});
