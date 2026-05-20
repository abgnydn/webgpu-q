// Pre-built molecule library tests.

import { describe, expect, test } from "vitest";
import { molecules, h2o, nh3, ch4, h2 } from "../../src/chemistry/molecules.js";
import { bondLength, bondAngle } from "../../src/chemistry/geometry.js";

describe("Molecule library", () => {
  test("H₂O has experimental geometry (r_OH = 0.9572, ∠HOH = 104.52°)", () => {
    expect(bondLength(h2o, 0, 1)).toBeCloseTo(0.9572, 6);
    expect(bondLength(h2o, 0, 2)).toBeCloseTo(0.9572, 6);
    expect(bondAngle(h2o, 1, 0, 2)).toBeCloseTo(104.52, 4);
  });

  test("H₂ has r_e = 0.7414 Å", () => {
    expect(bondLength(h2, 0, 1)).toBeCloseTo(0.7414, 6);
  });

  test("NH₃ has equivalent N-H bonds and equivalent H-N-H angles", () => {
    // 3 N-H bonds, all length 1.0124.
    expect(bondLength(nh3, 0, 1)).toBeCloseTo(1.0124, 6);
    expect(bondLength(nh3, 0, 2)).toBeCloseTo(1.0124, 6);
    expect(bondLength(nh3, 0, 3)).toBeCloseTo(1.0124, 6);
    // 3 H-N-H bond angles, all 106.7° by C₃v symmetry.
    expect(bondAngle(nh3, 1, 0, 2)).toBeCloseTo(106.7, 4);
    expect(bondAngle(nh3, 2, 0, 3)).toBeCloseTo(106.7, 4);
    expect(bondAngle(nh3, 1, 0, 3)).toBeCloseTo(106.7, 4);
  });

  test("CH₄ has tetrahedral 109.47° H-C-H angle", () => {
    expect(bondLength(ch4, 0, 1)).toBeCloseTo(1.0870, 6);
    // Tetrahedral angle = arccos(-1/3) = 109.47122°
    expect(bondAngle(ch4, 1, 0, 2)).toBeCloseTo(109.4712, 3);
  });

  test("Library exports all expected molecules", () => {
    expect(Object.keys(molecules).sort()).toEqual(
      ["atomicH", "beh2", "ch4", "f", "h2", "h2o", "he", "hf", "li", "liH", "nh3"].sort(),
    );
  });
});
