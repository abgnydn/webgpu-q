// quickReport tests.

import { describe, expect, test } from "vitest";
import { quickReport } from "../../src/chemistry/quick-report.js";
import { h2o, li, ch4 } from "../../src/chemistry/molecules.js";

describe("quickReport", () => {
  test("H₂O closed-shell: produces full report in one call", () => {
    const r = quickReport(h2o);
    expect(r.kind).toBe("rhf");
    expect(r.scfEnergy).toBeLessThan(-70);
    expect(r.dipoleMagnitudeDebye).toBeGreaterThan(1.0);
    expect(r.dipoleMagnitudeDebye).toBeLessThan(3.0);
    expect(r.mullikenCharges.length).toBe(3);
    expect(r.noonStronglyOccupied).toBe(5);
    expect(r.multireferenceSeverity).toBe("single-reference");
  });

  test("Li atom doublet: open-shell path with nAlpha/nBeta", () => {
    const r = quickReport(li, { nAlpha: 2, nBeta: 1 });
    expect(r.kind).toBe("uhf");
    expect(r.s2).toBeCloseTo(0.75, 6);
    expect(r.mullikenSpinPopulations).toBeDefined();
    expect(r.multireferenceSeverity).toBe("single-reference");
  });

  test("CH₄: bond orders show C-H bonds > 0.5", () => {
    const r = quickReport(ch4);
    // Index 0 = C, 1-4 = H. Bond orders is row-major 5×5.
    const nAtoms = 5;
    for (let i = 1; i <= 4; i++) {
      expect(r.bondOrders[0 * nAtoms + i]!).toBeGreaterThan(0.5);
    }
  });

  test("addD2 adds dispersion to totalEnergy", () => {
    const r1 = quickReport(h2o);
    const r2 = quickReport(h2o, { addD2: true });
    expect(r1.dispersionEnergy).toBeUndefined();
    expect(r2.dispersionEnergy).toBeDefined();
    expect(r2.dispersionEnergy!).toBeLessThan(0);
    expect(r2.totalEnergy).toBeCloseTo(r1.totalEnergy + r2.dispersionEnergy!, 8);
  });

  test("Throws if only nAlpha specified", () => {
    expect(() => quickReport(h2o, { nAlpha: 5 })).toThrow(/both nAlpha and nBeta required/);
  });

  test("Throws if (nAlpha + nBeta) ≠ nElectrons", () => {
    expect(() => quickReport(li, { nAlpha: 3, nBeta: 1 })).toThrow(/nAlpha \+ nBeta/);
  });
});
