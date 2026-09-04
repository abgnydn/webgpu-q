import { describe, expect, test } from "vitest";
import {
  betheBornCrossSection,
  BOHR2_TO_CM2,
  HARTREE_TO_EV,
} from "../../src/chemistry/cross-sections.js";

describe("betheBornCrossSection", () => {
  test("zero oscillator strength gives zero cross-section", () => {
    const r = betheBornCrossSection(
      { excitationEnergiesHa: new Float64Array([1]), oscillatorStrengths: new Float64Array([0]), ionizationPotentialHa: 2 },
      10,
    );
    expect(r.sigmaTotal).toBe(0);
    expect(r.nExc + r.nIon).toBe(0);
  });

  test("unit point matches hand-computed value and unit conversions", () => {
    const r = betheBornCrossSection(
      { excitationEnergiesHa: new Float64Array([1]), oscillatorStrengths: new Float64Array([1]), ionizationPotentialHa: 2 },
      10,
    );
    const expected = (4 * Math.PI / 10) * Math.log(4 * 10 / 1);
    expect(r.sigmaTotal).toBeCloseTo(expected, 12);
    expect(r.sigmaTotal * BOHR2_TO_CM2).toBeCloseTo(expected * 2.8002852e-17, 12);
    expect(HARTREE_TO_EV).toBeCloseTo(27.21138625, 8);
  });

  test("broadened discrete spectrum integrates to a positive total", () => {
    const r = betheBornCrossSection(
      {
        excitationEnergiesHa: new Float64Array([0.2, 0.5, 0.9]),
        oscillatorStrengths: new Float64Array([0.1, 0.3, 0.2]),
        ionizationPotentialHa: 0.7,
      },
      5,
    );
    expect(r.sigmaTotal).toBeGreaterThan(0);
    expect(r.nExc + r.nIon).toBe(3);
  });
});
