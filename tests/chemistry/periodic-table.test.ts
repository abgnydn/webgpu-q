// Periodic table data tests.

import { describe, expect, test } from "vitest";
import {
  PERIODIC_TABLE, elementInfo, molecularWeight,
} from "../../src/chemistry/periodic-table.js";

describe("Periodic table", () => {
  test("All supported elements have data", () => {
    expect(PERIODIC_TABLE.H.atomicNumber).toBe(1);
    expect(PERIODIC_TABLE.He.atomicNumber).toBe(2);
    expect(PERIODIC_TABLE.F.atomicNumber).toBe(9);
    expect(Object.keys(PERIODIC_TABLE).length).toBe(8);   // H, He, Li, Be, C, N, O, F
  });

  test("Pauling electronegativity ordering: H < C < N < O < F", () => {
    const en = (s: keyof typeof PERIODIC_TABLE) => PERIODIC_TABLE[s].electronegativity!;
    expect(en("H")).toBeLessThan(en("C"));
    expect(en("C")).toBeLessThan(en("N"));
    expect(en("N")).toBeLessThan(en("O"));
    expect(en("O")).toBeLessThan(en("F"));
    expect(en("F")).toBe(3.98);   // Pauling reference value
  });

  test("Helium has null electronegativity (noble gas)", () => {
    expect(PERIODIC_TABLE.He.electronegativity).toBeNull();
  });

  test("Period / group classifications match standard PT", () => {
    expect(PERIODIC_TABLE.H.period).toBe(1);
    expect(PERIODIC_TABLE.He.period).toBe(1);
    expect(PERIODIC_TABLE.Li.period).toBe(2);
    expect(PERIODIC_TABLE.O.period).toBe(2);
    expect(PERIODIC_TABLE.Li.group).toBe(1);   // alkali
    expect(PERIODIC_TABLE.F.group).toBe(17);    // halogen
    expect(PERIODIC_TABLE.He.group).toBe(18);   // noble
  });

  test("Family classifications", () => {
    expect(PERIODIC_TABLE.Li.family).toBe("alkali");
    expect(PERIODIC_TABLE.Be.family).toBe("alkaline-earth");
    expect(PERIODIC_TABLE.F.family).toBe("halogen");
    expect(PERIODIC_TABLE.He.family).toBe("noble");
    expect(PERIODIC_TABLE.C.family).toBe("main");
  });

  test("elementInfo throws on unknown element", () => {
    expect(() => elementInfo("Xe" as any)).toThrow(/unknown element/);
  });

  test("molecularWeight: water H₂O ≈ 18.011 amu", () => {
    expect(molecularWeight(["H", "H", "O"])).toBeCloseTo(18.011, 2);
    expect(molecularWeight(["C", "H", "H", "H", "H"])).toBeCloseTo(16.031, 2);   // methane
  });
});
