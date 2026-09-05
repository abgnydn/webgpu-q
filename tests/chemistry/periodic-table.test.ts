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
    expect(PERIODIC_TABLE.B.atomicNumber).toBe(5);
    expect(PERIODIC_TABLE.Ne.atomicNumber).toBe(10);
    expect(PERIODIC_TABLE.Na.atomicNumber).toBe(11);
    expect(PERIODIC_TABLE.Ar.atomicNumber).toBe(18);
    // H…Ne plus Na, Mg, Al, Si, P, S, Cl, Ar — periods 1-3 complete.
    expect(Object.keys(PERIODIC_TABLE).length).toBe(18);
  });

  test("Pauling electronegativity ordering: H < C < N < O < F", () => {
    const en = (s: keyof typeof PERIODIC_TABLE) => PERIODIC_TABLE[s].electronegativity!;
    expect(en("H")).toBeLessThan(en("C"));
    expect(en("C")).toBeLessThan(en("N"));
    expect(en("N")).toBeLessThan(en("O"));
    expect(en("O")).toBeLessThan(en("F"));
    expect(en("F")).toBe(3.98);   // Pauling reference value
  });

  test("Noble gases have null electronegativity (no Pauling value)", () => {
    expect(PERIODIC_TABLE.He.electronegativity).toBeNull();
    expect(PERIODIC_TABLE.Ne.electronegativity).toBeNull();
    expect(PERIODIC_TABLE.Ne.family).toBe("noble");
    // Boron does have one, and it sits between Be (1.57) and C (2.55).
    expect(PERIODIC_TABLE.B.electronegativity).toBe(2.04);
    expect(PERIODIC_TABLE.B.group).toBe(13);
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
