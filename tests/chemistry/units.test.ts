// Unit conversion tests.

import { describe, expect, test } from "vitest";
import {
  HARTREE_TO_EV, HARTREE_TO_KCAL_PER_MOL, HARTREE_TO_WAVENUMBER,
  BOHR_TO_ANGSTROM, AU_DIPOLE_TO_DEBYE,
  fromHartree, toHartree, HARTREE_TO_NM, NM_TO_WAVENUMBER,
} from "../../src/chemistry/units.js";

describe("Unit conversions", () => {
  test("Hartree ↔ eV: 1 Ha = 27.211 eV", () => {
    expect(HARTREE_TO_EV).toBeCloseTo(27.2113862, 5);
    expect(fromHartree(0.5, "eV")).toBeCloseTo(13.605693, 5);   // Rydberg
  });

  test("Hartree → kcal/mol round-trip", () => {
    const E = 0.012345;
    const E_kcal = fromHartree(E, "kcal/mol");
    expect(E_kcal).toBeCloseTo(E * HARTREE_TO_KCAL_PER_MOL, 10);
    expect(toHartree(E_kcal, "kcal/mol")).toBeCloseTo(E, 10);
  });

  test("Chemical accuracy: 1 kcal/mol ≈ 1.594 mHa", () => {
    const mHa = toHartree(1, "kcal/mol") * 1000;
    expect(mHa).toBeCloseTo(1.594, 3);
  });

  test("Hartree → cm⁻¹ for an O-H vibration ~3700 cm⁻¹", () => {
    const Ha = 3700 / HARTREE_TO_WAVENUMBER;
    expect(fromHartree(Ha, "cm-1")).toBeCloseTo(3700, 6);
  });

  test("Bohr ↔ Å: 1 Bohr = 0.5292 Å", () => {
    expect(BOHR_TO_ANGSTROM).toBeCloseTo(0.529177, 5);
  });

  test("Dipole a.u. → Debye: 1 e·Bohr = 2.5417 D", () => {
    expect(AU_DIPOLE_TO_DEBYE).toBeCloseTo(2.54175, 4);
  });

  test("Excitation energy → wavelength (UV-vis)", () => {
    // 0.2 Ha excitation ≈ 5.44 eV ≈ 228 nm (deep UV).
    const E = 0.2;
    const nm = HARTREE_TO_NM(E);
    expect(nm).toBeCloseTo(227.8, 0);
  });

  test("nm → wavenumber: 500 nm green light = 20000 cm⁻¹", () => {
    expect(NM_TO_WAVENUMBER(500)).toBeCloseTo(20000, 6);
  });
});
