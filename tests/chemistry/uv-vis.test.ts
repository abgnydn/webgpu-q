// UV-vis pipeline tests.

import { describe, expect, test } from "vitest";
import { uvVisSpectrum } from "../../src/chemistry/uv-vis.js";
import { h2o } from "../../src/chemistry/molecules.js";

describe("UV-vis pipeline", () => {
  test("H₂O HF/STO-3G: excitations + broadened spectrum + peaks + TRK", () => {
    const r = uvVisSpectrum(h2o, { method: "hf", nStates: 5, fwhm: 0.3 });

    // Excitations: 5 lowest singlets.
    expect(r.excitations.length).toBe(5);
    // All positive energies, ascending.
    for (let i = 1; i < r.excitations.length; i++) {
      expect(r.excitations[i]!.energyHa).toBeGreaterThan(r.excitations[i - 1]!.energyHa);
    }
    // First excitation should be in the UV range (5–25 eV for STO-3G).
    expect(r.excitations[0]!.energyEv).toBeGreaterThan(5);
    expect(r.excitations[0]!.energyEv).toBeLessThan(30);

    // Wavelength conversion check.
    const e1 = r.excitations[0]!;
    expect(e1.wavelengthNm).toBeCloseTo(1240 / e1.energyEv, 1);   // hc/E in eV·nm

    // Spectrum: array of (energy, absorption) pairs.
    expect(r.spectrum.length).toBeGreaterThan(50);
    // At least one absorption peak.
    expect(r.peaks.length).toBeGreaterThanOrEqual(1);

    // TRK cross-check: status not "over-bound" (rigorous bound).
    expect(r.trk.status).not.toBe("over-bound");
    expect(r.trk.expected).toBe(10);     // H₂O has 10 electrons

    // SCF energy reasonable.
    expect(r.scfEnergy).toBeLessThan(-70);
    expect(r.scfEnergy).toBeGreaterThan(-80);
  }, 60_000);

  test("Wavelength conversion is consistent across excitations", () => {
    const r = uvVisSpectrum(h2o, { method: "hf", nStates: 3 });
    for (const e of r.excitations) {
      // λ[nm] · E[eV] ≈ 1240
      expect(e.wavelengthNm * e.energyEv).toBeCloseTo(1240, 0);
    }
  }, 60_000);
});
