// Tier 2 stage 18: Placzek Raman activities.
//
// Pass bars:
//   1. H₂O HF/STO-3G all 3 modes are Raman-active (S > 0).
//      For H₂O (C₂ᵥ), all 3 vibrations are both IR and Raman active
//      (no centrosymmetry → no rule of mutual exclusion).
//   2. H₂ HF/STO-3G stretch is RAMAN-ACTIVE (homonuclear diatomic
//      is Raman-active at the symmetric stretch — D∞h → mutual
//      exclusion makes it IR-inactive but RAMAN-ACTIVE, the
//      classic textbook example).
//   3. Symmetric stretches are typically the strongest Raman
//      modes for closed-shell molecules — sanity range check.
import { describe, expect, test } from "vitest";
import { ramanSpectrum } from "../../src/chemistry/raman.js";
import { optimizeGeometry } from "../../src/chemistry/geometry.js";
import type { Atom } from "../../src/chemistry/atoms.js";

describe("Raman activities — H₂O HF/STO-3G", () => {
  test("All 3 H₂O modes Raman-active in a sensible range", { timeout: 120000 }, () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const x = 0.9572 * Math.sin(half);
    const z = 0.9572 * Math.cos(half);
    const start: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ x, 0, z] },
      { symbol: "H", pos: [-x, 0, z] },
    ];
    const opt = optimizeGeometry(start, { method: "hf" });
    const spec = ramanSpectrum(opt.atoms, { method: "hf" });

    expect(spec.frequenciesCmInv.length).toBe(3);
    expect(spec.ramanActivitiesA4PerAmu.length).toBe(3);
    // All 3 Raman-active (C₂ᵥ has no centrosymmetry).
    for (let k = 0; k < 3; k++) {
      expect(spec.ramanActivitiesA4PerAmu[k]!).toBeGreaterThan(0);
      // Sensible range: 0.1–2000 Å⁴/amu (small basis inflates intensities;
      // H₂O symmetric stretch in particular is a very strong Raman mode).
      expect(spec.ramanActivitiesA4PerAmu[k]!).toBeLessThan(2000);
    }
    // The full bundle (frequencies + IR + Raman) round-trips.
    for (let k = 0; k < 3; k++) {
      expect(Number.isFinite(spec.frequenciesCmInv[k]!)).toBe(true);
      expect(Number.isFinite(spec.irIntensitiesKmPerMol[k]!)).toBe(true);
    }
  });
});

describe("Rule of mutual exclusion — H₂ symmetric stretch", () => {
  test("H₂ stretch is IR-INACTIVE but RAMAN-ACTIVE (centrosymmetric ⇒ mutual exclusion)", { timeout: 60000 }, () => {
    const start: Atom[] = [
      { symbol: "H", pos: [0, 0, -0.37] },
      { symbol: "H", pos: [0, 0,  0.37] },
    ];
    const opt = optimizeGeometry(start, { method: "hf" });
    const spec = ramanSpectrum(opt.atoms, { method: "hf", linear: true });

    expect(spec.frequenciesCmInv.length).toBe(1);
    // IR-inactive (forced by g/u parity in centrosymmetric D∞h).
    expect(spec.irIntensitiesKmPerMol[0]!).toBeLessThan(1e-3);
    // Raman-ACTIVE — the textbook example. The symmetric stretch
    // changes the molecular polarizability (bond elongation alters
    // ⟨α_zz⟩ along the bond axis) without changing the dipole.
    expect(spec.ramanActivitiesA4PerAmu[0]!).toBeGreaterThan(1);
  });
});
