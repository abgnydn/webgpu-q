// Closed-form thermodynamic-limit energies — sanity checks.
// These guard the formulas in src/manybody/analytical.ts against
// regressions, by cross-checking exact special values that the
// integrators must reproduce to ~12 digits.
import { describe, expect, test } from "vitest";
import {
  tfimPfeutyEnergyPerBond,
  heisenbergBetheEnergyPerBond,
  xxBetheEnergyPerBond,
} from "../../src/manybody/analytical.js";

describe("Pfeuty TFIM thermodynamic limit", () => {
  test("e_∞(λ=0) = -J (pure Ising)", () => {
    expect(tfimPfeutyEnergyPerBond(1, 0)).toBeCloseTo(-1, 12);
    expect(tfimPfeutyEnergyPerBond(2, 0)).toBeCloseTo(-2, 12);
  });

  test("e_∞(J=0) = -h (pure transverse field)", () => {
    expect(tfimPfeutyEnergyPerBond(1e-12, 1)).toBeCloseTo(-1, 6);
    expect(tfimPfeutyEnergyPerBond(1e-12, 2.5)).toBeCloseTo(-2.5, 6);
  });

  test("e_∞(λ=1) = -4J/π (critical point)", () => {
    expect(tfimPfeutyEnergyPerBond(1, 1)).toBeCloseTo(-4 / Math.PI, 10);
    expect(tfimPfeutyEnergyPerBond(2, 2)).toBeCloseTo(-2 * 4 / Math.PI, 10);
  });

  test("e_∞ is symmetric under (J, h) ↔ (h, J)", () => {
    // Pfeuty integrand is symmetric in J ↔ h up to swapping their roles.
    // e_∞(J=1, h=2) = e_∞(J=2, h=1)? Let's check the reduced form.
    const a = tfimPfeutyEnergyPerBond(1, 2);
    const b = tfimPfeutyEnergyPerBond(2, 1);
    expect(a).toBeCloseTo(b, 10);
  });

  test("e_∞ monotonically decreases with λ (more field → lower energy)", () => {
    let prev = tfimPfeutyEnergyPerBond(1, 0);
    for (let lam = 0.1; lam <= 3; lam += 0.1) {
      const cur = tfimPfeutyEnergyPerBond(1, lam);
      expect(cur).toBeLessThan(prev + 1e-12);
      prev = cur;
    }
  });
});

describe("Heisenberg / XX Bethe ansatz limits", () => {
  test("Heisenberg AFM (J=1) e_∞ = 1/4 - ln 2 ≈ -0.4431", () => {
    const e = heisenbergBetheEnergyPerBond(1);
    expect(e).toBeCloseTo(0.25 - Math.LN2, 14);
    expect(e).toBeCloseTo(-0.4431471805599453, 14);
  });

  test("Heisenberg energy scales linearly with J", () => {
    expect(heisenbergBetheEnergyPerBond(2)).toBeCloseTo(2 * (0.25 - Math.LN2), 14);
  });

  test("XX (Δ=0) e_∞ = -2/π ≈ -0.6366", () => {
    expect(xxBetheEnergyPerBond(1)).toBeCloseTo(-2 / Math.PI, 14);
  });

  test("XX is more negative than Heisenberg (gapless XY < frustrated AFM)", () => {
    expect(xxBetheEnergyPerBond(1)).toBeLessThan(heisenbergBetheEnergyPerBond(1));
  });
});
