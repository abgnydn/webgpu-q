// Tier 2 stage 15b-GGA-B88: triplet TDA + TDDFT for BVWN5 + B3VWN5.
//
// Pass bars:
//   1. Spin-polarized BVWN5 (Slater + B88 + VWN5) at the closed-shell
//      baseline ρ_↑ = ρ_↓ = ρ/2, γ_↑↑ = γ_↓↓ = γ_↑↓ = γ/4 collapses
//      to the closed-shell evalXC[bvwn5] for ε_xc, v_↑, and the
//      chain-rule-combined v_γ.
//   2. runTDA(method='bvwn5', spin='triplet') on H₂ STO-3G sits BELOW
//      the singlet (textbook ordering — same as LDA path).
//   3. Same for B3VWN5 (B3-style hybrid with VWN5).
//   4. BLYP / B3LYP5 + triplet still throws (LYP-bearing path open).
//   5. Triplet oscillator strengths are zero (spin-forbidden).
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";
import { runTDA, runTDDFT } from "../../src/chemistry/tda-dft.js";
import { evalXC } from "../../src/chemistry/dft/functional.js";
import { evalXC_GGA_spin, type XCSpinOutputs } from "../../src/chemistry/dft/functional-spin.js";

const H2_ATOMS: Atom[] = [
  { symbol: "H", pos: [0, 0,  0.37] },
  { symbol: "H", pos: [0, 0, -0.37] },
];

describe("Spin-polarized BLYP closed-shell limit (Miehlich spin-LYP)", () => {
  test("evalXC_GGA_spin[blyp] at closed-shell baseline matches evalXC[blyp]", () => {
    const rho = new Float64Array([0.05, 0.1, 0.5, 1, 2, 5, 10, 50]);
    const gamma = new Float64Array([1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1e4]);
    const n = rho.length;
    const half = new Float64Array(n);
    const quart = new Float64Array(n);
    for (let i = 0; i < n; i++) { half[i] = rho[i]! / 2; quart[i] = gamma[i]! / 4; }

    const epsCS = new Float64Array(n);
    const vRhoCS = new Float64Array(n);
    const vGamCS = new Float64Array(n);
    evalXC("blyp", rho, gamma, epsCS, vRhoCS, vGamCS);

    const out: XCSpinOutputs = {
      epsXc:    new Float64Array(n),
      vRhoUp:   new Float64Array(n),
      vRhoDn:   new Float64Array(n),
      vGammaUU: new Float64Array(n),
      vGammaUD: new Float64Array(n),
      vGammaDD: new Float64Array(n),
    };
    evalXC_GGA_spin("blyp", half, half, quart, quart, quart, out);

    for (let i = 0; i < n; i++) {
      // ε_xc per-particle: exact match.
      expect(Math.abs(out.epsXc[i]! - epsCS[i]!)).toBeLessThan(1e-9);
      // v_↑ = v_↓ = closed-shell v_ρ (LYP ρ-derivatives via FD ⇒ ~1e-5 step
      // ⇒ ~1e-5 absolute precision; closed-shell v ~ O(1) ⇒ relative 1e-5).
      expect(Math.abs(out.vRhoUp[i]! - vRhoCS[i]!)).toBeLessThan(2e-4);
      expect(Math.abs(out.vRhoDn[i]! - vRhoCS[i]!)).toBeLessThan(2e-4);
      // Closed-shell v_γ from chain rule:
      //   v_γ^cs = (v_γ↑↑ + v_γ↑↓ + v_γ↓↓) / 4 = (2·v_γ↑↑ + v_γ↑↓) / 4
      // (LYP has nonzero v_γ↑↓ unlike B88.)
      const vGamRecombined = (2 * out.vGammaUU[i]! + out.vGammaUD[i]!) / 4;
      expect(Math.abs(vGamRecombined - vGamCS[i]!)).toBeLessThan(1e-9);
    }
  });
});

describe("Spin-polarized BVWN5 closed-shell limit", () => {
  test("evalXC_GGA_spin[bvwn5] at closed-shell baseline matches evalXC[bvwn5]", () => {
    // Sample 8 (ρ, γ) pairs spanning core / valence / tail.
    const rho = new Float64Array([0.05, 0.1, 0.5, 1, 2, 5, 10, 50]);
    const gamma = new Float64Array([1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1e4]);
    const n = rho.length;
    const half = new Float64Array(n);
    const quart = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      half[i] = rho[i]! / 2;
      quart[i] = gamma[i]! / 4;
    }

    // Closed-shell reference.
    const epsCS = new Float64Array(n);
    const vRhoCS = new Float64Array(n);
    const vGamCS = new Float64Array(n);
    evalXC("bvwn5", rho, gamma, epsCS, vRhoCS, vGamCS);

    // Spin-polarized at closed-shell baseline.
    const out: XCSpinOutputs = {
      epsXc:    new Float64Array(n),
      vRhoUp:   new Float64Array(n),
      vRhoDn:   new Float64Array(n),
      vGammaUU: new Float64Array(n),
      vGammaUD: new Float64Array(n),
      vGammaDD: new Float64Array(n),
    };
    evalXC_GGA_spin("bvwn5", half, half, quart, quart, quart, out);

    for (let i = 0; i < n; i++) {
      // ε_xc per-particle: exact match (Slater + B88 collapse cleanly).
      expect(Math.abs(out.epsXc[i]! - epsCS[i]!)).toBeLessThan(1e-10);
      // v_↑ = v_↓: both should match the closed-shell v_ρ.
      expect(Math.abs(out.vRhoUp[i]! - vRhoCS[i]!)).toBeLessThan(1e-9);
      expect(Math.abs(out.vRhoDn[i]! - vRhoCS[i]!)).toBeLessThan(1e-9);
      // Closed-shell v_γ from chain rule:
      //   v_γ^cs = (∂F^cs/∂γ)|_{γ_↑↑=γ_↓↓=γ_↑↓=γ/4}
      //          = (v_γ↑↑ + v_γ↑↓ + v_γ↓↓) / 4
      //          = (2·v_γ↑↑ + v_γ↑↓) / 4   (closed-shell sym)
      // For B88: v_γ↑↓ = 0, so v_γ^cs = v_γ↑↑ / 2.
      const vGamRecombined = (2 * out.vGammaUU[i]! + out.vGammaUD[i]!) / 4;
      expect(Math.abs(vGamRecombined - vGamCS[i]!)).toBeLessThan(1e-9);
      // B88 has zero v_γ↑↓ contribution.
      expect(out.vGammaUD[i]!).toBe(0);
    }
  });
});

describe("Full GGA ladder triplet TDA — sits below singlet", () => {
  for (const functional of ["bvwn5", "b3vwn5", "blyp", "b3lyp5"] as const) {
    test(`H₂ STO-3G ${functional}: ω(triplet) < ω(singlet) for the lowest excitation`, () => {
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2_ATOMS);
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const symbols = H2_ATOMS.map((a) => a.symbol);
      const dft = runRKSDFT(integrals, nElectrons, symbols, {
        functional, energyTol: 1e-10, residualTol: 1e-7, maxIter: 200,
      });
      const ks = {
        nOccupied: dft.nOccupied, n: integrals.n,
        orbitalEnergies: dft.orbitalEnergies, C_MO: dft.C_MO, D: dft.D,
      };
      const sing = runTDA(integrals, ks, {
        method: functional, spin: "singlet", nucleiSymbols: symbols,
      });
      const trip = runTDA(integrals, ks, {
        method: functional, spin: "triplet", nucleiSymbols: symbols,
      });

      // Triplet sits below singlet, both positive (no instability).
      expect(trip.singletEnergies[0]!).toBeLessThan(sing.singletEnergies[0]!);
      expect(trip.singletEnergies[0]!).toBeGreaterThan(0);
      expect(sing.singletEnergies[0]!).toBeGreaterThan(0);

      // Spin echoed back; oscillator strengths zero for triplet.
      expect(trip.spin).toBe("triplet");
      for (let r = 0; r < trip.oscillatorStrengths.length; r++) {
        expect(trip.oscillatorStrengths[r]!).toBe(0);
      }

      // hfMix matches: 0 for pure GGA, 0.20 for B3-style hybrids.
      const expectedMix = (functional === "b3vwn5" || functional === "b3lyp5") ? 0.20 : 0;
      expect(trip.hfMix).toBeCloseTo(expectedMix, 10);
    });
  }
});

describe("Full GGA ladder triplet TDDFT (full Casida) ≤ TDA", () => {
  for (const functional of ["bvwn5", "b3vwn5", "blyp", "b3lyp5"] as const) {
    test(`H₂ STO-3G ${functional}: TDDFT triplet ≤ TDA triplet per state`, () => {
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2_ATOMS);
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const symbols = H2_ATOMS.map((a) => a.symbol);
      const dft = runRKSDFT(integrals, nElectrons, symbols, {
        functional, energyTol: 1e-10, residualTol: 1e-7, maxIter: 200,
      });
      const ks = {
        nOccupied: dft.nOccupied, n: integrals.n,
        orbitalEnergies: dft.orbitalEnergies, C_MO: dft.C_MO, D: dft.D,
      };
      const tda = runTDA(integrals, ks, {
        method: functional, spin: "triplet", nucleiSymbols: symbols,
      });
      const tddft = runTDDFT(integrals, ks, {
        method: functional, spin: "triplet", nucleiSymbols: symbols,
      });

      // Same length and TDDFT ≤ TDA per state (B-correction sign).
      expect(tddft.singletEnergies.length).toBe(tda.singletEnergies.length);
      for (let r = 0; r < tddft.singletEnergies.length; r++) {
        expect(tddft.singletEnergies[r]!).toBeLessThanOrEqual(
          tda.singletEnergies[r]! + 1e-10,
        );
        // No imaginary frequencies.
        expect(tddft.singletEnergies[r]!).toBeGreaterThan(0);
      }
    });
  }
});
