// UHF TDHF — frequency-dependent α(ω) for open-shell radicals.
//
// Pass bars:
//   1. Static limit (ω = 0): α_UHF-TDHF(0) matches α_UHF-CPHF to
//      ≤ 1e-7 per tensor element. The RPA response equation collapses
//      to the static CPHF problem at ω = 0.
//   2. Closed-shell limit (n_α = n_β, identical α/β orbitals): UHF
//      TDHF α(ω) matches RHF TDHF α(ω) to ≤ 1e-7 per element.
//   3. Imaginary axis: α(iω) is real, positive, decreases monotonically
//      from α(0) to ≈ 0 as ω → ∞. No poles.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import { tdhfPolarizability } from "../../src/chemistry/tdhf.js";
import { uhfCphfPolarizability } from "../../src/chemistry/uhf-cphf.js";
import {
  uhfTdhfPolarizability,
  uhfTdhfPolarizabilityImag,
} from "../../src/chemistry/uhf-tdhf.js";

describe("UHF TDHF — frequency-dependent α(ω) for open-shell radicals", () => {
  test("Static limit (ω=0): UHF-TDHF matches UHF-CPHF to ≤ 1e-7", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 5, 5, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      symmetryBreaking: 0,
    });

    const cphf = uhfCphfPolarizability(uhf, integrals, shells);
    const tdhf0 = uhfTdhfPolarizability(uhf, integrals, shells, 0.0);

    for (let k = 0; k < 9; k++) {
      expect(Math.abs(tdhf0.alpha[k]! - cphf.alpha[k]!)).toBeLessThan(1e-7);
    }
  });

  test("Closed-shell limit on H₂O (n_α=n_β=5): UHF-TDHF matches RHF-TDHF to ≤ 1e-7 at ω = 0.1 Ha", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const rhf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const uhf = runUHFSCF(integrals, 5, 5, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      symmetryBreaking: 0,
    });

    const omega = 0.1;
    const rhf_a = tdhfPolarizability(rhf, integrals, shells, omega);
    const uhf_a = uhfTdhfPolarizability(uhf, integrals, shells, omega);

    for (let k = 0; k < 9; k++) {
      expect(Math.abs(uhf_a.alpha[k]! - rhf_a.alpha[k]!)).toBeLessThan(1e-7);
    }
  });

  test("Li atom doublet cc-pVDZ: α(iω) decreases monotonically from α(0)", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ], "cc-pvdz");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 1, {
      maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(uhf.converged).toBe(true);
    const a0 = uhfTdhfPolarizability(uhf, integrals, shells, 0).isotropic;
    const samples = [0, 0.05, 0.1, 0.3, 1.0, 5.0].map((w) =>
      uhfTdhfPolarizabilityImag(uhf, integrals, shells, w).isotropic,
    );
    // α(i·0) = α(0).
    expect(Math.abs(samples[0]! - a0)).toBeLessThan(1e-10);
    // Monotonic decrease.
    for (let i = 1; i < samples.length; i++) {
      expect(samples[i]!).toBeLessThan(samples[i - 1]!);
    }
    // All real, positive, finite.
    for (const v of samples) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }
  }, 60_000);
});
