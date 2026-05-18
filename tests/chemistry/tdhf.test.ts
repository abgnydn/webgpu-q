// TDHF frequency-dependent α(ω) — cross-checks.
//
// Pass bars:
//   1. Static limit (ω = 0): α_TDHF(0) matches α_CPHF (the static
//      analytical CPHF result) to ≤ 1e-7 per tensor element. This
//      validates that [(A−B)(A+B)] X = (A−B)F reduces to (A+B)X = F
//      cleanly, including the (A−B) prefactor cancellation.
//   2. Monotonic frequency dependence below the first pole: α(ω) is
//      real, positive, and isotropic α grows monotonically as
//      ω → ω₁⁻ (the first RPA excitation). Standard textbook
//      dispersion behavior — α(ω) diverges as ω approaches a pole
//      from below.
//   3. Reproduces TDDFT/RPA first pole position (within numerical
//      precision): at frequencies just below the first known RPA
//      excitation energy, α should be large; just above (rejected
//      regime — we don't test there directly).

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { cphfPolarizability } from "../../src/chemistry/cphf.js";
import { tdhfPolarizability } from "../../src/chemistry/tdhf.js";
import { runTDDFT } from "../../src/chemistry/tda-dft.js";

describe("TDHF — frequency-dependent α(ω)", () => {
  test("Static limit (ω=0): α_TDHF(0) matches α_CPHF to ≤ 1e-7", () => {
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
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });

    const aCPHF = cphfPolarizability(hf, integrals, shells);
    const aTDHF0 = tdhfPolarizability(hf, integrals, shells, 0.0);

    for (let k = 0; k < 9; k++) {
      expect(Math.abs(aTDHF0.alpha[k]! - aCPHF.alpha[k]!)).toBeLessThan(1e-7);
    }
    expect(Math.abs(aTDHF0.isotropic - aCPHF.isotropic)).toBeLessThan(1e-7);
  });

  test("H₂O dispersion: α(ω) grows monotonically as ω → first RPA pole from below", () => {
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
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });

    // Get the first RPA excitation energy from runTDDFT.
    const tddft = runTDDFT(integrals, hf, { method: "hf", spin: "singlet" });
    const omega1 = tddft.singletEnergies[0]!;
    expect(omega1).toBeGreaterThan(0);

    // Sample below the first pole at 0%, 30%, 60%, 90%, 95% of ω₁.
    const fractions = [0.0, 0.3, 0.6, 0.9, 0.95];
    const isotropics = fractions.map((f) =>
      tdhfPolarizability(hf, integrals, shells, f * omega1).isotropic,
    );

    // Monotonically increasing.
    for (let i = 1; i < isotropics.length; i++) {
      expect(isotropics[i]!).toBeGreaterThan(isotropics[i - 1]!);
    }
    // All real, positive, finite.
    for (const v of isotropics) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }
    // Pole signature: α(0.95·ω₁) is enhanced vs α(0). H₂O's lowest
    // RPA singlet is weak (f ≈ 0.003) and the bright pole sits at
    // ω₅, so the enhancement at 0.95·ω₁ is modest — observed ~1.5×
    // on H₂O STO-3G. Asserting > 1.3× captures the dispersion
    // without overfitting the basis.
    expect(isotropics[4]! / isotropics[0]!).toBeGreaterThan(1.3);
  });

  test("H₂O: α(ω) tensor symmetric to 1e-10 at non-zero ω", () => {
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
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    // ω = 0.0656 Ha = 1.79 eV = 694 nm (red light, safely sub-resonance).
    const r = tdhfPolarizability(hf, integrals, shells, 0.0656);
    for (let x = 0; x < 3; x++) {
      for (let y = x + 1; y < 3; y++) {
        expect(Math.abs(r.alpha[x * 3 + y]! - r.alpha[y * 3 + x]!)).toBeLessThan(1e-10);
      }
    }
    expect(r.isotropic).toBeGreaterThan(0);
  });
});
