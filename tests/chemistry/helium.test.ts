// Helium atom STO-3G — basis-set + HF + dispersion sanity.
//
// Pass bars:
//   1. He STO-3G HF: 1-shell basis, 1 occupied MO, 2 electrons → HF
//      total energy in the literature range (~−2.84 Ha for He
//      STO-3G; experimental −2.9037 Ha).
//   2. Closed-shell, ⟨S²⟩ = 0.
//   3. C₆ self-dispersion (HF route): positive, finite. He literature
//      CCSD-F12 C₆ ≈ 1.46 a.u.; STO-3G HF will undershoot.
//   4. Counterpoise on He···He at 4 Å: BSSE small, ΔE ≈ 0 (no
//      interaction at minimal basis).

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCounterpoise } from "../../src/chemistry/counterpoise.js";
import { c6Coefficient } from "../../src/chemistry/dispersion.js";

describe("Helium STO-3G", () => {
  test("He atom HF: 1 shell, 2 electrons, E ≈ −2.84 Ha (literature)", () => {
    const atoms: Atom[] = [{ symbol: "He", pos: [0, 0, 0] }];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    expect(shells.length).toBe(1);
    expect(nElectrons).toBe(2);

    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(hf.converged).toBe(true);
    // Literature: He STO-3G HF = −2.80778 Ha (Pople's original numbers).
    expect(hf.energy).toBeLessThan(-2.7);
    expect(hf.energy).toBeGreaterThan(-2.9);
    expect(hf.nOccupied).toBe(1);
  });

  test("He-He STO-3G at 4 Å: counterpoise BSSE small (atom-pair sanity)", () => {
    // C₆ via CPHF/TDHF response is N/A for He STO-3G: only 1 AO (1s),
    // hence zero virtual space → CPHF orbital Hessian has dim 0 and
    // throws "empty virtual space". A proper He polarizability would
    // require cc-pVDZ (with 2p augmentation) or aug-cc-pVDZ — basis
    // data not yet wired for He.

    // Counterpoise on He···He works (no virtual-space requirement).
    const dimerAtoms: Atom[] = [
      { symbol: "He", pos: [0, 0, 0]    },
      { symbol: "He", pos: [0, 0, 4.0] },
    ];
    const cp = runCounterpoise(
      dimerAtoms,
      [
        { atomIndices: [0] },
        { atomIndices: [1] },
      ],
      "sto-3g",
      { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 },
    );
    expect(cp.allConverged).toBe(true);
    // Long-range He-He: ΔE essentially zero.
    expect(Math.abs(cp.interactionEnergy)).toBeLessThan(5e-4);
    // BSSE non-negative.
    expect(cp.bsseCorrection).toBeGreaterThanOrEqual(0);
  }, 60_000);

  test("He cc-pVDZ HF: 5-shell basis (1s + 2s + 2p), E lower than STO-3G", () => {
    const atoms: Atom[] = [{ symbol: "He", pos: [0, 0, 0] }];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, "cc-pvdz");
    expect(shells.length).toBe(5);
    expect(nElectrons).toBe(2);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(hf.converged).toBe(true);
    // He cc-pVDZ HF ≈ −2.8552 Ha (literature). Variationally below STO-3G.
    expect(hf.energy).toBeLessThan(-2.84);
    expect(hf.energy).toBeGreaterThan(-2.87);
  }, 60_000);

  test("He cc-pVDZ self-dispersion: C₆ positive, in physical range", () => {
    const atoms: Atom[] = [{ symbol: "He", pos: [0, 0, 0] }];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, "cc-pvdz");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const c6 = c6Coefficient(hf, integrals, shells, hf, integrals, shells).c6;
    // Literature He-He CCSD-F12 / experimental: C₆ ≈ 1.46 a.u.
    // TDHF/cc-pVDZ underestimates (no correlation), expect ≈ 0.5-1.5.
    expect(c6).toBeGreaterThan(0.1);
    expect(c6).toBeLessThan(2.5);
  }, 60_000);
});
