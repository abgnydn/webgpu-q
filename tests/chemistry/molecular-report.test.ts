// molecularReport — comprehensive property aggregator tests.
//
// Pass bars:
//   1. RHF report on H₂O: every field populated, sensible values.
//   2. UHF report on Li doublet: spin-resolved fields present
//      (spin populations, ⟨S²⟩).
//   3. Optional D2 adds dispersion to totalEnergy.
//   4. Closed-shell singlet has multireferenceSeverity = "single-reference".

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import { molecularReport } from "../../src/chemistry/molecular-report.js";

describe("molecularReport", () => {
  test("H₂O HF/STO-3G: report has all closed-shell fields", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const report = molecularReport(
      { kind: "rhf", hf }, integrals, shells, atoms, shellAtomIdx,
    );

    expect(report.kind).toBe("rhf");
    expect(report.scfEnergy).toBeCloseTo(hf.energy, 8);
    expect(report.totalEnergy).toBeCloseTo(hf.energy, 8);
    expect(report.nOccupied.alpha).toBe(5);
    expect(report.nOccupied.beta).toBe(5);
    expect(report.homoLumoGap).toBeGreaterThan(0);
    expect(report.koopmansIPEv).toBeGreaterThan(10);  // H₂O Koopmans IP ~13 eV
    expect(report.koopmansIPEv).toBeLessThan(20);

    // Dipole non-zero (polar molecule).
    expect(report.dipoleMagnitudeAU).toBeGreaterThan(0.5);
    expect(report.dipoleMagnitudeDebye).toBeGreaterThan(1.0);
    expect(report.dipoleMagnitudeDebye).toBeLessThan(3.0);

    // Mulliken: 3 atoms, sum to 0.
    expect(report.mullikenCharges.length).toBe(3);
    let qSum = 0;
    for (let A = 0; A < 3; A++) qSum += report.mullikenCharges[A]!;
    expect(Math.abs(qSum)).toBeLessThan(1e-9);

    // Bond orders: 3x3 row-major.
    expect(report.bondOrders.length).toBe(9);
    expect(report.bondOrders[0 * 3 + 1]!).toBeGreaterThan(0.5);   // O-H

    // No spin info on closed-shell.
    expect(report.mullikenSpinPopulations).toBeUndefined();
    expect(report.s2).toBeUndefined();

    // NOON: 5 strongly occupied + 2 virtual.
    expect(report.noonStronglyOccupied).toBe(5);
    expect(report.noonActive).toBe(0);
    expect(Math.abs(report.noonTotalElectrons - 10)).toBeLessThan(1e-9);

    // Multireference verdict: single-reference (closed-shell H₂O).
    expect(report.multireferenceSeverity).toBe("single-reference");
    expect(report.multireferenceFlags.length).toBe(0);
  });

  test("Li doublet UHF/STO-3G: report has spin-resolved fields", () => {
    const { shells, nuclei, shellAtomIdx } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 1, {
      maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const report = molecularReport(
      { kind: "uhf", uhf }, integrals, shells,
      [{ symbol: "Li", pos: [0, 0, 0] }], shellAtomIdx,
    );

    expect(report.kind).toBe("uhf");
    expect(report.nOccupied.alpha).toBe(2);
    expect(report.nOccupied.beta).toBe(1);
    expect(report.s2).toBeCloseTo(0.75, 6);
    expect(report.s2Exact).toBeCloseTo(0.75, 6);

    // Mulliken spin populations: total = (n_α - n_β)/2 = 0.5.
    expect(report.mullikenSpinPopulations).toBeDefined();
    let spinSum = 0;
    for (let A = 0; A < report.mullikenSpinPopulations!.length; A++) {
      spinSum += report.mullikenSpinPopulations![A]!;
    }
    expect(Math.abs(spinSum - 0.5)).toBeLessThan(1e-6);

    // NOON total electrons = 3.
    expect(Math.abs(report.noonTotalElectrons - 3)).toBeLessThan(1e-8);

    // Single-reference (clean doublet, no spin contamination).
    expect(report.multireferenceSeverity).toBe("single-reference");
  });

  test("D2 dispersion added to total energy", () => {
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [1.1, 0, 0] },
      { symbol: "H", pos: [-1.1, 0, 0] },
      { symbol: "H", pos: [0, 1.1, 0] },
      { symbol: "H", pos: [0, -1.1, 0] },
    ];
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const r1 = molecularReport(
      { kind: "rhf", hf }, integrals, shells, atoms, shellAtomIdx,
    );
    const r2 = molecularReport(
      { kind: "rhf", hf }, integrals, shells, atoms, shellAtomIdx,
      { addD2: true, d2Opts: { functional: "b3lyp5" } },
    );

    expect(r1.dispersionEnergy).toBeUndefined();
    expect(r2.dispersionEnergy).toBeDefined();
    expect(r2.dispersionEnergy!).toBeLessThan(0);
    expect(r2.totalEnergy).toBeCloseTo(r1.totalEnergy + r2.dispersionEnergy!, 14);
  });
});
