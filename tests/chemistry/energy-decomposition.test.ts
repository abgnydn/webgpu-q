// Energy decomposition tests.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import {
  decomposeHFEnergy, decomposeUHFEnergy,
} from "../../src/chemistry/energy-decomposition.js";
import { h2o, li } from "../../src/chemistry/molecules.js";

describe("Energy decomposition", () => {
  test("H₂O HF: decomposition sums to total energy (to 1e-9 Ha)", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(h2o);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const d = decomposeHFEnergy(hf, integrals);
    // Components should reconstruct the total.
    expect(Math.abs(d.checkResidual)).toBeLessThan(1e-9);
    // Sanity: nuclear repulsion is positive, one-electron is negative,
    // coulomb is positive, exchange is negative.
    expect(d.nuclearRepulsion).toBeGreaterThan(0);
    expect(d.oneElectron).toBeLessThan(0);
    expect(d.coulomb).toBeGreaterThan(0);
    expect(d.exchange).toBeLessThan(0);
    // Total ≈ -75 Ha for H₂O/STO-3G.
    expect(d.total).toBeLessThan(-70);
    expect(d.total).toBeGreaterThan(-80);
  });

  test("Li doublet UHF: decomposition sums to total", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(li);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 1, {
      maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const d = decomposeUHFEnergy(uhf, integrals);
    expect(Math.abs(d.checkResidual)).toBeLessThan(1e-9);
    expect(d.oneElectron).toBeLessThan(0);
    expect(d.coulomb).toBeGreaterThan(0);
    expect(d.exchange).toBeLessThan(0);
    expect(d.nuclearRepulsion).toBe(0);   // single atom — no nuclear pairs
  });
});
