// QCSchema export — structural validation.
//
// Pass bars:
//   1. Output is valid JSON.
//   2. Top-level fields: schema_name = "qcschema_output", schema_version = 1,
//      driver = "energy", success = true, return_result = totalEnergy.
//   3. molecule sub-object: symbols / geometry (3·N reals in bohr) /
//      atomic_numbers matches input atoms.
//   4. properties sub-object: nuclear_repulsion_energy + scf_total_energy
//      + calcinfo_natom / nbasis / nmo / nalpha / nbeta.
//   5. Open-shell: molecular_multiplicity = nα − nβ + 1; calcinfo_nalpha /
//      nbeta from explicit spin counts; scf_s_squared if provided.
//   6. Method tagged correlation_energy in MP2/CCSD method names.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import {
  toQCSchemaClosedShell, toQCSchemaOpenShell,
} from "../../src/chemistry/qcschema.js";

describe("QCSchema export", () => {
  test("H₂O HF/STO-3G: valid AtomicResult JSON, closed-shell shape", () => {
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

    const json = toQCSchemaClosedShell({
      atoms,
      basis: "sto-3g",
      method: "hf",
      scfEnergy: hf.energy,
      totalEnergy: hf.energy,
      oneElectron: undefined,
      twoElectron: undefined,
      nuclearRepulsion: integrals.Vnn,
      nBasis: integrals.n,
      nMO: integrals.n,
      nElectrons,
      nOccupied: hf.nOccupied,
      success: hf.converged,
    });

    const parsed = JSON.parse(json);
    expect(parsed.schema_name).toBe("qcschema_output");
    expect(parsed.schema_version).toBe(1);
    expect(parsed.driver).toBe("energy");
    expect(parsed.success).toBe(true);
    expect(parsed.return_result).toBeCloseTo(hf.energy, 8);
    expect(parsed.model.method).toBe("hf");
    expect(parsed.model.basis).toBe("sto-3g");

    expect(parsed.molecule.schema_name).toBe("qcschema_molecule");
    expect(parsed.molecule.symbols).toEqual(["O", "H", "H"]);
    expect(parsed.molecule.atomic_numbers).toEqual([8, 1, 1]);
    expect(parsed.molecule.geometry.length).toBe(9);

    expect(parsed.properties.calcinfo_natom).toBe(3);
    expect(parsed.properties.calcinfo_nbasis).toBe(integrals.n);
    expect(parsed.properties.calcinfo_nalpha).toBe(hf.nOccupied);
    expect(parsed.properties.calcinfo_nbeta).toBe(hf.nOccupied);
    expect(parsed.properties.nuclear_repulsion_energy).toBeCloseTo(integrals.Vnn, 10);
    expect(parsed.properties.scf_total_energy).toBeCloseTo(hf.energy, 10);
  });

  test("CCSD method-tagged correlation_energy propagates correctly", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const eCorr = -0.012345;  // synthetic CCSD correlation
    const json = toQCSchemaClosedShell({
      atoms, basis: "sto-3g", method: "ccsd",
      scfEnergy: hf.energy,
      correlationEnergy: eCorr,
      totalEnergy: hf.energy + eCorr,
      nuclearRepulsion: integrals.Vnn,
      nBasis: integrals.n, nMO: integrals.n,
      nElectrons, nOccupied: hf.nOccupied,
      success: true,
    });
    const parsed = JSON.parse(json);
    expect(parsed.properties.scf_correlation_energy).toBeCloseTo(eCorr, 10);
    expect(parsed.properties.ccsd_correlation_energy).toBeCloseTo(eCorr, 10);
    // mp2_correlation_energy should be absent for ccsd method tag.
    expect(parsed.properties.mp2_correlation_energy).toBeUndefined();
  });

  test("Open-shell H atom UHF: multiplicity + nalpha/nbeta + s_squared", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "H", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 1, 0, {
      maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const json = toQCSchemaOpenShell({
      atoms: [{ symbol: "H", pos: [0, 0, 0] }],
      basis: "sto-3g", method: "uhf",
      scfEnergy: uhf.energy,
      totalEnergy: uhf.energy,
      nuclearRepulsion: integrals.Vnn,
      nBasis: integrals.n, nMO: integrals.n,
      nAlpha: 1, nBeta: 0,
      s2: uhf.s2,
      success: uhf.converged,
    });
    const parsed = JSON.parse(json);
    expect(parsed.molecule.molecular_multiplicity).toBe(2);  // 1 - 0 + 1
    expect(parsed.properties.calcinfo_nalpha).toBe(1);
    expect(parsed.properties.calcinfo_nbeta).toBe(0);
    expect(parsed.properties.scf_s_squared).toBeCloseTo(0.75, 6);
  });

  test("Extras field is preserved", () => {
    const atoms: Atom[] = [{ symbol: "H", pos: [0, 0, 0] }];
    const json = toQCSchemaClosedShell({
      atoms, basis: "sto-3g", method: "hf",
      scfEnergy: -0.5, totalEnergy: -0.5,
      nuclearRepulsion: 0,
      nBasis: 1, nMO: 1, nElectrons: 2, nOccupied: 1,
      success: true,
      extras: { custom_field: "hello", iteration_history: [1, 2, 3] },
    });
    const parsed = JSON.parse(json);
    expect(parsed.extras.custom_field).toBe("hello");
    expect(parsed.extras.iteration_history).toEqual([1, 2, 3]);
    expect(parsed.extras.webgpu_q_version).toBeDefined();
  });
});
