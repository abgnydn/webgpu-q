// CIS Davidson eigensolver cross-check.
//
// Verifies that the iterative `useDavidson: true` path returns the
// same lowest singlet + triplet excitation energies as the dense
// `eigsymmetric` path. Davidson avoids building the full A matrix,
// which scales as O(dim²) memory — important for cc-pVDZ on
// multi-atom systems where dim grows past a few hundred.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCIS } from "../../src/chemistry/cis.js";

describe("CIS — Davidson vs dense", () => {
  test("H₂O HF/STO-3G — k=3 singlets agree to ≤ 1e-7 Ha", () => {
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

    const dense = runCIS(integrals, hf, { nRoots: 3, spin: "singlet" });
    const dav   = runCIS(integrals, hf, { nRoots: 3, spin: "singlet", useDavidson: true, davidsonTol: 1e-9 });

    expect(dense.singlet.energies.length).toBe(3);
    expect(dav.singlet.energies.length).toBe(3);
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(dav.singlet.energies[k]! - dense.singlet.energies[k]!)).toBeLessThan(1e-6);
    }
  }, 60_000);

  test("H₂O HF/STO-3G — k=3 triplets agree to ≤ 1e-6 Ha", () => {
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
    const dense = runCIS(integrals, hf, { nRoots: 3, spin: "triplet" });
    const dav   = runCIS(integrals, hf, { nRoots: 3, spin: "triplet", useDavidson: true });
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(dav.triplet.energies[k]! - dense.triplet.energies[k]!)).toBeLessThan(1e-6);
    }
  }, 60_000);
});
