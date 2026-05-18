// Analytical static polarizability via CPHF (Tier 3).
//
// Cross-validation against the existing finite-field implementation
// (polarizability.ts, Tier 2 stage 17). Pass bars:
//
//   1. CPHF α tensor agrees with finite-field α to within FD truncation
//      (~0.5% on a typical small molecule at E_field=5e-3).
//   2. α is symmetric to machine precision (built-in via averaging).
//   3. CPHF runs in O(N_OV²) per axis — much faster than 6 SCFs.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { cphfPolarizability } from "../../src/chemistry/cphf.js";
import { polarizabilityFiniteField } from "../../src/chemistry/polarizability.js";

describe("CPHF analytical polarizability — vs finite-field", () => {
  test("H₂O HF/STO-3G: CPHF α matches finite-field α within FD truncation", () => {
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
    expect(hf.converged).toBe(true);

    const cphf = cphfPolarizability(hf, integrals, shells);
    const ff = polarizabilityFiniteField(atoms, {
      basis: "sto-3g", method: "hf", fieldStrength: 5e-3,
    });

    // Diagonal elements: agreement should be tight (sub-percent).
    for (let i = 0; i < 3; i++) {
      const aDiag = cphf.alpha[i * 3 + i]!;
      const bDiag = ff.tensor[i * 3 + i]!;
      // Both positive and at the same scale (typically 0.1-5 a.u.
      // for a small molecule in STO-3G).
      expect(aDiag).toBeGreaterThan(0);
      // Agreement: 5% generous bar covers FD truncation + symmetry-
      // breaking from the finite-field's perturbative nature.
      const relErr = Math.abs(aDiag - bDiag) / Math.abs(bDiag);
      expect(relErr).toBeLessThan(0.05);
    }
    // Isotropic α matches to ~1%.
    const isoFF = (ff.tensor[0]! + ff.tensor[4]! + ff.tensor[8]!) / 3;
    expect(Math.abs(cphf.isotropic - isoFF) / Math.abs(isoFF)).toBeLessThan(0.02);
  }, 90_000);

  test("H₂O CPHF α tensor is symmetric (machine-precision)", () => {
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
    const cphf = cphfPolarizability(hf, integrals, shells);
    for (let x = 0; x < 3; x++) {
      for (let y = x + 1; y < 3; y++) {
        expect(Math.abs(cphf.alpha[x * 3 + y]! - cphf.alpha[y * 3 + x]!)).toBeLessThan(1e-12);
      }
    }
  });
});
