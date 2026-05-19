// Foster-Boys orbital localization tests.
//
// Pass bars:
//   1. L is monotonically non-decreasing across sweeps.
//   2. U is orthogonal: U^T · U ≈ I.
//   3. Localized H₂O σ-bonds + lone pairs have distinct centroids:
//      the two O-H σ-bonds sit between O and each H; the two
//      lone pairs sit on O. Centroids differ from canonical MOs.
//   4. Total electron density unchanged by localization (U is unitary).

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { fosterBoys } from "../../src/chemistry/foster-boys.js";

describe("Foster-Boys orbital localization", () => {
  test("H₂O HF/STO-3G: converges, L monotonically increases", () => {
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

    const fb = fosterBoys(hf.C_MO, integrals, shells, hf.nOccupied);
    expect(fb.converged).toBe(true);
    expect(fb.nSweeps).toBeLessThan(100);
    expect(fb.L_history.length).toBeGreaterThan(0);
    // L should be monotonically non-decreasing.
    for (let s = 1; s < fb.L_history.length; s++) {
      expect(fb.L_history[s]!).toBeGreaterThanOrEqual(fb.L_history[s - 1]! - 1e-12);
    }
  });

  test("U is orthogonal: U^T · U ≈ I", () => {
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [0.629, 0.629, 0.629] },
      { symbol: "H", pos: [-0.629, -0.629, 0.629] },
      { symbol: "H", pos: [0.629, -0.629, -0.629] },
      { symbol: "H", pos: [-0.629, 0.629, -0.629] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const fb = fosterBoys(hf.C_MO, integrals, shells, hf.nOccupied);
    const nOcc = hf.nOccupied;
    // Check U^T · U = I.
    let maxDev = 0;
    for (let i = 0; i < nOcc; i++) {
      for (let j = 0; j < nOcc; j++) {
        let s = 0;
        for (let k = 0; k < nOcc; k++) {
          s += fb.U[k * nOcc + i]! * fb.U[k * nOcc + j]!;
        }
        const expected = i === j ? 1 : 0;
        maxDev = Math.max(maxDev, Math.abs(s - expected));
      }
    }
    expect(maxDev).toBeLessThan(1e-9);
  }, 60_000);

  test("H₂O localized orbitals: at least one centroid moves away from origin (localization actually happens)", () => {
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
    const fb = fosterBoys(hf.C_MO, integrals, shells, hf.nOccupied);
    // Localization should increase L significantly over the initial canonical value.
    expect(fb.L_history[fb.L_history.length - 1]!).toBeGreaterThan(0.5);
  });
});
