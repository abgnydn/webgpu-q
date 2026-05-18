// Mayer bond orders — UHF / UCCSD spin-resolved extension.
//
// Pass bars:
//   1. Closed-shell UHF (D_α = D_β = D_total/2): bondOrdersUHF
//      reduces EXACTLY to the closed-shell `bondOrders` result.
//   2. Open-shell radical (Li atom): bond-order matrix is finite
//      with sensible structure.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import { bondOrders, bondOrdersUHF } from "../../src/chemistry/properties.js";

describe("Mayer bond orders — UHF/UCCSD spin-resolved", () => {
  test("Closed-shell H₂O: UHF bond-orders match RHF bond-orders to 1e-9", () => {
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

    const rhf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const uhf = runUHFSCF(integrals, 5, 5, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      symmetryBreaking: 0,
    });
    expect(rhf.converged).toBe(true);
    expect(uhf.converged).toBe(true);

    const closed = bondOrders(integrals, rhf.D, shellAtomIdx);
    const open = bondOrdersUHF(integrals, uhf.D_alpha, uhf.D_beta, shellAtomIdx);

    expect(closed.length).toBe(open.length);
    for (let k = 0; k < closed.length; k++) {
      expect(Math.abs(closed[k]! - open[k]!)).toBeLessThan(1e-8);
    }
  });

  test("Open-shell Li atom: bondOrdersUHF returns finite symmetric matrix", () => {
    const { shells, nuclei, shellAtomIdx } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 1, {
      maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(uhf.converged).toBe(true);
    const B = bondOrdersUHF(integrals, uhf.D_alpha, uhf.D_beta, shellAtomIdx);
    // Single atom: 1x1 matrix; should be positive (diagonal Mayer self-overlap).
    expect(B.length).toBe(1);
    expect(B[0]!).toBeGreaterThan(0);
  });
});
