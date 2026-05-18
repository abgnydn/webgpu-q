// UHF CPHF analytical static polarizability — cross-checks.
//
// Pass bars:
//   1. Closed-shell UHF (n_α = n_β = 5, no symmetry breaking) on
//      H₂O matches the RHF CPHF result from `cphf.ts` to ≤ 1e-7
//      per tensor element (spin-orbital basis independence).
//   2. UHF CPHF on Li atom doublet: returns a real, positive,
//      finite isotropic α. Lithium's STO-3G isotropic α is ~25-50
//      a.u. order of magnitude.
//   3. α tensor is symmetric to machine precision.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import { cphfPolarizability } from "../../src/chemistry/cphf.js";
import { uhfCphfPolarizability } from "../../src/chemistry/uhf-cphf.js";

describe("UHF CPHF — analytical static polarizability", () => {
  test("Closed-shell H₂O: UHF CPHF α matches RHF CPHF α to ≤ 1e-7 per element", () => {
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

    const rhfA = cphfPolarizability(rhf, integrals, shells);
    const uhfA = uhfCphfPolarizability(uhf, integrals, shells);

    for (let k = 0; k < 9; k++) {
      expect(Math.abs(uhfA.alpha[k]! - rhfA.alpha[k]!)).toBeLessThan(1e-7);
    }
    expect(Math.abs(uhfA.isotropic - rhfA.isotropic)).toBeLessThan(1e-7);
  });

  test("Li atom doublet cc-pVDZ: real, positive isotropic α", () => {
    // Li at minimal STO-3G has no α-virtual (Li STO-3G = 1s + 2s, no 2p),
    // and the only β transition 1s → 2s is parity-forbidden — so α = 0
    // in that basis. Use cc-pVDZ which adds p functions; the 2s → 2p
    // β transition is dipole-allowed.
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ], "cc-pvdz");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 1, {
      maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(uhf.converged).toBe(true);
    const r = uhfCphfPolarizability(uhf, integrals, shells);
    expect(Number.isFinite(r.isotropic)).toBe(true);
    expect(r.isotropic).toBeGreaterThan(0);
    // Symmetric.
    for (let x = 0; x < 3; x++) {
      for (let y = x + 1; y < 3; y++) {
        expect(Math.abs(r.alpha[x * 3 + y]! - r.alpha[y * 3 + x]!)).toBeLessThan(1e-10);
      }
    }
  }, 60_000);
});
