// aug-cc-pVDZ first-row expansion verification.
//
// Companion to ccpvdz-firstrow.test.ts. With the diffuse functions
// for Li/Be/C/N/F wired in, every cc-pVDZ first-row molecule should
// be runnable at aug-cc-pVDZ, and the augmented basis should give
// a lower (or equal) HF energy than the bare cc-pVDZ run (basis-set
// monotonicity is variational).

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";

function hf(atoms: Atom[], basis: "cc-pvdz" | "aug-cc-pvdz"): number {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const res = runRHFSCF(integrals, nElectrons, {
    useDIIS: true, maxIter: 200, energyTol: 1e-9, densityTol: 1e-7,
  });
  expect(res.converged).toBe(true);
  return res.energy;
}

describe("aug-cc-pVDZ — first-row atom diffuse expansion", () => {
  test("LiH: aug-cc-pVDZ HF lies variationally below cc-pVDZ HF", () => {
    const atoms: Atom[] = [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.6] },
    ];
    const ccpvdz = hf(atoms, "cc-pvdz");
    const aug    = hf(atoms, "aug-cc-pvdz");
    expect(aug).toBeLessThanOrEqual(ccpvdz + 1e-12);
    // Diffuse shift is small for neutral closed-shell at equilibrium.
    expect(ccpvdz - aug).toBeLessThan(0.05); // < 50 mHa shift
  }, 60_000);

  test("CH₄: aug-cc-pVDZ HF lies variationally below cc-pVDZ HF", () => {
    const r3 = 1.09 / Math.sqrt(3);
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [ r3,  r3,  r3] },
      { symbol: "H", pos: [ r3, -r3, -r3] },
      { symbol: "H", pos: [-r3,  r3, -r3] },
      { symbol: "H", pos: [-r3, -r3,  r3] },
    ];
    const ccpvdz = hf(atoms, "cc-pvdz");
    const aug    = hf(atoms, "aug-cc-pvdz");
    expect(aug).toBeLessThanOrEqual(ccpvdz + 1e-12);
    expect(ccpvdz - aug).toBeLessThan(0.05);
  }, 120_000);

  test("HF molecule: aug-cc-pVDZ HF lies variationally below cc-pVDZ HF", () => {
    const atoms: Atom[] = [
      { symbol: "F", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.917] },
    ];
    const ccpvdz = hf(atoms, "cc-pvdz");
    const aug    = hf(atoms, "aug-cc-pvdz");
    expect(aug).toBeLessThanOrEqual(ccpvdz + 1e-12);
    expect(ccpvdz - aug).toBeLessThan(0.05);
  }, 60_000);
});
