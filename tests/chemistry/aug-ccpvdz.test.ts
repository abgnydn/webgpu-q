// aug-cc-pVDZ basis. Tier 1 bundle: cc-pVDZ + 1 diffuse function per
// angular momentum class on each atom. EMSL Dunning convention.
//
// Pass bars:
//   • H₂O / aug-cc-pvdz spherical: HF matches PySCF to ≤ 0.5 mHa.
//     The diffuse functions buy ~14 mHa of HF energy beyond cc-pVDZ
//     by capturing long-range electron density.
//   • aug-cc-pVDZ HF lies BELOW cc-pVDZ HF (variational principle:
//     larger basis → lower energy).
//   • Spherical aug-cc-pvdz has n = Cartesian n − 2 (one d-shell
//     redundancy in the cc-pVDZ block + one in the aug block).
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";

const half = (104.52 / 2) * Math.PI / 180;
const xH = 0.9572 * Math.sin(half);
const zH = 0.9572 * Math.cos(half);
const h2o: Atom[] = [
  { symbol: "O", pos: [0, 0, 0] },
  { symbol: "H", pos: [ xH, 0, zH] },
  { symbol: "H", pos: [-xH, 0, zH] },
];

describe("aug-cc-pVDZ basis (cc-pVDZ + diffuse)", () => {
  test("H₂O Cartesian n=43, Spherical n=41 (2 d-shell groups → drop 2)", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "aug-cc-pvdz");
    const cart = computeMolecularIntegrals(shells, nuclei);
    const sph  = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    expect(cart.n).toBe(43);
    expect(sph.n).toBe(41);
  }, 30_000);

  test("H₂O HF/aug-cc-pvdz spherical matches PySCF to ≤ 0.5 mHa", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "aug-cc-pvdz");
    const sph = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    const hf = runRHFSCF(sph, 10, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(hf.converged).toBe(true);
    // PySCF aug-cc-pVDZ HF/H₂O at experimental geometry = -76.041358 Ha.
    expect(Math.abs(hf.energy - (-76.041358))).toBeLessThan(5e-4);
  }, 30_000);

  test("aug-cc-pVDZ HF < cc-pVDZ HF (variational: larger basis lowers energy)", () => {
    const cc  = moleculeToShellsNuclei(h2o, "cc-pvdz");
    const aug = moleculeToShellsNuclei(h2o, "aug-cc-pvdz");
    const dzInt  = computeMolecularIntegrals(cc.shells,  cc.nuclei,  { spherical: true });
    const augInt = computeMolecularIntegrals(aug.shells, aug.nuclei, { spherical: true });
    const hfDz  = runRHFSCF(dzInt,  10, { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
    const hfAug = runRHFSCF(augInt, 10, { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
    expect(hfAug.energy).toBeLessThan(hfDz.energy);
    expect((hfDz.energy - hfAug.energy) * 1000).toBeGreaterThan(5);  // gain > 5 mHa
  }, 30_000);
});
