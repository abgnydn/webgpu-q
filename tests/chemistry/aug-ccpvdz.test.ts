// aug-cc-pVDZ basis. Tier 1 bundle: cc-pVDZ + 1 diffuse function per
// angular momentum class on each atom. EMSL Dunning convention.
//
// Pass bars:
//   • H₂O / aug-cc-pvdz spherical: HF matches PySCF to ≤ 0.001 mHa.
//     Reference is READ from the committed fixture
//     tests/chemistry/elements/pyscf-reference.json (H2O · aug-cc-pvdz ·
//     spherical, E_HF = -76.04142796052426, PySCF 2.13 @ conv_tol 1e-12),
//     not retyped. The old hardcoded -76.041358 disagreed with that
//     fixture by 0.070 mHa — two "PySCF references" in one repo.
//     Measured |Δ| for this code path: 2.6e-4 mHa (2.6e-7 Ha).
//     The diffuse functions buy ~14 mHa of HF energy beyond cc-pVDZ
//     by capturing long-range electron density.
//   • aug-cc-pVDZ HF lies BELOW cc-pVDZ HF (variational principle:
//     larger basis → lower energy).
//   • Spherical aug-cc-pvdz has n = Cartesian n − 2 (one d-shell
//     redundancy in the cc-pVDZ block + one in the aug block).
//
// Convention pairing (same rule as elements/reference-agreement.test.ts):
//   PySCF spherical (default 5d) <-> computeMolecularIntegrals(..., {spherical:true})
//   PySCF cartesian (mol.cart)   <-> computeMolecularIntegrals(...)  [default]
import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";

interface RefRow {
  molecule: string; basis: string; convention: string;
  atoms: { symbol: string; pos: number[] }[];
  ok: boolean; E_HF?: number; nao?: number;
}
const fixture = JSON.parse(
  readFileSync(new URL("./elements/pyscf-reference.json", import.meta.url), "utf8"),
) as { rows: RefRow[] };

function refRow(molecule: string, basis: string, convention: string): RefRow {
  const r = fixture.rows.find(
    (x) => x.molecule === molecule && x.basis === basis && x.convention === convention && x.ok,
  );
  if (!r) throw new Error(`no fixture row for ${molecule}/${basis}/${convention}`);
  return r;
}

// The reference is only a reference at ITS geometry. Guard against a future
// fixture regeneration silently moving the nuclei out from under this test.
function expectSameGeometry(row: RefRow, atoms: Atom[]): void {
  expect(row.atoms.length).toBe(atoms.length);
  row.atoms.forEach((a, i) => {
    expect(a.symbol).toBe(atoms[i]!.symbol);
    a.pos.forEach((c, k) => expect(c).toBeCloseTo(atoms[i]!.pos[k]!, 12));
  });
}

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
  }, 360_000);

  test("H₂O HF/aug-cc-pvdz spherical matches PySCF to ≤ 0.001 mHa", () => {
    const row = refRow("H2O", "aug-cc-pvdz", "spherical");
    expectSameGeometry(row, h2o);
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "aug-cc-pvdz");
    const sph = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    expect(sph.n).toBe(row.nao);
    const hf = runRHFSCF(sph, 10, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(hf.converged).toBe(true);
    // Full-precision PySCF value straight from the fixture; measured
    // |Δ| = 2.6e-4 mHa, bar set ~4× above it.
    expect(Math.abs((hf.energy - row.E_HF!) * 1000)).toBeLessThan(0.001);
  }, 360_000);

  test("aug-cc-pVDZ HF < cc-pVDZ HF (variational: larger basis lowers energy)", () => {
    const cc  = moleculeToShellsNuclei(h2o, "cc-pvdz");
    const aug = moleculeToShellsNuclei(h2o, "aug-cc-pvdz");
    const dzInt  = computeMolecularIntegrals(cc.shells,  cc.nuclei,  { spherical: true });
    const augInt = computeMolecularIntegrals(aug.shells, aug.nuclei, { spherical: true });
    const hfDz  = runRHFSCF(dzInt,  10, { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
    const hfAug = runRHFSCF(augInt, 10, { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
    expect(hfAug.energy).toBeLessThan(hfDz.energy);
    expect((hfDz.energy - hfAug.energy) * 1000).toBeGreaterThan(5);  // gain > 5 mHa
  }, 360_000);
});
