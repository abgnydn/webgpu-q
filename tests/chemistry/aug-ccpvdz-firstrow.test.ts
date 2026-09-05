// aug-cc-pVDZ first-row expansion verification.
//
// Companion to ccpvdz-firstrow.test.ts. With the diffuse functions
// for Li/Be/C/N/F wired in, every cc-pVDZ first-row molecule should
// be runnable at aug-cc-pVDZ, and the augmented basis should give
// a lower (or equal) HF energy than the bare cc-pVDZ run (basis-set
// monotonicity is variational).
//
// Pass bars:
//   1. PRIMARY — both the cc-pVDZ and the aug-cc-pVDZ total energy
//      match the committed PySCF reference to ≤ 0.01 mHa, with the
//      AO count matched too.
//   2. SUPPLEMENTARY — aug lies at or below plain cc-pVDZ, and the
//      diffuse shift equals the reference shift.
//
// HISTORY — why bar 1 exists. This file used to assert ONLY the
// variational ordering plus "shift < 50 mHa", with no absolute
// reference anywhere. The real shifts are 0.56 mHa (LiH), 0.92 mHa
// (CH₄) and 14.59 mHa (HF), so the 50 mHa guard could not fail even
// if every diffuse exponent were wrong; and an ordering-only test
// stays green when BOTH energies are wrong by the same amount —
// exactly how the 1.29 mHa Li/Be cc-pVDZ table defect survived.
//
// References READ (never retyped) from the committed fixture
//   tests/chemistry/elements/pyscf-reference.json
// PySCF 2.13, conv_tol 1e-12; geometries taken from the same rows so
// reference and test always describe the same molecule.
//
// Convention pairing: this file runs the default Cartesian integral
// path, so it pairs with the "cartesian" fixture rows only.
//
// Measured |Δ| on this code path (mHa), Cartesian:
//   cc-pVDZ      LiH 5.6e-5 · CH₄ 9.0e-4 · HF 2.6e-5
//   aug-cc-pVDZ  LiH 6.3e-5 · CH₄ 9.6e-4 · HF 1.9e-4
// Bar is 0.01 mHa — ~10× the worst measurement.

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

function refRow(molecule: string, basis: string): RefRow {
  const r = fixture.rows.find(
    (x) => x.molecule === molecule && x.basis === basis && x.convention === "cartesian" && x.ok,
  );
  if (!r) throw new Error(`no cartesian fixture row for ${molecule}/${basis}`);
  return r;
}

function atomsOf(row: RefRow): Atom[] {
  return row.atoms.map((a) => ({
    symbol: a.symbol as Atom["symbol"],
    pos: [a.pos[0]!, a.pos[1]!, a.pos[2]!],
  }));
}

function hf(atoms: Atom[], basis: "cc-pvdz" | "aug-cc-pvdz"): { e: number; n: number } {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const res = runRHFSCF(integrals, nElectrons, {
    useDIIS: true, maxIter: 200, energyTol: 1e-9, densityTol: 1e-7,
  });
  expect(res.converged).toBe(true);
  return { e: res.energy, n: integrals.n };
}

const BAR_mHa = 0.01;

describe("aug-cc-pVDZ — first-row atom diffuse expansion", () => {
  for (const { label, molecule } of [
    { label: "LiH", molecule: "LiH" },
    { label: "CH₄", molecule: "CH4" },
    { label: "HF molecule", molecule: "HF" },
  ]) {
    test(`${label}: cc-pVDZ and aug-cc-pVDZ both within ${BAR_mHa} mHa of PySCF`, () => {
      const ccRow = refRow(molecule, "cc-pvdz");
      const augRow = refRow(molecule, "aug-cc-pvdz");
      const atoms = atomsOf(ccRow);

      const cc = hf(atoms, "cc-pvdz");
      const aug = hf(atoms, "aug-cc-pvdz");

      expect(cc.n).toBe(ccRow.nao);
      expect(aug.n).toBe(augRow.nao);
      expect(Math.abs((cc.e - ccRow.E_HF!) * 1000)).toBeLessThan(BAR_mHa);
      expect(Math.abs((aug.e - augRow.E_HF!) * 1000)).toBeLessThan(BAR_mHa);

      // Supplementary: variational ordering, and the diffuse shift
      // reproduces the reference shift (no magic window).
      expect(aug.e).toBeLessThanOrEqual(cc.e + 1e-12);
      const refShift = (ccRow.E_HF! - augRow.E_HF!) * 1000;   // mHa, > 0
      expect((cc.e - aug.e) * 1000).toBeCloseTo(refShift, 2);
    }, 360_000);
  }
});
