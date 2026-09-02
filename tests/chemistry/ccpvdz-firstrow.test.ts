// cc-pVDZ basis-set expansion to Li, Be, C, N, F (Tier 3).
//
// Closes the "Basis-set atom coverage" gap in LIMITATIONS.md §1.
// Before this commit, cc-pVDZ was wired only for H and O (the
// minimum needed for water). This adds Li, Be, C, N, F so the
// comparison harness can run cc-pVDZ on LiH, BeH₂, CH₄, NH₃, HF.
//
// Pass bars per molecule:
//   1. PRIMARY — the cc-pVDZ RHF total energy matches the committed
//      PySCF reference to ≤ 0.01 mHa, and the AO count matches too.
//   2. SUPPLEMENTARY — the cc-pVDZ HF energy is lower than the
//      STO-3G HF energy on the same geometry (variational: bigger
//      basis = lower or equal energy).
//   3. No NaN, no throws, no integral pathology.
//
// HISTORY — why bar 1 exists. This file used to claim "verified vs
// PySCF" with "loose (10 mHa)" tolerances while actually asserting
// windows 200-300 mHa wide against 2-decimal literals: LiH 250
// ("≈ -7.98" checked as -8.10 < E < -7.85), BeH₂ 300, CH₄ 200,
// NH₃ 200, HF 300. A 250 mHa window cannot see a wrong basis table:
// the Li/Be cc-pVDZ tables were in fact wrong by up to 1.29 mHa and
// this suite passed throughout. The window is now the same 0.01 mHa
// class of bar the rest of the chemistry suite uses.
//
// (This comment previously said "116-170 mHa wide" while printing an
// example spanning 250 mHa. 116 mHa is the ONE-SIDED distance from the
// LiH reference -7.983651 to the lower edge -8.10 — a half-width read
// as a width. Corrected 2026-09-02; the widths above are differences of
// the literals on lines 57-58, 73-74, 92-93, 118-119 and 133-134.)
//
// References are READ (never retyped) from the committed fixture
//   tests/chemistry/elements/pyscf-reference.json
// PySCF 2.13, conv_tol 1e-12. Geometries come from the fixture rows
// as well, so reference and test always describe the same molecule —
// the previous hardcoded geometries drifted from the reference ones
// (and the old NH₃ construction actually built an HNH angle of 88°,
// not the 106.7° its comment claimed).
//
// Convention pairing (Gate 0.1, same rule as
// tests/chemistry/elements/reference-agreement.test.ts):
//   PySCF cartesian (mol.cart=True) <-> computeMolecularIntegrals(...) [default]
// This file runs the default Cartesian path, so it pairs with the
// "cartesian" fixture rows only. Crossing conventions would inject a
// phantom ~0.1-0.3 mHa disagreement on any d-carrying element.
//
// Measured |Δ| on this code path (mHa), cc-pVDZ / Cartesian:
//   LiH 5.6e-5 · BeH₂ 1.7e-4 · CH₄ 9.0e-4 · NH₃ 7.6e-4 · HF 2.6e-5
// Bar is 0.01 mHa — ~11× the worst measurement, 130× below the
// 1.29 mHa defect that motivated this rewrite.

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

function runHF(atoms: Atom[], basis: "sto-3g" | "cc-pvdz"): { e: number; n: number; converged: boolean } {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons, {
    useDIIS: true, maxIter: 200, energyTol: 1e-9, densityTol: 1e-7,
  });
  return { e: hf.energy, n: integrals.n, converged: hf.converged };
}

// 0.01 mHa = 10 µHa. See the header for how this was set.
const BAR_mHa = 0.01;

const MOLECULES: { label: string; molecule: string; timeout: number }[] = [
  { label: "LiH (Li)", molecule: "LiH", timeout: 60_000 },
  { label: "BeH₂ (Be)", molecule: "BeH2", timeout: 60_000 },
  { label: "CH₄ (C)", molecule: "CH4", timeout: 120_000 },
  { label: "NH₃ (N)", molecule: "NH3", timeout: 120_000 },
  { label: "HF (F)", molecule: "HF", timeout: 60_000 },
];

describe("cc-pVDZ — first-row atom expansion (Li, Be, C, N, F)", () => {
  for (const { label, molecule, timeout } of MOLECULES) {
    test(`${label} cc-pVDZ: |ΔE_HF| ≤ ${BAR_mHa} mHa vs PySCF (Cartesian)`, () => {
      const row = refRow(molecule, "cc-pvdz");
      const { e, n, converged } = runHF(atomsOf(row), "cc-pvdz");
      // A non-converged SCF is a failure, never a datum.
      expect(converged, `SCF did not converge for ${label}`).toBe(true);
      expect(Number.isFinite(e)).toBe(true);
      expect(n).toBe(row.nao);
      expect(Math.abs((e - row.E_HF!) * 1000)).toBeLessThan(BAR_mHa);
    }, timeout);

    test(`${label}: cc-pVDZ lies below STO-3G (variational, supplementary)`, () => {
      const row = refRow(molecule, "cc-pvdz");
      const atoms = atomsOf(row);
      const sto3g = runHF(atoms, "sto-3g");
      const ccpvdz = runHF(atoms, "cc-pvdz");
      expect(sto3g.converged).toBe(true);
      expect(ccpvdz.converged).toBe(true);
      expect(ccpvdz.e).toBeLessThanOrEqual(sto3g.e);
    }, timeout);
  }
});
