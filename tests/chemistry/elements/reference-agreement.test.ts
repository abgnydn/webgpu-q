// Per-element RHF agreement against a committed PySCF reference table.
//
// This is the permanent verifier for element coverage: every supported
// element is exercised in a real molecule (hydrides, not bare atoms, so
// two-center integrals actually run) across every supported basis, under
// BOTH d-conventions.
//
// Convention pairing (Gate 0.1, docs/RUN-PLAN-24H-ELEMENTS.md):
//   PySCF spherical (default, 5d)  <-> computeMolecularIntegrals(..., {spherical:true})
//   PySCF cartesian (mol.cart=True) <-> computeMolecularIntegrals(...)  [default path]
// Crossing the conventions produces a ~0.34 mHa phantom disagreement on
// any element carrying d functions -- a basis-set difference, not a bug.
// STO-3G has no d functions below the third row, so both conventions
// coincide there (asserted below).
//
// Regenerate the fixture with:
//   cd ~/dev/ml-research && uv run python \
//     <repo>/scripts/run-element-reference.py --out <repo>/tests/chemistry/elements/pyscf-reference.json
import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { computeMolecularIntegrals } from "../../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type BasisName } from "../../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../../src/chemistry/hf-scf.js";

interface RefRow {
  element: string; molecule: string; basis: string; convention: string;
  atoms: { symbol: string; pos: number[] }[];
  ok: boolean; E_HF?: number; nao?: number;
}

const fixture = JSON.parse(
  readFileSync(new URL("./pyscf-reference.json", import.meta.url), "utf8"),
) as { rows: RefRow[] };

// Same primitives + same convention should agree far tighter than
// chemical accuracy. 0.1 mHa is the repo's existing HF-vs-PySCF bar
// (CLAUDE.md); anything near it means a real basis-data difference.
const BAR_mHa = 0.1;

function ourEnergy(row: RefRow): { energy: number; n: number; converged: boolean } {
  const atoms: Atom[] = row.atoms.map((a) => ({
    symbol: a.symbol as Atom["symbol"],
    pos: [a.pos[0]!, a.pos[1]!, a.pos[2]!],
  }));
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, row.basis as BasisName);
  const integrals = computeMolecularIntegrals(shells, nuclei, {
    spherical: row.convention === "spherical",
  });
  const hf = runRHFSCF(integrals, nElectrons, {
    useDIIS: true, maxIter: 300, energyTol: 1e-11, densityTol: 1e-9,
  });
  return { energy: hf.energy, n: integrals.n, converged: hf.converged };
}

describe("Element coverage: RHF vs PySCF, convention-matched", () => {
  // The fixture is the trust root for every per-element number in this repo,
  // and the loop below emits ZERO tests for a row that is missing or not ok.
  // Without this assertion a regenerated fixture that quietly dropped elements
  // would go green with fewer cells -- the identical failure mode to
  // check-basis-vs-pyscf.py covering only 8 of 18 elements when it was first
  // wired, which is what this whole branch exists to stop repeating.
  test("fixture is complete: 18 elements x 3 bases x 2 conventions, all ok", () => {
    const EXPECTED_ELEMENTS = 18, EXPECTED_BASES = 3, EXPECTED_CONVENTIONS = 2;
    expect(fixture.rows.length).toBe(EXPECTED_ELEMENTS * EXPECTED_BASES * EXPECTED_CONVENTIONS);
    expect(fixture.rows.every((r) => r.ok)).toBe(true);
    expect(new Set(fixture.rows.map((r) => r.element)).size).toBe(EXPECTED_ELEMENTS);
    expect(new Set(fixture.rows.map((r) => r.basis)).size).toBe(EXPECTED_BASES);
    expect(new Set(fixture.rows.map((r) => r.convention)).size).toBe(EXPECTED_CONVENTIONS);
  });

  for (const row of fixture.rows) {
    if (!row.ok) continue;
    const label = `${row.element} · ${row.molecule} · ${row.basis} · ${row.convention}`;
    test(`${label}: |ΔE| ≤ ${BAR_mHa} mHa`, () => {
      const { energy, n, converged } = ourEnergy(row);
      // A non-converged SCF is a failure, never a datum.
      expect(converged).toBe(true);
      expect(Number.isFinite(energy)).toBe(true);
      expect(n).toBe(row.nao);
      const dmHa = (energy - row.E_HF!) * 1000;
      expect(Math.abs(dmHa)).toBeLessThan(BAR_mHa);
    }, 300_000);
  }

  test("STO-3G is d-convention-free for rows 1-2 (reference sanity)", () => {
    const sph = new Map<string, number>();
    const cart = new Map<string, number>();
    for (const r of fixture.rows) {
      if (r.basis !== "sto-3g" || !r.ok) continue;
      (r.convention === "spherical" ? sph : cart).set(r.molecule, r.E_HF!);
    }
    expect(sph.size).toBeGreaterThan(0);
    for (const [mol, e] of sph) {
      expect(cart.get(mol)).toBeCloseTo(e, 12);
    }
  });
});
