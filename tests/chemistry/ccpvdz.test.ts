// cc-pVDZ basis validation. Phase E stage 2 — first non-minimal
// basis set. Validates:
//   1. d-shell ERIs work via the existing CG framework (sanity).
//   2. H₂O cc-pVDZ HF matches PySCF to ≤ 0.001 mHa, in BOTH d
//      conventions, each against its own PySCF reference.
//   3. H₂O cc-pVDZ MP2 captures dramatically more correlation
//      than STO-3G (207 mHa vs 50 mHa).
//
// Note on Cartesian vs spherical d: this codebase uses 6 Cartesian
// d-functions (d_xx, d_yy, d_zz, d_xy, d_xz, d_yz) per d-shell by
// default; `{spherical: true}` switches to the 5 spherical-harmonic
// d functions PySCF uses by default. The redundant linear combination
// (d_xx + d_yy + d_zz)/√3 is an extra "3s" component that makes the
// Cartesian basis slightly bigger and thus variationally LOWER — for
// H₂O/cc-pVDZ that is worth 0.340 mHa.
//
// That 0.340 mHa is a BASIS difference, not an error budget: it must
// never be hidden inside a tolerance. This file used to compare a
// Cartesian run against the SPHERICAL reference (-76.0267) inside a
// 1 mHa window, which absorbed the whole convention gap. Each run is
// now paired with its own reference, read from the committed fixture
// tests/chemistry/elements/pyscf-reference.json (PySCF 2.13,
// conv_tol 1e-12):
//   H2O · cc-pvdz · cartesian  E_HF = -76.02713907180959  (nao 25)
//   H2O · cc-pvdz · spherical  E_HF = -76.02679869746876  (nao 24)
// Measured |Δ| for this code path: 2.7e-4 mHa (cart), 2.8e-4 mHa (sph).
import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runMP2 } from "../../src/chemistry/mp2.js";
import { S_cg, makeCGShell } from "../../src/chemistry/integrals-cg.js";

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

// A reference is only a reference at ITS geometry.
function expectSameGeometry(row: RefRow, atoms: Atom[]): void {
  expect(row.atoms.length).toBe(atoms.length);
  row.atoms.forEach((a, i) => {
    expect(a.symbol).toBe(atoms[i]!.symbol);
    a.pos.forEach((c, k) => expect(c).toBeCloseTo(atoms[i]!.pos[k]!, 12));
  });
}

const ORIGIN = [0, 0, 0] as const;

describe("d-shell CG-integral sanity", () => {
  // A normalized single-primitive d Gaussian (α = 1) at the origin
  // should self-overlap to 1 by construction.
  const SINGLE = { alpha: [1.0] as const, c: [1.0] as const };

  test("each Cartesian d_ii self-overlap is 1", () => {
    for (const t of [[2, 0, 0], [0, 2, 0], [0, 0, 2]] as const) {
      const d = makeCGShell(SINGLE, ORIGIN, t);
      expect(S_cg(d, d)).toBeCloseTo(1, 10);
    }
  });

  test("each Cartesian d_ij (i≠j) self-overlap is 1", () => {
    for (const t of [[1, 1, 0], [1, 0, 1], [0, 1, 1]] as const) {
      const d = makeCGShell(SINGLE, ORIGIN, t);
      expect(S_cg(d, d)).toBeCloseTo(1, 10);
    }
  });

  test("Cartesian d_xx ⊥ d_xy by parity", () => {
    const dxx = makeCGShell(SINGLE, ORIGIN, [2, 0, 0]);
    const dxy = makeCGShell(SINGLE, ORIGIN, [1, 1, 0]);
    expect(S_cg(dxx, dxy)).toBeCloseTo(0, 12);
  });

  test("⟨d_xx | d_yy⟩ = 1/3 (Cartesian d's are not mutually orthogonal)", () => {
    // The "redundant" overlap that makes 6 Cartesian d's bigger
    // than 5 spherical d's. Löwdin orthogonalization in SCF takes
    // care of this transparently.
    const dxx = makeCGShell(SINGLE, ORIGIN, [2, 0, 0]);
    const dyy = makeCGShell(SINGLE, ORIGIN, [0, 2, 0]);
    expect(S_cg(dxx, dyy)).toBeCloseTo(1 / 3, 10);
  });
});

describe("H₂O in cc-pVDZ", () => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half);
  const z = 0.9572 * Math.cos(half);
  const atoms: Atom[] = [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];

  test("basis size: 15 shells on O + 5 on each H = 25 CG shells", () => {
    const { shells } = moleculeToShellsNuclei(atoms, "cc-pvdz");
    expect(shells.length).toBe(25);
  });

  test("HF energy matches PySCF to ≤ 0.001 mHa in each d convention", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, "cc-pvdz");
    const opts = { maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8 };

    for (const convention of ["cartesian", "spherical"] as const) {
      const row = refRow("H2O", "cc-pvdz", convention);
      expectSameGeometry(row, atoms);
      const integrals = computeMolecularIntegrals(
        shells, nuclei, convention === "spherical" ? { spherical: true } : undefined,
      );
      expect(integrals.n).toBe(row.nao);
      const hf = runRHFSCF(integrals, nElectrons, opts);
      expect(hf.converged, `SCF did not converge (${convention})`).toBe(true);
      // Measured |Δ|: 2.7e-4 mHa (cart) / 2.8e-4 mHa (sph). Bar ~4× above.
      expect(Math.abs((hf.energy - row.E_HF!) * 1000)).toBeLessThan(0.001);
    }
  }, 60_000);

  test("Cartesian d lies 0.340 mHa below spherical d (basis difference, not error)", () => {
    // The gap the old ≤1 mHa cross-convention window used to swallow.
    // Reference gap from the fixture rows themselves — no magic number.
    const refGap = (refRow("H2O", "cc-pvdz", "cartesian").E_HF!
      - refRow("H2O", "cc-pvdz", "spherical").E_HF!) * 1000;   // -0.3404 mHa
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, "cc-pvdz");
    const opts = { maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8 };
    const cart = runRHFSCF(computeMolecularIntegrals(shells, nuclei), nElectrons, opts);
    const sph = runRHFSCF(
      computeMolecularIntegrals(shells, nuclei, { spherical: true }), nElectrons, opts,
    );
    expect(cart.energy).toBeLessThan(sph.energy);   // more functions → lower
    expect((cart.energy - sph.energy) * 1000).toBeCloseTo(refGap, 3);
  }, 60_000);

  test("MP2 captures ~4× more correlation in cc-pVDZ vs STO-3G", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, "cc-pvdz");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
    });
    const mp2 = runMP2(hf, integrals);
    // STO-3G H₂O MP2 captures ~36 mHa of correlation; cc-pVDZ
    // captures ~205 mHa — much more because the bigger basis can
    // resolve more dynamic correlation.
    expect(mp2.correlationEnergy).toBeLessThan(-0.15);   // ≥ 150 mHa
    expect(mp2.correlationEnergy).toBeGreaterThan(-0.25); // ≤ 250 mHa
    // The window above is a shape check; pin the absolute value too.
    // All-electron MP2/cc-pVDZ, CARTESIAN d (matching the integrals
    // built above), PySCF 2.13.1, conv_tol 1e-12, at this exact geometry:
    //   gto.M(atom=[('O',(0,0,0)),('H',(x,0,z)),('H',(-x,0,z))],
    //         basis='cc-pvdz', unit='Angstrom', cart=True)
    //   scf.RHF -> mp.MP2  =>  E_corr = -0.20751927093106037
    // Measured |Δ| = 2.4e-5 mHa on E_corr; bar ~4× above.
    expect(Math.abs((mp2.correlationEnergy - (-0.20751927093106037)) * 1000))
      .toBeLessThan(0.0001);
  }, 60_000);
});
