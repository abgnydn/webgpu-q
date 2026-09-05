// CCSD(T) validation. Strategy:
//   1. Structural zero: H₂ (NOCC=2 < 3) and LiH s-only (NVIRT=2 < 3)
//      cannot support triple excitations. (T) must be exactly 0,
//      independent of formula. Strong correctness check.
//   2. Variational direction: for closed-shell molecules near
//      equilibrium, (T) is negative (lowers energy further past
//      CCSD). E_CCSD(T) < E_CCSD.
//   3. Magnitude sanity: in STO-3G, (T) is sub-mHa for these
//      molecules (small basis → small dynamic correlation).
//      |E_CCSD(T) − E_FCI| ≤ 0.5 mHa for every molecule with
//      defined triples.
//   4. CCSD(T) closes most of the CCSD−FCI gap (it doesn't have
//      to close all — Q_4 quadruples and higher are still missing).
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runCCSDT } from "../../src/chemistry/ccsd-t.js";
import { buildMoleculeFCI } from "../../src/chemistry/molecule-builder.js";

interface MolRef {
  name: string;
  atoms: Atom[];
  E_FCI: number;
}

const MOLECULES: MolRef[] = [
  { name: "H2", atoms: [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ], E_FCI: -1.137270 },
  { name: "LiH (s-only)", atoms: [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ], E_FCI: -7.843394 },
  { name: "BeH2 (full)", atoms: [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -1.34] },
      { symbol: "H",  pos: [0, 0,  1.34] },
    ], E_FCI: -15.594861 },
  { name: "H2O", atoms: (() => {
      const half = (104.52 / 2) * Math.PI / 180;
      const x = 0.9572 * Math.sin(half);
      const z = 0.9572 * Math.cos(half);
      return [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ x, 0, z] },
        { symbol: "H", pos: [-x, 0, z] },
      ] as Atom[];
    })(), E_FCI: -75.012403 },
];

function runStack(atoms: Atom[]) {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons, {
    maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
  });
  const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-10 });
  const ccsdt = runCCSDT(ccsd, hf, integrals);
  return { hf, ccsd, ccsdt, integrals, nElectrons };
}

describe("CCSD(T) — structural zero (no triples possible)", () => {
  test("H₂ STO-3G: NOCC=2 < 3 → E_(T) is exactly 0", () => {
    const m = MOLECULES.find((x) => x.name === "H2")!;
    const { ccsdt } = runStack(m.atoms);
    expect(ccsdt.tripleCorrection).toBe(0);
  });

  test("LiH s-only STO-3G: NVIRT=2 < 3 → E_(T) is exactly 0", () => {
    const m = MOLECULES.find((x) => x.name === "LiH (s-only)")!;
    const { ccsdt } = runStack(m.atoms);
    expect(ccsdt.tripleCorrection).toBe(0);
  });
});

describe("CCSD(T) — non-trivial molecules", () => {
  for (const m of MOLECULES.filter((x) => x.name === "BeH2 (full)" || x.name === "H2O")) {
    test(`${m.name}: (T) is negative and CCSD(T) is within 0.5 mHa of FCI`, () => {
      const { ccsd, ccsdt } = runStack(m.atoms);
      // (T) lowers energy further past CCSD.
      expect(ccsdt.tripleCorrection).toBeLessThan(0);
      expect(ccsdt.totalEnergy).toBeLessThan(ccsd.totalEnergy);
      // CCSD(T) is within 0.5 mHa of FCI for these closed-shell
      // molecules in STO-3G near equilibrium.
      expect(Math.abs(ccsdt.totalEnergy - m.E_FCI)).toBeLessThan(0.5e-3);
    }, 30_000);
  }
});

describe("CCSD(T) — closes most of the CCSD−FCI gap", () => {
  for (const m of MOLECULES.filter((x) => x.name === "BeH2 (full)" || x.name === "H2O")) {
    test(`${m.name}: |CCSD(T)−FCI| < |CCSD−FCI|`, () => {
      const { ccsd, ccsdt } = runStack(m.atoms);
      const ccsdGap = Math.abs(ccsd.totalEnergy - m.E_FCI);
      const ccsdtGap = Math.abs(ccsdt.totalEnergy - m.E_FCI);
      expect(ccsdtGap).toBeLessThan(ccsdGap);
    }, 30_000);
  }
});

describe("CCSD(T) — CH₄ chemistry-grade closure", () => {
  test("CH₄ STO-3G: CCSD(T) within 0.5 mHa of cached FCI (PySCF-validated)", () => {
    const r3 = 1.09 / Math.sqrt(3);
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [ r3,  r3,  r3] },
      { symbol: "H", pos: [ r3, -r3, -r3] },
      { symbol: "H", pos: [-r3,  r3, -r3] },
      { symbol: "H", pos: [-r3, -r3,  r3] },
    ];
    const { ccsdt } = runStack(atoms);
    // FCI cached from Phase C v5 sparse run (matches PySCF to 0.76 mHa).
    const E_FCI_CH4 = -39.806036;
    expect(Math.abs(ccsdt.totalEnergy - E_FCI_CH4)).toBeLessThan(0.5e-3);
    expect(ccsdt.tripleCorrection).toBeLessThan(0);
  }, 60_000);
});

describe("CCSD(T) — full FCI cross-check via molecule-builder", () => {
  test("BeH₂ full: CCSD(T) brings energy within 0.3 mHa of fresh FCI run", () => {
    const m = MOLECULES.find((x) => x.name === "BeH2 (full)")!;
    const { ccsdt } = runStack(m.atoms);
    const mol = buildMoleculeFCI(m.atoms);
    const fci = mol.fci();
    expect(Math.abs(ccsdt.totalEnergy - fci.energy)).toBeLessThan(0.3e-3);
  }, 60_000);
});

// cc-pVDZ on H₂O takes ~95 s wall — skipped by default in vitest.
// Enable with: PHASE_E5_CCPVDZ=1 npx vitest run tests/chemistry/ccsd-t.test.ts
describe.skipIf(process.env.PHASE_E5_CCPVDZ !== "1")(
  "CCSD(T) — cc-pVDZ chemistry-grade on H₂O (slow, opt-in)", () => {
    test("H₂O cc-pVDZ CCSD(T) within 5 mHa of PySCF spherical-d ref", async () => {
      const half = (104.52 / 2) * Math.PI / 180;
      const x = 0.9572 * Math.sin(half);
      const z = 0.9572 * Math.cos(half);
      const atoms: Atom[] = [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ x, 0, z] },
        { symbol: "H", pos: [-x, 0, z] },
      ];
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const hf = runRHFSCF(integrals, nElectrons, {
        maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
      });
      const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-8 });
      const ccsdt = runCCSDT(ccsd, hf, integrals);
      // PySCF spherical-d cc-pVDZ CCSD(T) ≈ -76.2438. Our Cartesian-d is
      // variationally lower (extra (xx+yy+zz)/√3 component), so we sit
      // ~3-4 mHa below — this is a known basis-format slack, not a bug.
      const PYSCF_CCSDT = -76.2438;
      expect(ccsdt.totalEnergy).toBeLessThan(PYSCF_CCSDT);  // we're below (Cartesian d)
      expect(Math.abs(ccsdt.totalEnergy - PYSCF_CCSDT)).toBeLessThan(5e-3);
      expect(ccsdt.tripleCorrection).toBeLessThan(0);
    // 600 s: measured 362 s on 4-core CI runners (M2 ≈ 95 s). Numbers, not code.
    }, 600_000);
  },
);
