// MP2 validation. Strategy:
//   1. Algorithmic correctness: MO ERI tensor symmetry, energy
//      formula sanity (E_MP2_corr < 0 for closed-shell molecules
//      near equilibrium).
//   2. Variational consistency: HF + MP2 must lie BETWEEN HF and
//      FCI for our standard set (no over-correction).
//   3. Capture ratio: MP2 captures ≥ 50% of the HF→FCI correlation
//      gap for every molecule. This is the canonical empirical
//      property of MP2 on closed-shell systems near equilibrium.
//   4. Cross-check H₂O against the published PySCF STO-3G MP2
//      number to ~1 mHa (geometry/precision tolerance).
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runMP2, transformERIToMO } from "../../src/chemistry/mp2.js";
import { buildMoleculeFCI } from "../../src/chemistry/molecule-builder.js";

interface MolRef {
  name: string;
  atoms: Atom[];
  E_FCI: number;
}

const MOLECULES: MolRef[] = [
  {
    name: "H2",
    atoms: [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ],
    E_FCI: -1.137270,
  },
  {
    name: "LiH (s-only)",
    atoms: [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ],
    E_FCI: -7.8434,
  },
  {
    name: "BeH2 (full)",
    atoms: [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -1.34] },
      { symbol: "H",  pos: [0, 0,  1.34] },
    ],
    E_FCI: -15.5949,
  },
  {
    name: "H2O",
    atoms: (() => {
      const half = (104.52 / 2) * Math.PI / 180;
      const x = 0.9572 * Math.sin(half);
      const z = 0.9572 * Math.cos(half);
      return [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ x, 0, z] },
        { symbol: "H", pos: [-x, 0, z] },
      ] as Atom[];
    })(),
    E_FCI: -75.0124,
  },
  {
    name: "CH4",
    atoms: (() => {
      const r3 = 1.09 / Math.sqrt(3);
      return [
        { symbol: "C", pos: [0, 0, 0] },
        { symbol: "H", pos: [ r3,  r3,  r3] },
        { symbol: "H", pos: [ r3, -r3, -r3] },
        { symbol: "H", pos: [-r3,  r3, -r3] },
        { symbol: "H", pos: [-r3, -r3,  r3] },
      ] as Atom[];
    })(),
    E_FCI: -39.8068,
  },
];

describe("MO ERI transformation — symmetry", () => {
  test("8-fold chemist-notation symmetry preserved on H₂O", () => {
    const m = MOLECULES.find((x) => x.name === "H2O")!;
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { maxIter: 500, damping: 0.3 });
    const eri_MO = transformERIToMO(integrals.eri_AO, hf.C_MO, integrals.n);
    const n = integrals.n;
    const idx = (a: number, b: number, c: number, d: number): number =>
      ((a * n + b) * n + c) * n + d;
    let maxAsym = 0;
    for (let p = 0; p < n; p++) {
      for (let q = 0; q < n; q++) {
        for (let r = 0; r < n; r++) {
          for (let s = 0; s < n; s++) {
            const v = eri_MO[idx(p, q, r, s)]!;
            // (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq).
            maxAsym = Math.max(maxAsym, Math.abs(v - eri_MO[idx(q, p, r, s)]!));
            maxAsym = Math.max(maxAsym, Math.abs(v - eri_MO[idx(p, q, s, r)]!));
            maxAsym = Math.max(maxAsym, Math.abs(v - eri_MO[idx(r, s, p, q)]!));
          }
        }
      }
    }
    expect(maxAsym).toBeLessThan(1e-10);
  });
});

describe("MP2 — variational and capture properties", () => {
  for (const m of MOLECULES.filter((x) => x.name !== "CH4")) {
    test(`${m.name}: HF > HF+MP2 > FCI; MP2 captures ≥ 50% of correlation`, () => {
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const hf = runRHFSCF(integrals, nElectrons, {
        maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
      });
      const mp2 = runMP2(hf, integrals);

      // MP2 correlation is negative.
      expect(mp2.correlationEnergy).toBeLessThan(0);

      // Variational ordering: HF > HF+MP2 > FCI for these closed-shell
      // molecules near equilibrium (MP2 doesn't over-correct).
      expect(hf.energy).toBeGreaterThan(mp2.totalEnergy);
      expect(mp2.totalEnergy).toBeGreaterThan(m.E_FCI);

      // Capture ratio: MP2 picks up ≥ 50% of the HF→FCI gap.
      const totalCorr = m.E_FCI - hf.energy;        // negative
      const mp2Corr = mp2.correlationEnergy;        // negative
      const capture = mp2Corr / totalCorr;
      expect(capture).toBeGreaterThan(0.5);
      expect(capture).toBeLessThan(1.0);            // MP2 doesn't over-correct
    }, 30_000);
  }

  test("CH4: HF + MP2 captures ≥ 50% of FCI correlation (no FCI rerun)", () => {
    // Use the already-validated CH₄ FCI from Phase C v5: -39.806036 Ha.
    // Skip rebuilding the 80 s sparse FCI here.
    const E_FCI_CH4 = -39.806036;
    const m = MOLECULES.find((x) => x.name === "CH4")!;
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { maxIter: 500, damping: 0.3 });
    const mp2 = runMP2(hf, integrals);
    expect(mp2.correlationEnergy).toBeLessThan(0);
    expect(hf.energy).toBeGreaterThan(mp2.totalEnergy);
    expect(mp2.totalEnergy).toBeGreaterThan(E_FCI_CH4);
    const capture = mp2.correlationEnergy / (E_FCI_CH4 - hf.energy);
    expect(capture).toBeGreaterThan(0.5);
    expect(capture).toBeLessThan(1.0);
  });
});

describe("MP2 — published-PySCF cross-check on H₂O", () => {
  test("H₂O MP2/STO-3G total energy matches PySCF to ≤ 2 mHa", () => {
    // PySCF H₂O/STO-3G MP2 at R_OH=0.9572, ∠=104.52°: ~ -74.998 to -74.999.
    // Tolerance covers small geometry / convention spread.
    const m = MOLECULES.find((x) => x.name === "H2O")!;
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { maxIter: 500, damping: 0.3 });
    const mp2 = runMP2(hf, integrals);
    const PYSCF_H2O_MP2 = -74.999;
    expect(Math.abs(mp2.totalEnergy - PYSCF_H2O_MP2)).toBeLessThan(2e-3);
  });
});

describe("MP2 — buildMoleculeFCI vs runMP2 + buildMoleculeFCI shows ordering", () => {
  test("H₂ HF/MP2/FCI: prove the three-tier hierarchy", () => {
    const m = MOLECULES[0]!;  // H2
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { maxIter: 200 });
    const mp2 = runMP2(hf, integrals);
    const mol = buildMoleculeFCI(m.atoms);
    const fci = mol.fci();
    // Strict variational hierarchy.
    expect(hf.energy).toBeGreaterThan(mp2.totalEnergy);
    expect(mp2.totalEnergy).toBeGreaterThan(fci.energy);
    // H₂ MP2 captures most of the correlation (~64% canonical for STO-3G).
    const capture = mp2.correlationEnergy / (fci.energy - hf.energy);
    expect(capture).toBeGreaterThan(0.6);
  });
});
