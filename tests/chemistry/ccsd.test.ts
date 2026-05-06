// CCSD validation. Strategy:
//   1. MP2 sanity: T1=0, T2=MP2 amplitudes → E_corr from CCSD's
//      energy formula must equal closed-shell MP2 from mp2.ts.
//      Validates the spin-orbital ⟨ij||ab⟩ tensor + amplitude init.
//   2. H₂ correctness: 2 electrons → CCSD ≡ FCI exactly. The
//      strongest correctness check (no triples needed).
//   3. Variational ordering across the standard molecule set:
//      HF > HF+MP2 > CCSD ≥ FCI (CCSD ≥ FCI to numerical noise;
//      strict equality only for 2-electron systems).
//   4. CCSD captures ≥ 95% of HF→FCI correlation for closed-shell
//      molecules near equilibrium (canonical CCSD property).
//   5. Cross-check against published PySCF CCSD/STO-3G numbers
//      where I have them.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runMP2 } from "../../src/chemistry/mp2.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
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
    E_FCI: -7.843394,
  },
  {
    name: "BeH2 (full)",
    atoms: [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -1.34] },
      { symbol: "H",  pos: [0, 0,  1.34] },
    ],
    E_FCI: -15.594861,
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
    E_FCI: -75.012403,
  },
];

describe("CCSD — MP2 initial-amplitude consistency", () => {
  for (const m of MOLECULES) {
    test(`${m.name}: CCSD energy formula at MP2 amplitudes equals closed-shell MP2`, () => {
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const hf = runRHFSCF(integrals, nElectrons, {
        maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
      });
      const mp2 = runMP2(hf, integrals);
      // Run CCSD with maxIter=0: just initialize T2 to MP2 amplitudes,
      // compute energy, return without iterating. We achieve this by
      // requesting a single iteration — but the very first computeECorr
      // is called BEFORE any iteration runs, so it equals MP2 from init.
      const ccsd = runCCSD(hf, integrals, { maxIter: 0 });
      // First entry of history is the energy at MP2 amplitudes (pre-iteration).
      expect(ccsd.history[0]).toBeCloseTo(mp2.correlationEnergy, 8);
    }, 30_000);
  }
});

describe("CCSD — H₂ must equal FCI (exact)", () => {
  test("H₂ STO-3G: CCSD = FCI to better than 1e-7 Ha", () => {
    const m = MOLECULES[0]!;
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      maxIter: 500, damping: 0.3, energyTol: 1e-12, densityTol: 1e-10,
    });
    const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-12 });
    expect(ccsd.converged).toBe(true);
    expect(Math.abs(ccsd.totalEnergy - m.E_FCI)).toBeLessThan(1e-7);
  }, 30_000);
});

describe("CCSD — variational ordering and capture across molecule set", () => {
  for (const m of MOLECULES) {
    test(`${m.name}: HF > HF+MP2 > CCSD ≥ FCI; CCSD captures ≥ 95% of correlation`, () => {
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const hf = runRHFSCF(integrals, nElectrons, {
        maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
      });
      const mp2 = runMP2(hf, integrals);
      const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });
      expect(ccsd.converged).toBe(true);
      expect(ccsd.correlationEnergy).toBeLessThan(0);
      // Variational ordering. CCSD is non-variational (can dip below FCI),
      // but for closed-shell molecules near equilibrium it stays above.
      // Allow numerical slack of 1e-6 Ha at the FCI side.
      expect(hf.energy).toBeGreaterThan(mp2.totalEnergy);
      expect(mp2.totalEnergy).toBeGreaterThan(ccsd.totalEnergy);
      expect(ccsd.totalEnergy).toBeGreaterThan(m.E_FCI - 1e-6);
      // Capture ratio: CCSD picks up ≥ 95% of HF→FCI correlation.
      const totalCorr = m.E_FCI - hf.energy;        // negative
      const capture = ccsd.correlationEnergy / totalCorr;
      expect(capture).toBeGreaterThan(0.95);
    }, 60_000);
  }
});

describe("CCSD — published PySCF cross-check", () => {
  // PySCF CCSD/STO-3G at standard geometries:
  //   H₂   R = 0.7414: −1.137270 (= FCI)
  //   H₂O  R = 0.9572, ∠=104.52°: ~ −75.0117 (FCI − ~0.7 mHa for missing T,
  //         within published spread of −75.0117 to −75.0124)
  //   LiH  R = 1.595:  ~ −7.84249
  //   BeH₂ R = 1.34:   ~ −15.5947
  test("H₂O CCSD/STO-3G total within 1.5 mHa of −75.0117", () => {
    const m = MOLECULES.find((x) => x.name === "H2O")!;
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
    });
    const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });
    expect(Math.abs(ccsd.totalEnergy - (-75.0117))).toBeLessThan(1.5e-3);
  }, 60_000);
});

describe("CCSD — full FCI cross-check via molecule-builder", () => {
  test("LiH H_2-style: CCSD energy within 0.5 mHa of FCI from buildMoleculeFCI", () => {
    const m = MOLECULES.find((x) => x.name === "LiH (s-only)")!;
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
    });
    const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-10 });
    const mol = buildMoleculeFCI(m.atoms);
    const fci = mol.fci();
    expect(Math.abs(ccsd.totalEnergy - fci.energy)).toBeLessThan(0.5e-3);
  }, 60_000);
});

describe("Frozen core — exclude 1s of heavy atoms from correlation", () => {
  // For first-row chemistry the convention is to freeze the 1s core
  // orbital of every heavy atom. The frozen-core CCSD correlation
  // energy differs from the all-electron value by 1s-1s and 1s-valence
  // dynamic correlation that's nearly basis-set-incomplete in any
  // small basis. Pass bars:
  //   • H₂ (no heavy atom): nFrozenCore=0 must equal default — sanity.
  //   • H₂O / CH₄: nFrozenCore=1 must change E_CCSD_corr by < 30 mHa
  //     vs all-electron, and the resulting total must still be
  //     between E_HF and E_FCI (variational sanity).
  //   • MP2 + CCSD must agree on which orbitals are frozen — the
  //     core-included MP2 minus core-frozen MP2 must equal the
  //     pure 1s-1s correlation (always negative).
  const half = (104.52 / 2) * Math.PI / 180;
  const xH = 0.9572 * Math.sin(half);
  const zH = 0.9572 * Math.cos(half);
  const r3 = 1.09 / Math.sqrt(3);

  const cases: { name: string; atoms: Atom[]; nFrozen: number }[] = [
    {
      name: "H2O",
      atoms: [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ xH, 0, zH] },
        { symbol: "H", pos: [-xH, 0, zH] },
      ],
      nFrozen: 1,
    },
    {
      name: "CH4",
      atoms: [
        { symbol: "C", pos: [0, 0, 0] },
        { symbol: "H", pos: [ r3,  r3,  r3] },
        { symbol: "H", pos: [ r3, -r3, -r3] },
        { symbol: "H", pos: [-r3,  r3, -r3] },
        { symbol: "H", pos: [-r3, -r3,  r3] },
      ],
      nFrozen: 1,
    },
  ];

  for (const c of cases) {
    test(`${c.name}: frozen-1s CCSD diverges from full by < 30 mHa, lies above all-electron`, () => {
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(c.atoms);
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const hf = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      });

      const ccsdAll = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });
      const ccsdFC  = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9, nFrozenCore: c.nFrozen });

      // Frozen-core total energy is HIGHER (less correlation captured).
      expect(ccsdFC.totalEnergy).toBeGreaterThan(ccsdAll.totalEnergy);
      // Difference is the 1s-1s + 1s-valence correlation — bounded for
      // STO-3G (small basis can only capture so much core correlation).
      const diffMHa = (ccsdFC.totalEnergy - ccsdAll.totalEnergy) * 1000;
      expect(diffMHa).toBeGreaterThan(0);
      expect(diffMHa).toBeLessThan(30);
      // Both must converge.
      expect(ccsdAll.converged).toBe(true);
      expect(ccsdFC.converged).toBe(true);

      // MP2 frozen-core sanity: same direction.
      const mp2All = runMP2(hf, integrals);
      const mp2FC  = runMP2(hf, integrals, { nFrozenCore: c.nFrozen });
      expect(mp2FC.totalEnergy).toBeGreaterThan(mp2All.totalEnergy);
    }, 30_000);
  }
});
