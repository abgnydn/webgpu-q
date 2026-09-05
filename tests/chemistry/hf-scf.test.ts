// RHF SCF validation. Cross-checks our HF energies against PySCF
// references for the standard molecule set, AND validates that
// the HF energy + correlation gap = FCI for each molecule (the
// most basis-independent check — proves both HF and FCI are
// internally consistent).
//
// REFERENCE PROVENANCE — this is the headline "matches PySCF" suite,
// so every literal below is full precision and sourced, and every
// molecule is checked in cc-pVDZ as well as STO-3G. It previously
// stored references to 4 decimals and asserted a 0.5 mHa bar, which
// is 5000× looser than the digits it carried, and had no cc-pVDZ row
// at all — so the Li/Be cc-pVDZ table defect (up to 1.29 mHa) could
// not be seen from here.
//
// STO-3G column:
//   H₂, LiH, BeH₂  — experiments/results/2026-07-06/level-6/E34-pyscf.json,
//                    rows {basis:"sto-3g", method:"HF"} (PySCF 2.13.0,
//                    same geometries as below).
//   H₂O            — tests/chemistry/elements/pyscf-reference.json,
//                    row H2O · sto-3g (geometry byte-identical to the one
//                    constructed below; conv_tol 1e-12).
//   CH₄            — not present in either committed artifact at R=1.09 Å,
//                    so regenerated here (see the cc-pVDZ note below).
//   NOTE on LiH: this repo's STO-3G table for Li is s-only (nao = 3, no
//   2p), matching both committed artifacts. PySCF's stock STO-3G Li adds
//   2p and gives -7.862023860127111 at the same geometry — do NOT use
//   that number here; it describes a different basis.
//
// cc-pVDZ column (Cartesian d — this suite runs the default Cartesian
// integral path, so it must pair with PySCF mol.cart=True):
//   H₂, H₂O — tests/chemistry/elements/pyscf-reference.json,
//             convention "cartesian" (geometries identical to below).
//   LiH, BeH₂, CH₄ — regenerated at THESE geometries with PySCF 2.13.1
//             (the committed fixture uses slightly different bond
//             lengths: LiH 1.5949, BeH₂ 1.3264, CH₄ 1.087):
//               m = gto.M(atom=..., basis=..., unit='Angstrom', cart=True)
//               mf = scf.RHF(m); mf.conv_tol = 1e-12; mf.kernel()
//
// Measured |Δ| for this code path (mHa):
//   STO-3G   H₂ 1.0e-4 · LiH 7.6e-5 · BeH₂ 1.3e-4 · H₂O 7.0e-4 · CH₄ 1.2e-4
//   cc-pVDZ  H₂ 1.1e-4 · LiH 5.6e-5 · BeH₂ 1.5e-4 · H₂O 2.7e-4 · CH₄ 8.9e-4
// Bar is 0.01 mHa in both columns — ~11× the worst measurement.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { buildMoleculeFCI } from "../../src/chemistry/molecule-builder.js";

interface MolRef {
  name: string;
  atoms: Atom[];
  E_HF_PySCF: number;       // PySCF RHF/STO-3G reference (Ha)
  E_FCI_PySCF: number;      // PySCF FCI/STO-3G reference (Ha)
  E_HF_ccpvdz: number;      // PySCF RHF/cc-pVDZ reference, Cartesian d (Ha)
  nao_ccpvdz: number;       // PySCF AO count, Cartesian d
}

// 0.01 mHa = 10 µHa. See the provenance block above.
const BAR_mHa = 0.01;

const MOLECULES: MolRef[] = [
  {
    name: "H2",
    atoms: [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ],
    E_HF_PySCF: -1.116684387085341,
    E_FCI_PySCF: -1.137270,
    E_HF_ccpvdz: -1.128714959029653,
    nao_ccpvdz: 10,
  },
  {
    name: "LiH (s-only)",
    atoms: [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ],
    E_HF_PySCF: -7.804242631940016,
    E_FCI_PySCF: -7.8434,
    E_HF_ccpvdz: -7.983650704720753,
    nao_ccpvdz: 20,
  },
  {
    name: "BeH2 (full)",
    atoms: [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -1.34] },
      { symbol: "H",  pos: [0, 0,  1.34] },
    ],
    E_HF_PySCF: -15.559405412277242,
    E_FCI_PySCF: -15.5949,
    E_HF_ccpvdz: -15.767442032106226,
    nao_ccpvdz: 25,
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
    E_HF_PySCF: -74.96292824643288,
    E_FCI_PySCF: -75.0124,
    E_HF_ccpvdz: -76.02713907180959,
    nao_ccpvdz: 25,
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
    E_HF_PySCF: -39.726700052055236,
    E_FCI_PySCF: -39.8068,
    E_HF_ccpvdz: -40.19877842686372,
    nao_ccpvdz: 35,
  },
];

describe("RHF SCF — energy matches PySCF for the standard molecule set", () => {
  for (const m of MOLECULES) {
    test(`${m.name} STO-3G: |E_HF − E_PySCF| ≤ ${BAR_mHa} mHa`, () => {
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const r = runRHFSCF(integrals, nElectrons, {
        maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
      });
      expect(r.converged, `SCF did not converge for ${m.name}`).toBe(true);
      expect(Math.abs((r.energy - m.E_HF_PySCF) * 1000)).toBeLessThan(BAR_mHa);
    }, 30_000);
  }

  // cc-pVDZ coverage. Same molecules, same geometries, first non-minimal
  // basis — this is the column where a wrong per-element basis table
  // (the Li/Be defect) actually shows up. Cartesian d on both sides.
  for (const m of MOLECULES) {
    test(`${m.name} cc-pVDZ: |E_HF − E_PySCF| ≤ ${BAR_mHa} mHa (Cartesian d)`, () => {
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei);
      expect(integrals.n, `AO count mismatch for ${m.name}`).toBe(m.nao_ccpvdz);
      const r = runRHFSCF(integrals, nElectrons, {
        maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
      });
      expect(r.converged, `cc-pVDZ SCF did not converge for ${m.name}`).toBe(true);
      expect(Math.abs((r.energy - m.E_HF_ccpvdz) * 1000)).toBeLessThan(BAR_mHa);
      // Variational sanity across bases (supplementary).
      expect(r.energy).toBeLessThan(m.E_HF_PySCF);
    }, 120_000);
  }
});

describe("HF + correlation = FCI (variational consistency)", () => {
  // For each molecule, compute both HF and FCI, then check that
  // E_HF > E_FCI (variational principle) and that the correlation
  // gap E_FCI − E_HF is in the canonical 10-100 mHa range for
  // these closed-shell minimal-basis molecules. CH₄ is excluded
  // here — its sparse Hsec build is 80 s synchronous which trips
  // vitest's worker-IPC heartbeat timeout. CH₄ HF is validated in
  // the test above and CH₄ FCI is validated by the Phase C v5
  // publishable artifact.
  for (const m of MOLECULES.filter((x) => x.name !== "CH4")) {
    test(`${m.name}: HF > FCI by 10–150 mHa (canonical correlation)`, () => {
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const hf = runRHFSCF(integrals, nElectrons, {
        maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
      });
      const mol = buildMoleculeFCI(m.atoms);
      const fci = mol.fci();
      const corrMHa = (hf.energy - fci.energy) * 1000;
      expect(hf.energy).toBeGreaterThan(fci.energy);
      expect(corrMHa).toBeGreaterThan(10);
      expect(corrMHa).toBeLessThan(150);
    }, 60_000);
  }
});

describe("RHF SCF — orbital structure", () => {
  test("H2O orbital energies match canonical ordering", () => {
    const m = MOLECULES.find((x) => x.name === "H2O")!;
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const r = runRHFSCF(integrals, nElectrons, {
      maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
    });
    // Canonical H₂O STO-3G HF orbital energies:
    //   1a₁ (O 1s core):   ε ≈ -20.24 Ha
    //   2a₁ (O 2s + bond): ε ≈ -1.27
    //   1b₂ (in-plane):    ε ≈ -0.62
    //   3a₁ (out-of-plane):ε ≈ -0.45
    //   1b₁ (lone pair):   ε ≈ -0.39
    //   4a₁ (antibonding): ε ≈  +0.61
    //   2b₂ (antibonding): ε ≈  +0.74
    const eps = r.orbitalEnergies;
    expect(eps[0]!).toBeCloseTo(-20.24, 1);
    expect(eps[1]!).toBeCloseTo(-1.27, 1);
    expect(eps[2]!).toBeCloseTo(-0.62, 1);
    expect(eps[3]!).toBeCloseTo(-0.45, 1);
    expect(eps[4]!).toBeCloseTo(-0.39, 1);
    expect(eps[5]!).toBeGreaterThan(0);  // antibonding
    expect(eps[6]!).toBeGreaterThan(0);
  }, 30_000);
});

describe("DIIS extrapolation — converges in fewer iterations than damping", () => {
  // DIIS is the default convergence accelerator. The pass bar:
  //   STO-3G H₂O    DIIS ≤ 15 iter (vs ~92 with damping)
  //   cc-pVDZ H₂O   DIIS ≤ 25 iter (vs ~101 with damping)
  // Both DIIS and damping must converge to the same energy to ≤ 1e-9 Ha.
  const half = (104.52 / 2) * Math.PI / 180;
  const xH = 0.9572 * Math.sin(half);
  const zH = 0.9572 * Math.cos(half);
  const h2o: Atom[] = [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ xH, 0, zH] },
    { symbol: "H", pos: [-xH, 0, zH] },
  ];

  for (const { basis, diisCap } of [
    { basis: "sto-3g" as const, diisCap: 15 },
    { basis: "cc-pvdz" as const, diisCap: 25 },
  ]) {
    test(`H₂O ${basis}: DIIS ≤ ${diisCap} iter, energies agree to 1e-9`, () => {
      const { shells, nuclei } = moleculeToShellsNuclei(h2o, basis);
      const integrals = computeMolecularIntegrals(shells, nuclei);

      const diis = runRHFSCF(integrals, 10, {
        useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      });
      expect(diis.converged, "DIIS did not converge").toBe(true);
      expect(diis.iter).toBeLessThanOrEqual(diisCap);

      const damp = runRHFSCF(integrals, 10, {
        useDIIS: false, damping: 0.3, maxIter: 500, energyTol: 1e-10, densityTol: 1e-8,
      });
      expect(damp.converged, "damping did not converge").toBe(true);
      expect(Math.abs(diis.energy - damp.energy)).toBeLessThan(1e-9);
    }, 30_000);
  }
});
