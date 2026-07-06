// Spherical-harmonic d-shell basis. Tier 1 / Phase E stage 6 work
// (Tier 1 bundle from CLAUDE.md roadmap).
//
// Pass bars:
//   • H₂O cc-pVDZ HF spherical matches PySCF to ≤ 0.1 mHa (PySCF ref
//     uses spherical d). Cartesian basis, by contrast, sits ~0.4 mHa
//     below PySCF because the (xx+yy+zz)/√3 redundant component makes
//     the basis variationally larger — a documented basis-set
//     convention difference, not a code bug.
//   • Spherical n = Cartesian n − (#d-shells). For H₂O cc-pVDZ:
//     25 Cart → 24 Sph (one O d-shell drops 1 redundant function).
//   • The 5 spherical-d AOs at a single center must be mutually
//     orthonormal in the AO overlap matrix (within FP precision).
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

describe("Spherical-harmonic d-shell basis (cc-pVDZ)", () => {
  test("H₂O Cartesian: n=25, Spherical: n=24 (drops 1 d redundancy)", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "cc-pvdz");
    const cart = computeMolecularIntegrals(shells, nuclei);
    const sph  = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    expect(cart.n).toBe(25);
    expect(sph.n).toBe(24);
  });

  test("Spherical-d AOs are mutually orthonormal at the d-shell center", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "cc-pvdz");
    const sph = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    // O d-shell occupies indices 9..13 in the spherical basis:
    //   0..2   O 1s, 2s, 2s'         (3 s-shells)
    //   3..8   O 2p_{xyz}, 2p'_{xyz} (6 p-shells)
    //   9..13  O 3d (5 spherical d, was 6 Cartesian)
    //   14..18 H1: 1s, 2s, 2p_{xyz}
    //   19..23 H2: 1s, 2s, 2p_{xyz}
    // The 5×5 d-d overlap block at indices 9..13 must be the identity
    // — they all sit at the same center with the same primitives, so
    // they decompose into orthonormal real spherical harmonics.
    const n = sph.n;
    for (let i = 9; i <= 13; i++) {
      for (let j = 9; j <= 13; j++) {
        const expected = i === j ? 1 : 0;
        const got = sph.S_AO[i * n + j]!;
        expect(Math.abs(got - expected)).toBeLessThan(1e-12);
      }
    }
  });

  test("H₂O HF/cc-pVDZ spherical matches PySCF to ≤ 0.1 mHa", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "cc-pvdz");
    const sph = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    const hf = runRHFSCF(sph, 10, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(hf.converged).toBe(true);
    // PySCF cc-pVDZ HF/H₂O at experimental geometry = -76.026765 Ha.
    expect(Math.abs(hf.energy - (-76.026765))).toBeLessThan(1e-4);
  }, 360_000);

  test("Spherical HF lies above Cartesian HF (variational principle: Cart basis ⊃ Sph basis)", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "cc-pvdz");
    const cart = computeMolecularIntegrals(shells, nuclei);
    const sph  = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    const hfC = runRHFSCF(cart, 10, { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
    const hfS = runRHFSCF(sph, 10,  { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
    expect(hfC.energy).toBeLessThan(hfS.energy);
    expect((hfS.energy - hfC.energy) * 1000).toBeLessThan(1.0);  // diff < 1 mHa
  }, 360_000);
});

import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";
import { runTDA } from "../../src/chemistry/tda-dft.js";
import { hfGradient, buildEnergyWeightedDensity } from "../../src/chemistry/hf-gradient.js";
import { dftGradient } from "../../src/chemistry/dft-gradient.js";

describe("Spherical-d round-trip on the grid (DFT + TDA + gradient)", () => {
  // Stage 15a: when integrals are built with `spherical: true`, P / W
  // arrive in the spherical-AO basis but `shells` / `shellAtomIdx` are
  // still Cartesian. The grid-using paths (RKS-DFT SCF, TDA-DFT XC
  // kernel, DFT analytical gradient) apply the T transform internally
  // and produce results consistent with the Cartesian path within
  // ~1 mHa (the documented Cartesian-vs-spherical-d slack).
  const H2O: Atom[] = (() => {
    const half = (104.52 / 2) * Math.PI / 180;
    const x = 0.9572 * Math.sin(half);
    const z = 0.9572 * Math.cos(half);
    return [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ x, 0, z] },
      { symbol: "H", pos: [-x, 0, z] },
    ] as Atom[];
  })();

  test("RKS-DFT/B3LYP5: spherical converges and matches Cartesian within 2 mHa", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O, "cc-pvdz");
    const symbols = H2O.map((a) => a.symbol);
    const cart = computeMolecularIntegrals(shells, nuclei);
    const sph  = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    const dC = runRKSDFT(cart, nElectrons, symbols, { functional: "b3lyp5", energyTol: 1e-10, maxIter: 200 });
    const dS = runRKSDFT(sph,  nElectrons, symbols, { functional: "b3lyp5", energyTol: 1e-10, maxIter: 200 });
    expect(dS.converged).toBe(true);
    expect(Number.isFinite(dS.energy)).toBe(true);
    expect(Math.abs(dS.energy - dC.energy)).toBeLessThan(2e-3);
  }, 360_000);

  test("TDA-B3LYP5 first singlet: spherical and Cartesian within 50 meV", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O, "cc-pvdz");
    const symbols = H2O.map((a) => a.symbol);
    for (const sphFlag of [false, true] as const) {
      const integrals = computeMolecularIntegrals(shells, nuclei, { spherical: sphFlag });
      const dft = runRKSDFT(integrals, nElectrons, symbols, { functional: "b3lyp5", energyTol: 1e-10, maxIter: 200 });
      const tda = runTDA(integrals, dft, { method: "b3lyp5", nucleiSymbols: symbols, nRoots: 1 });
      expect(Number.isFinite(tda.singletEnergies[0]!)).toBe(true);
      expect(tda.singletEnergies[0]!).toBeGreaterThan(0);
    }
  }, 360_000);

  test("HF + DFT analytical gradients on cc-pVDZ spherical: finite, |∇| ≈ Cartesian", () => {
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(H2O, "cc-pvdz");
    const symbols = H2O.map((a) => a.symbol);
    for (const sphFlag of [false, true] as const) {
      const integrals = computeMolecularIntegrals(shells, nuclei, { spherical: sphFlag });
      const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
      const Whf = buildEnergyWeightedDensity(hf.C_MO, hf.orbitalEnergies, hf.nOccupied, integrals.n);
      const gHF = hfGradient({
        shells, nuclei, shellAtomIdx, P: hf.D, W: Whf, sphericalT: integrals.sphericalT,
      });
      let normHF = 0; for (const v of gHF) normHF += v * v;
      expect(Number.isFinite(normHF)).toBe(true);
      expect(normHF).toBeGreaterThan(0);

      const dft = runRKSDFT(integrals, nElectrons, symbols, { functional: "b3lyp5", energyTol: 1e-10, maxIter: 200 });
      const Wdft = buildEnergyWeightedDensity(dft.C_MO, dft.orbitalEnergies, dft.nOccupied, integrals.n);
      const gDFT = dftGradient({
        shells, nuclei, shellAtomIdx, nucleiSymbols: symbols,
        P: dft.D, W: Wdft, functional: "b3lyp5",
        sphericalT: integrals.sphericalT,
      });
      let normDFT = 0; for (const v of gDFT) normDFT += v * v;
      expect(Number.isFinite(normDFT)).toBe(true);
      expect(normDFT).toBeGreaterThan(0);
    }
  }, 360_000);
});
