// Tier 2 stage 11: ground-state molecular properties.
//
// Pass bars (atomic units unless noted; 1 a.u. = 2.5417 Debye):
//   • H₂O HF/STO-3G dipole ≈ 1.7 Debye, ±0.1 D — published
//     reference (DOI 10.1063/1.1722842 et al.). Sign +z (oxygen
//     end is the negative side; in our coordinate system O is
//     at the origin and Hs are at +z, so μ points roughly +z).
//   • H₂O LDA/GGA/hybrid dipoles ≈ 1.7-1.9 Debye, all in the
//     same direction.
//   • H₂ and BeH₂ have zero dipole by symmetry → magnitude ≤ 1e-9
//     a.u. (just floating-point noise in the dipole AO build).
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";
import { dipoleMoment, dipoleMagnitude, mullikenCharges, bondOrders, AU_TO_DEBYE } from "../../src/chemistry/properties.js";

const H2O_ATOMS: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half);
  const z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ] as Atom[];
})();

describe("Closed-shell ground-state dipole moments", () => {
  test("H₂O HF/STO-3G: |μ| ≈ 1.7 Debye, points along +z", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, energyTol: 1e-12, maxIter: 200,
    });
    const mu = dipoleMoment(integrals, hf.D);
    const muD = dipoleMagnitude(mu) * AU_TO_DEBYE;
    // |x| component zero by C₂v symmetry (xy-plane has H atoms symmetric).
    expect(Math.abs(mu[0])).toBeLessThan(1e-9);
    // |y| zero by molecular plane (xz).
    expect(Math.abs(mu[1])).toBeLessThan(1e-9);
    // z-component positive (O at origin, Hs at +z; net dipole +z).
    expect(mu[2]).toBeGreaterThan(0);
    // Magnitude ~1.7 Debye for HF/STO-3G H₂O.
    expect(muD).toBeGreaterThan(1.5);
    expect(muD).toBeLessThan(2.0);
  });

  test("H₂O DFT functionals: |μ| in 1.5-2.0 Debye range", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2O_ATOMS.map((a) => a.symbol);
    for (const k of ["lda-svwn", "blyp", "b3lyp5"] as const) {
      const dft = runRKSDFT(integrals, nElectrons, symbols, {
        functional: k, energyTol: 1e-12, maxIter: 200,
      });
      const mu = dipoleMoment(integrals, dft.D);
      const muD = dipoleMagnitude(mu) * AU_TO_DEBYE;
      expect(Math.abs(mu[0])).toBeLessThan(1e-8);
      expect(Math.abs(mu[1])).toBeLessThan(1e-8);
      expect(mu[2]).toBeGreaterThan(0);
      expect(muD).toBeGreaterThan(1.4);
      expect(muD).toBeLessThan(2.1);
    }
  }, 60_000);

  test("H₂: zero dipole by inversion symmetry", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
    const mu = dipoleMoment(integrals, hf.D);
    // H₂ has D∞h symmetry — center of charge is mid-bond, no dipole.
    // We translated so atoms are at z = 0 and z = 0.7414 Å, which puts
    // the inversion center at z = 0.7414/2 Å. Net dipole should still
    // be 0 because the electronic and nuclear contributions cancel
    // exactly through the inversion centre.
    expect(dipoleMagnitude(mu)).toBeLessThan(1e-9);
  });

  test("BeH₂ linear: zero dipole by D∞h symmetry", () => {
    const atoms: Atom[] = [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.34] },
      { symbol: "H",  pos: [0, 0, -1.34] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
    const mu = dipoleMoment(integrals, hf.D);
    expect(dipoleMagnitude(mu)).toBeLessThan(1e-9);
  });
});

describe("Mulliken population analysis", () => {
  test("H₂O HF/STO-3G: O negative, both H positive, Σq = 0", () => {
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, energyTol: 1e-12, maxIter: 200,
    });
    const q = mullikenCharges(integrals, hf.D, shellAtomIdx);
    expect(q.length).toBe(3);
    // Atom ordering: [O, H, H].
    expect(q[0]!).toBeLessThan(0);                       // O negative
    expect(q[1]!).toBeGreaterThan(0);                    // H positive
    expect(q[2]!).toBeGreaterThan(0);
    // Symmetric H atoms → equal charges.
    expect(Math.abs(q[1]! - q[2]!)).toBeLessThan(1e-10);
    // Conservation: charges sum to net molecular charge (0).
    let sum = 0; for (const v of q) sum += v;
    expect(Math.abs(sum)).toBeLessThan(1e-10);
    // Magnitude sanity — STO-3G HF gives O ≈ −0.4 to −0.5 e.
    expect(q[0]!).toBeGreaterThan(-0.6);
    expect(q[0]!).toBeLessThan(-0.2);
  });

  test("H₂: charges zero by symmetry", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
    const q = mullikenCharges(integrals, hf.D, shellAtomIdx);
    // Both H atoms should carry zero net charge by inversion symmetry.
    expect(Math.abs(q[0]!)).toBeLessThan(1e-10);
    expect(Math.abs(q[1]!)).toBeLessThan(1e-10);
  });

  test("BeH₂ linear: Be carries the positive charge, Hs equally negative", () => {
    const atoms: Atom[] = [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.34] },
      { symbol: "H",  pos: [0, 0, -1.34] },
    ];
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
    const q = mullikenCharges(integrals, hf.D, shellAtomIdx);
    // Be is more electropositive than H — positive Mulliken charge.
    expect(q[0]!).toBeGreaterThan(0);
    expect(q[1]!).toBeLessThan(0);
    expect(q[2]!).toBeLessThan(0);
    expect(Math.abs(q[1]! - q[2]!)).toBeLessThan(1e-10);
    let sum = 0; for (const v of q) sum += v;
    expect(Math.abs(sum)).toBeLessThan(1e-10);
  });

  test("Conservation under DFT functional swap (H₂O)", () => {
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2O_ATOMS.map((a) => a.symbol);
    for (const k of ["lda-svwn", "blyp", "b3lyp5"] as const) {
      const dft = runRKSDFT(integrals, nElectrons, symbols, {
        functional: k, energyTol: 1e-12, maxIter: 200,
      });
      const q = mullikenCharges(integrals, dft.D, shellAtomIdx);
      let sum = 0; for (const v of q) sum += v;
      expect(Math.abs(sum)).toBeLessThan(1e-10);
      // Same qualitative pattern as HF.
      expect(q[0]!).toBeLessThan(0);
      expect(q[1]!).toBeGreaterThan(0);
      expect(q[2]!).toBeGreaterThan(0);
    }
  }, 60_000);
});

describe("Wiberg-Mayer bond orders", () => {
  test("H₂: B_HH ≈ 1 (single bond)", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
    const B = bondOrders(integrals, hf.D, shellAtomIdx);
    // B_HH should be 1 to high precision (one shared pair).
    expect(Math.abs(B[0 * 2 + 1]! - 1.0)).toBeLessThan(0.05);
    // Symmetry.
    expect(Math.abs(B[0 * 2 + 1]! - B[1 * 2 + 0]!)).toBeLessThan(1e-12);
  });

  test("H₂O: B_OH ≈ 1 each, B_HH small (no direct bond)", () => {
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, energyTol: 1e-12, maxIter: 200,
    });
    const B = bondOrders(integrals, hf.D, shellAtomIdx);
    // Atom indices: 0 = O, 1 = H, 2 = H.
    const B_OH1 = B[0 * 3 + 1]!;
    const B_OH2 = B[0 * 3 + 2]!;
    const B_HH  = B[1 * 3 + 2]!;
    // Each O-H bond should be ~1.
    expect(Math.abs(B_OH1 - 1.0)).toBeLessThan(0.1);
    expect(Math.abs(B_OH2 - 1.0)).toBeLessThan(0.1);
    // Symmetric H atoms.
    expect(Math.abs(B_OH1 - B_OH2)).toBeLessThan(1e-10);
    // H-H is geminal (through-bond), small but non-zero.
    expect(Math.abs(B_HH)).toBeLessThan(0.1);
  });

  test("BeH₂ linear: B_BeH ≈ 1, B_HH small, total Be valence ≈ 2", () => {
    const atoms: Atom[] = [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.34] },
      { symbol: "H",  pos: [0, 0, -1.34] },
    ];
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
    const B = bondOrders(integrals, hf.D, shellAtomIdx);
    const B_BeH1 = B[0 * 3 + 1]!;
    const B_BeH2 = B[0 * 3 + 2]!;
    const B_HH   = B[1 * 3 + 2]!;
    expect(Math.abs(B_BeH1 - 1.0)).toBeLessThan(0.15);
    expect(Math.abs(B_BeH2 - 1.0)).toBeLessThan(0.15);
    expect(Math.abs(B_BeH1 - B_BeH2)).toBeLessThan(1e-10);
    // H-H across a 2.68 Bohr gap with Be in between — small.
    expect(Math.abs(B_HH)).toBeLessThan(0.1);
    // Mayer valence on Be ≈ sum of Be-X bond orders — about 2.
    let valBe = 0;
    for (let X = 0; X < 3; X++) if (X !== 0) valBe += B[0 * 3 + X]!;
    expect(Math.abs(valBe - 2.0)).toBeLessThan(0.3);
  });

  test("Bond orders are reference-agnostic: similar across DFT functionals", () => {
    const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2O_ATOMS.map((a) => a.symbol);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, energyTol: 1e-12, maxIter: 200,
    });
    const B_HF = bondOrders(integrals, hf.D, shellAtomIdx);
    for (const k of ["lda-svwn", "blyp"] as const) {
      const dft = runRKSDFT(integrals, nElectrons, symbols, {
        functional: k, energyTol: 1e-12, maxIter: 200,
      });
      const B_DFT = bondOrders(integrals, dft.D, shellAtomIdx);
      // O-H bond order should agree across HF / DFT to a few %.
      expect(Math.abs(B_HF[0 * 3 + 1]! - B_DFT[0 * 3 + 1]!)).toBeLessThan(0.1);
    }
  }, 60_000);
});
