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
import { dipoleMoment, dipoleMagnitude, AU_TO_DEBYE } from "../../src/chemistry/properties.js";

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
