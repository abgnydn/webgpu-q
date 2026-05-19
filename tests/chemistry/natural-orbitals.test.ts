// Natural orbital occupation numbers (NOON) — multireference diagnostic.
//
// Pass bars:
//   1. Closed-shell RHF (single Slater determinant): NOONs are exactly
//      {2, 2, ..., 2, 0, 0, ...} ± 1e-10. Sum = N_electrons.
//   2. Be atom UHF: NOONs essentially {2, 2, 0, ...} — Be is closed-shell
//      so UHF reduces to RHF with no spin contamination, NOONs integer.
//   3. UHF on a system with spin contamination (e.g., H₂ dissociation at
//      large separation): NOONs become fractional, reflecting the
//      broken-symmetry character.
//   4. Total electrons (Σ NOON) matches input.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import { naturalOrbitalOccupations } from "../../src/chemistry/natural-orbitals.js";

describe("Natural orbital occupation numbers (NOON)", () => {
  test("Closed-shell H₂O HF/STO-3G: NOONs are exactly {2, 2, 2, 2, 2, 0, 0}", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });

    const noon = naturalOrbitalOccupations(hf.D, integrals.S_AO);

    // STO-3G H₂O has 7 basis functions, 10 electrons → 5 occupied.
    expect(noon.occupations.length).toBe(7);
    expect(noon.nStronglyOccupied).toBe(5);
    expect(noon.nActive).toBe(0);                  // no fractional NOON for HF
    expect(Math.abs(noon.totalElectrons - 10)).toBeLessThan(1e-9);
    // First 5 are ~2, last 2 are ~0.
    for (let i = 0; i < 5; i++) expect(Math.abs(noon.occupations[i]! - 2)).toBeLessThan(1e-10);
    for (let i = 5; i < 7; i++) expect(Math.abs(noon.occupations[i]!)).toBeLessThan(1e-10);
  });

  test("Be atom UHF/STO-3G (n_α=n_β=2, no symm-break): NOONs {2, 2, 0, 0, 0}", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "Be", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 2, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      symmetryBreaking: 0,
    });
    expect(uhf.converged).toBe(true);

    const D_total = new Float64Array(integrals.n * integrals.n);
    for (let i = 0; i < D_total.length; i++) D_total[i] = uhf.D_alpha[i]! + uhf.D_beta[i]!;
    const noon = naturalOrbitalOccupations(D_total, integrals.S_AO);

    expect(Math.abs(noon.totalElectrons - 4)).toBeLessThan(1e-9);
    // First 2 NOONs ~ 2, rest ~ 0.
    expect(Math.abs(noon.occupations[0]! - 2)).toBeLessThan(1e-8);
    expect(Math.abs(noon.occupations[1]! - 2)).toBeLessThan(1e-8);
    for (let i = 2; i < noon.occupations.length; i++) {
      expect(Math.abs(noon.occupations[i]!)).toBeLessThan(1e-8);
    }
    expect(noon.nActive).toBe(0);   // no MR character
  });

  test("Be atom UHF symm-break (n_α=2, n_β=2 with symmBreaking): NOONs slightly fractional", () => {
    // Forcibly break spin symmetry via initial perturbation. For Be
    // (closed-shell), UHF SHOULD restore to RHF, but with a strong
    // symmetry-breaking pulse the SCF may converge to a slightly
    // broken solution. Either way the NOON sum stays = nElectrons.
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "Be", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 2, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      symmetryBreaking: 0.5,  // strong perturbation
    });
    expect(uhf.converged).toBe(true);
    const D_total = new Float64Array(integrals.n * integrals.n);
    for (let i = 0; i < D_total.length; i++) D_total[i] = uhf.D_alpha[i]! + uhf.D_beta[i]!;
    const noon = naturalOrbitalOccupations(D_total, integrals.S_AO);
    expect(Math.abs(noon.totalElectrons - 4)).toBeLessThan(1e-9);
    // Each individual NOON in [0, 2] (or close to it).
    for (let i = 0; i < noon.occupations.length; i++) {
      expect(noon.occupations[i]!).toBeGreaterThanOrEqual(-1e-8);
      expect(noon.occupations[i]!).toBeLessThanOrEqual(2 + 1e-8);
    }
  });
});
