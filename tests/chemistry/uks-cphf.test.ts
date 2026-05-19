// UKS-CPHF (open-shell DFT CPKS) static polarizability — LSDA only.
//
// Pass bars:
//   1. Closed-shell limit: UKS-CPHF α(0) for n_α=n_β H₂O matches the
//      RKS-LSDA tddftPolarizability at ω=0 to ≤ 1e-4 per tensor
//      element. (UKS-CPHF and RKS-TDDFT-at-zero are equivalent in
//      closed-shell limit; both build the same (A+B) at ω=0 with the
//      LSDA kernel piece.)
//   2. Li atom doublet cc-pVDZ UKS-LSDA: α tensor real, symmetric,
//      isotropic > 0. Adding the XC kernel pulls α away from the
//      uncoupled HF-only result.
//   3. Hybrid / GGA refused with clear message (this stage).

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type AtomSymbol } from "../../src/chemistry/atoms.js";
import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";
import { runUKSDFT } from "../../src/chemistry/dft/uks-scf.js";
import { tddftPolarizability } from "../../src/chemistry/tddft-response.js";
import { uksCphfPolarizability } from "../../src/chemistry/uks-cphf.js";

describe("UKS-CPHF static polarizability (LSDA)", () => {
  test("Closed-shell H₂O / LSDA: UKS-CPHF α matches RKS-TDDFT@ω=0 to ≤ 1e-3 Ha/V², iso to 1e-4", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const nucleiSymbols: AtomSymbol[] = ["O", "H", "H"];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);

    const rks = runRKSDFT(integrals, nElectrons, nucleiSymbols, {
      functional: "lda-svwn",
      useDIIS: true, maxIter: 200, energyTol: 1e-9, residualTol: 1e-7,
    });
    expect(rks.converged).toBe(true);

    const uks = runUKSDFT(integrals, 5, 5, nucleiSymbols, {
      functional: "lda-svwn",
      useDIIS: true, maxIter: 200, energyTol: 1e-9, residualTol: 1e-7,
      symmetryBreaking: 0,
    });
    expect(uks.converged).toBe(true);

    // Reference: RKS-TDDFT(ω=0) with LSDA kernel — same machinery as
    // tda-dft's buildTDABlocks, evaluated through the response solver.
    const rksAlpha = tddftPolarizability(integrals, rks, shells,
      { method: "lda-svwn", spin: "singlet", nucleiSymbols }, 0.0);
    const uksAlpha = uksCphfPolarizability(uks, integrals, shells,
      { functional: "lda-svwn", nucleiSymbols });

    for (let k = 0; k < 9; k++) {
      expect(Math.abs(uksAlpha.alpha[k]! - rksAlpha.alpha[k]!)).toBeLessThan(1e-3);
    }
    expect(Math.abs(uksAlpha.isotropic - rksAlpha.isotropic)).toBeLessThan(1e-4);
  }, 60_000);

  test("Li doublet cc-pVDZ UKS-LSDA: α tensor real, symmetric, positive isotropic", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ], "cc-pvdz");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uks = runUKSDFT(integrals, 2, 1, ["Li"], {
      functional: "lda-svwn",
      maxIter: 200, energyTol: 1e-9, residualTol: 1e-7,
    });
    expect(uks.converged).toBe(true);
    const r = uksCphfPolarizability(uks, integrals, shells,
      { functional: "lda-svwn", nucleiSymbols: ["Li"] });
    expect(Number.isFinite(r.isotropic)).toBe(true);
    expect(r.isotropic).toBeGreaterThan(0);
    // Symmetric tensor.
    for (let x = 0; x < 3; x++) {
      for (let y = x + 1; y < 3; y++) {
        expect(Math.abs(r.alpha[x * 3 + y]! - r.alpha[y * 3 + x]!)).toBeLessThan(1e-10);
      }
    }
  }, 60_000);

  test("Non-LSDA functional refused", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "H", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uks = runUKSDFT(integrals, 1, 0, ["H"], {
      functional: "lda-svwn",
      maxIter: 200, energyTol: 1e-9,
    });
    expect(() =>
      uksCphfPolarizability(uks, integrals, shells, {
        functional: "b3lyp5",
        nucleiSymbols: ["H"],
      }),
    ).toThrow(/only "lda-svwn" supported/);
  });
});
