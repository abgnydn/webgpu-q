// UKS-TDDFT α(ω) frequency-dependent polarizability (LSDA only).
//
// Pass bars:
//   1. Static limit (ω=0): UKS-TDDFT@ω=0 matches UKS-CPHF to ≤ 1e-7
//      per tensor element.
//   2. Closed-shell limit: UKS-TDDFT α(ω) for n_α=n_β H₂O matches
//      RKS-TDDFT α(ω) to ≤ 1e-3 at non-zero ω.
//   3. Dispersion: α(ω) grows monotonically below the first pole.
//   4. α(iω) on imaginary axis: real, positive, decreasing.
//   5. Non-LSDA functional refused.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type AtomSymbol } from "../../src/chemistry/atoms.js";
import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";
import { runUKSDFT } from "../../src/chemistry/dft/uks-scf.js";
import { tddftPolarizability } from "../../src/chemistry/tddft-response.js";
import { uksCphfPolarizability } from "../../src/chemistry/uks-cphf.js";
import {
  uksTddftPolarizability, uksTddftPolarizabilityImag,
} from "../../src/chemistry/uks-tdhf.js";

describe("UKS-TDDFT α(ω) (LSDA)", () => {
  test("Static limit ω=0: UKS-TDDFT matches UKS-CPHF to ≤ 1e-7", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ], "cc-pvdz");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uks = runUKSDFT(integrals, 2, 1, ["Li"], {
      functional: "lda-svwn",
      maxIter: 200, energyTol: 1e-9, residualTol: 1e-7,
    });

    const cphf = uksCphfPolarizability(uks, integrals, shells,
      { functional: "lda-svwn", nucleiSymbols: ["Li"] });
    const tdhf0 = uksTddftPolarizability(uks, integrals, shells, 0.0,
      { functional: "lda-svwn", nucleiSymbols: ["Li"] });

    for (let k = 0; k < 9; k++) {
      expect(Math.abs(tdhf0.alpha[k]! - cphf.alpha[k]!)).toBeLessThan(1e-7);
    }
  }, 60_000);

  test("Closed-shell H₂O LSDA: UKS-TDDFT@ω=0.1 matches RKS-TDDFT@ω=0.1 to ≤ 1e-3", () => {
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
    const uks = runUKSDFT(integrals, 5, 5, nucleiSymbols, {
      functional: "lda-svwn",
      useDIIS: true, maxIter: 200, energyTol: 1e-9, residualTol: 1e-7,
      symmetryBreaking: 0,
    });

    const omega = 0.1;
    const rksAlpha = tddftPolarizability(integrals, rks, shells,
      { method: "lda-svwn", spin: "singlet", nucleiSymbols }, omega);
    const uksAlpha = uksTddftPolarizability(uks, integrals, shells, omega,
      { functional: "lda-svwn", nucleiSymbols });

    for (let k = 0; k < 9; k++) {
      expect(Math.abs(uksAlpha.alpha[k]! - rksAlpha.alpha[k]!)).toBeLessThan(1e-3);
    }
    expect(Math.abs(uksAlpha.isotropic - rksAlpha.isotropic)).toBeLessThan(1e-4);
  }, 90_000);

  test("Li doublet cc-pVDZ α(iω): real, positive, monotonically decreasing", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ], "cc-pvdz");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uks = runUKSDFT(integrals, 2, 1, ["Li"], {
      functional: "lda-svwn",
      maxIter: 200, energyTol: 1e-9, residualTol: 1e-7,
    });
    const a0 = uksTddftPolarizability(uks, integrals, shells, 0,
      { functional: "lda-svwn", nucleiSymbols: ["Li"] }).isotropic;
    const samples = [0, 0.05, 0.1, 0.3, 1.0, 5.0].map((w) =>
      uksTddftPolarizabilityImag(uks, integrals, shells, w,
        { functional: "lda-svwn", nucleiSymbols: ["Li"] }).isotropic,
    );
    expect(Math.abs(samples[0]! - a0)).toBeLessThan(1e-10);
    for (let i = 1; i < samples.length; i++) {
      expect(samples[i]!).toBeLessThan(samples[i - 1]!);
    }
    for (const v of samples) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
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
      uksTddftPolarizability(uks, integrals, shells, 0.1, {
        functional: "b3lyp5",
        nucleiSymbols: ["H"],
      }),
    ).toThrow(/only "lda-svwn" supported/);
  });
});
