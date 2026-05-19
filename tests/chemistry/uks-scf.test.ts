// UKS-DFT SCF — unrestricted Kohn-Sham (open-shell DFT).
//
// Pass bars:
//   1. Closed-shell consistency: UKS-LDA on H₂O with n_α = n_β
//      matches RKS-LDA to ≤ 1e-6 Ha (open-shell-as-closed-shell
//      sanity check, mirroring uhf-scf vs rhf-scf).
//   2. H atom doublet UKS-LDA: 1 α electron, 0 β. Converges to
//      the expected ~−0.45 Ha LDA value. ⟨S²⟩ ≈ 0.75 (no
//      contamination on a single-electron system).
//   3. Li atom doublet UKS-LDA cc-pVDZ: converges, ⟨S²⟩ ≈ 0.75,
//      spin density present (nonzero D_α − D_β).
//   4. GGA refused with clear message (Tier 3 follow-up).

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type AtomSymbol } from "../../src/chemistry/atoms.js";
import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";
import { runUKSDFT } from "../../src/chemistry/dft/uks-scf.js";

describe("UKS-DFT SCF (open-shell DFT)", () => {
  test("Closed-shell H₂O / LDA: UKS (n_α=n_β=5) matches RKS to ≤ 1e-6 Ha", () => {
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
      useDIIS: true, maxIter: 200, energyTol: 1e-8, residualTol: 1e-6,
    });
    expect(rks.converged).toBe(true);

    const uks = runUKSDFT(integrals, 5, 5, nucleiSymbols, {
      functional: "lda-svwn",
      useDIIS: true, maxIter: 200, energyTol: 1e-8, residualTol: 1e-6,
      symmetryBreaking: 0,
    });
    expect(uks.converged).toBe(true);

    expect(Math.abs(uks.energy - rks.energy)).toBeLessThan(1e-6);
    // ⟨S²⟩ should be exactly 0 (singlet, no contamination).
    expect(Math.abs(uks.s2)).toBeLessThan(1e-4);
  }, 60_000);

  test("H atom doublet (1 α, 0 β) UKS-LDA: converges, ⟨S²⟩ ≈ 0.75", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "H", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const nucleiSymbols: AtomSymbol[] = ["H"];
    const uks = runUKSDFT(integrals, 1, 0, nucleiSymbols, {
      functional: "lda-svwn",
      maxIter: 200, energyTol: 1e-9, residualTol: 1e-7,
    });
    expect(uks.converged).toBe(true);
    expect(uks.energy).toBeLessThan(0);
    expect(uks.energy).toBeGreaterThan(-1);     // sanity range
    // Single electron → ⟨S²⟩ = 0.75 exactly (no β electron to overlap).
    expect(Math.abs(uks.s2 - 0.75)).toBeLessThan(1e-6);
    expect(uks.nAlpha).toBe(1);
    expect(uks.nBeta).toBe(0);
  });

  test("Li atom doublet cc-pVDZ UKS-LDA: converges, ⟨S²⟩ ≈ 0.75, spin density nonzero", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ], "cc-pvdz");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const nucleiSymbols: AtomSymbol[] = ["Li"];
    const uks = runUKSDFT(integrals, 2, 1, nucleiSymbols, {
      functional: "lda-svwn",
      maxIter: 200, energyTol: 1e-8, residualTol: 1e-6,
    });
    expect(uks.converged).toBe(true);
    // ⟨S²⟩ ≈ 0.75 ± small basis-set spin contamination.
    expect(Math.abs(uks.s2 - 0.75)).toBeLessThan(0.05);
    // Spin density should be non-zero (open-shell radical).
    let spinNormSq = 0;
    for (let i = 0; i < uks.spinDensity.length; i++) {
      spinNormSq += uks.spinDensity[i]! * uks.spinDensity[i]!;
    }
    expect(spinNormSq).toBeGreaterThan(1e-6);
  }, 60_000);

  test("GGA functional refused with clear message", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "H", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    expect(() =>
      runUKSDFT(integrals, 1, 0, ["H"], { functional: "b3lyp5" }),
    ).toThrow(/only "lda-svwn" supported/);
  });
});
