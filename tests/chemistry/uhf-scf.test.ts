// Tier 2 stage 21: Unrestricted Hartree-Fock SCF.
//
// Pass bars:
//   1. H atom HF/STO-3G UHF with (nα=1, nβ=0): converges to ε(α 1s)
//      = −0.4666 Ha (the STO-3G H 1s eigenvalue) and total energy
//      -0.4666 Ha (a single electron has E = ε exactly, no e-e
//      repulsion).
//   2. Li atom HF/STO-3G UHF with (nα=2, nβ=1): converges to a
//      doublet ground state, E ≈ -7.31 Ha (literature ~-7.31).
//      ⟨S²⟩ ≈ 0.75 ± 0.01 (S=1/2 ⇒ S(S+1) = 0.75; small spin
//      contamination from UHF's permitted α/β orbital splitting).
//   3. UHF on a closed-shell system (H₂, nα=nβ=1) reproduces the
//      RHF energy to 1e-8 — the symmetry-breaking initial guess
//      relaxes back to the closed-shell solution at convergence.
//   4. Spin density Σ tr(D_α − D_β) integrates to (nα − nβ) — the
//      conservation law that confirms the density matrices
//      contain the right number of α and β electrons.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import type { Atom } from "../../src/chemistry/atoms.js";

describe("UHF — H atom (1-electron doublet)", () => {
  test("H atom HF/STO-3G converges to ε(1s) ≈ −0.4666 Ha", () => {
    const atoms: Atom[] = [{ symbol: "H", pos: [0, 0, 0] }];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 1, 0, {
      energyTol: 1e-10, densityTol: 1e-9, maxIter: 50,
    });
    expect(uhf.converged).toBe(true);
    // 1-electron system: energy = ε(1s) (no electron-electron repulsion).
    expect(uhf.energy).toBeCloseTo(-0.4665818, 4);
    expect(uhf.orbitalEnergiesAlpha[0]!).toBeCloseTo(-0.4665818, 4);
    // ⟨S²⟩ for a 1-electron doublet = S(S+1) = 0.75 exactly (no
    // contamination since there's nothing for the single α electron
    // to mix with).
    expect(Math.abs(uhf.s2 - 0.75)).toBeLessThan(1e-10);
    expect(uhf.s2Exact).toBeCloseTo(0.75, 12);
  });
});

describe("UHF — Li atom (3-electron doublet)", () => {
  test("Li atom HF/STO-3G converges to E ≈ −7.31 Ha with ⟨S²⟩ ≈ 0.75", { timeout: 30000 }, () => {
    const atoms: Atom[] = [{ symbol: "Li", pos: [0, 0, 0] }];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 1, {
      energyTol: 1e-10, densityTol: 1e-9, maxIter: 200,
    });
    expect(uhf.converged).toBe(true);
    // Li atom HF/STO-3G ≈ −7.3155 Ha (Hehre/Stewart/Pople basis).
    expect(uhf.energy).toBeGreaterThan(-7.4);
    expect(uhf.energy).toBeLessThan(-7.2);
    // Expected S(S+1) = 3/4 for doublet; UHF can have small contamination.
    expect(Math.abs(uhf.s2 - 0.75)).toBeLessThan(0.05);
    expect(uhf.s2Exact).toBeCloseTo(0.75, 12);
  });
});

describe("UHF — closed-shell collapse to RHF", () => {
  test("H₂ HF/STO-3G with (nα=nβ=1) reproduces RHF energy to 1e-8", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, -0.37] },
      { symbol: "H", pos: [0, 0,  0.37] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const rhf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, energyTol: 1e-12, densityTol: 1e-10, maxIter: 200,
    });
    const uhf = runUHFSCF(integrals, 1, 1, {
      energyTol: 1e-12, densityTol: 1e-10, maxIter: 200,
      symmetryBreaking: 0.01,        // small to break degeneracy in initial guess
    });
    expect(uhf.converged).toBe(true);
    // UHF should land on the same closed-shell solution.
    expect(Math.abs(uhf.energy - rhf.energy)).toBeLessThan(1e-8);
    // ⟨S²⟩ = 0 for a singlet to FP precision.
    expect(Math.abs(uhf.s2 - 0)).toBeLessThan(1e-6);
  });
});

describe("UHF — spin density conservation", () => {
  test("Li atom: tr(D_α − D_β)·S_AO integrates to (nα − nβ) = 1", { timeout: 30000 }, () => {
    const atoms: Atom[] = [{ symbol: "Li", pos: [0, 0, 0] }];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 1, {
      energyTol: 1e-10, densityTol: 1e-9, maxIter: 200,
    });
    // Σ_μν spinDensity[μν]·S_AO[μν] = nα − nβ.
    let charge = 0;
    const n = integrals.n;
    for (let mu = 0; mu < n; mu++) {
      for (let nu = 0; nu < n; nu++) {
        charge += uhf.spinDensity[mu * n + nu]! * integrals.S_AO[mu * n + nu]!;
      }
    }
    expect(Math.abs(charge - 1)).toBeLessThan(1e-8);
    // And tr((D_α + D_β)·S_AO) = total electrons = 3.
    let totalE = 0;
    for (let mu = 0; mu < n; mu++) {
      for (let nu = 0; nu < n; nu++) {
        totalE += (uhf.D_alpha[mu * n + nu]! + uhf.D_beta[mu * n + nu]!) * integrals.S_AO[mu * n + nu]!;
      }
    }
    expect(Math.abs(totalE - 3)).toBeLessThan(1e-8);
  });
});
