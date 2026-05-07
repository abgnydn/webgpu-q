// Tier 2 stage 9: TDA-DFT singlet excitation energies.
//
// Pass bars:
//   • method = "hf" reproduces runCIS singlet energies to 1e-10
//     (just generalizes the CIS A matrix with hfMix = 1; no XC
//     kernel; pure CIS algebra).
//   • method = "lda-svwn" gives a different, lower H₂O first
//     singlet than HF/CIS (LDA orbitals + XC kernel pull the
//     excitation toward the experimental ~7 eV regime — a known
//     LDA improvement over CIS for valence transitions).
//   • GGA / hybrid functionals throw a clear "needs GGA TDA"
//     error.
//   • Amplitudes are normalized.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";
import { runCIS } from "../../src/chemistry/cis.js";
import { runTDA, runTDDFT } from "../../src/chemistry/tda-dft.js";

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

describe("TDA-HF reproduces CIS exactly", () => {
  test("H₂O STO-3G: runTDA(method='hf') singlet energies match runCIS singlet to 1e-10", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, energyTol: 1e-12, densityTol: 1e-10, maxIter: 200,
    });
    const cis = runCIS(integrals, hf, { spin: "singlet" });
    const tda = runTDA(integrals, hf, { method: "hf" });
    expect(tda.singletEnergies.length).toBe(cis.singlet.energies.length);
    for (let r = 0; r < cis.singlet.energies.length; r++) {
      expect(Math.abs(tda.singletEnergies[r]! - cis.singlet.energies[r]!)).toBeLessThan(1e-10);
    }
    expect(tda.hfMix).toBe(1.0);
    expect(tda.usedXCKernel).toBe(false);
  });
});

describe("TDA-LDA: shifts H₂O first singlet below the HF/CIS value", () => {
  test("H₂O STO-3G: TDA-LDA first singlet is lower than HF/CIS (LDA over-binds)", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2O_ATOMS.map((a) => a.symbol);

    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, energyTol: 1e-12, maxIter: 200,
    });
    const cisHF = runCIS(integrals, hf, { spin: "singlet" });

    const dft = runRKSDFT(integrals, nElectrons, symbols, {
      functional: "lda-svwn", energyTol: 1e-12, maxIter: 200,
    });
    const tdaLDA = runTDA(integrals, dft, {
      method: "lda-svwn",
      nucleiSymbols: symbols,
    });

    // Both should produce sensible positive excitation energies.
    expect(tdaLDA.singletEnergies[0]!).toBeGreaterThan(0);
    expect(tdaLDA.usedXCKernel).toBe(true);
    expect(tdaLDA.hfMix).toBe(0);
    // LDA orbitals have a smaller HOMO-LUMO gap than HF, and the
    // XC kernel adds a positive correction — net effect on H₂O is
    // a lower TDA-LDA first singlet than HF/CIS. Both are in the
    // 0.2-0.5 Ha range for STO-3G.
    expect(tdaLDA.singletEnergies[0]!).toBeLessThan(cisHF.singlet.energies[0]!);
  }, 60_000);
});

describe("TDA-DFT amplitudes: normalization", () => {
  test("LDA H₂O: every singlet eigenvector has ‖c‖² = 1 to 1e-10", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2O_ATOMS.map((a) => a.symbol);
    const dft = runRKSDFT(integrals, nElectrons, symbols, {
      functional: "lda-svwn", energyTol: 1e-12, maxIter: 200,
    });
    const tda = runTDA(integrals, dft, { method: "lda-svwn", nucleiSymbols: symbols });
    const dim = tda.nOccupied * tda.nVirtual;
    for (let r = 0; r < tda.singletEnergies.length; r++) {
      let nrm = 0;
      for (let k = 0; k < dim; k++) {
        const c = tda.singletAmplitudes[r * dim + k]!;
        nrm += c * c;
      }
      expect(Math.abs(nrm - 1)).toBeLessThan(1e-10);
    }
  }, 60_000);
});

describe("Full TDDFT (Casida): excitations real and lower than TDA", () => {
  // Full RPA / TDDFT includes B-block coupling — for closed-shell
  // ground states, full TDDFT eigenvalues are STRICTLY ≤ the TDA
  // eigenvalues at the same level (well-known TDA→full RPA shift).
  // Same-system check: the Bauernschmitt-Ahlrichs 1996 ordering.

  test("H₂O STO-3G TDHF: eigenvalues real, positive, ≤ TDA-HF", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, energyTol: 1e-12, maxIter: 200,
    });
    const tda = runTDA(integrals, hf, { method: "hf", nRoots: 5 });
    const rpa = runTDDFT(integrals, hf, { method: "hf", nRoots: 5 });
    expect(rpa.singletEnergies.length).toBe(5);
    // All eigenvalues should be real and positive (no instability).
    for (let r = 0; r < 5; r++) {
      expect(rpa.singletEnergies[r]!).toBeGreaterThan(0);
    }
    // Full RPA ≤ TDA (per excited state, for stable closed-shell).
    // Tolerance 1e-9 allows fp noise above the strict ≤.
    for (let r = 0; r < 5; r++) {
      expect(rpa.singletEnergies[r]! - tda.singletEnergies[r]!).toBeLessThan(1e-9);
    }
  });

  test("H₂O STO-3G TDDFT-LDA: eigenvalues real, positive, ≤ TDA-LDA", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2O_ATOMS.map((a) => a.symbol);
    const dft = runRKSDFT(integrals, nElectrons, symbols, {
      functional: "lda-svwn", energyTol: 1e-12, maxIter: 200,
    });
    const tda = runTDA(integrals, dft, { method: "lda-svwn", nucleiSymbols: symbols, nRoots: 5 });
    const rpa = runTDDFT(integrals, dft, { method: "lda-svwn", nucleiSymbols: symbols, nRoots: 5 });
    for (let r = 0; r < 5; r++) {
      expect(rpa.singletEnergies[r]!).toBeGreaterThan(0);
      expect(rpa.singletEnergies[r]! - tda.singletEnergies[r]!).toBeLessThan(1e-9);
    }
  }, 60_000);
});

describe("TDA-DFT: GGA / hybrid not yet supported", () => {
  test.each([
    ["bvwn5"],
    ["blyp"],
    ["b3vwn5"],
    ["b3lyp5"],
  ])("method = %s throws with clear TODO", (kind) => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, maxIter: 200 });
    expect(() => runTDA(integrals, hf, {
      // Casting via "as never" because TDAMethod doesn't accept these
      // values at compile time — runtime check is the safety net.
      method: kind as never,
      nucleiSymbols: H2O_ATOMS.map((a) => a.symbol),
    })).toThrow(/GGA.*XC kernel/);
  });
});
