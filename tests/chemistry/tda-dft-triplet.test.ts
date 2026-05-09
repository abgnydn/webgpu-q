// Tier 2 stage 15b: triplet TDA + TDDFT for HF and LDA.
//
// Pass bars:
//   1. Spin-polarized LSDA at ρ_↑ = ρ_↓ = ρ/2 collapses to the
//      closed-shell evalXC for "lda-svwn" — both ε_xc and v_xc.
//   2. runTDA(method = "hf", spin = "triplet") matches
//      runCIS({ spin: "triplet" }) up to floating-point noise
//      (1e-10).
//   3. runTDA(method = "lda-svwn", spin = "triplet") on H₂ STO-3G
//      sits BELOW the singlet excitation (the standard energy
//      ordering: triplet states are lower than singlets at the same
//      orbital configuration — Hund's rule analog).
//   4. runTDA + GGA + triplet throws a clear "needs spin-polarized
//      GGA kernel" message.
//   5. Triplet oscillator strengths are zero (spin-forbidden ΔS=1
//      from a closed-shell ground state in the dipole approximation).
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";
import { runCIS } from "../../src/chemistry/cis.js";
import { runTDA } from "../../src/chemistry/tda-dft.js";
import { evalXC } from "../../src/chemistry/dft/functional.js";
import { evalXC_LSDA } from "../../src/chemistry/dft/functional-spin.js";

const H2_ATOMS: Atom[] = [
  { symbol: "H", pos: [0, 0,  0.37] },
  { symbol: "H", pos: [0, 0, -0.37] },
];

describe("LSDA closed-shell limit (sanity check)", () => {
  test("evalXC_LSDA at ρ_↑ = ρ_↓ = ρ/2 matches closed-shell evalXC for lda-svwn", () => {
    // Sample 10 densities spanning core to tail.
    const rho = new Float64Array([1e-3, 1e-2, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100]);
    const n = rho.length;
    const half = new Float64Array(n);
    for (let i = 0; i < n; i++) half[i] = rho[i]! / 2;

    // Closed-shell reference.
    const epsCS = new Float64Array(n);
    const vCS   = new Float64Array(n);
    evalXC("lda-svwn", rho, null, epsCS, vCS, null);

    // Spin-polarized at closed-shell baseline.
    const epsLSDA = new Float64Array(n);
    const vUp = new Float64Array(n);
    const vDn = new Float64Array(n);
    evalXC_LSDA(half, half, epsLSDA, vUp, vDn);

    for (let i = 0; i < n; i++) {
      // ε_xc per-particle (per-particle convention, NOT per-volume): exact match.
      expect(Math.abs(epsLSDA[i]! - epsCS[i]!)).toBeLessThan(1e-12);
      // v_↑ = v_↓ = closed-shell v_xc at ζ = 0.
      expect(Math.abs(vUp[i]! - vCS[i]!)).toBeLessThan(1e-10);
      expect(Math.abs(vDn[i]! - vCS[i]!)).toBeLessThan(1e-10);
    }
  });
});

describe("TDA triplet matches CIS triplet for HF reference", () => {
  test("H₂ STO-3G: runTDA(method='hf', spin='triplet') == runCIS({spin:'triplet'}) to 1e-10", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, energyTol: 1e-12, densityTol: 1e-10, maxIter: 200,
    });
    const cis = runCIS(integrals, hf, { spin: "triplet" });
    const tda = runTDA(integrals, hf, { method: "hf", spin: "triplet" });
    expect(tda.singletEnergies.length).toBe(cis.triplet.energies.length);
    for (let r = 0; r < cis.triplet.energies.length; r++) {
      expect(
        Math.abs(tda.singletEnergies[r]! - cis.triplet.energies[r]!),
      ).toBeLessThan(1e-10);
    }
    expect(tda.spin).toBe("triplet");
    expect(tda.hfMix).toBe(1.0);
    expect(tda.usedXCKernel).toBe(false);
    // Triplet ↔ closed-shell singlet GS is dipole-forbidden.
    for (let r = 0; r < tda.oscillatorStrengths.length; r++) {
      expect(tda.oscillatorStrengths[r]!).toBe(0);
    }
  });
});

describe("TDA-LDA triplet sits below TDA-LDA singlet", () => {
  test("H₂ STO-3G: ω(LDA, triplet) < ω(LDA, singlet) for the lowest excitation", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2_ATOMS.map((a) => a.symbol);
    const dft = runRKSDFT(integrals, nElectrons, symbols, {
      functional: "lda-svwn", energyTol: 1e-10, residualTol: 1e-7, maxIter: 200,
    });
    // Adapt KS solution to the HFLike interface buildTDABlocks expects.
    const ks = {
      nOccupied: dft.nOccupied, n: integrals.n,
      orbitalEnergies: dft.orbitalEnergies, C_MO: dft.C_MO, D: dft.D,
    };
    const sing = runTDA(integrals, ks, {
      method: "lda-svwn", spin: "singlet", nucleiSymbols: symbols,
    });
    const trip = runTDA(integrals, ks, {
      method: "lda-svwn", spin: "triplet", nucleiSymbols: symbols,
    });
    expect(trip.singletEnergies[0]!).toBeLessThan(sing.singletEnergies[0]!);
    // And both should be positive (no instability).
    expect(trip.singletEnergies[0]!).toBeGreaterThan(0);
    expect(sing.singletEnergies[0]!).toBeGreaterThan(0);
    expect(trip.usedXCKernel).toBe(true);
    expect(trip.hfMix).toBe(0);
  });
});

describe("Triplet supported across the full functional ladder", () => {
  test("runTDA + spin='triplet' produces real, positive energies for blyp / b3lyp5", () => {
    // After stage 15b-LYP, BLYP and B3LYP5 triplets are wired through
    // (Miehlich spin-polarized LYP). They no longer throw; the energies
    // are real and positive (no instability) and sit below the singlet.
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2_ATOMS.map((a) => a.symbol);
    for (const functional of ["blyp", "b3lyp5"] as const) {
      const dft = runRKSDFT(integrals, nElectrons, symbols, {
        functional, energyTol: 1e-10, residualTol: 1e-7, maxIter: 200,
      });
      const ks = {
        nOccupied: dft.nOccupied, n: integrals.n,
        orbitalEnergies: dft.orbitalEnergies, C_MO: dft.C_MO, D: dft.D,
      };
      const trip = runTDA(integrals, ks, {
        method: functional, spin: "triplet", nucleiSymbols: symbols,
      });
      const sing = runTDA(integrals, ks, {
        method: functional, spin: "singlet", nucleiSymbols: symbols,
      });
      expect(trip.singletEnergies[0]!).toBeGreaterThan(0);
      expect(trip.singletEnergies[0]!).toBeLessThan(sing.singletEnergies[0]!);
    }
  });
});
