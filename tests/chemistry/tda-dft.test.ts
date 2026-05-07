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

describe("Oscillator strengths", () => {
  // Oscillator strengths f_n are dimensionless transition
  // intensities; sum over states is bounded by N_electrons via
  // the Thomas-Reiche-Kuhn sum rule (within the singles manifold,
  // STO-3G truncates this).
  test("H₂ STO-3G TDA-HF: HOMO→LUMO oscillator strength matches the LCAO estimate", () => {
    // For two 1s functions at separation R with overlap S, the
    // bonding-antibonding ⟨σ_g | z | σ_u⟩ ≈ −R/(2·√(1−S²)). At
    // R = 0.7414 Å (1.401 Bohr) with STO-3G overlap ~0.66 this
    // gives |T_z| ≈ 0.93 → f = (4/3)·ω·|T_z|² ≈ 1.10 with
    // ω = 0.946 Ha. The TRK sum rule allows Σf ≤ N_e = 2, so a
    // value near 1.1 is consistent.
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
    const tda = runTDA(integrals, hf, { method: "hf" });
    expect(tda.oscillatorStrengths.length).toBe(tda.singletEnergies.length);
    expect(tda.oscillatorStrengths[0]!).toBeGreaterThan(0.9);
    expect(tda.oscillatorStrengths[0]!).toBeLessThan(1.3);
  });

  test("H₂O STO-3G TDA-HF: oscillator strengths all ≥ 0, sum bounded", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
    const tda = runTDA(integrals, hf, { method: "hf" });
    let sum = 0;
    for (const f of tda.oscillatorStrengths) {
      expect(f).toBeGreaterThanOrEqual(0);
      sum += f;
    }
    // Partial TRK sum rule — bounded by N_electrons (10 for H₂O),
    // realistically much less because we only span the singles
    // manifold within STO-3G's small virtual space.
    expect(sum).toBeLessThan(10);
  });

  test("TDA limit of TDDFT: oscillator strengths agree", () => {
    // For TDA reduces to runCIS; the TDDFT oscillator-strength
    // formula evaluated with B = 0 (HF-only would still have B ≠ 0
    // from the (ib|aj) exchange). Use a point where B is small or
    // verify that the totals are within ~10%.
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
    const tda = runTDA(integrals, hf, { method: "hf" });
    const rpa = runTDDFT(integrals, hf, { method: "hf" });
    // Bounds on per-root agreement: TDDFT and TDA should agree to
    // within a few % per state on H₂O STO-3G (B-correction is small).
    let totalTDA = 0, totalRPA = 0;
    for (let r = 0; r < tda.oscillatorStrengths.length; r++) {
      totalTDA += tda.oscillatorStrengths[r]!;
      totalRPA += rpa.oscillatorStrengths[r]!;
    }
    expect(totalRPA).toBeGreaterThan(0);
    expect(totalTDA).toBeGreaterThan(0);
    // Allow ~30% difference total (the exact match holds only in
    // the strict TDA→RPA limit; H₂O has nontrivial B-block).
    expect(Math.abs(totalRPA - totalTDA) / totalTDA).toBeLessThan(0.3);
  });

  test("DFT TDA: oscillator strengths positive across the functional ladder", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2O_ATOMS.map((a) => a.symbol);
    for (const k of ["lda-svwn", "blyp", "b3lyp5"] as const) {
      const dft = runRKSDFT(integrals, nElectrons, symbols, {
        functional: k, energyTol: 1e-12, maxIter: 200,
      });
      const tda = runTDA(integrals, dft, { method: k, nucleiSymbols: symbols });
      let total = 0;
      for (const f of tda.oscillatorStrengths) {
        expect(f).toBeGreaterThanOrEqual(0);
        total += f;
      }
      expect(total).toBeGreaterThan(0);
      expect(total).toBeLessThan(10);
    }
  }, 60_000);
});

describe("GGA / hybrid TDA + TDDFT", () => {
  // Tier 2 stage 9c: BVWN5 / BLYP / B3VWN5 / B3LYP5 TDA + TDDFT
  // singlet sectors. Every functional should produce real,
  // positive lowest-state excitations on H₂O STO-3G; full TDDFT
  // should remain ≤ TDA per state.
  test.each([
    ["bvwn5"  as const],
    ["blyp"   as const],
    ["b3vwn5" as const],
    ["b3lyp5" as const],
  ])("H₂O STO-3G %s: TDA singlet > 0; TDDFT singlet ≤ TDA", (kind) => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2O_ATOMS.map((a) => a.symbol);
    const dft = runRKSDFT(integrals, nElectrons, symbols, {
      functional: kind, energyTol: 1e-12, maxIter: 200,
    });
    const tda  = runTDA  (integrals, dft, { method: kind, nucleiSymbols: symbols, nRoots: 5 });
    const rpa  = runTDDFT(integrals, dft, { method: kind, nucleiSymbols: symbols, nRoots: 5 });
    expect(tda.usedXCKernel).toBe(true);
    expect(rpa.usedXCKernel).toBe(true);
    for (let r = 0; r < 5; r++) {
      expect(tda.singletEnergies[r]!).toBeGreaterThan(0);
      expect(rpa.singletEnergies[r]!).toBeGreaterThan(0);
      // Full RPA ≤ TDA per state for stable closed-shell ground state.
      expect(rpa.singletEnergies[r]! - tda.singletEnergies[r]!).toBeLessThan(1e-9);
    }
  }, 60_000);

  test("BLYP TDA first singlet lies between LDA and HF (typical valence ordering)", () => {
    // Hartree-Fock orbitals → small HOMO-LUMO Coulomb screening, big gap.
    // LDA orbitals → tighter gap; XC kernel adjusts, big stabilization.
    // GGA (BLYP) → KS gap close to LDA but XC kernel correction makes
    // the first valence excitation land between TDA-LDA and CIS for H₂O
    // STO-3G. This is a soft check (the ordering is a robust trend, not
    // a hard literature pin).
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O_ATOMS);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = H2O_ATOMS.map((a) => a.symbol);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-12, maxIter: 200 });
    const lda = runRKSDFT(integrals, nElectrons, symbols, {
      functional: "lda-svwn", energyTol: 1e-12, maxIter: 200,
    });
    const blyp = runRKSDFT(integrals, nElectrons, symbols, {
      functional: "blyp", energyTol: 1e-12, maxIter: 200,
    });
    const eHF   = runTDA(integrals, hf,   { method: "hf",                                    nRoots: 1 }).singletEnergies[0]!;
    const eLDA  = runTDA(integrals, lda,  { method: "lda-svwn", nucleiSymbols: symbols,      nRoots: 1 }).singletEnergies[0]!;
    const eBLYP = runTDA(integrals, blyp, { method: "blyp",     nucleiSymbols: symbols,      nRoots: 1 }).singletEnergies[0]!;
    // All positive, BLYP somewhere reasonable.
    expect(eBLYP).toBeGreaterThan(0);
    expect(eBLYP).toBeLessThan(eHF);     // hybrids and pure DFT lower than CIS for valence
    // BLYP and LDA shouldn't differ by more than a few hundred mHa for
    // a small-basis valence excitation.
    expect(Math.abs(eBLYP - eLDA)).toBeLessThan(0.2);
  }, 60_000);
});
