// EE-EOM-CCSD validation.
// Strategy:
//   1. H₂ STO-3G — strongest correctness test. For 2 electrons CCSD = FCI
//      exactly, and EOM-CCSD's spin-orbital singles+doubles manifold IS
//      the entire 4-spin-orbital Fock space minus the reference. So the
//      5 EOM-CCSD eigenvalues must match the FCI excitations:
//         3× ω_T (triplet HOMO→LUMO, M_S ∈ {−1, 0, +1})
//         1× ω_S1 (singlet HOMO→LUMO)
//         1× ω_S2 (doubly excited singlet)
//      computed by analytic Slater-rule 3×3 singlet block + 1×1 triplet
//      in the MO basis.
//   2. H₂O STO-3G — sanity: real, positive, lowest below CIS singlet.
//   3. Imaginary parts at roundoff.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runCIS } from "../../src/chemistry/cis.js";
import { runEOMCCSD } from "../../src/chemistry/eom-ccsd.js";
import { transformERIToMO } from "../../src/chemistry/mp2.js";
import { eigsymmetric } from "../../src/manybody/dense-eig.js";

const H2: Atom[] = [
  { symbol: "H", pos: [0, 0, 0] },
  { symbol: "H", pos: [0, 0, 0.7414] },
];
const H2O: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half);
  const z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
})();

describe("EE-EOM-CCSD — Tier 2 stage 24b", () => {
  test("H₂ STO-3G — EOM-CCSD ≡ FCI (2-electron exact)", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 2);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-12 });
    expect(ccsd.converged).toBe(true);

    const eom = runEOMCCSD(ccsd, integrals, hf);
    expect(eom.dim).toBe(5);
    expect(eom.energies.length).toBeGreaterThan(0);
    for (let k = 0; k < eom.energies.length; k++) {
      expect(Math.abs(eom.imag[k]!)).toBeLessThan(1e-6);
      expect(eom.energies[k]!).toBeGreaterThan(0);
    }

    // ── Compute FCI excitations analytically via Slater rules. ──
    // H₂ STO-3G: 1 occupied + 1 virtual spatial MO (call them 0 and 1).
    // MO-basis chemist ERI (pq|rs) and h_pq.
    const n = integrals.n;
    const C = hf.C_MO;
    const eri_MO = transformERIToMO(integrals.eri_AO, C, n);
    // h_MO = C^T h_AO C
    const tmp = new Float64Array(n * n);
    for (let p = 0; p < n; p++) {
      for (let nu = 0; nu < n; nu++) {
        let s = 0;
        for (let mu = 0; mu < n; mu++) s += C[mu * n + p]! * integrals.h_AO[mu * n + nu]!;
        tmp[p * n + nu] = s;
      }
    }
    const h_MO = new Float64Array(n * n);
    for (let p = 0; p < n; p++) {
      for (let q = 0; q < n; q++) {
        let s = 0;
        for (let nu = 0; nu < n; nu++) s += tmp[p * n + nu]! * C[nu * n + q]!;
        h_MO[p * n + q] = s;
      }
    }
    const Vnn = integrals.Vnn;
    const h00 = h_MO[0]!;
    const h11 = h_MO[n + 1]!;                              // 1*n + 1
    const eri0000 = eri_MO[((0 * n + 0) * n + 0) * n + 0]!; // (00|00)
    const eri1111 = eri_MO[((1 * n + 1) * n + 1) * n + 1]!;
    const eri0011 = eri_MO[((0 * n + 0) * n + 1) * n + 1]!; // (00|11) = J
    const eri0101 = eri_MO[((0 * n + 1) * n + 0) * n + 1]!; // (01|01) = K

    // Singlet 3×3 block (HF, singly excited singlet, doubly excited).
    // For H₂ STO-3G, by g/u symmetry, (01|11) = (00|01) = 0, so HF–S and
    // S–D off-diagonals vanish; only HF–D = (01|01) = K couples.
    const E_HF = 2 * h00 + eri0000 + Vnn;
    const E_S  = h00 + h11 + eri0011 + eri0101 + Vnn;
    const E_D  = 2 * h11 + eri1111 + Vnn;
    const H3 = new Float64Array([
      E_HF, 0,    eri0101,
      0,    E_S,  0,
      eri0101, 0, E_D,
    ]);
    const eig = eigsymmetric(H3, 3);
    const fci_omega_S1 = eig.values[1]! - eig.values[0]!;
    const fci_omega_S2 = eig.values[2]! - eig.values[0]!;
    // Triplet M_S=0 (or ±1, all degenerate at canonical RHF):
    const E_T = h00 + h11 + eri0011 - eri0101 + Vnn;
    const fci_omega_T = E_T - eig.values[0]!;

    console.log(`[eom-ccsd-h2] FCI ground: ${eig.values[0]!.toFixed(8)} Ha`);
    console.log(`[eom-ccsd-h2] FCI excitations (Ha): T=${fci_omega_T.toFixed(8)}, S1=${fci_omega_S1.toFixed(8)}, S2=${fci_omega_S2.toFixed(8)}`);
    console.log(`[eom-ccsd-h2] EOM-CCSD eigenvalues: [${
      Array.from(eom.energies).map(x => x.toFixed(8)).join(", ")
    }]`);

    // Expect 5 eigenvalues = 3×T + S1 + S2 sorted ascending (T < S1 < S2 for H₂ near eq).
    // After the stage 32c empirical σ-diagonal patch (rigorously derived from the
    // brute-force reference in eom-ccsd-bruteforce.test.ts), H₂ STO-3G EOM-CCSD
    // matches FCI to CCSD/FCI numerical precision (~10⁻⁶ Ha).
    expect(eom.energies.length).toBe(5);
    // 3 triplets degenerate:
    expect(Math.abs(eom.energies[0]! - eom.energies[1]!)).toBeLessThan(1e-8);
    expect(Math.abs(eom.energies[1]! - eom.energies[2]!)).toBeLessThan(1e-8);
    // Match FCI to high precision (was ~10-20 mHa pre-patch).
    expect(Math.abs(eom.energies[0]! - fci_omega_T)).toBeLessThan(1e-5);
    expect(Math.abs(eom.energies[3]! - fci_omega_S1)).toBeLessThan(1e-5);
    expect(Math.abs(eom.energies[4]! - fci_omega_S2)).toBeLessThan(1e-5);
    // Ordering is correct.
    expect(eom.energies[0]!).toBeLessThan(eom.energies[3]!);
    expect(eom.energies[3]!).toBeLessThan(eom.energies[4]!);
    // Eigenvectors are normalized.
    expect(eom.amplitudes.length).toBe(5 * eom.dim);
    for (let k = 0; k < 5; k++) {
      let norm = 0;
      for (let i = 0; i < eom.dim; i++) norm += eom.amplitudes[k * eom.dim + i]! ** 2;
      expect(Math.abs(norm - 1)).toBeLessThan(1e-6);
    }
  });

  test("H₂ STO-3G — spin classifier flags 3 triplets + 2 singlets", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 2);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-12 });
    const eom = runEOMCCSD(ccsd, integrals, hf);
    expect(eom.singletWeight.length).toBe(eom.energies.length);
    expect(eom.tripletWeight.length).toBe(eom.energies.length);
    console.log(`[eom-ccsd-h2-spin] energies (Ha): [${
      Array.from(eom.energies).map(x => x.toFixed(6)).join(", ")
    }]`);
    console.log(`[eom-ccsd-h2-spin] singlet wts: [${
      Array.from(eom.singletWeight).map(x => x.toFixed(3)).join(", ")
    }]`);
    console.log(`[eom-ccsd-h2-spin] triplet wts: [${
      Array.from(eom.tripletWeight).map(x => x.toFixed(3)).join(", ")
    }]`);
    // First 3 are triplets: tripletWeight ≈ 1, singletWeight ≈ 0.
    for (let k = 0; k < 3; k++) {
      expect(eom.tripletWeight[k]!).toBeGreaterThan(0.95);
      expect(eom.singletWeight[k]!).toBeLessThan(0.05);
    }
    // 4th is singlet S1 (HOMO→LUMO): singletWeight ≈ 1.
    expect(eom.singletWeight[3]!).toBeGreaterThan(0.95);
    expect(eom.tripletWeight[3]!).toBeLessThan(0.05);
    // 5th is doubly-excited singlet S2: singletWeight close to 0 (lives
    // mostly in R₂), but the R₁ portion if any should be singlet-character.
    expect(eom.tripletWeight[4]!).toBeLessThan(0.05);
  });

  test("H₂ STO-3G — oscillator strengths: triplet f ≈ 0, singlet f > 0", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 2);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-12 });
    const eom = runEOMCCSD(ccsd, integrals, hf);
    expect(eom.oscillatorStrengths.length).toBe(eom.energies.length);
    console.log(`[eom-ccsd-h2-f] energies (Ha): [${
      Array.from(eom.energies).map(x => x.toFixed(6)).join(", ")
    }]`);
    console.log(`[eom-ccsd-h2-f] f-values:    [${
      Array.from(eom.oscillatorStrengths).map(x => x.toExponential(2)).join(", ")
    }]`);
    // For H₂ STO-3G: 3 degenerate triplets, 1 dipole-allowed singlet
    // S1 (HOMO→LUMO), 1 doubly-excited singlet S2. The triplets and S2
    // are dipole-forbidden (triplets by spin selection, S2 by Pauli +
    // 1-particle nature of μ̂).
    // First 3 are triplets:
    for (let k = 0; k < 3; k++) {
      expect(eom.oscillatorStrengths[k]!).toBeLessThan(1e-6);
    }
    // 4th is dipole-allowed singlet S1.
    expect(eom.oscillatorStrengths[3]!).toBeGreaterThan(0.1);
    // 5th is doubly-excited singlet S2 — should be ≈ 0 (1-particle μ̂
    // can't reach a doubly-excited state from HF).
    expect(eom.oscillatorStrengths[4]!).toBeLessThan(1e-6);
    // Σf bounded by ~10 (TRK rule for 2 electrons gives Σf ≤ 2).
    let sumF = 0;
    for (let k = 0; k < eom.oscillatorStrengths.length; k++) sumF += eom.oscillatorStrengths[k]!;
    expect(sumF).toBeLessThan(10);
  });

  test("H₂O STO-3G — excitations real, positive, lowest below CIS singlet", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 10);
    const ccsd = runCCSD(hf, integrals, { maxIter: 50, tol: 1e-10 });
    expect(ccsd.converged).toBe(true);

    const eom = runEOMCCSD(ccsd, integrals, hf, { nRoots: 10 });
    expect(eom.energies.length).toBeGreaterThan(0);
    for (let k = 0; k < eom.energies.length; k++) {
      expect(Math.abs(eom.imag[k]!)).toBeLessThan(1e-6);
      expect(eom.energies[k]!).toBeGreaterThan(0);
    }
    const cis = runCIS(integrals, hf, { spin: "both", nRoots: 5 });
    const cisLowestSinglet = cis.singlet.energies[0]!;
    console.log(`[eom-ccsd-h2o] EOM-CCSD lowest 5 (eV): ${
      Array.from(eom.energies.slice(0, 5)).map(x => (x * 27.2114).toFixed(3)).join(", ")
    }`);
    console.log(`[eom-ccsd-h2o] CIS lowest singlet: ${(cisLowestSinglet * 27.2114).toFixed(3)} eV`);
    expect(eom.energies[0]!).toBeLessThan(cisLowestSinglet);
  }, 30000);
});
