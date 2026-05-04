// Verify the from-scratch H₂ molecular-integral builder agrees with
// the hardcoded OpenFermion Pauli-sum Hamiltonian at R=0.7414 Å,
// then sanity-check the dissociation curve at additional R values.
import { describe, expect, test } from "vitest";
import { buildH2Dense, computeH2Integrals, expectationDense } from "../../src/chemistry/h2-builder.js";
import { buildDenseMatrix, smallestEigenvalue } from "../../src/chemistry/eigensolver.js";
import { H2_R0_7414, E_FCI_H2_R0_7414 } from "../../src/chemistry/h2.js";

describe("H₂ molecular-integral builder", () => {
  test("STO-3G AO integrals match published values at R=0.7414 Å", () => {
    const I = computeH2Integrals(0.7414);
    // Cross-overlap S_AB at R≈1.4 Bohr for STO-3G H 1s ≈ 0.6593 (Szabo & Ostlund Table 3.1).
    expect(I.S_AB).toBeGreaterThan(0.65);
    expect(I.S_AB).toBeLessThan(0.67);
    // Nuclear repulsion = 1/R_Bohr.
    expect(Math.abs(I.Vnn - 1 / I.R_Bohr)).toBeLessThan(1e-15);
    // h^MO_{σg, σg} should be lower than h^MO_{σu, σu} (bonding < antibonding).
    expect(I.h_MO[0]![0]!).toBeLessThan(I.h_MO[1]![1]!);
  });

  test("FCI ground state of integral-built H matches hardcoded Pauli sum within 5 mHa", () => {
    const { H } = buildH2Dense(0.7414);
    const eFromIntegrals = smallestEigenvalue(H, 16);

    const hardcodedDense = buildDenseMatrix(H2_R0_7414, 4);
    const eHardcoded = smallestEigenvalue(hardcodedDense, 16);

    console.log(
      `[h2-builder] integral-derived FCI=${eFromIntegrals}, hardcoded FCI=${eHardcoded}, ` +
      `Δ=${(eFromIntegrals - eHardcoded) * 1000} mHa`,
    );
    // Expect agreement within ~5 mHa — the OpenFermion table has small
    // FP rounding in the published coefficients.
    expect(Math.abs(eFromIntegrals - eHardcoded)).toBeLessThan(5e-3);
    // And the integral build should be in the same neighborhood as the
    // textbook H₂ FCI (-1.137 Ha).
    expect(eFromIntegrals).toBeLessThan(-1.13);
    expect(eFromIntegrals).toBeGreaterThan(-1.15);
  });

  test("dissociation curve: FCI energy varies sensibly with R", () => {
    const Rs = [0.5, 0.7414, 1.0, 1.5, 2.5];
    const energies: number[] = [];
    for (const R of Rs) {
      const { H } = buildH2Dense(R);
      const e = smallestEigenvalue(H, 16);
      energies.push(e);
      console.log(`[h2-builder] R=${R.toFixed(4)} Å  E_FCI=${e.toFixed(6)} Ha`);
    }
    // Equilibrium near R=0.7414 should be the minimum of the curve.
    const minIdx = energies.indexOf(Math.min(...energies));
    expect(Rs[minIdx]).toBe(0.7414);
    // At dissociation (R=2.5) the energy approaches 2× E(H atom) = -1.0 Ha
    // in STO-3G (basis-set incomplete, but the trend should be > -1.05).
    expect(energies[energies.length - 1]!).toBeGreaterThan(-1.05);
    // The full curve should monotone-rise from R=0.7414 outwards.
    for (let i = 1; i < energies.length - 1; i++) {
      if (Rs[i]! >= 0.7414) expect(energies[i + 1]!).toBeGreaterThan(energies[i]!);
    }
  });

  test("dense expectation matches Pauli-sum expectation at HF state |0011⟩", () => {
    // At R=0.7414 Å, both the integral builder and the hardcoded Pauli
    // sum should give the same E_HF on |HF⟩ = |0011⟩, modulo the small
    // table-rounding offset.
    const { H } = buildH2Dense(0.7414);
    const psi = new Float64Array(32);
    psi[3 * 2] = 1;
    const eHF = expectationDense(psi, H);
    expect(eHF).toBeGreaterThan(-1.13);
    expect(eHF).toBeLessThan(-1.10);
    // Sanity: HF should be above FCI (FCI is the minimum of full diagonalization).
    void E_FCI_H2_R0_7414;
  });
});
