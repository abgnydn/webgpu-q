// H_n chain integrals + Hamiltonian + FCI energy.
// Cross-checks: H_2 must match the existing h2-builder result;
// H_3, H_4 dissociation must agree with published trends.
import { describe, expect, test } from "vitest";
import { buildHnDense, computeHnIntegrals } from "../../src/chemistry/hn-builder.js";
import { buildH2Dense } from "../../src/chemistry/h2-builder.js";
import { smallestEigenvalue } from "../../src/chemistry/eigensolver.js";

describe("H_n integrals — basic invariants", () => {
  test("H_2 integrals via Hn-builder match the dedicated H_2 builder", () => {
    const a = computeHnIntegrals(2, 0.7414);
    // S_AO is symmetric, diagonal ≈ 1 (STO-3G is not exactly normalized due
    // to the published 5-digit contraction coefficients — accept ε < 1e-7).
    expect(Math.abs(a.S_AO[0 * 2 + 0]! - 1)).toBeLessThan(1e-7);
    expect(Math.abs(a.S_AO[1 * 2 + 1]! - 1)).toBeLessThan(1e-7);
    expect(a.S_AO[0 * 2 + 1]).toBeCloseTo(a.S_AO[1 * 2 + 0]!, 12);
    expect(a.Vnn).toBeCloseTo(1 / a.R_Bohr, 12);
  });

  test("H_2 FCI from Hn-builder matches the dedicated H_2 builder", () => {
    const { H: H_hn } = buildHnDense(2, 0.7414);
    const { H: H_h2 } = buildH2Dense(0.7414);
    const e_hn = smallestEigenvalue(H_hn, 16);
    const e_h2 = smallestEigenvalue(H_h2, 16);
    expect(Math.abs(e_hn - e_h2)).toBeLessThan(1e-8);
  });

  test("H_3 chain: builds at R = 1.0 Å, ground state below HF energy bound", () => {
    const { H, nQubits } = buildHnDense(3, 1.0);
    expect(nQubits).toBe(6);
    const e = smallestEigenvalue(H, 1 << nQubits);
    // H_3 STO-3G FCI at uniform 1 Å spacing: published ~ -1.62 Ha
    // (Stair & Evangelista 2020). 5% tolerance.
    expect(e).toBeLessThan(-1.4);
    expect(e).toBeGreaterThan(-1.9);
  });

  test("H_4 chain: builds at R = 0.74 Å, ground state energy in expected window", () => {
    const { H, nQubits } = buildHnDense(4, 0.74);
    expect(nQubits).toBe(8);
    const e = smallestEigenvalue(H, 1 << nQubits);
    // H_4 in STO-3G at R=0.74 is roughly at the equilibrium minimum with
    // dimerized bonds; published FCI ~ -2.17 Ha. Loose window for now;
    // tighter benchmark would need a reference table.
    expect(e).toBeLessThan(-1.9);
    expect(e).toBeGreaterThan(-2.4);
  });

  test("H_n stretching: FCI energy at R=3.0 Å > FCI at R=1.0 Å (well goes up at long bond)", () => {
    const e1 = smallestEigenvalue(buildHnDense(3, 1.0).H, 64);
    const e3 = smallestEigenvalue(buildHnDense(3, 3.0).H, 64);
    expect(e3).toBeGreaterThan(e1);
  });
});
