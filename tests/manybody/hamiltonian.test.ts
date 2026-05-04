// Smoke tests for the Hamiltonian1D library: shape, hermiticity,
// and that buildDense agrees with manual single-bond expectations.
import { describe, expect, test } from "vitest";
import { heisenberg, tfim, xxz, buildDense } from "../../src/manybody/hamiltonian.js";
import { eigsymmetric } from "../../src/manybody/dense-eig.js";

describe("Hamiltonian1D — basic invariants", () => {
  test("Heisenberg bonds are real-symmetric", () => {
    const H = heisenberg(6, 1);
    for (const b of H.bonds) {
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          expect(Math.abs(b.h[i * 4 + j]! - b.h[j * 4 + i]!)).toBeLessThan(1e-15);
        }
      }
    }
  });

  test("TFIM dense matrix is hermitian (real-symmetric)", () => {
    const H = tfim(6, 1, 1);
    const M = buildDense(H);
    const dim = 1 << H.nQubits;
    let maxAsym = 0;
    for (let i = 0; i < dim; i++) {
      for (let j = i + 1; j < dim; j++) {
        maxAsym = Math.max(maxAsym, Math.abs(M[i * dim + j]! - M[j * dim + i]!));
      }
    }
    expect(maxAsym).toBeLessThan(1e-12);
  });

  test("XXZ at Δ=1 matches Heisenberg up to overall scale (J·4)", () => {
    // H_XXZ(J=1, Δ=1) = (XX + YY + ZZ); H_Heis(J=4) = (XX + YY + ZZ).
    const Hx = xxz(4, 1, 1);
    const Hh = heisenberg(4, 4);
    const Mx = buildDense(Hx);
    const Mh = buildDense(Hh);
    let maxDiff = 0;
    for (let i = 0; i < Mx.length; i++) maxDiff = Math.max(maxDiff, Math.abs(Mx[i]! - Mh[i]!));
    expect(maxDiff).toBeLessThan(1e-12);
  });
});

describe("Exact diagonalization vs textbook references (small chains)", () => {
  test("Heisenberg N=2 ground state: E_0 = -3J/4 (singlet)", () => {
    // The 2-spin Heisenberg ground state is the singlet |↑↓⟩ - |↓↑⟩,
    // E_singlet = J · (S₁·S₂) where S = σ/2 → ⟨S·S⟩_singlet = -3/4.
    const M = buildDense(heisenberg(2, 1));
    const { values } = eigsymmetric(M, 4);
    expect(values[0]).toBeCloseTo(-0.75, 9);
  });

  test("TFIM N=2 at h=J=1: E_0 = -√5  (closed form, open BC)", () => {
    // OBC, so H = -Z_0Z_1 - X_0 - X_1. In the Z₂-even parity sector
    // {|00⟩+|11⟩, |01⟩+|10⟩} the 2×2 block is [[-1,-2],[-2,1]] with
    // eigenvalues ±√5; in the odd sector eigenvalues are ±1. Smallest = -√5.
    const M = buildDense(tfim(2, 1, 1));
    const { values } = eigsymmetric(M, 4);
    expect(values[0]).toBeCloseTo(-Math.sqrt(5), 9);
  });

  test("Heisenberg N=4: E_0 ≈ -1.6160 (Bethe-ansatz / exact)", () => {
    // 4-site Heisenberg with periodic BC E_0 = -2; with OPEN BC E_0 ≈ -1.6160.
    // We use OPEN boundary (no bond between 3 and 0).
    const M = buildDense(heisenberg(4, 1));
    const { values } = eigsymmetric(M, 16);
    expect(values[0]).toBeCloseTo(-1.6160254037844, 6);
  });
});
