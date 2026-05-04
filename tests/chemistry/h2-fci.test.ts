// Verify the H₂ Hamiltonian we hard-coded actually has FCI energy
// ≈ −1.137 Ha. If this fails, the coefficient table or the Pauli
// expectation routine is wrong — every downstream chemistry test
// is meaningless until this passes.
import { describe, expect, test } from "vitest";
import { H2_R0_7414, E_FCI_H2_R0_7414, E_HF_H2_R0_7414 } from "../../src/chemistry/h2.js";
import { buildDenseMatrix, diagonalsMatch, smallestEigenvalue } from "../../src/chemistry/eigensolver.js";
import { expectationHamiltonian } from "../../src/chemistry/pauli.js";

describe("H₂ Hamiltonian — ground truth", () => {
  test("dense diagonals match per-Pauli expectation (sanity)", () => {
    const M = buildDenseMatrix(H2_R0_7414, 4);
    expect(diagonalsMatch(H2_R0_7414, M, 4)).toBe(true);
  });

  test("real-symmetric within roundoff", () => {
    const N = 16;
    const M = buildDenseMatrix(H2_R0_7414, 4);
    let maxAsym = 0;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        maxAsym = Math.max(maxAsym, Math.abs(M[i * N + j]! - M[j * N + i]!));
      }
    }
    expect(maxAsym).toBeLessThan(1e-12);
  });

  test("Jacobi diagonalization is deterministic across runs", () => {
    const M = buildDenseMatrix(H2_R0_7414, 4);
    const e1 = smallestEigenvalue(M, 16);
    const e2 = smallestEigenvalue(M, 16);
    expect(e1).toBe(e2);
    console.log(`[h2-fci-test] Jacobi smallest eigenvalue: ${e1}`);
  });

  test("FCI ground state matches the published constant within Jacobi tolerance", () => {
    const M = buildDenseMatrix(H2_R0_7414, 4);
    const e = smallestEigenvalue(M, 16);
    console.log(`[fci-eigenvalue] e=${e}, expected ${E_FCI_H2_R0_7414}, diff ${e - E_FCI_H2_R0_7414}`);
    expect(Math.abs(e - E_FCI_H2_R0_7414)).toBeLessThan(1e-6);
    expect(e).toBeLessThan(-1.13);
    expect(e).toBeGreaterThan(-1.14);
  });

  test("HF reference state |0011⟩ has E ≈ E_HF (qubits 0,1 occupied)", () => {
    const psi = new Float64Array(32);
    psi[3 * 2] = 1;
    const e = expectationHamiltonian(psi, H2_R0_7414);
    expect(Math.abs(e - E_HF_H2_R0_7414)).toBeLessThan(1e-6);
  });
});
