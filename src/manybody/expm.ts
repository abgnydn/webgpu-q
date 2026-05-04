// ─────────────────────────────────────────────────────────────
// expm.ts — matrix exponential exp(τ · M) for a real-symmetric
// matrix via eigendecomposition:
//
//   M = V Λ V^T  ⇒  exp(τ M) = V exp(τ Λ) V^T
//
// Used to build Trotter-step gates exp(-τ h_{i,i+1}) for
// imaginary-time evolution. The 4×4 case is the hot path; the
// generic version is also handy for exact-diagonalization
// tests (full 2^N × 2^N is too expensive but we only call this
// inside Jacobi-already path for small dim).
// ─────────────────────────────────────────────────────────────

import { eigsymmetric } from "./dense-eig.js";
import { type ComplexMatrix, zeros } from "../linalg.js";

/** Compute exp(τ · M) for a real-symmetric M, returning a real row-major matrix. */
export function expmReal(M: Float64Array, N: number, tau: number): Float64Array {
  const { values, vectors } = eigsymmetric(M, N);
  // exp(τΛ) on the diagonal.
  const e = new Float64Array(N);
  for (let i = 0; i < N; i++) e[i] = Math.exp(tau * values[i]!);

  // V exp(τΛ) V^T. vectors is column-major, i.e. vectors[col*N + row].
  // (V D V^T)[r,c] = Σ_k V[r,k] · e[k] · V[c,k]
  const out = new Float64Array(N * N);
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      let s = 0;
      for (let k = 0; k < N; k++) {
        s += vectors[k * N + r]! * e[k]! * vectors[k * N + c]!;
      }
      out[r * N + c] = s;
    }
  }
  return out;
}

/** Convenience: lift a real 4×4 result into a ComplexMatrix for the MPS API. */
export function realToComplex(real: Float64Array, n: number): ComplexMatrix {
  const out = zeros(n, n);
  for (let i = 0; i < n * n; i++) out.data[i * 2] = real[i]!;
  return out;
}
