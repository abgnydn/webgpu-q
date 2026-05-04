// ─────────────────────────────────────────────────────────────
// real-time.ts — real-time evolution exp(-i τ H) of a 1D MPS
// state under a Hamiltonian1D, Trotter-stepped over the bonds.
//
// Used for quench-dynamics demos: prepare a product state,
// evolve under a different Hamiltonian, watch entanglement
// spread in a light cone (Lieb-Robinson bound).
//
// Math: H_q is real-symmetric. exp(-i τ H_q) = V exp(-i τ Λ) V^T
// where (V, Λ) is the eigendecomposition. This produces a
// complex 4×4 matrix; the existing MPS.applyTwoSite handles
// complex inputs natively.
// ─────────────────────────────────────────────────────────────

import { eigsymmetric } from "./dense-eig.js";
import { type ComplexMatrix, zeros } from "../linalg.js";

/** Build exp(-i τ M) for a real-symmetric M. Returns ComplexMatrix (4×4 in our path). */
export function expmComplex(M: Float64Array, N: number, tau: number): ComplexMatrix {
  const { values, vectors } = eigsymmetric(M, N);
  // Diagonal exp(-i τ λ) split into real + imag.
  const reD = new Float64Array(N);
  const imD = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    reD[i] = Math.cos(tau * values[i]!);
    imD[i] = -Math.sin(tau * values[i]!);
  }
  // (V D V^T)[r,c] = Σ_k V[r,k] · D[k] · V[c,k]; vectors is column-major
  // (vectors[k*N + r] is row r of column k).
  const out = zeros(N, N);
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      let re = 0, im = 0;
      for (let k = 0; k < N; k++) {
        const w = vectors[k * N + r]! * vectors[k * N + c]!;
        re += w * reD[k]!;
        im += w * imD[k]!;
      }
      out.data[(r * N + c) * 2] = re;
      out.data[(r * N + c) * 2 + 1] = im;
    }
  }
  return out;
}
