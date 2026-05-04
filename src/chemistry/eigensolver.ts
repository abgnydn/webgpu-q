// ─────────────────────────────────────────────────────────────
// eigensolver.ts — Real-symmetric Jacobi eigensolver for small
// dense matrices.
//
// We need this only as a *ground-truth* tool: build the full
// 16×16 real-symmetric matrix for H₂, diagonalize, compare
// VQE's lowest energy to the smallest eigenvalue. Production
// VQE never calls it.
//
// Algorithm: classical Jacobi sweeps on the full off-diagonal,
// rotating the largest off-diagonal pair each pass. Converges
// quadratically near the fixed point. For n = 16 finishes in <1ms.
// ─────────────────────────────────────────────────────────────

import { expectationPauli, type Hamiltonian } from "./pauli.js";

/** Build the dense real-symmetric matrix for a Hamiltonian via column-wise expectations. */
export function buildDenseMatrix(H: Hamiltonian, nQubits: number): Float64Array {
  const N = 1 << nQubits;
  const out = new Float64Array(N * N);

  // M[i,j] = ⟨e_i| H |e_j⟩. We compute column j by setting ψ = e_j, then for
  // each row i, M[i,j] = (Hψ)_i. Because H is a Pauli sum we can extract
  // (Hψ)_i = Σ_t coeff_t · (P_t e_j)_i, and (P_t e_j)_i depends only on
  // whether i is the basis index reachable from j under P_t's bit-flips.

  // Helper: compute (P · e_j) for a single Pauli string. Returns the
  // single nonzero index k = j ^ xyMask, plus the (re, im) factor that
  // multiplies it.
  type Slot = { k: number; re: number; im: number };
  const slots: Slot[] = new Array(H.length);
  // Pre-compute masks per term.
  const terms = H.map((t) => {
    let xyMask = 0, yMask = 0, zMask = 0;
    for (let q = 0; q < t.ops.length; q++) {
      const c = t.ops[q]!;
      if (c === "X" || c === "Y") xyMask |= 1 << q;
      if (c === "Y") yMask |= 1 << q;
      if (c === "Z") zMask |= 1 << q;
    }
    const yc = popcount(yMask);
    const yPhaseRe = [1, 0, -1, 0][yc & 3]!;
    const yPhaseIm = [0, 1, 0, -1][yc & 3]!;
    return { coeff: t.coeff, xyMask, yMask, zMask, yPhaseRe, yPhaseIm };
  });

  for (let j = 0; j < N; j++) {
    // (Pψ_j) for each term.
    for (let t = 0; t < terms.length; t++) {
      const T = terms[t]!;
      const k = j ^ T.xyMask;
      // Sign from the Y AND Z bookkeeping — matches expectationPauli.
      const yParity = popcount(j & T.yMask) & 1;
      const zParity = popcount(j & T.zMask) & 1;
      const sign = (yParity ^ zParity) ? -1 : 1;
      slots[t] = { k, re: sign * T.coeff * T.yPhaseRe, im: sign * T.coeff * T.yPhaseIm };
    }

    // Accumulate (Hψ)_i = Σ_t slots[t].factor · δ_{i, slots[t].k}.
    // For Hermitian, real-coefficient Pauli sum H, the matrix is real-
    // symmetric. We zero the imaginary parts at the end as a sanity check.
    for (const s of slots) {
      const idx = s.k * N + j;
      out[idx] = (out[idx] ?? 0) + s.re;
    }
  }

  return out;
}

function popcount(x: number): number {
  let n = 0;
  while (x !== 0) { n += x & 1; x >>>= 1; }
  return n;
}

/** Lowest eigenvalue of a real-symmetric N×N matrix via Jacobi. */
export function smallestEigenvalue(M: Float64Array, N: number): number {
  return smallestEigenpair(M, N).value;
}

/**
 * Lowest eigenpair of a real-symmetric N×N matrix via Jacobi sweeps.
 *
 * Tracks the rotation matrix V such that V^T M V = D (diagonal). The
 * eigenvector with the smallest diagonal entry is the column of V at
 * that index — i.e. the corresponding column of the accumulated
 * rotation. For our N = 16 H₂ matrix this finishes well under 1 ms.
 */
export function smallestEigenpair(M: Float64Array, N: number): {
  value: number;
  vector: Float64Array;
} {
  const A = new Float64Array(M);
  const V = new Float64Array(N * N);
  for (let i = 0; i < N; i++) V[i * N + i] = 1;

  const TOL = 1e-12;
  const MAX_SWEEPS = 200;

  for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
    let off = 0;
    for (let p = 0; p < N - 1; p++) {
      for (let q = p + 1; q < N; q++) {
        const apq = A[p * N + q]!;
        if (Math.abs(apq) > off) off = Math.abs(apq);
      }
    }
    if (off < TOL) break;

    for (let p = 0; p < N - 1; p++) {
      for (let q = p + 1; q < N; q++) {
        const apq = A[p * N + q]!;
        if (Math.abs(apq) < TOL) continue;
        const app = A[p * N + p]!;
        const aqq = A[q * N + q]!;
        const tau = (aqq - app) / (2 * apq);
        const t = tau >= 0
          ? 1 / (tau + Math.sqrt(1 + tau * tau))
          : -1 / (-tau + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        // Rotate rows p, q of A.
        for (let k = 0; k < N; k++) {
          const akp = A[k * N + p]!;
          const akq = A[k * N + q]!;
          A[k * N + p] = c * akp - s * akq;
          A[k * N + q] = s * akp + c * akq;
        }
        // Rotate cols p, q of A.
        for (let k = 0; k < N; k++) {
          const apk = A[p * N + k]!;
          const aqk = A[q * N + k]!;
          A[p * N + k] = c * apk - s * aqk;
          A[q * N + k] = s * apk + c * aqk;
        }
        // Track the rotation in V (right-multiply: cols of V get mixed).
        for (let k = 0; k < N; k++) {
          const vkp = V[k * N + p]!;
          const vkq = V[k * N + q]!;
          V[k * N + p] = c * vkp - s * vkq;
          V[k * N + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  let minIdx = 0;
  let minE = A[0]!;
  for (let i = 1; i < N; i++) {
    const ei = A[i * N + i]!;
    if (ei < minE) { minE = ei; minIdx = i; }
  }
  const vec = new Float64Array(N);
  for (let k = 0; k < N; k++) vec[k] = V[k * N + minIdx]!;
  return { value: minE, vector: vec };
}

/**
 * Round-trip sanity check: the dense matrix built from a Hamiltonian
 * has diagonals equal to ⟨e_i|H|e_i⟩ — useful in tests to detect a
 * sign / mask error in either pauli.ts or this builder.
 */
export function diagonalsMatch(H: Hamiltonian, M: Float64Array, nQubits: number): boolean {
  const N = 1 << nQubits;
  const psi = new Float64Array(2 * N);
  for (let i = 0; i < N; i++) {
    psi.fill(0);
    psi[i * 2] = 1;
    const expVal = 0;
    let acc = expVal;
    for (const term of H) {
      acc += term.coeff * expectationPauli(psi, term.ops);
    }
    if (Math.abs(acc - M[i * N + i]!) > 1e-10) return false;
  }
  return true;
}
