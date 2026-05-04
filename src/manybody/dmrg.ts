// ─────────────────────────────────────────────────────────────
// dmrg.ts — ground-state finder via direct diagonalization +
// statevector-to-MPS conversion.
//
// This is the small-N gold standard: build the dense Hamiltonian,
// find its smallest eigenvalue with a real-symmetric Jacobi
// eigensolver, then compress the ground-state vector into a
// canonical-form MPS by sweeping SVDs from left to right with
// chiMax truncation.
//
// For N ≤ 14 this is the cleanest path — no DMRG sweeps needed
// because the full Hamiltonian fits in memory and we can solve
// it exactly. The MPS conversion is the same building block a
// full two-site DMRG (with Lanczos local solver, MPO H_eff, and
// environment tracking) would use to project the global ground
// state onto the bond-truncated manifold.
//
// Validation target: ITensor reference energies in
// tests/manybody/itensor-reference.json — the energy returned by
// `exactGroundStateMPS` should agree to f64 precision for any
// chiMax that exceeds the model's required bond dimension.
// ─────────────────────────────────────────────────────────────

import { type Hamiltonian1D, buildDense } from "./hamiltonian.js";
import { eigsymmetric } from "./dense-eig.js";
import { MPS } from "../mps.js";
import { type ComplexMatrix, svd, zeros } from "../linalg.js";

export interface ExactGroundStateOpts {
  /** MPS bond-dim cap. Defaults to 2^(N/2) (no truncation for N ≤ 12). */
  chiMax?: number;
}

export interface ExactGroundStateResult {
  /** Ground-state energy E_0 = smallest eigenvalue of H (exact, in f64). */
  energy: number;
  /** Ground-state vector ψ as a flat real Float64Array, length 2^N. */
  psi: Float64Array;
  /** Compressed MPS form of ψ, with each bond capped at chiMax. */
  mps: MPS;
  /** Fidelity |⟨ψ_exact | ψ_mps⟩|² of the truncated MPS vs the exact state. */
  truncationFidelity: number;
}

/**
 * Exact ground-state of a 1D bond-decomposed Hamiltonian, returned both as a
 * flat statevector (the input to any post-processing) and as a chiMax-bounded
 * MPS (the form DMRG / time-evolution / measurement code consumes).
 *
 * Capped at N ≤ 14 by the dense Hamiltonian's 2^(2N) memory footprint.
 */
export function exactGroundStateMPS(
  H: Hamiltonian1D,
  opts: ExactGroundStateOpts = {},
): ExactGroundStateResult {
  const N = H.nQubits;
  if (N > 14) {
    throw new Error(`exactGroundStateMPS: N=${N} too large (dense H is 2^${2 * N} f64).`);
  }
  const chiMax = opts.chiMax ?? Math.max(1, 1 << Math.ceil(N / 2));

  // 1. Dense diagonalization.
  const Hdense = buildDense(H);
  const dim = 1 << N;
  const { values, vectors } = eigsymmetric(Hdense, dim);
  const energy = values[0]!;
  // Smallest eigenvector lives in column 0 (eigsymmetric returns ascending).
  const psi = new Float64Array(dim);
  for (let i = 0; i < dim; i++) psi[i] = vectors[i]!;

  // 2. Statevector → canonical-form MPS via SVD chain.
  const mps = statevectorToMPS(psi, N, chiMax);

  // 3. Truncation-fidelity diagnostic.
  const psiMps = mps.statevector();
  let dot = 0;
  for (let i = 0; i < dim; i++) {
    // psiMps is interleaved [re, im]; psi is real → dot is real.
    dot += psi[i]! * psiMps[i * 2]!;
  }
  const truncationFidelity = dot * dot;

  return { energy, psi, mps, truncationFidelity };
}

/**
 * Compress a flat real statevector ψ ∈ ℝ^(2^N) into an MPS in
 * left-canonical form, truncating each bond at chiMax via SVD.
 *
 * Algorithm (single left-to-right sweep):
 *   reshape ψ as (2 × 2^(N-1)) → SVD → T_0 = U[:, :k], rest = σ_k V_k^H
 *   reshape rest as (2k × 2^(N-2)) → SVD → T_1 = U[:, :k] reshaped,
 *     rest = σ V^H continues
 *   …repeat until we've consumed all sites.
 *
 * Each T_q ends up shape (chi_L · 2, chi_R), the same layout MPS expects.
 */
function statevectorToMPS(psi: Float64Array, N: number, chiMax: number): MPS {
  const mps = new MPS({ nQubits: N, chiMax });
  if (N === 1) {
    // Special case: just stuff [psi[0], psi[1]] into T_0 of shape (2, 1).
    const T = zeros(2, 1);
    T.data[0] = psi[0]!; T.data[1] = 0;
    T.data[2] = psi[1]!; T.data[3] = 0;
    setSiteTensor(mps, 0, T);
    return mps;
  }

  // "Remaining" matrix: starts as ψ reshaped to 2 × 2^(N-1), then grows.
  // Stored as a ComplexMatrix (zero imag).
  let chiL = 1;
  let rest = zeros(2 * chiL, 1 << (N - 1));
  for (let i = 0; i < rest.data.length; i++) rest.data[i] = 0;
  for (let row = 0; row < 2; row++) {
    for (let col = 0; col < (1 << (N - 1)); col++) {
      // Basis index of full ψ corresponding to (s_0=row, rest_bits=col).
      // ψ index convention: |s_0 s_1 ... s_{N-1}⟩ at index Σ_q s_q · 2^q.
      // Here rest_bits encodes (s_1, …, s_{N-1}) in their natural order.
      const idx = row | (col << 1);
      rest.data[(row * rest.cols + col) * 2] = psi[idx]!;
    }
  }

  for (let q = 0; q < N - 1; q++) {
    const m = rest.rows;
    const n = rest.cols;
    const { U, sigma, Vh } = svd(rest);
    // Truncate to k = min(chiMax, rank).
    const rank = Math.min(m, n, sigma.length);
    let k = Math.min(chiMax, rank);
    // Drop near-zero σ.
    const sigma0 = sigma[0] ?? 1;
    while (k > 1 && (sigma[k - 1] ?? 0) < 1e-14 * sigma0) k--;

    // T_q = U[:, :k] reshaped as (chiL · 2, k).
    const T = zeros(m, k);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < k; j++) {
        T.data[(i * k + j) * 2]     = U.data[(i * U.cols + j) * 2]!;
        T.data[(i * k + j) * 2 + 1] = U.data[(i * U.cols + j) * 2 + 1]!;
      }
    }
    setSiteTensor(mps, q, T);

    // rest_new = σ_k · V^H[:k, :] reshaped to (k · 2, n / 2)
    // (so the next two physical-site bits become the row dim's high bit
    // for the next iteration). For the last bond (q == N-2) we just store
    // the residual into T_{N-1} as shape (k · 2, 1).
    const restRows = k;
    const restCols = n;
    const r = zeros(restRows, restCols);
    for (let i = 0; i < restRows; i++) {
      const s = sigma[i] ?? 0;
      for (let j = 0; j < restCols; j++) {
        r.data[(i * restCols + j) * 2]     = s * Vh.data[(i * Vh.cols + j) * 2]!;
        r.data[(i * restCols + j) * 2 + 1] = s * Vh.data[(i * Vh.cols + j) * 2 + 1]!;
      }
    }

    if (q === N - 2) {
      // Last site: T_{N-1} has shape (k · 2, 1). Reshape r (k × 2) → (k·2, 1).
      const T_last = zeros(restRows * 2, 1);
      for (let i = 0; i < restRows; i++) {
        for (let s = 0; s < 2; s++) {
          T_last.data[((i * 2 + s) * 1 + 0) * 2]     = r.data[(i * 2 + s) * 2]!;
          T_last.data[((i * 2 + s) * 1 + 0) * 2 + 1] = r.data[(i * 2 + s) * 2 + 1]!;
        }
      }
      setSiteTensor(mps, N - 1, T_last);
    } else {
      // For the next iteration: reshape r (k × 2^(N-q-1)) so its row dim
      // absorbs the next physical bit, becoming (k · 2) × 2^(N-q-2).
      const newRows = k * 2;
      const newCols = 1 << (N - q - 2);
      const next = zeros(newRows, newCols);
      for (let i = 0; i < k; i++) {
        for (let s = 0; s < 2; s++) {
          for (let j = 0; j < newCols; j++) {
            // r's column index encodes (s_q+1, s_q+2, …, s_{N-1}) with s_q+1
            // in bit 0. We absorb s_q+1 = s into the new row dim and keep
            // (s_q+2, …) as the new column j.  So oldCol = s + (j << 1).
            const oldCol = s | (j << 1);
            next.data[((i * 2 + s) * newCols + j) * 2]     = r.data[(i * restCols + oldCol) * 2]!;
            next.data[((i * 2 + s) * newCols + j) * 2 + 1] = r.data[(i * restCols + oldCol) * 2 + 1]!;
          }
        }
      }
      rest = next;
      chiL = k;
    }
  }
  return mps;
}

/**
 * Replace site q's tensor inside an MPS. The MPS class doesn't expose a
 * setter, so we poke at the internal `tensors` array via a brand assertion.
 * Same trick the existing trajectory module uses.
 */
function setSiteTensor(mps: MPS, q: number, T: ComplexMatrix): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const internal = mps as unknown as { tensors: ComplexMatrix[] };
  internal.tensors[q] = T;
}
