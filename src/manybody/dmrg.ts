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
import { type MPO } from "./mpo.js";
import {
  type SiteTensor,
  mpoToReal,
  leftEnvSeed, rightEnvSeed,
  leftEnvUpdate, rightEnvUpdate,
  makeApplyHEff,
} from "./dmrg-env.js";
import { lanczosGroundState } from "./lanczos.js";

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

// ─────────────────────────────────────────────────────────────
// Phase A.3: real two-site DMRG with matrix-free Lanczos.
// Operates on real-only MPOs (TFIM in this codebase); throws on
// any non-zero imaginary entry. Heisenberg / XXZ extensions are
// a follow-on (complex-Hermitian Lanczos + complex environments).
// ─────────────────────────────────────────────────────────────

export interface DmrgOpts {
  /** MPS bond-dim cap. */
  chiMax: number;
  /** Max number of full sweeps (left+right). Default 16. */
  maxSweeps?: number;
  /** Energy-convergence tolerance between sweeps. Default 1e-8. */
  tol?: number;
  /** Lanczos max iterations per H_eff diagonalization. Default 60. */
  lanczosMax?: number;
  /** Lanczos tolerance per H_eff diagonalization. Default 1e-10. */
  lanczosTol?: number;
  /** Seed for the random initial MPS. */
  seed?: number;
}

export interface DmrgResult {
  /** Lowest energy after final sweep. */
  energy: number;
  /** Per-sweep energy trajectory. */
  history: number[];
  /** Final MPS in canonical form, capped at chiMax. */
  mps: MPS;
  /** Total Lanczos matvec count (matvec ≈ N · sweeps · iters). */
  lanczosMatvecs: number;
}

function rng32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
}

function makeRandomMPS(N: number, chiMax: number, seed: number): SiteTensor[] {
  // Allocate per-site bond dims following the natural growth law for
  // a 1D chain: chi_q = min(2^q, 2^(N-q), chiMax). Boundary chi = 1.
  // Once q ≥ log₂(chiMax) the cap dominates, so we cap the shift at 30
  // to avoid 1 << 31 wrapping to a negative int (JS bit shifts are 32-bit
  // signed). This is the bug that blocked Phase B's N ≥ 32 runs.
  const safeShift = (k: number): number =>
    k >= 31 ? Number.MAX_SAFE_INTEGER : 1 << k;
  const chi: number[] = [1];
  for (let q = 1; q < N; q++) {
    chi.push(Math.min(safeShift(q), safeShift(N - q), chiMax));
  }
  chi.push(1);
  const r = rng32(seed);
  const sites: SiteTensor[] = [];
  for (let q = 0; q < N; q++) {
    const cL = chi[q]!;
    const cR = chi[q + 1]!;
    const data = new Float64Array(cL * 2 * cR);
    for (let i = 0; i < data.length; i++) data[i] = r() * 0.1;
    sites.push({ chiL: cL, chiR: cR, data });
  }
  return sites;
}

/** Lift a real SiteTensor to ComplexMatrix shape (chiL · 2, chiR). */
function siteToComplexMatrix(T: SiteTensor): ComplexMatrix {
  const M = zeros(T.chiL * 2, T.chiR);
  for (let i = 0; i < T.chiL * 2 * T.chiR; i++) {
    M.data[i * 2] = T.data[i]!;
  }
  return M;
}

/**
 * Build initial mixed-canonical MPS at left edge (orthogonality center
 * at site 0). Sweeps left-canonicalization right-to-left so every site
 * 1..N-1 ends up right-canonical and site 0 carries the residual norm.
 */
function rightCanonicalizeChain(sites: SiteTensor[]): void {
  // Right-canonical means T[a, σ, b] satisfies Σ_{σ, b} T[a, σ, b] · T[a', σ, b] = δ_{a,a'}.
  // Equivalent: reshape T to (chiL, 2 · chiR) and treat as (chiL × m); want rows orthonormal.
  // SVD: M = U σ V^H → set T ← V^H (rows orthonormal of length chiL ≤ rank).
  // Push U σ into the previous site's right index.
  for (let q = sites.length - 1; q >= 1; q--) {
    const T = sites[q]!;
    const m = T.chiL;
    const n = 2 * T.chiR;
    // Reshape T (chiL × 2 × chiR) to M (chiL × 2·chiR).
    const M = zeros(m, n);
    for (let a = 0; a < m; a++) {
      for (let s = 0; s < 2; s++) {
        for (let b = 0; b < T.chiR; b++) {
          M.data[(a * n + (s * T.chiR + b)) * 2] = T.data[a * 2 * T.chiR + s * T.chiR + b]!;
        }
      }
    }
    const { U, sigma, Vh } = svd(M);
    const k = Math.min(m, n, sigma.length);
    // New T_q = top-k rows of V^H, reshape to (k × 2 × chiR).
    const newData = new Float64Array(k * 2 * T.chiR);
    for (let r = 0; r < k; r++) {
      for (let s = 0; s < 2; s++) {
        for (let b = 0; b < T.chiR; b++) {
          newData[r * 2 * T.chiR + s * T.chiR + b] = Vh.data[(r * Vh.cols + (s * T.chiR + b)) * 2]!;
        }
      }
    }
    sites[q] = { chiL: k, chiR: T.chiR, data: newData };
    // Push U · σ into site q-1's right index.
    const prev = sites[q - 1]!;
    // newPrev[a, σ, r] = Σ_b prev[a, σ, b] · (U · diag(σ))[b, r]
    // Note prev.chiR was equal to old m (= sites[q].chiL).
    if (prev.chiR !== m) {
      throw new Error(`rightCanonicalizeChain: bond mismatch at q-1=${q - 1}: chiR=${prev.chiR}, m=${m}`);
    }
    const newPrev = new Float64Array(prev.chiL * 2 * k);
    for (let a = 0; a < prev.chiL; a++) {
      for (let s = 0; s < 2; s++) {
        for (let r = 0; r < k; r++) {
          let acc = 0;
          for (let b = 0; b < m; b++) {
            const Uv = U.data[(b * U.cols + r) * 2]!;
            const sV = sigma[r]!;
            acc += prev.data[a * 2 * prev.chiR + s * prev.chiR + b]! * Uv * sV;
          }
          newPrev[a * 2 * k + s * k + r] = acc;
        }
      }
    }
    sites[q - 1] = { chiL: prev.chiL, chiR: k, data: newPrev };
  }
}

/**
 * Run two-site DMRG to find the ground state of an MPO Hamiltonian.
 * MPO must have all imaginary parts zero (TFIM is supported in v1;
 * Heisenberg/XXZ require a complex-Hermitian extension).
 */
export function dmrgGroundState(M: MPO, opts: DmrgOpts): DmrgResult {
  const N = M.nQubits;
  const chiMax = opts.chiMax;
  const maxSweeps = opts.maxSweeps ?? 16;
  const tol = opts.tol ?? 1e-8;
  const lanczosMax = opts.lanczosMax ?? 60;
  const lanczosTol = opts.lanczosTol ?? 1e-10;
  const seed = opts.seed ?? 0xDEAD_BEEF;

  const W = mpoToReal(M);
  const sites = makeRandomMPS(N, chiMax, seed);
  rightCanonicalizeChain(sites);
  // After rightCanonicalize, sites 1..N-1 are right-canonical and site 0
  // carries the residual norm. Normalize site 0 to unit norm.
  {
    const T0 = sites[0]!;
    let n2 = 0;
    for (let i = 0; i < T0.data.length; i++) n2 += T0.data[i]! * T0.data[i]!;
    const inv = 1 / Math.sqrt(Math.max(n2, 1e-300));
    for (let i = 0; i < T0.data.length; i++) T0.data[i] = T0.data[i]! * inv;
  }

  // Build R[2..N] right environments. R[N] = seed.
  const R: Float64Array[] = new Array(N + 1);
  R[N] = rightEnvSeed();
  for (let q = N - 1; q >= 2; q--) {
    R[q] = rightEnvUpdate(R[q + 1]!, sites[q]!, W[q]!);
  }
  // L[0] seed; L[q] for q ≥ 1 are filled during right-sweep.
  const L: Float64Array[] = new Array(N);
  L[0] = leftEnvSeed();

  const history: number[] = [];
  let energy = Infinity;
  let lanczosMatvecs = 0;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    // ── Right sweep: optimize bonds 0, 1, …, N-2 ───────────
    for (let q = 0; q < N - 1; q++) {
      const Tq = sites[q]!;
      const Tq1 = sites[q + 1]!;
      const chiL = Tq.chiL;
      const chiR = Tq1.chiR;
      const apply = makeApplyHEff(L[q]!, R[q + 2]!, W[q]!, W[q + 1]!, chiL, chiR);

      // Seed Lanczos with the current 2-site state ψ_local = T_q · T_{q+1}.
      const dim = chiL * 4 * chiR;
      const psi0 = new Float64Array(dim);
      for (let a = 0; a < chiL; a++) {
        for (let sq = 0; sq < 2; sq++) {
          for (let b1 = 0; b1 < Tq.chiR; b1++) {
            const tq = Tq.data[a * 2 * Tq.chiR + sq * Tq.chiR + b1]!;
            for (let sq1 = 0; sq1 < 2; sq1++) {
              for (let b = 0; b < chiR; b++) {
                const tq1 = Tq1.data[b1 * 2 * Tq1.chiR + sq1 * Tq1.chiR + b]!;
                const idx = a * 4 * chiR + sq * 2 * chiR + sq1 * chiR + b;
                psi0[idx] = (psi0[idx] ?? 0) + tq * tq1;
              }
            }
          }
        }
      }
      const lan = lanczosGroundState(apply, psi0, { maxIter: lanczosMax, tol: lanczosTol });
      lanczosMatvecs += lan.iter;
      const psi = lan.vector;

      // Reshape ψ (chiL · 2, 2 · chiR) for SVD.
      const M2 = zeros(chiL * 2, 2 * chiR);
      for (let a = 0; a < chiL; a++) {
        for (let sq = 0; sq < 2; sq++) {
          for (let sq1 = 0; sq1 < 2; sq1++) {
            for (let b = 0; b < chiR; b++) {
              const v = psi[a * 4 * chiR + sq * 2 * chiR + sq1 * chiR + b]!;
              M2.data[((a * 2 + sq) * (2 * chiR) + (sq1 * chiR + b)) * 2] = v;
            }
          }
        }
      }
      const { U, sigma, Vh } = svd(M2);
      const fullRank = Math.min(M2.rows, M2.cols, sigma.length);
      const k = Math.min(chiMax, fullRank);
      // Right-sweep: T_q ← U[:, :k] (left-canonical), T_{q+1} ← σ V^H[:k, :].
      const newTq = new Float64Array(chiL * 2 * k);
      for (let row = 0; row < chiL * 2; row++) {
        for (let r = 0; r < k; r++) {
          newTq[row * k + r] = U.data[(row * U.cols + r) * 2]!;
        }
      }
      // Reshape (chiL · 2, k) → (chiL, 2, k).
      const newTqShaped = new Float64Array(chiL * 2 * k);
      for (let a = 0; a < chiL; a++) {
        for (let s = 0; s < 2; s++) {
          for (let r = 0; r < k; r++) {
            newTqShaped[a * 2 * k + s * k + r] = newTq[(a * 2 + s) * k + r]!;
          }
        }
      }
      sites[q] = { chiL, chiR: k, data: newTqShaped };
      // T_{q+1} ← σ · V^H[:k, :] reshape (k, 2, chiR)
      const newTq1 = new Float64Array(k * 2 * chiR);
      for (let r = 0; r < k; r++) {
        const s = sigma[r]!;
        for (let sq1 = 0; sq1 < 2; sq1++) {
          for (let b = 0; b < chiR; b++) {
            const v = Vh.data[(r * Vh.cols + (sq1 * chiR + b)) * 2]!;
            newTq1[r * 2 * chiR + sq1 * chiR + b] = s * v;
          }
        }
      }
      sites[q + 1] = { chiL: k, chiR, data: newTq1 };
      // Update L[q+1] from L[q] + new T_q + W_q.
      L[q + 1] = leftEnvUpdate(L[q]!, sites[q]!, W[q]!);
      energy = lan.energy;
    }
    // ── Left sweep: optimize bonds N-2, N-3, …, 0 ─────────
    for (let q = N - 2; q >= 0; q--) {
      const Tq = sites[q]!;
      const Tq1 = sites[q + 1]!;
      const chiL = Tq.chiL;
      const chiR = Tq1.chiR;
      const apply = makeApplyHEff(L[q]!, R[q + 2]!, W[q]!, W[q + 1]!, chiL, chiR);

      const dim = chiL * 4 * chiR;
      const psi0 = new Float64Array(dim);
      for (let a = 0; a < chiL; a++) {
        for (let sq = 0; sq < 2; sq++) {
          for (let b1 = 0; b1 < Tq.chiR; b1++) {
            const tq = Tq.data[a * 2 * Tq.chiR + sq * Tq.chiR + b1]!;
            for (let sq1 = 0; sq1 < 2; sq1++) {
              for (let b = 0; b < chiR; b++) {
                const tq1 = Tq1.data[b1 * 2 * Tq1.chiR + sq1 * Tq1.chiR + b]!;
                const idx = a * 4 * chiR + sq * 2 * chiR + sq1 * chiR + b;
                psi0[idx] = (psi0[idx] ?? 0) + tq * tq1;
              }
            }
          }
        }
      }
      const lan = lanczosGroundState(apply, psi0, { maxIter: lanczosMax, tol: lanczosTol });
      lanczosMatvecs += lan.iter;
      const psi = lan.vector;

      const M2 = zeros(chiL * 2, 2 * chiR);
      for (let a = 0; a < chiL; a++) {
        for (let sq = 0; sq < 2; sq++) {
          for (let sq1 = 0; sq1 < 2; sq1++) {
            for (let b = 0; b < chiR; b++) {
              const v = psi[a * 4 * chiR + sq * 2 * chiR + sq1 * chiR + b]!;
              M2.data[((a * 2 + sq) * (2 * chiR) + (sq1 * chiR + b)) * 2] = v;
            }
          }
        }
      }
      const { U, sigma, Vh } = svd(M2);
      const fullRank = Math.min(M2.rows, M2.cols, sigma.length);
      const k = Math.min(chiMax, fullRank);
      // Left-sweep: T_q ← U · σ (carries residual), T_{q+1} ← V^H (right-canonical).
      const newTq = new Float64Array(chiL * 2 * k);
      for (let row = 0; row < chiL * 2; row++) {
        for (let r = 0; r < k; r++) {
          newTq[row * k + r] = U.data[(row * U.cols + r) * 2]! * sigma[r]!;
        }
      }
      const newTqShaped = new Float64Array(chiL * 2 * k);
      for (let a = 0; a < chiL; a++) {
        for (let s = 0; s < 2; s++) {
          for (let r = 0; r < k; r++) {
            newTqShaped[a * 2 * k + s * k + r] = newTq[(a * 2 + s) * k + r]!;
          }
        }
      }
      sites[q] = { chiL, chiR: k, data: newTqShaped };
      const newTq1 = new Float64Array(k * 2 * chiR);
      for (let r = 0; r < k; r++) {
        for (let sq1 = 0; sq1 < 2; sq1++) {
          for (let b = 0; b < chiR; b++) {
            newTq1[r * 2 * chiR + sq1 * chiR + b] = Vh.data[(r * Vh.cols + (sq1 * chiR + b)) * 2]!;
          }
        }
      }
      sites[q + 1] = { chiL: k, chiR, data: newTq1 };
      // Update R[q+1] from R[q+2] + new T_{q+1} + W_{q+1}.
      R[q + 1] = rightEnvUpdate(R[q + 2]!, sites[q + 1]!, W[q + 1]!);
      energy = lan.energy;
    }

    history.push(energy);
    if (history.length >= 2) {
      const prev = history[history.length - 2]!;
      if (Math.abs(energy - prev) < tol) break;
    }
  }

  // Wrap into MPS for return.
  const mps = new MPS({ nQubits: N, chiMax });
  for (let q = 0; q < N; q++) {
    setSiteTensor(mps, q, siteToComplexMatrix(sites[q]!));
  }
  return { energy, history, mps, lanczosMatvecs };
}
