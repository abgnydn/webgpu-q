// ─────────────────────────────────────────────────────────────
// mpo.ts — Matrix Product Operator representation for 1D
// nearest-neighbor Hamiltonians. The standard format every
// DMRG / TDVP routine expects.
//
// Tensor convention: the chain is N rank-4 tensors W[q] with
// shape (D_l, D_r, 2, 2). The two physical legs are indexed by
// (σ', σ) with σ the input physical index (column) and σ' the
// output (row), so applying H to a state contracts σ on the
// MPS site and produces σ' on the new site.
//
// Boundary convention: tensors[0].D_l = 1 absorbs the v_L row
// vector; tensors[N-1].D_r = 1 absorbs the v_R column vector.
// This keeps the representation uniform — the full operator
// is recovered by contracting the chain with no extra boundary
// vectors needed.
//
// Storage: each W is a flat Float64Array, length 8 · D_l · D_r,
// laid out as
//   data[((((al * D_r + ar) * 2) + sp) * 2 + s) * 2 + part]
// with part ∈ {0, 1} = {re, im}. This is the same complex
// interleaving the rest of the codebase uses; YY etc. stay real
// but Heisenberg's individual Y components are imaginary.
//
// Builders use the standard "left/right boundary + bulk" form
// (Schollwöck 2011, eq. 167):
//   • TFIM   D = 3
//   • XXZ    D = 5  (Heisenberg = XXZ at Δ=1, J/4)
// ─────────────────────────────────────────────────────────────

import { type ComplexMatrix, zeros } from "../linalg.js";

export interface MPOTensor {
  readonly D_l: number;
  readonly D_r: number;
  /** length = 8 · D_l · D_r ; layout above. */
  readonly data: Float64Array;
}

export interface MPO {
  readonly nQubits: number;
  readonly modelName: string;
  /** Length-N. tensors[0].D_l = tensors[N-1].D_r = 1 (boundary-absorbed). */
  readonly tensors: readonly MPOTensor[];
}

// ── Index helpers for the (D_l, D_r, 2, 2) complex tensor ────

function idx(D_r: number, al: number, ar: number, sp: number, s: number): number {
  return ((((al * D_r + ar) * 2 + sp) * 2 + s) * 2);
}

/** Allocate a zero-filled MPO tensor of the given shape. */
export function zeroMPOTensor(D_l: number, D_r: number): MPOTensor {
  return { D_l, D_r, data: new Float64Array(8 * D_l * D_r) };
}

/** Set the 2×2 operator block at virtual position (al, ar) from a real 4-tuple. */
function setReal2x2(W: MPOTensor, al: number, ar: number, m: readonly [number, number, number, number]): void {
  const base = idx(W.D_r, al, ar, 0, 0);
  W.data[base + 0]      = m[0]!;  // [0,0] re
  W.data[base + 2]      = m[1]!;  // [0,1] re
  W.data[base + 4]      = m[2]!;  // [1,0] re
  W.data[base + 6]      = m[3]!;  // [1,1] re
  // im parts already 0
}

/** Set the 2×2 operator block from imaginary entries (real part stays 0). */
function setImag2x2(W: MPOTensor, al: number, ar: number, m: readonly [number, number, number, number]): void {
  const base = idx(W.D_r, al, ar, 0, 0);
  W.data[base + 1] = m[0]!;
  W.data[base + 3] = m[1]!;
  W.data[base + 5] = m[2]!;
  W.data[base + 7] = m[3]!;
}

// ── Single-site primitives ───────────────────────────────────
const I_OP: readonly [number, number, number, number] = [1, 0, 0, 1];
const X_OP: readonly [number, number, number, number] = [0, 1, 1, 0];
const Z_OP: readonly [number, number, number, number] = [1, 0, 0, -1];
// Y = i · [[0, -1], [1, 0]] in the standard convention.
// Stored as imaginary part [0, -1, 1, 0].
const Y_IMAG: readonly [number, number, number, number] = [0, -1, 1, 0];

function scaledReal(m: readonly [number, number, number, number], s: number): [number, number, number, number] {
  return [m[0] * s, m[1] * s, m[2] * s, m[3] * s];
}

function scaledImag(m: readonly [number, number, number, number], s: number): [number, number, number, number] {
  return [m[0] * s, m[1] * s, m[2] * s, m[3] * s];
}

// ── TFIM MPO ─────────────────────────────────────────────────
//
// H = -h Σ X_i - J Σ Z_i Z_{i+1}
//
// Bulk W (D = 3):
//   ⎛ I    Z    -hX ⎞
//   ⎜ 0    0    -JZ ⎟
//   ⎝ 0    0     I  ⎠
// v_L = (1, 0, 0), v_R = (0, 0, 1)^T.
//
// Boundary tensors absorb v_L / v_R: tensors[0] keeps row 0 only,
// tensors[N-1] keeps column 2 only.

export function tfimMPO(N: number, J = 1, hField = 1): MPO {
  if (N < 2) throw new Error(`tfimMPO: N must be ≥ 2`);
  const D = 3;
  const tensors: MPOTensor[] = [];

  // Left boundary (D_l = 1, D_r = 3): row 0 of the bulk W.
  const W0 = zeroMPOTensor(1, D);
  setReal2x2(W0, 0, 0, I_OP);
  setReal2x2(W0, 0, 1, Z_OP);
  setReal2x2(W0, 0, 2, scaledReal(X_OP, -hField));
  tensors.push(W0);

  // Bulk (D_l = 3, D_r = 3) for sites 1..N-2.
  for (let q = 1; q < N - 1; q++) {
    const W = zeroMPOTensor(D, D);
    setReal2x2(W, 0, 0, I_OP);
    setReal2x2(W, 0, 1, Z_OP);
    setReal2x2(W, 0, 2, scaledReal(X_OP, -hField));
    setReal2x2(W, 1, 2, scaledReal(Z_OP, -J));
    setReal2x2(W, 2, 2, I_OP);
    tensors.push(W);
  }

  // Right boundary (D_l = 3, D_r = 1): column 2 of the bulk W.
  const Wlast = zeroMPOTensor(D, 1);
  setReal2x2(Wlast, 0, 0, scaledReal(X_OP, -hField));
  setReal2x2(Wlast, 1, 0, scaledReal(Z_OP, -J));
  setReal2x2(Wlast, 2, 0, I_OP);
  tensors.push(Wlast);

  return { nQubits: N, modelName: `TFIM(N=${N}, J=${J}, h=${hField})`, tensors };
}

// ── XXZ MPO ──────────────────────────────────────────────────
//
// H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1})
//
// Bulk W (D = 5):
//   ⎛ I    JX    JY    JΔZ    0  ⎞
//   ⎜ 0    0     0     0      X  ⎟
//   ⎜ 0    0     0     0      Y  ⎟
//   ⎜ 0    0     0     0      Z  ⎟
//   ⎝ 0    0     0     0      I  ⎠
// v_L = (1, 0, 0, 0, 0), v_R = (0, 0, 0, 0, 1)^T.
//
// Y is purely imaginary; the (J Y) and bare Y entries go through
// setImag2x2. All others are real.

export function xxzMPO(N: number, J = 1, delta = 1): MPO {
  if (N < 2) throw new Error(`xxzMPO: N must be ≥ 2`);
  const D = 5;
  const tensors: MPOTensor[] = [];

  // Left boundary (D_l = 1, D_r = 5): row 0 of the bulk W.
  const W0 = zeroMPOTensor(1, D);
  setReal2x2(W0, 0, 0, I_OP);
  setReal2x2(W0, 0, 1, scaledReal(X_OP, J));
  setImag2x2(W0, 0, 2, scaledImag(Y_IMAG, J));
  setReal2x2(W0, 0, 3, scaledReal(Z_OP, J * delta));
  tensors.push(W0);

  // Bulk for sites 1..N-2.
  for (let q = 1; q < N - 1; q++) {
    const W = zeroMPOTensor(D, D);
    setReal2x2(W, 0, 0, I_OP);
    setReal2x2(W, 0, 1, scaledReal(X_OP, J));
    setImag2x2(W, 0, 2, scaledImag(Y_IMAG, J));
    setReal2x2(W, 0, 3, scaledReal(Z_OP, J * delta));
    setReal2x2(W, 1, 4, X_OP);
    setImag2x2(W, 2, 4, Y_IMAG);
    setReal2x2(W, 3, 4, Z_OP);
    setReal2x2(W, 4, 4, I_OP);
    tensors.push(W);
  }

  // Right boundary (D_l = 5, D_r = 1): column 4 of the bulk W.
  const Wlast = zeroMPOTensor(D, 1);
  setReal2x2(Wlast, 1, 0, X_OP);
  setImag2x2(Wlast, 2, 0, Y_IMAG);
  setReal2x2(Wlast, 3, 0, Z_OP);
  setReal2x2(Wlast, 4, 0, I_OP);
  tensors.push(Wlast);

  return { nQubits: N, modelName: `XXZ(N=${N}, J=${J}, Δ=${delta})`, tensors };
}

// ── Heisenberg MPO ──────────────────────────────────────────
//
// H = J Σ S_i · S_{i+1} = (J/4) Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
// = XXZ(N, J/4, 1).

export function heisenbergMPO(N: number, J = 1): MPO {
  const inner = xxzMPO(N, J / 4, 1);
  return { ...inner, modelName: `Heisenberg(N=${N}, J=${J})` };
}

// ── Dense contraction (cross-validation oracle) ─────────────
//
// Contract the entire MPO chain to a 2^N × 2^N complex dense
// matrix. Used by tests to confirm the MPO equals the bond-list
// Hamiltonian. Cost: O(2^{2N} · D · N) with D = max bond dim.
// Practical for N ≤ 10.

export function mpoToDense(M: MPO): ComplexMatrix {
  const N = M.nQubits;
  const dim = 1 << N;
  const out = zeros(dim, dim);
  for (const W of M.tensors) {
    if (!Number.isFinite(W.D_l) || !Number.isFinite(W.D_r)) {
      throw new Error("mpoToDense: invalid tensor shape");
    }
  }

  // For each (sp, s) pair, walk the chain accumulating a
  // (1, D_curr) complex row vector, where D_curr = D_l of the
  // next tensor. After the last site, D_curr = 1 and the row
  // vector's single entry is the matrix element.
  for (let sp = 0; sp < dim; sp++) {
    for (let s = 0; s < dim; s++) {
      // Vector A of length D_curr (initially 1), interleaved [re, im].
      let A = new Float64Array(2);
      A[0] = 1; // (1 + 0i)
      let D_curr = 1;
      for (let q = 0; q < N; q++) {
        const W = M.tensors[q]!;
        if (W.D_l !== D_curr) {
          throw new Error(`mpoToDense: bond mismatch at site ${q}: D_l=${W.D_l}, prev D_r=${D_curr}`);
        }
        const sq = (s >> q) & 1;
        const spq = (sp >> q) & 1;
        const next = new Float64Array(2 * W.D_r);
        for (let ar = 0; ar < W.D_r; ar++) {
          let nr = 0, ni = 0;
          for (let al = 0; al < W.D_l; al++) {
            const wIdx = idx(W.D_r, al, ar, spq, sq);
            const wr = W.data[wIdx]!;
            const wi = W.data[wIdx + 1]!;
            const ar_re = A[al * 2]!;
            const ar_im = A[al * 2 + 1]!;
            // (ar_re + i ar_im) · (wr + i wi)
            nr += ar_re * wr - ar_im * wi;
            ni += ar_re * wi + ar_im * wr;
          }
          next[ar * 2] = nr;
          next[ar * 2 + 1] = ni;
        }
        A = next;
        D_curr = W.D_r;
      }
      if (D_curr !== 1) {
        throw new Error(`mpoToDense: trailing bond ≠ 1 (got ${D_curr})`);
      }
      const k = (sp * dim + s) * 2;
      out.data[k]     = A[0]!;
      out.data[k + 1] = A[1]!;
    }
  }
  return out;
}

/** Frobenius distance ‖A − B‖_F for complex matrices. */
export function complexMatrixFrobDistance(A: ComplexMatrix, B: ComplexMatrix): number {
  if (A.rows !== B.rows || A.cols !== B.cols) {
    throw new Error(`shape mismatch: ${A.rows}×${A.cols} vs ${B.rows}×${B.cols}`);
  }
  let s = 0;
  for (let i = 0; i < A.data.length; i++) {
    const d = A.data[i]! - B.data[i]!;
    s += d * d;
  }
  return Math.sqrt(s);
}
