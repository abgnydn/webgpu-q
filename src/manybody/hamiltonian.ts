// ─────────────────────────────────────────────────────────────
// hamiltonian.ts — generic 1-D nearest-neighbor Hamiltonian as
// a per-bond list of 4×4 two-site operators. Built-in models:
//
//   • Heisenberg:        H = J Σ S_i · S_{i+1}
//                          = (J/4) Σ (X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1})
//
//   • Transverse-field Ising (TFIM):
//                        H = -J Σ Z_iZ_{i+1} - h Σ X_i
//                          (single-site X is split half-on-each-bond
//                           with edge corrections; see assemble())
//
//   • XXZ:               H = J Σ (X_iX_{i+1} + Y_iY_{i+1} + Δ Z_iZ_{i+1})
//
// Convention matches MPS.applyTwoSite: the 4×4 matrix is
// indexed by i = s_q · 2 + s_{q+1} with site q as the MSB.
// All models are real-symmetric in this basis (Y⊗Y is real
// because both factors of i cancel), so we store them as
// real arrays and lift to ComplexMatrix only when handing to
// the MPS API.
// ─────────────────────────────────────────────────────────────

import { type ComplexMatrix, zeros } from "../linalg.js";

/** Real 4×4 matrix, row-major, length 16. */
export type Real4 = Float64Array;

export interface BondTerm {
  /** Two-site operator h_{q, q+1} as a row-major real 4×4 matrix. */
  readonly h: Real4;
}

export interface Hamiltonian1D {
  readonly nQubits: number;
  readonly modelName: string;
  /** One bond term per nearest-neighbor pair (length N − 1). */
  readonly bonds: readonly BondTerm[];
  /** Optional single-site terms baked into the leftmost bond
   *  via h_0 += (1-site_left ⊗ I)/2; see edge handling below. */
}

// ── Pauli matrices, as flat row-major real 4×4 ─────────────
//
// Each `pauli2(A, B)` returns A ⊗ B in the (s_q · 2 + s_{q+1}) basis.
// Helpers below build XX, YY, ZZ, etc. from primitive 2×2 reals.

type Real2 = [number, number, number, number];

const I2: Real2 = [1, 0, 0, 1];
const X2: Real2 = [0, 1, 1, 0];
// Y is imaginary, but Y⊗Y is real:
//   (Y⊗Y)[i,j] = Y[a,b] · Y[c,d] where (i,j) decode bits.
//   Since each Y has a single 'i' factor, (i)·(i) = -1: real result.
// We hard-code the result for YY rather than carrying complex through.
const Z2: Real2 = [1, 0, 0, -1];

function kron2x2(A: Real2, B: Real2): Real4 {
  // (A ⊗ B)[(i,j), (k,l)] = A[i,k] · B[j,l].
  // Row-major flat index for the 4×4: row = 2i + j, col = 2k + l.
  const out = new Float64Array(16);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      for (let k = 0; k < 2; k++) {
        for (let l = 0; l < 2; l++) {
          out[(2 * i + j) * 4 + (2 * k + l)] = A[i * 2 + k]! * B[j * 2 + l]!;
        }
      }
    }
  }
  return out;
}

const XX = kron2x2(X2, X2);
// YY by direct construction: only the antidiagonal is non-zero.
//   YY = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
const YY: Real4 = (() => {
  const m = new Float64Array(16);
  m[0 * 4 + 3] = -1;
  m[1 * 4 + 2] = +1;
  m[2 * 4 + 1] = +1;
  m[3 * 4 + 0] = -1;
  return m;
})();
const ZZ = kron2x2(Z2, Z2);
const XI = kron2x2(X2, I2);
const IX = kron2x2(I2, X2);

function add(...ms: readonly Real4[]): Real4 {
  const out = new Float64Array(16);
  for (const m of ms) {
    for (let i = 0; i < 16; i++) out[i] = (out[i] ?? 0) + m[i]!;
  }
  return out;
}

function scale(m: Real4, s: number): Real4 {
  const out = new Float64Array(16);
  for (let i = 0; i < 16; i++) out[i] = m[i]! * s;
  return out;
}

/** Convert a real 4×4 to the ComplexMatrix layout the MPS API uses. */
export function asComplex4(m: Real4): ComplexMatrix {
  const out = zeros(4, 4);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      out.data[(r * 4 + c) * 2]     = m[r * 4 + c]!;   // re
      out.data[(r * 4 + c) * 2 + 1] = 0;               // im
    }
  }
  return out;
}

// ── Model factories ─────────────────────────────────────────

/**
 * Heisenberg chain: H = J Σ S_i · S_{i+1}, with S = σ/2 so
 *   S_iS_{i+1} = (1/4) (XX + YY + ZZ).
 */
export function heisenberg(N: number, J = 1): Hamiltonian1D {
  if (N < 2) throw new Error(`heisenberg: N must be ≥ 2`);
  const h = scale(add(XX, YY, ZZ), J * 0.25);
  const bonds: BondTerm[] = [];
  for (let q = 0; q < N - 1; q++) bonds.push({ h });
  return { nQubits: N, modelName: `Heisenberg(N=${N}, J=${J})`, bonds };
}

/**
 * Transverse-field Ising: H = −J Σ Z_iZ_{i+1} − h Σ X_i.
 *
 * The single-site X term is split between adjacent bonds
 * (each bond carries h/2 of each of its two sites). Edge
 * sites get their full X added to the unique bond touching
 * them. This way, summing over all bonds yields exactly the
 * full Hamiltonian.
 */
export function tfim(N: number, J = 1, hField = 1): Hamiltonian1D {
  if (N < 2) throw new Error(`tfim: N must be ≥ 2`);
  const bonds: BondTerm[] = [];
  for (let q = 0; q < N - 1; q++) {
    let term = scale(ZZ, -J);
    // Distribute single-site X across this bond.
    const leftFactor = q === 0 ? 1 : 0.5;
    const rightFactor = q === N - 2 ? 1 : 0.5;
    term = add(term, scale(XI, -hField * leftFactor), scale(IX, -hField * rightFactor));
    bonds.push({ h: term });
  }
  return { nQubits: N, modelName: `TFIM(N=${N}, J=${J}, h=${hField})`, bonds };
}

/**
 * XXZ chain: H = J Σ (X_iX_{i+1} + Y_iY_{i+1} + Δ Z_iZ_{i+1}).
 * Δ = 1 reduces to Heisenberg (up to overall scale).
 */
export function xxz(N: number, J = 1, delta = 1): Hamiltonian1D {
  if (N < 2) throw new Error(`xxz: N must be ≥ 2`);
  const h = scale(add(XX, YY, scale(ZZ, delta)), J);
  const bonds: BondTerm[] = [];
  for (let q = 0; q < N - 1; q++) bonds.push({ h });
  return { nQubits: N, modelName: `XXZ(N=${N}, J=${J}, Δ=${delta})`, bonds };
}

// ── Exact-diagonalization helper (small N only) ────────────

/**
 * Build the full 2^N × 2^N dense Hermitian matrix for a given
 * Hamiltonian1D, summing per-bond two-site operators. Useful
 * as a ground-truth oracle for tests at N ≤ 12 or so.
 *
 * Convention: |s_0 s_1 … s_{N-1}⟩ at index Σ_q s_q · 2^q.
 */
export function buildDense(H: Hamiltonian1D): Float64Array {
  const N = H.nQubits;
  const dim = 1 << N;
  const M = new Float64Array(dim * dim);

  for (let q = 0; q < N - 1; q++) {
    const bond = H.bonds[q]!.h;
    // Bond q acts on (q, q+1). For each basis state |b⟩, decode
    // the 2-bit pair (s_q, s_{q+1}) → idx = 2 s_q + s_{q+1}.
    const bitQ = 1 << q;
    const bitQ1 = 1 << (q + 1);
    for (let b = 0; b < dim; b++) {
      const sq = (b >> q) & 1;
      const sq1 = (b >> (q + 1)) & 1;
      const inIdx = 2 * sq + sq1;
      // For each output (sq', sq1'), add bond[outIdx, inIdx] to M[b', b].
      for (let outIdx = 0; outIdx < 4; outIdx++) {
        const v = bond[outIdx * 4 + inIdx]!;
        if (v === 0) continue;
        const outSq = (outIdx >> 1) & 1;
        const outSq1 = outIdx & 1;
        const bp = (b & ~bitQ & ~bitQ1) | (outSq ? bitQ : 0) | (outSq1 ? bitQ1 : 0);
        M[bp * dim + b] = (M[bp * dim + b] ?? 0) + v;
      }
    }
  }
  return M;
}
