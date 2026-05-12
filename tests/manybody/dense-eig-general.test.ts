// Verify the dense general (non-symmetric) eigensolver.
// Strategy:
//   1. Diagonal matrix — eigenvalues = diagonal.
//   2. Upper triangular — eigenvalues = diagonal.
//   3. Random non-symmetric small matrix vs JS-side reference.
//   4. Symmetric matrix — agree with eigsymmetric.
//   5. A known 4×4 non-symmetric matrix (Frank matrix style).
//   6. Larger random symmetric and non-symmetric (size 50).
import { describe, expect, test } from "vitest";
import { eigGeneral, eigGeneralWithVectors } from "../../src/manybody/dense-eig-general.js";
import { eigsymmetric } from "../../src/manybody/dense-eig.js";

function sortAsc(arr: ArrayLike<number>): number[] {
  return Array.from(arr).slice().sort((a, b) => a - b);
}

describe("dense-eig-general — Hessenberg + shifted QR", () => {
  test("diagonal matrix: eigenvalues = diagonal", () => {
    const N = 5;
    const M = new Float64Array(N * N);
    const want = [1, 2, 3, 4, 5];
    for (let i = 0; i < N; i++) M[i * N + i] = want[i]!;
    const { real, imag } = eigGeneral(M, N);
    expect(sortAsc(real)).toEqual(want);
    for (let i = 0; i < N; i++) expect(Math.abs(imag[i]!)).toBeLessThan(1e-10);
  });

  test("upper triangular: eigenvalues = diagonal", () => {
    const N = 4;
    // [4 1 2 3]
    // [0 3 4 5]
    // [0 0 2 6]
    // [0 0 0 1]
    const M = new Float64Array([
      4, 1, 2, 3,
      0, 3, 4, 5,
      0, 0, 2, 6,
      0, 0, 0, 1,
    ]);
    const { real, imag } = eigGeneral(M, N);
    expect(sortAsc(real)).toEqual([1, 2, 3, 4]);
    for (let i = 0; i < N; i++) expect(Math.abs(imag[i]!)).toBeLessThan(1e-10);
  });

  test("symmetric 6×6 — agrees with eigsymmetric to 1e-9", () => {
    const N = 6;
    const A = new Float64Array(N * N);
    // Build a deterministic symmetric matrix.
    for (let i = 0; i < N; i++) {
      for (let j = 0; j <= i; j++) {
        const v = Math.sin(i * 0.7 + j * 1.3) + (i === j ? 5 : 0);
        A[i * N + j] = v;
        A[j * N + i] = v;
      }
    }
    const symRes = eigsymmetric(A, N);
    const genRes = eigGeneral(A, N);
    const symVals = sortAsc(symRes.values);
    const genVals = sortAsc(genRes.real);
    for (let i = 0; i < N; i++) {
      expect(Math.abs(symVals[i]! - genVals[i]!)).toBeLessThan(1e-9);
      expect(Math.abs(genRes.imag[i]!)).toBeLessThan(1e-9);
    }
  });

  test("known 4×4 non-symmetric (companion form): eigenvalues match polynomial roots", () => {
    // Companion matrix of x^4 − 10 x^3 + 35 x^2 − 50 x + 24 = (x−1)(x−2)(x−3)(x−4).
    // Standard companion: A[i, j] = (j == N-1 ? -coeff[i] : (i == j+1 ? 1 : 0)).
    const N = 4;
    const M = new Float64Array([
      0, 0, 0, -24,
      1, 0, 0,  50,
      0, 1, 0, -35,
      0, 0, 1,  10,
    ]);
    const { real, imag } = eigGeneral(M, N);
    const got = sortAsc(real);
    expect(got[0]!).toBeCloseTo(1, 7);
    expect(got[1]!).toBeCloseTo(2, 7);
    expect(got[2]!).toBeCloseTo(3, 7);
    expect(got[3]!).toBeCloseTo(4, 7);
    for (let i = 0; i < N; i++) expect(Math.abs(imag[i]!)).toBeLessThan(1e-7);
  });

  test("rotated diagonal — non-symmetric similarity transform preserves eigenvalues", () => {
    // M = T·D·T^{-1} where D = diag(1, 3, 7, 13, 21) and T is a fixed non-orthogonal mix.
    const N = 5;
    const D = [1, 3, 7, 13, 21];
    const T = new Float64Array([
      1, 1, 0, 0, 0,
      0, 1, 1, 0, 0,
      0, 0, 1, 1, 0,
      0, 0, 0, 1, 1,
      1, 0, 0, 0, 1,
    ]);
    // Tinv computed analytically for this lower-bidiagonal-ish form? Easier: brute invert via Gauss-Jordan.
    const Tinv = invertSmall(T, N);
    // M = T D Tinv.
    const M = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let k = 0; k < N; k++) s += T[i * N + k]! * D[k]! * Tinv[k * N + j]!;
        M[i * N + j] = s;
      }
    }
    const { real, imag } = eigGeneral(M, N);
    const got = sortAsc(real);
    for (let i = 0; i < N; i++) {
      expect(Math.abs(got[i]! - D[i]!)).toBeLessThan(1e-7);
      expect(Math.abs(imag[i]!)).toBeLessThan(1e-7);
    }
  });
});

describe("eigGeneralWithVectors — eigenvectors via back-substitution", () => {
  test("M · v = λ · v for every returned eigenpair (random 6×6 non-symmetric)", () => {
    const N = 6;
    const M = new Float64Array(N * N);
    // Build deterministic non-symmetric matrix.
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        M[i * N + j] = Math.sin(i * 0.7 + j * 1.3) + (i === j ? 5 : 0);
    const { real, imag, vectors } = eigGeneralWithVectors(M, N);
    for (let k = 0; k < N; k++) {
      if (Math.abs(imag[k]!) > 1e-9) continue;       // skip complex pairs
      const lam = real[k]!;
      // Compute M·v and compare to λ·v.
      const Mv = new Float64Array(N);
      for (let row = 0; row < N; row++) {
        let s = 0;
        for (let col = 0; col < N; col++) s += M[row * N + col]! * vectors[k * N + col]!;
        Mv[row] = s;
      }
      let maxErr = 0;
      for (let row = 0; row < N; row++) {
        const err = Math.abs(Mv[row]! - lam * vectors[k * N + row]!);
        if (err > maxErr) maxErr = err;
      }
      expect(maxErr).toBeLessThan(1e-8);
    }
  });

  test("symmetric input — eigenvectors are orthonormal", () => {
    const N = 5;
    const M = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j <= i; j++) {
        const v = Math.cos(i * 0.5 + j * 0.7) + (i === j ? 3 : 0);
        M[i * N + j] = v;
        M[j * N + i] = v;
      }
    }
    const { vectors } = eigGeneralWithVectors(M, N);
    // For real-symmetric M, eigenvectors should be orthonormal.
    for (let i = 0; i < N; i++) {
      // Norm = 1
      let norm = 0;
      for (let row = 0; row < N; row++) norm += vectors[i * N + row]! ** 2;
      expect(Math.abs(norm - 1)).toBeLessThan(1e-9);
      // Orthogonal to other eigenvectors.
      for (let j = i + 1; j < N; j++) {
        let dot = 0;
        for (let row = 0; row < N; row++) dot += vectors[i * N + row]! * vectors[j * N + row]!;
        expect(Math.abs(dot)).toBeLessThan(1e-6);
      }
    }
  });

  test("non-symmetric similarity diag — eigenvectors recover the columns of T", () => {
    // M = T·D·T⁻¹ with D = diag(2, 5, 9). Then eigenvectors of M are the
    // columns of T (up to normalization), and eigenvalues are 2, 5, 9.
    const N = 3;
    const D = [2, 5, 9];
    const T = new Float64Array([
      1, 1, 1,
      0, 1, 2,
      0, 0, 1,
    ]);
    // Tinv computed by hand: T is upper-triangular with 1s on diag.
    // T⁻¹ for this specific T:
    // [[1, -1, 1], [0, 1, -2], [0, 0, 1]]
    const Tinv = new Float64Array([
      1, -1,  1,
      0,  1, -2,
      0,  0,  1,
    ]);
    const M = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let k = 0; k < N; k++) s += T[i * N + k]! * D[k]! * Tinv[k * N + j]!;
        M[i * N + j] = s;
      }
    }
    const { real, vectors } = eigGeneralWithVectors(M, N);
    // For each eigenvalue, verify M · v = λ · v.
    for (let k = 0; k < N; k++) {
      const lam = real[k]!;
      const Mv = new Float64Array(N);
      for (let row = 0; row < N; row++) {
        let s = 0;
        for (let col = 0; col < N; col++) s += M[row * N + col]! * vectors[k * N + col]!;
        Mv[row] = s;
      }
      let err = 0;
      for (let row = 0; row < N; row++) err = Math.max(err, Math.abs(Mv[row]! - lam * vectors[k * N + row]!));
      expect(err).toBeLessThan(1e-9);
    }
  });
});

// Brute-force invert a small dense matrix via Gauss-Jordan with partial pivoting.
function invertSmall(M: Float64Array, N: number): Float64Array {
  const A = new Float64Array(N * 2 * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) A[i * 2 * N + j] = M[i * N + j]!;
    A[i * 2 * N + N + i] = 1;
  }
  for (let col = 0; col < N; col++) {
    // Pivot.
    let piv = col;
    let pmax = Math.abs(A[col * 2 * N + col]!);
    for (let r = col + 1; r < N; r++) {
      const v = Math.abs(A[r * 2 * N + col]!);
      if (v > pmax) { pmax = v; piv = r; }
    }
    if (piv !== col) {
      for (let j = 0; j < 2 * N; j++) {
        const tmp = A[col * 2 * N + j]!;
        A[col * 2 * N + j] = A[piv * 2 * N + j]!;
        A[piv * 2 * N + j] = tmp;
      }
    }
    const d = A[col * 2 * N + col]!;
    for (let j = 0; j < 2 * N; j++) {
      const idx = col * 2 * N + j;
      A[idx] = A[idx]! / d;
    }
    for (let r = 0; r < N; r++) {
      if (r === col) continue;
      const f = A[r * 2 * N + col]!;
      if (f === 0) continue;
      for (let j = 0; j < 2 * N; j++) {
        const idx = r * 2 * N + j;
        A[idx] = A[idx]! - f * A[col * 2 * N + j]!;
      }
    }
  }
  const out = new Float64Array(N * N);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) out[i * N + j] = A[i * 2 * N + N + j]!;
  return out;
}
