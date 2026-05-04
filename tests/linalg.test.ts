import { describe, expect, test } from "vitest";
import {
  svd,
  svdReconstruct,
  frobeniusDistance,
  identity,
  zeros,
  cset,
  cget,
  clone,
  matmul,
  type ComplexMatrix,
} from "../src/linalg";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomMatrix(m: number, n: number, seed: number): ComplexMatrix {
  const rng = mulberry32(seed);
  const M = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      cset(M, i, j, rng() * 2 - 1, rng() * 2 - 1);
    }
  }
  return M;
}

function hermitianTranspose(M: ComplexMatrix): ComplexMatrix {
  const out = zeros(M.cols, M.rows);
  for (let r = 0; r < M.rows; r++) {
    for (let c = 0; c < M.cols; c++) {
      const [re, im] = cget(M, r, c);
      cset(out, c, r, re, -im);
    }
  }
  return out;
}

describe("SVD — identity and sparse cases", () => {
  test("SVD of identity returns σ = [1, 1, ..., 1]", () => {
    const n = 5;
    const I = identity(n);
    const { U, sigma, Vh } = svd(I);
    expect(U.rows).toBe(n);
    expect(U.cols).toBe(n);
    expect(Vh.rows).toBe(n);
    expect(Vh.cols).toBe(n);
    for (let i = 0; i < n; i++) expect(sigma[i]).toBeCloseTo(1, 12);
    // Reconstruction must equal I
    const reconstructed = svdReconstruct({ U, sigma, Vh });
    expect(frobeniusDistance(reconstructed, I)).toBeLessThan(1e-10);
  });

  test("SVD of diagonal matrix returns σ = descending diagonal entries", () => {
    const D = zeros(4, 4);
    cset(D, 0, 0, 3, 0);
    cset(D, 1, 1, 1, 0);
    cset(D, 2, 2, 5, 0);
    cset(D, 3, 3, 2, 0);
    const { sigma } = svd(D);
    expect(Array.from(sigma)).toEqual([5, 3, 2, 1]);
  });

  test("SVD of rank-1 matrix has one nonzero σ", () => {
    // M = u · v^H where u = [1, 2], v = [1, 1, 1] → M is 2x3 rank 1.
    const M = zeros(2, 3);
    cset(M, 0, 0, 1, 0); cset(M, 0, 1, 1, 0); cset(M, 0, 2, 1, 0);
    cset(M, 1, 0, 2, 0); cset(M, 1, 1, 2, 0); cset(M, 1, 2, 2, 0);
    const { sigma } = svd(M);
    expect(sigma.length).toBe(2); // min(m, n) = 2
    expect(sigma[0]).toBeGreaterThan(0);
    expect(sigma[1]).toBeLessThan(1e-10);
  });
});

describe("SVD — round-trip U Σ V^H ≈ M", () => {
  for (const [m, n, seed] of [
    [4, 4, 1],
    [5, 3, 2],
    [3, 5, 3],
    [8, 8, 4],
    [10, 6, 5],
    [6, 10, 6],
    [16, 16, 7],
    [32, 32, 8],
  ] as const) {
    test(`random ${m}×${n} matrix reconstructs`, () => {
      const M = randomMatrix(m, n, seed);
      const result = svd(M);
      const reconstructed = svdReconstruct(result);
      const normM = Math.sqrt(frobeniusDistance(M, zeros(m, n)) ** 2);
      const err = frobeniusDistance(reconstructed, M);
      // Jacobi SVD should hit ~machine precision
      expect(err / Math.max(normM, 1)).toBeLessThan(1e-10);
    });
  }

  test("singular values are non-negative and descending", () => {
    const M = randomMatrix(16, 16, 99);
    const { sigma } = svd(M);
    for (let i = 0; i < sigma.length; i++) {
      expect(sigma[i]).toBeGreaterThanOrEqual(0);
      if (i > 0) expect(sigma[i]!).toBeLessThanOrEqual(sigma[i - 1]! + 1e-12);
    }
  });

  test("U has orthonormal columns (U^H U = I_k)", () => {
    const M = randomMatrix(10, 6, 123);
    const { U } = svd(M);
    const UhU = matmul(hermitianTranspose(U), U);
    const I = identity(U.cols);
    expect(frobeniusDistance(UhU, I)).toBeLessThan(1e-10);
  });

  test("V^H has orthonormal rows (Vh · Vh^H = I_k)", () => {
    const M = randomMatrix(10, 6, 456);
    const { Vh } = svd(M);
    const VVh = matmul(Vh, hermitianTranspose(Vh));
    const I = identity(Vh.rows);
    expect(frobeniusDistance(VVh, I)).toBeLessThan(1e-10);
  });
});

describe("SVD — handles wide (m < n) and complex entries", () => {
  test("wide 4×8 complex matrix round-trips", () => {
    const M = randomMatrix(4, 8, 77);
    const r = svd(M);
    expect(r.U.rows).toBe(4);
    expect(r.U.cols).toBe(4);   // k = min(m, n) = 4
    expect(r.Vh.rows).toBe(4);
    expect(r.Vh.cols).toBe(8);
    const err = frobeniusDistance(svdReconstruct(r), M);
    expect(err).toBeLessThan(1e-10);
  });

  test("pure-imaginary matrix round-trips", () => {
    const M = zeros(3, 3);
    cset(M, 0, 1, 0, 1);
    cset(M, 1, 2, 0, 2);
    cset(M, 2, 0, 0, 3);
    const result = svd(M);
    expect(Array.from(result.sigma).map((s) => +s.toFixed(10))).toEqual([3, 2, 1]);
    const reconstructed = svdReconstruct(result);
    expect(frobeniusDistance(reconstructed, M)).toBeLessThan(1e-10);
  });
});

describe("matmul", () => {
  test("I · A = A", () => {
    const A = randomMatrix(5, 3, 42);
    const I = identity(5);
    const C = matmul(I, A);
    expect(frobeniusDistance(C, A)).toBeLessThan(1e-12);
  });

  test("shape guard", () => {
    const A = zeros(2, 3);
    const B = zeros(4, 2);
    expect(() => matmul(A, B)).toThrow();
  });

  test("complex multiplication: (1+i)(1-i) = 2", () => {
    const A = zeros(1, 1);
    const B = zeros(1, 1);
    cset(A, 0, 0, 1, 1);   // 1 + i
    cset(B, 0, 0, 1, -1);  // 1 - i
    const C = matmul(A, B);
    expect(C.data[0]).toBeCloseTo(2, 12);
    expect(C.data[1]).toBeCloseTo(0, 12);
  });
});

describe("clone and getters", () => {
  test("clone yields independent copy", () => {
    const A = randomMatrix(3, 3, 9);
    const B = clone(A);
    cset(B, 0, 0, 999, 999);
    expect(cget(A, 0, 0)[0]).not.toBe(999);
  });
});
