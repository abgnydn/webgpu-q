// Unit tests for the matrix-free Lanczos ground-state solver.
// We feed it dense real-symmetric matrices via a closure and
// check the returned energy/vector match a Jacobi reference.
import { describe, expect, test } from "vitest";
import { lanczosGroundState } from "../../src/manybody/lanczos.js";
import { eigsymmetric } from "../../src/manybody/dense-eig.js";

function densMatVec(M: Float64Array, dim: number) {
  return (x: Float64Array, out: Float64Array): void => {
    for (let i = 0; i < dim; i++) {
      let s = 0;
      for (let j = 0; j < dim; j++) s += M[i * dim + j]! * x[j]!;
      out[i] = s;
    }
  };
}

function randomSymmetric(dim: number, seed: number): Float64Array {
  let s = seed >>> 0;
  const rng = (): number => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
  const M = new Float64Array(dim * dim);
  for (let i = 0; i < dim; i++) {
    for (let j = i; j < dim; j++) {
      const v = rng();
      M[i * dim + j] = v;
      M[j * dim + i] = v;
    }
  }
  return M;
}

describe("Lanczos ground state vs Jacobi reference", () => {
  test("dim=8 random symmetric: smallest eigenvalue agrees to 1e-10", () => {
    const dim = 8;
    const M = randomSymmetric(dim, 0xCAFE0008);
    const { values: ref } = eigsymmetric(M, dim);
    const x0 = new Float64Array(dim);
    for (let i = 0; i < dim; i++) x0[i] = 1;
    const res = lanczosGroundState(densMatVec(M, dim), x0, { tol: 1e-12, maxIter: 30 });
    expect(Math.abs(res.energy - ref[0]!)).toBeLessThan(1e-10);
  });

  test("dim=32 random symmetric: smallest eigenvalue agrees to 1e-9", () => {
    const dim = 32;
    const M = randomSymmetric(dim, 0xCAFE0020);
    const { values: ref } = eigsymmetric(M, dim);
    const x0 = new Float64Array(dim);
    for (let i = 0; i < dim; i++) x0[i] = 1;
    const res = lanczosGroundState(densMatVec(M, dim), x0, { tol: 1e-12, maxIter: 50 });
    expect(Math.abs(res.energy - ref[0]!)).toBeLessThan(1e-9);
  });

  test("dim=4 explicit eigenvalue (diagonal matrix)", () => {
    // Diagonal matrix with eigenvalues -3, -1, 1, 5; smallest = -3.
    const dim = 4;
    const M = new Float64Array(dim * dim);
    M[0] = -3; M[5] = -1; M[10] = 1; M[15] = 5;
    const x0 = new Float64Array([1, 1, 1, 1]);
    const res = lanczosGroundState(densMatVec(M, dim), x0, { tol: 1e-12 });
    expect(res.energy).toBeCloseTo(-3, 9);
    // Eigenvector should be ±e_0 (concentrated on index 0).
    expect(Math.abs(Math.abs(res.vector[0]!) - 1)).toBeLessThan(1e-9);
  });

  test("returned vector is unit-norm", () => {
    const dim = 16;
    const M = randomSymmetric(dim, 0xBEEF);
    const x0 = new Float64Array(dim);
    for (let i = 0; i < dim; i++) x0[i] = i + 1;
    const res = lanczosGroundState(densMatVec(M, dim), x0, { tol: 1e-12 });
    let n2 = 0;
    for (let i = 0; i < dim; i++) n2 += res.vector[i]! * res.vector[i]!;
    expect(n2).toBeCloseTo(1, 9);
  });
});
