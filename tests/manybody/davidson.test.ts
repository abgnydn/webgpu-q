import { describe, expect, it } from "vitest";
import { davidson } from "../../src/manybody/davidson.js";
import { eigGeneralWithVectors } from "../../src/manybody/dense-eig-general.js";

/**
 * Cross-checks block-Davidson against the existing dense
 * Hessenberg-QR eigensolver. The pair shares an interface: M·v
 * matvec + M diagonal. For matrices with all-real spectrum
 * (the EOM-CCSD regime), the k lowest eigenvalues should agree
 * to ≤ 1e-6 Ha, and the corresponding right eigenvectors should
 * satisfy ‖M·y − λ·y‖ ≤ 1e-6.
 */

function buildMatvec(M: Float64Array, N: number): (v: Float64Array) => Float64Array {
  return (v) => {
    const out = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += M[i * N + j]! * v[j]!;
      out[i] = s;
    }
    return out;
  };
}

function diagOf(M: Float64Array, N: number): Float64Array {
  const d = new Float64Array(N);
  for (let i = 0; i < N; i++) d[i] = M[i * N + i]!;
  return d;
}

function smallestKReal(real: Float64Array, imag: Float64Array, k: number): number[] {
  const idx: Array<{ v: number; i: number }> = [];
  for (let i = 0; i < real.length; i++) {
    if (Math.abs(imag[i]!) < 1e-6) idx.push({ v: real[i]!, i });
  }
  idx.sort((a, b) => a.v - b.v);
  return idx.slice(0, k).map((x) => x.v);
}

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

describe("davidson", () => {
  it("matches dense eigGeneral on a 50x50 symmetric matrix (k=3)", () => {
    // Use a moderate dim where Davidson genuinely helps (subspace
    // much smaller than dim). For dim ≤ ~2k Davidson degenerates
    // to dense and the test isn't meaningful.
    const N = 50;
    const rng = mulberry32(0xD0FF);
    // Diagonally dominant symmetric M with well-spaced spectrum.
    const M = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      M[i * N + i] = i * 2.0 + 1.0; // dominant diagonal: 1, 3, 5, ...
      for (let j = i + 1; j < N; j++) {
        const off = (rng() - 0.5) * 0.4;
        M[i * N + j] = off;
        M[j * N + i] = off; // symmetric
      }
    }

    const dense = eigGeneralWithVectors(M, N);
    const k = 3;
    const dav = davidson(N, buildMatvec(M, N), diagOf(M, N), { k, tol: 1e-8, maxIter: 200 });

    expect(dav.converged).toBe(true);
    expect(dav.iterations).toBeLessThan(150);

    const denseLow = smallestKReal(dense.real, dense.imag, k);
    for (let p = 0; p < k; p++) {
      expect(Math.abs(dav.energies[p]! - denseLow[p]!)).toBeLessThan(1e-6);
    }
  });

  it("matches dense on a non-symmetric matrix with all-real spectrum (k=2)", () => {
    // M with real spectrum but non-symmetric: take a diagonal D with
    // distinct entries and conjugate by a non-orthogonal similarity
    // matrix S. M = S · D · S⁻¹ has spectrum equal to D.
    const N = 8;
    const rng = mulberry32(0xC0DE);
    const D = new Float64Array(N * N);
    const eigvals: number[] = [];
    for (let i = 0; i < N; i++) {
      const v = i + 1.0;
      D[i * N + i] = v;
      eigvals.push(v);
    }
    // Random near-identity S.
    const S = new Float64Array(N * N);
    const Sinv = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === j) S[i * N + j] = 1.0;
        else S[i * N + j] = (rng() - 0.5) * 0.2;
      }
    }
    // Compute S⁻¹ via Gauss-Jordan (small N).
    {
      const aug = new Float64Array(N * 2 * N);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) aug[i * 2 * N + j] = S[i * N + j]!;
        aug[i * 2 * N + N + i] = 1;
      }
      for (let i = 0; i < N; i++) {
        // Pivot.
        const pivot = aug[i * 2 * N + i]!;
        if (Math.abs(pivot) < 1e-15) throw new Error("singular S in test");
        for (let j = 0; j < 2 * N; j++) aug[i * 2 * N + j]! /= pivot;
        for (let r = 0; r < N; r++) {
          if (r === i) continue;
          const f = aug[r * 2 * N + i]!;
          for (let j = 0; j < 2 * N; j++) aug[r * 2 * N + j]! -= f * aug[i * 2 * N + j]!;
        }
      }
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) Sinv[i * N + j] = aug[i * 2 * N + N + j]!;
      }
    }
    // M = S · D · S⁻¹.
    const tmp = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let p = 0; p < N; p++) s += S[i * N + p]! * D[p * N + j]!;
        tmp[i * N + j] = s;
      }
    }
    const M = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let p = 0; p < N; p++) s += tmp[i * N + p]! * Sinv[p * N + j]!;
        M[i * N + j] = s;
      }
    }

    const k = 2;
    const dav = davidson(N, buildMatvec(M, N), diagOf(M, N), { k, tol: 1e-9, maxIter: 100 });
    expect(dav.converged).toBe(true);
    // Two smallest exact eigenvalues are 1 and 2.
    expect(Math.abs(dav.energies[0]! - 1.0)).toBeLessThan(1e-6);
    expect(Math.abs(dav.energies[1]! - 2.0)).toBeLessThan(1e-6);
  });

  it("returns right eigenvectors satisfying M·y ≈ λ·y", () => {
    const N = 12;
    const rng = mulberry32(0xBADF00D);
    const M = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      M[i * N + i] = i + 0.3;
      for (let j = 0; j < N; j++) if (i !== j) M[i * N + j] = (rng() - 0.5) * 0.1;
    }

    const k = 3;
    const dav = davidson(N, buildMatvec(M, N), diagOf(M, N), { k, tol: 1e-9, maxIter: 100 });
    expect(dav.converged).toBe(true);

    for (let p = 0; p < k; p++) {
      const y = new Float64Array(N);
      for (let i = 0; i < N; i++) y[i] = dav.vectors[p * N + i]!;
      // M · y
      const My = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let j = 0; j < N; j++) s += M[i * N + j]! * y[j]!;
        My[i] = s;
      }
      // ‖M·y − λ·y‖
      let r2 = 0;
      for (let i = 0; i < N; i++) {
        const d = My[i]! - dav.energies[p]! * y[i]!;
        r2 += d * d;
      }
      expect(Math.sqrt(r2)).toBeLessThan(1e-6);
    }
  });

  it("filter accepts only real-positive eigenvalues by default", () => {
    // Build a small problem where some eigenvalues are negative;
    // davidson with the default filter should still return k positives.
    const N = 6;
    const M = new Float64Array(N * N);
    // Diagonal: -2, -1, 1, 2, 3, 4 — three negative, three positive.
    M[0] = -2; M[7] = -1; M[14] = 1; M[21] = 2; M[28] = 3; M[35] = 4;
    // Tiny non-symmetric off-diagonal so it's interesting.
    M[1] = 0.02; M[6] = 0.01;

    const dav = davidson(N, buildMatvec(M, N), diagOf(M, N), {
      k: 3,
      tol: 1e-10,
      maxIter: 100,
    });
    expect(dav.converged).toBe(true);
    // All returned energies should be > 0 and ascending.
    for (let p = 0; p < 3; p++) {
      expect(dav.energies[p]!).toBeGreaterThan(0);
      if (p > 0) expect(dav.energies[p]!).toBeGreaterThanOrEqual(dav.energies[p - 1]!);
    }
    // Three smallest positives are 1, 2, 3.
    expect(Math.abs(dav.energies[0]! - 1)).toBeLessThan(1e-6);
    expect(Math.abs(dav.energies[1]! - 2)).toBeLessThan(1e-6);
    expect(Math.abs(dav.energies[2]! - 3)).toBeLessThan(1e-6);
  });

  it("matvec count is far below dim·k for a well-conditioned problem", () => {
    // Sanity that Davidson is doing many fewer matvecs than the
    // dense-probe path (which would be exactly N matvecs).
    const N = 50;
    const rng = mulberry32(0xFE5D);
    const M = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      M[i * N + i] = i + 1.0;
      for (let j = 0; j < N; j++) {
        if (i !== j) M[i * N + j] = (rng() - 0.5) * 0.05;
      }
    }

    const k = 4;
    const dav = davidson(N, buildMatvec(M, N), diagOf(M, N), { k, tol: 1e-8, maxIter: 100 });
    expect(dav.converged).toBe(true);
    // Dense-probe path would be N = 50 matvecs just to build the matrix.
    // Davidson should converge well below that.
    expect(dav.matvecs).toBeLessThan(N);
  });
});
