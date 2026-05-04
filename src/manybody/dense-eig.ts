// ─────────────────────────────────────────────────────────────
// dense-eig.ts — full real-symmetric eigendecomposition via
// classical Jacobi sweeps. Returns ALL eigenpairs sorted
// ascending, matching standard Lapack `eigh` output.
//
// For our use cases the matrix is small (4×4 for Trotter steps,
// up to 4096×4096 for exact-diagonalization tests at N ≤ 12).
// Jacobi is the right tool: simple, stable, no LAPACK dep.
// ─────────────────────────────────────────────────────────────

export interface EigDecomp {
  /** Eigenvalues, ascending. Length N. */
  values: Float64Array;
  /** Column-major eigenvectors. vectors[col*N + row]. Each column = an eigenvector. */
  vectors: Float64Array;
}

/** Full eigendecomposition of a real-symmetric N×N matrix M (row-major). */
export function eigsymmetric(M: Float64Array, N: number): EigDecomp {
  const A = new Float64Array(M);
  const V = new Float64Array(N * N);
  for (let i = 0; i < N; i++) V[i * N + i] = 1; // identity

  const TOL = 1e-14;
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

        for (let k = 0; k < N; k++) {
          const akp = A[k * N + p]!;
          const akq = A[k * N + q]!;
          A[k * N + p] = c * akp - s * akq;
          A[k * N + q] = s * akp + c * akq;
        }
        for (let k = 0; k < N; k++) {
          const apk = A[p * N + k]!;
          const aqk = A[q * N + k]!;
          A[p * N + k] = c * apk - s * aqk;
          A[q * N + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < N; k++) {
          const vkp = V[k * N + p]!;
          const vkq = V[k * N + q]!;
          V[k * N + p] = c * vkp - s * vkq;
          V[k * N + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  // Extract diagonal as eigenvalues, sort ascending, permute columns of V.
  const idx = Array.from({ length: N }, (_, i) => i);
  idx.sort((a, b) => A[a * N + a]! - A[b * N + b]!);
  const values = new Float64Array(N);
  const vectors = new Float64Array(N * N);
  for (let col = 0; col < N; col++) {
    const src = idx[col]!;
    values[col] = A[src * N + src]!;
    for (let row = 0; row < N; row++) {
      vectors[col * N + row] = V[row * N + src]!;
    }
  }
  return { values, vectors };
}
