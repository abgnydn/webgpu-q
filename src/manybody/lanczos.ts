// ─────────────────────────────────────────────────────────────
// lanczos.ts — matrix-free Lanczos ground-state solver for
// real-symmetric (or complex-Hermitian represented as real-pair)
// linear operators on flat Float64Array vectors.
//
// Used as the local eigensolver inside two-site DMRG: H_eff is
// never materialized; we hand Lanczos a function that maps a
// 2-site state to H_eff applied to it, and Lanczos returns the
// smallest eigenvalue / eigenvector of H_eff.
//
// Algorithm: standard 3-term recurrence on a real symmetric
// tridiagonal projection of H_eff. Each iteration:
//   1. w = A·v_k − β_k · v_{k-1}
//   2. α_k = ⟨v_k, w⟩
//   3. w ← w − α_k · v_k
//   4. (full reorthogonalization vs all earlier v_j — robust against
//      losing Lanczos orthogonality at the cost of O(k) extra dots)
//   5. β_{k+1} = ‖w‖, v_{k+1} = w / β_{k+1}
// then diagonalize the (k+1) × (k+1) tridiagonal {α, β} for the
// smallest Ritz pair (θ, s), reconstruct the full vector as
// V · s, and stop when |θ_new − θ_old| < tol or β_{k+1} < tol.
//
// Convergence at chemical accuracy on textbook 1D models is
// typically ≤ 30 iterations for N ≤ 20.
// ─────────────────────────────────────────────────────────────

/** Default Lanczos ground-state solver convergence tolerance. */
export const LANCZOS_TOL = 1e-10;

export interface LanczosResult {
  /** Smallest eigenvalue of A. */
  energy: number;
  /** Corresponding (unit-norm) eigenvector, real-Float64. */
  vector: Float64Array;
  /** Number of Lanczos iterations actually performed. */
  iter: number;
  /** Last energy delta (|θ_k − θ_{k-1}|). */
  residual: number;
}

export interface LanczosOpts {
  maxIter?: number;
  tol?: number;
  /**
   * If set, abort if ‖A x‖ ≈ 0 (zero matvec) and return the supplied
   * `x0` energy as a noop. Defaults to false.
   */
  allowZeroOp?: boolean;
}

/** v ← v / ‖v‖. Returns the original norm. */
function normalize(v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i]! * v[i]!;
  const n = Math.sqrt(s);
  if (n > 0) {
    const inv = 1 / n;
    for (let i = 0; i < v.length; i++) v[i] = v[i]! * inv;
  }
  return n;
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

// In-place: y ← y − α · x
function axpyMinus(y: Float64Array, x: Float64Array, alpha: number): void {
  for (let i = 0; i < y.length; i++) y[i] = y[i]! - alpha * x[i]!;
}

/** Diagonalize a real-symmetric tridiagonal {α, β} of dim k. Returns {values asc, vectors row-major}. */
function eigTridiag(alpha: Float64Array, beta: Float64Array, k: number): { values: Float64Array; vectors: Float64Array } {
  // Build dense and run a small Jacobi diagonalization. k is small (≤ 60).
  const M = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    M[i * k + i] = alpha[i]!;
    if (i + 1 < k) {
      M[i * k + (i + 1)] = beta[i + 1]!;
      M[(i + 1) * k + i] = beta[i + 1]!;
    }
  }
  // V ← I.
  const V = new Float64Array(k * k);
  for (let i = 0; i < k; i++) V[i * k + i] = 1;

  // Classic Jacobi rotation sweeps on a tiny matrix.
  const maxSweeps = 100;
  const tol = 1e-15;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) off += Math.abs(M[i * k + j]!);
    }
    if (off < tol) break;
    for (let p = 0; p < k - 1; p++) {
      for (let q = p + 1; q < k; q++) {
        const apq = M[p * k + q]!;
        if (Math.abs(apq) < tol) continue;
        const app = M[p * k + p]!, aqq = M[q * k + q]!;
        const theta = (aqq - app) / (2 * apq);
        const t = (theta >= 0)
          ? 1 / (theta + Math.sqrt(1 + theta * theta))
          : 1 / (theta - Math.sqrt(1 + theta * theta));
        const c = 1 / Math.sqrt(1 + t * t);
        const sn = t * c;
        // Rotate rows/cols p, q.
        for (let r = 0; r < k; r++) {
          const Mrp = M[r * k + p]!, Mrq = M[r * k + q]!;
          M[r * k + p] = c * Mrp - sn * Mrq;
          M[r * k + q] = sn * Mrp + c * Mrq;
        }
        for (let c2 = 0; c2 < k; c2++) {
          const Mpc = M[p * k + c2]!, Mqc = M[q * k + c2]!;
          M[p * k + c2] = c * Mpc - sn * Mqc;
          M[q * k + c2] = sn * Mpc + c * Mqc;
        }
        // Update V.
        for (let r = 0; r < k; r++) {
          const Vrp = V[r * k + p]!, Vrq = V[r * k + q]!;
          V[r * k + p] = c * Vrp - sn * Vrq;
          V[r * k + q] = sn * Vrp + c * Vrq;
        }
      }
    }
  }
  const values = new Float64Array(k);
  for (let i = 0; i < k; i++) values[i] = M[i * k + i]!;
  // Sort ascending; permute eigenvectors.
  const idx = Array.from({ length: k }, (_, i) => i).sort((a, b) => values[a]! - values[b]!);
  const sortedV = new Float64Array(values);
  const sortedVec = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    sortedV[i] = values[idx[i]!]!;
    for (let r = 0; r < k; r++) {
      sortedVec[r * k + i] = V[r * k + idx[i]!]!;
    }
  }
  return { values: sortedV, vectors: sortedVec };
}

/**
 * Find the smallest eigenpair of a real-symmetric linear operator
 * `apply: (x) -> A x`. The vector dimension is inferred from `x0`.
 * Returns {energy, vector, iter, residual}.
 */
export function lanczosGroundState(
  apply: (x: Float64Array, out: Float64Array) => void,
  x0: Float64Array,
  opts: LanczosOpts = {},
): LanczosResult {
  const dim = x0.length;
  const maxIter = Math.min(opts.maxIter ?? 60, dim);
  const tol = opts.tol ?? LANCZOS_TOL;

  // Krylov basis V[0..k-1], each row a vector of length dim.
  // Stored as a flat (maxIter × dim) Float64Array, row-major.
  const V = new Float64Array(maxIter * dim);
  // tridiagonal coefficients
  const alpha = new Float64Array(maxIter);
  const beta = new Float64Array(maxIter + 1);

  // v_0 = x0 / ‖x0‖ — if x0 is zero, seed with all-ones.
  let v = new Float64Array(x0);
  const initN = normalize(v);
  if (initN === 0) {
    for (let i = 0; i < dim; i++) v[i] = 1;
    normalize(v);
  }
  for (let i = 0; i < dim; i++) V[i] = v[i]!;

  const w = new Float64Array(dim);
  let prevE = Infinity;
  let lastE = 0;
  let lastResidual = Infinity;
  let actualIter = 0;
  // Reusable workspace for the eigenvector reconstruction.
  let lastVecKrylov = new Float64Array(0);

  for (let k = 0; k < maxIter; k++) {
    // w = A v_k
    apply(v, w);
    // Subtract β_k · v_{k-1}
    if (k > 0) {
      const prev = V.subarray((k - 1) * dim, k * dim);
      axpyMinus(w, prev, beta[k]!);
    }
    // α_k = ⟨v_k, w⟩
    const a = dot(v, w);
    alpha[k] = a;
    axpyMinus(w, v, a);
    // Full reorthogonalization vs all earlier v_j. Cheap (k * dim) per iter.
    for (let j = 0; j <= k; j++) {
      const vj = V.subarray(j * dim, (j + 1) * dim);
      const c = dot(w, vj);
      axpyMinus(w, vj, c);
    }
    // β_{k+1} = ‖w‖
    const bn = normalize(w);
    beta[k + 1] = bn;

    // Project to tridiagonal & extract smallest Ritz value.
    const m = k + 1;
    const { values, vectors } = eigTridiag(alpha, beta, m);
    const e = values[0]!;
    actualIter = m;
    lastE = e;
    lastResidual = Math.abs(e - prevE);
    // Keep the Krylov-basis combination of the smallest Ritz vector.
    lastVecKrylov = new Float64Array(m);
    for (let i = 0; i < m; i++) lastVecKrylov[i] = vectors[i * m + 0]!;

    if (lastResidual < tol) {
      // Converged.
      break;
    }
    if (bn < 1e-14) {
      // Krylov subspace closed up — H_eff invariant subspace.
      break;
    }

    prevE = e;
    // v_{k+1} = w (already normalized)
    if (k + 1 < maxIter) {
      const next = V.subarray((k + 1) * dim, (k + 2) * dim);
      for (let i = 0; i < dim; i++) next[i] = w[i]!;
      v = new Float64Array(next);
    }
  }

  // Reconstruct ψ = Σ_j s_j · v_j
  const psi = new Float64Array(dim);
  for (let j = 0; j < lastVecKrylov.length; j++) {
    const vj = V.subarray(j * dim, (j + 1) * dim);
    const c = lastVecKrylov[j]!;
    for (let i = 0; i < dim; i++) psi[i] = psi[i]! + c * vj[i]!;
  }
  // Sanitize norm — small drift from finite-precision Krylov.
  normalize(psi);

  if (!opts.allowZeroOp && !Number.isFinite(lastE)) {
    throw new Error("lanczosGroundState: did not converge to a finite eigenvalue");
  }

  return { energy: lastE, vector: psi, iter: actualIter, residual: lastResidual };
}
