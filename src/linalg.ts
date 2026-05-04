// ─────────────────────────────────────────────────────────────
// linalg.ts — Complex dense linear algebra for MPS (Level 2).
//
// Zero external deps. For the sizes MPS needs (≤ 2χ × 2χ, with
// χ ≤ 128 in practice), one-sided Jacobi SVD is fast and numerically
// stable — error is O(ε · cond(M)), no LAPACK required.
//
// Storage convention: ComplexMatrix.data is a single Float64Array
// of length 2 · rows · cols, row-major, with [re, im] per element.
// ─────────────────────────────────────────────────────────────

export interface ComplexMatrix {
  readonly rows: number;
  readonly cols: number;
  /** Row-major interleaved [re, im, re, im, ...], length = 2·rows·cols. */
  data: Float64Array;
}

export function zeros(rows: number, cols: number): ComplexMatrix {
  return { rows, cols, data: new Float64Array(2 * rows * cols) };
}

export function identity(n: number): ComplexMatrix {
  const m = zeros(n, n);
  for (let i = 0; i < n; i++) m.data[(i * n + i) * 2] = 1;
  return m;
}

export function clone(M: ComplexMatrix): ComplexMatrix {
  return { rows: M.rows, cols: M.cols, data: new Float64Array(M.data) };
}

/** Set element (r, c) = re + i·im. */
export function cset(M: ComplexMatrix, r: number, c: number, re: number, im: number): void {
  const k = (r * M.cols + c) * 2;
  M.data[k] = re;
  M.data[k + 1] = im;
}

/** Read element (r, c) as [re, im]. */
export function cget(M: ComplexMatrix, r: number, c: number): [number, number] {
  const k = (r * M.cols + c) * 2;
  return [M.data[k]!, M.data[k + 1]!];
}

// ── SVD result ────────────────────────────────────────────────
export interface SVDResult {
  /** m × k, with k = min(m, n). Columns = left singular vectors. */
  readonly U: ComplexMatrix;
  /** Length k, descending. */
  readonly sigma: Float64Array;
  /** k × n. Rows of Vh are conjugate-transpose of right singular vectors. */
  readonly Vh: ComplexMatrix;
}

// ─────────────────────────────────────────────────────────────
// Complex one-sided Jacobi SVD of M (m × n).
//
// Algorithm (Hestenes 1958 / Brent-Luk-Van Loan): orthogonalize
// columns of M by applying a unitary V on the right, so M' = M V
// has mutually orthogonal columns. Then σ_i = ||M'[:,i]||, U[:,i]
// = M'[:,i] / σ_i, and V^H is the transpose-conjugate of V.
//
// For complex matrices we eliminate the cross-inner-product in two
// sub-steps: (1) phase-rotate column q so that ⟨col p, col q⟩ is
// real positive, (2) apply a real Jacobi rotation on (p, q) to zero
// it. Both sub-rotations accumulate into V unitarily.
//
// Converges at quadratic rate once near the fixed point. For
// matrices ≤ 128 × 128 this finishes well within 30 sweeps.
// ─────────────────────────────────────────────────────────────
export function svd(M: ComplexMatrix): SVDResult {
  // If wide (m < n), SVD(M) = U Σ V^H with U (m × m), Σ (m × n), V^H (n × n).
  // One-sided Jacobi on wide M would produce up to m nonzero columns and
  // leave n - m zero columns — a waste. Transpose-trick: SVD(M^H) = V Σ U^H,
  // so we compute SVD of M^H (tall n × m) and swap U ↔ V at the end.
  if (M.rows < M.cols) {
    const mh = hermitianTranspose(M);
    const sub = svdTallOrSquare(mh);
    // sub.U: n × m (left singular vectors of M^H) — these are RIGHT singular vectors of M.
    // sub.Vh: m × m — rows of Vh are conj-trans of LEFT singular vectors of M^H,
    //                 so Vh^H (columns) are RIGHT of M^H = LEFT of M.
    const Ufinal = hermitianTranspose(sub.Vh);      // m × m  → columns are U of M
    const Vfinal_h = hermitianTranspose(sub.U);     // m × n  → rows = V^H of M
    return { U: Ufinal, sigma: sub.sigma, Vh: Vfinal_h };
  }
  return svdTallOrSquare(M);
}

/** Hermitian transpose: out[i,j] = conj(M[j,i]). */
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

function svdTallOrSquare(M: ComplexMatrix): SVDResult {
  const m = M.rows;
  const n = M.cols;
  // We mutate a working copy of M; V is built up as identity and accumulates
  // all column operations. Final V is the unitary that maps original column
  // space to orthogonal columns of the working matrix.
  const A = clone(M);
  const V = identity(n);

  const MAX_SWEEPS = 60;
  const TOL = 1e-14;

  let sweeps = 0;
  let off: number;
  do {
    off = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        off += jacobiSweepPair(A, V, p, q, TOL);
      }
    }
    sweeps++;
  } while (off > TOL && sweeps < MAX_SWEEPS);

  // Now A has orthogonal columns. Extract sigmas and U.
  const k = Math.min(m, n);
  const sigmaAll = new Float64Array(n);
  for (let c = 0; c < n; c++) sigmaAll[c] = Math.sqrt(colNormSq(A, c));

  // Sort descending
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => sigmaAll[b]! - sigmaAll[a]!);

  // U: m × k, columns = A[:, order[i]] / sigma[order[i]], for first k.
  // V^H: k × n, rows = conj(V[:, order[i]])^T for first k.
  const sigma = new Float64Array(k);
  const U = zeros(m, k);
  const Vh = zeros(k, n);
  for (let i = 0; i < k; i++) {
    const srcCol = order[i]!;
    const s = sigmaAll[srcCol]!;
    sigma[i] = s;
    if (s > 0) {
      for (let r = 0; r < m; r++) {
        const [re, im] = cget(A, r, srcCol);
        cset(U, r, i, re / s, im / s);
      }
    } else {
      // Degenerate: leave as zero. Downstream MPS truncation drops these.
    }
    for (let c = 0; c < n; c++) {
      const [re, im] = cget(V, c, srcCol);
      cset(Vh, i, c, re, -im);
    }
  }
  return { U, sigma, Vh };
}

function colNormSq(A: ComplexMatrix, c: number): number {
  let s = 0;
  for (let r = 0; r < A.rows; r++) {
    const k = (r * A.cols + c) * 2;
    const re = A.data[k]!, im = A.data[k + 1]!;
    s += re * re + im * im;
  }
  return s;
}

/** Inner product ⟨col p, col q⟩ = Σ conj(A[i,p]) · A[i,q]. */
function colDotConj(A: ComplexMatrix, p: number, q: number): [number, number] {
  let re = 0, im = 0;
  for (let r = 0; r < A.rows; r++) {
    const kp = (r * A.cols + p) * 2;
    const kq = (r * A.cols + q) * 2;
    const pr = A.data[kp]!, pi = A.data[kp + 1]!;
    const qr = A.data[kq]!, qi = A.data[kq + 1]!;
    // conj(p) * q = (pr - i·pi)·(qr + i·qi) = (pr·qr + pi·qi) + i·(pr·qi - pi·qr)
    re += pr * qr + pi * qi;
    im += pr * qi - pi * qr;
  }
  return [re, im];
}

/** Apply to columns p, q of M: new_p = c·p − s·q,  new_q = s·p + c·q   (c, s real). */
function realJacobiRotate(M: ComplexMatrix, p: number, q: number, c: number, s: number): void {
  for (let r = 0; r < M.rows; r++) {
    const kp = (r * M.cols + p) * 2;
    const kq = (r * M.cols + q) * 2;
    const pr = M.data[kp]!, pi = M.data[kp + 1]!;
    const qr = M.data[kq]!, qi = M.data[kq + 1]!;
    M.data[kp]     = c * pr - s * qr;
    M.data[kp + 1] = c * pi - s * qi;
    M.data[kq]     = s * pr + c * qr;
    M.data[kq + 1] = s * pi + c * qi;
  }
}

/** Multiply column c of M by complex scalar (zr, zi). */
function colScale(M: ComplexMatrix, c: number, zr: number, zi: number): void {
  for (let r = 0; r < M.rows; r++) {
    const k = (r * M.cols + c) * 2;
    const re = M.data[k]!, im = M.data[k + 1]!;
    // (re + i·im)·(zr + i·zi) = (re·zr − im·zi) + i·(re·zi + im·zr)
    M.data[k]     = re * zr - im * zi;
    M.data[k + 1] = re * zi + im * zr;
  }
}

/** One Jacobi rotation on pair (p, q). Returns |a_pq|² / (a_pp · a_qq) as "off" indicator.
 *
 * Numerical stability notes:
 *   - Skip if either column has near-zero norm (can't rotate against a zero vector).
 *   - Skip if mag is below an absolute floor — comparing against tol·max(‖p‖, ‖q‖)
 *     catches the case where the *relative* ratio passes but absolute mag is
 *     just FP noise from cancellation (this otherwise drives the whole rotation
 *     into a 45° mix of two near-zero columns, propagating cancellation
 *     errors into the parent state and on rare V8 inputs zeroing out the norm
 *     entirely — observed at brick-depth-4 N=8 trial=3 in Chromium).
 *   - Bail on any non-finite intermediate; corrupting A and V with NaN is far
 *     worse than skipping a single rotation.
 */
function jacobiSweepPair(A: ComplexMatrix, V: ComplexMatrix, p: number, q: number, tol: number): number {
  const [dre, dim] = colDotConj(A, p, q);
  const mag = Math.hypot(dre, dim);
  const app = colNormSq(A, p);
  const aqq = colNormSq(A, q);
  if (!(app > 0) || !(aqq > 0)) return 0;
  const ratio = (mag * mag) / (app * aqq);
  if (ratio < tol * tol) return ratio;

  // Absolute floor: if mag is at FP noise relative to the column norms,
  // the columns are already orthogonal up to round-off — rotating only
  // mixes noise. tol=1e-14 already, this just adds an absolute backstop.
  const normFloor = Math.sqrt(Math.max(app, aqq)) * 1e-15;
  if (mag <= normFloor) return ratio;

  const zr = dre / mag;
  const zi = dim / mag;
  if (!Number.isFinite(zr) || !Number.isFinite(zi)) return ratio;

  // Phase-align col q so ⟨p, q⟩ becomes real positive = mag.
  colScale(A, q, zr, -zi);
  colScale(V, q, zr, -zi);

  // Real Jacobi: tan(2θ) = 2·mag / (aqq − app). Use smaller-root form for stability.
  const tau = (aqq - app) / (2 * mag);
  const t = tau >= 0
    ? 1 / (tau + Math.sqrt(1 + tau * tau))
    : -1 / (-tau + Math.sqrt(1 + tau * tau));
  const cj = 1 / Math.sqrt(1 + t * t);
  const sj = t * cj;
  if (!Number.isFinite(cj) || !Number.isFinite(sj)) return ratio;

  realJacobiRotate(A, p, q, cj, sj);
  realJacobiRotate(V, p, q, cj, sj);

  return ratio;
}

// ─────────────────────────────────────────────────────────────
// Matrix multiplication: C = A · B (complex).
// ─────────────────────────────────────────────────────────────
export function matmul(A: ComplexMatrix, B: ComplexMatrix): ComplexMatrix {
  if (A.cols !== B.rows) {
    throw new Error(`matmul shape mismatch: ${A.rows}×${A.cols} · ${B.rows}×${B.cols}`);
  }
  const C = zeros(A.rows, B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let k = 0; k < A.cols; k++) {
      const aRe = A.data[(i * A.cols + k) * 2]!;
      const aIm = A.data[(i * A.cols + k) * 2 + 1]!;
      if (aRe === 0 && aIm === 0) continue;
      for (let j = 0; j < B.cols; j++) {
        const bRe = B.data[(k * B.cols + j) * 2]!;
        const bIm = B.data[(k * B.cols + j) * 2 + 1]!;
        // (aRe + i·aIm)·(bRe + i·bIm)
        const pr = aRe * bRe - aIm * bIm;
        const pi = aRe * bIm + aIm * bRe;
        const off = (i * B.cols + j) * 2;
        C.data[off]     = C.data[off]! + pr;
        C.data[off + 1] = C.data[off + 1]! + pi;
      }
    }
  }
  return C;
}

/** Frobenius distance ||A - B||_F, for SVD reconstruction tests. */
export function frobeniusDistance(A: ComplexMatrix, B: ComplexMatrix): number {
  if (A.rows !== B.rows || A.cols !== B.cols) {
    throw new Error("shape mismatch in frobeniusDistance");
  }
  let s = 0;
  for (let i = 0; i < A.data.length; i += 2) {
    const dr = A.data[i]! - B.data[i]!;
    const di = A.data[i + 1]! - B.data[i + 1]!;
    s += dr * dr + di * di;
  }
  return Math.sqrt(s);
}

/** Reconstruct M = U · diag(sigma) · Vh, for round-trip tests. */
export function svdReconstruct(svd: SVDResult): ComplexMatrix {
  const { U, sigma, Vh } = svd;
  const k = sigma.length;
  // Bake diag(sigma) into Vh.
  const SVh = clone(Vh);
  for (let i = 0; i < k; i++) {
    const si = sigma[i]!;
    for (let j = 0; j < Vh.cols; j++) {
      const idx = (i * Vh.cols + j) * 2;
      SVh.data[idx]     = SVh.data[idx]! * si;
      SVh.data[idx + 1] = SVh.data[idx + 1]! * si;
    }
  }
  return matmul(U, SVh);
}
