// ─────────────────────────────────────────────────────────────
// dense-eig-general.ts — real general (non-symmetric) eigenvalues
// via Hessenberg reduction + implicit-shift QR.
//
// Returns real and imaginary parts of all eigenvalues. For
// matrices arising from stable EOM-CCSD-style problems the
// spectrum is real; imaginary parts should be at roundoff.
//
// Algorithm (textbook — Wilkinson 1965 / Golub-Van Loan ch. 7):
//   1. Householder reduction H = Q^T M Q to upper Hessenberg form.
//      O(N^3), in-place.
//   2. QR iteration on H with a Wilkinson shift drawn from the
//      trailing 2×2 block. Each step:
//        a. shift the diagonal by σ
//        b. apply n−1 Givens rotations from the left to reduce
//           (H − σI) to upper triangular R
//        c. apply the same rotations from the right (RQ), preserving
//           Hessenberg form
//        d. add σ back
//      Subdiagonals shrink as iterates converge; deflate when
//      |H[k, k−1]| < tol · (|H[k−1, k−1]| + |H[k, k]|).
//   3. After deflation the matrix is quasi-upper-triangular: 1×1
//      blocks give real eigenvalues, 2×2 blocks give either two
//      real eigenvalues (disc ≥ 0) or a complex pair (disc < 0).
//
// We do NOT track Q (we only need eigenvalues). Adding eigenvectors
// is a follow-up — that would track Q during the Hessenberg reduction
// and accumulate the right-Givens rotations during QR iterations.
// ─────────────────────────────────────────────────────────────

export interface EigGeneralResult {
  /** Real parts of the eigenvalues. Length N. Unsorted. */
  readonly real: Float64Array;
  /** Imaginary parts. For a complex-conjugate pair, sign(im) and
   *  −sign(im) appear adjacent. */
  readonly imag: Float64Array;
}

export interface EigGeneralVectorsResult extends EigGeneralResult {
  /** Right eigenvectors of M, column-major: vectors[k * N + row] is
   *  the row-th component of the eigenvector for eigenvalue k.
   *  For complex eigenvalue pairs (re, ±im at columns k, k+1), the
   *  two columns hold the real and imaginary parts of one complex
   *  eigenvector. For our use case (EOM-CCSD on stable closed-shell
   *  ref) all eigenvalues are real and the complex case doesn't fire. */
  readonly vectors: Float64Array;
}

export interface EigGeneralOpts {
  /** Convergence tolerance on subdiagonal deflation. Default 1e-12. */
  readonly tol?: number;
  /** Max QR iterations before giving up. Default 30·N. */
  readonly maxIter?: number;
}

/**
 * Real general eigenvalues of an N×N real matrix M (row-major).
 * For matrices with all-real spectrum, imaginary parts will be at
 * roundoff. For matrices with complex-conjugate pairs, those pairs
 * are emitted as (re, ±im) adjacent entries.
 */
export function eigGeneral(M: Float64Array, N: number, opts: EigGeneralOpts = {}): EigGeneralResult {
  const tol = opts.tol ?? 1e-12;
  const maxIter = opts.maxIter ?? 30 * N;
  const H = new Float64Array(M);
  hessenbergReduce(H, N, null);
  qrIterate(H, N, tol, maxIter, null);
  return extractEigenvalues(H, N, tol);
}

/**
 * Eigenvalues AND right eigenvectors of an N×N real matrix M.
 *
 * Tracks Q through Hessenberg reduction (apply each Householder
 * reflector to Q from the right) and through QR iteration (each
 * right-Givens rotation acts on Q from the right). After QR
 * converges, M is similar to a quasi-upper-triangular T via
 * M = Q · T · Q^T. Real eigenvalues sit on T's diagonal; their
 * eigenvectors come from back-substitution on T, then transformed
 * back to M's basis via v_M = Q · v_T.
 *
 * Degenerate eigenvalues: when Y_jj = λ_k for some j < k along
 * the back-substitution chain, the standard recurrence has a 0
 * denominator. We set v_j = 0 in that case, which picks ONE
 * valid eigenvector from the degenerate eigenspace.
 *
 * IMPORTANT CAVEAT: the routine returns individually unit-normalized
 * eigenvectors for each eigenvalue, but for an N-fold degenerate
 * eigenvalue the N returned vectors are NOT guaranteed to be
 * mutually orthogonal — they span the degenerate eigenspace but
 * are arbitrary representatives within it. Callers that need an
 * orthonormal basis of a degenerate subspace must Gram-Schmidt
 * the returned vectors explicitly. For uses that sum |c_i|² over
 * components (oscillator strengths, spin weights), the choice of
 * representative is irrelevant.
 */
export function eigGeneralWithVectors(
  M: Float64Array,
  N: number,
  opts: EigGeneralOpts = {},
): EigGeneralVectorsResult {
  const tol = opts.tol ?? 1e-12;
  const maxIter = opts.maxIter ?? 30 * N;
  const H = new Float64Array(M);
  // Q starts as identity; accumulated as similarity transforms are applied.
  const Q = new Float64Array(N * N);
  for (let i = 0; i < N; i++) Q[i * N + i] = 1;
  hessenbergReduce(H, N, Q);
  qrIterate(H, N, tol, maxIter, Q);
  const { real, imag } = extractEigenvalues(H, N, tol);
  const vectors = backSubstituteEigenvectors(H, Q, real, imag, N, tol);
  return { real, imag, vectors };
}

// ─────────────────────────────────────────────────────────────
// Step 1 — Householder reduction to upper Hessenberg form.
// When Q is non-null, accumulates each Householder reflector into Q
// from the right: Q_new = Q · (I − 2 v v^T / vnorm²). This makes
// Q hold the similarity transform that brought M to Hessenberg form,
// so M = Q · H · Q^T at the end.
// ─────────────────────────────────────────────────────────────
function hessenbergReduce(H: Float64Array, N: number, Q: Float64Array | null): void {
  for (let k = 0; k < N - 2; k++) {
    // Build Householder reflector for column k below row k+1.
    let normx = 0;
    for (let i = k + 1; i < N; i++) {
      const x = H[i * N + k]!;
      normx += x * x;
    }
    normx = Math.sqrt(normx);
    if (normx < 1e-30) continue;
    const xk1 = H[(k + 1) * N + k]!;
    const sign = xk1 >= 0 ? 1 : -1;
    const alpha = -sign * normx;
    // v = x - alpha·e_1 (where x is the column below row k+1).
    // Store v in H[k+2..N-1, k] in-place; H[k+1, k] becomes v[0] = xk1 - alpha.
    const v = new Float64Array(N - k - 1);
    v[0] = xk1 - alpha;
    for (let i = k + 2; i < N; i++) v[i - k - 1] = H[i * N + k]!;
    let vnorm2 = 0;
    for (let i = 0; i < v.length; i++) vnorm2 += v[i]! * v[i]!;
    if (vnorm2 < 1e-30) continue;
    // Apply (I − 2 vv^T/vnorm2) from the LEFT to H[k+1.., k..].
    const beta = 2 / vnorm2;
    // For each column j ≥ k: w = v^T · H[k+1.., j]; subtract β·v·w from that column.
    for (let j = k; j < N; j++) {
      let w = 0;
      for (let i = 0; i < v.length; i++) w += v[i]! * H[(k + 1 + i) * N + j]!;
      w *= beta;
      for (let i = 0; i < v.length; i++) {
        const idx = (k + 1 + i) * N + j;
        H[idx] = H[idx]! - v[i]! * w;
      }
    }
    // Apply (I − 2 vv^T/vnorm2) from the RIGHT to H[.., k+1..].
    for (let i = 0; i < N; i++) {
      let w = 0;
      for (let j = 0; j < v.length; j++) w += H[i * N + k + 1 + j]! * v[j]!;
      w *= beta;
      for (let j = 0; j < v.length; j++) {
        const idx = i * N + k + 1 + j;
        H[idx] = H[idx]! - w * v[j]!;
      }
    }
    // Force exact Hessenberg structure (numerical hygiene).
    H[(k + 1) * N + k] = alpha;
    for (let i = k + 2; i < N; i++) H[i * N + k] = 0;
    // Accumulate the Householder reflector into Q from the right.
    if (Q !== null) {
      for (let i = 0; i < N; i++) {
        let w = 0;
        for (let j = 0; j < v.length; j++) w += Q[i * N + k + 1 + j]! * v[j]!;
        w *= beta;
        for (let j = 0; j < v.length; j++) {
          const idx = i * N + k + 1 + j;
          Q[idx] = Q[idx]! - w * v[j]!;
        }
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Step 2 — QR iteration with Wilkinson shift and deflation.
// When Q is non-null, each right-Givens rotation is also applied
// to Q from the right (Q tracks the cumulative similarity that
// transforms M into the (quasi-)triangular form held by H).
// ─────────────────────────────────────────────────────────────
function qrIterate(H: Float64Array, N: number, tol: number, maxIter: number, Q: Float64Array | null): void {
  let hi = N - 1;
  let iter = 0;
  while (hi > 0 && iter < maxIter) {
    iter++;
    // Deflation: if the last subdiagonal is tiny, peel off a 1×1 block.
    if (
      Math.abs(H[hi * N + hi - 1]!) <
      tol * (Math.abs(H[(hi - 1) * N + hi - 1]!) + Math.abs(H[hi * N + hi]!) + 1e-30)
    ) {
      H[hi * N + hi - 1] = 0;
      hi -= 1;
      continue;
    }
    // Two-step deflation: peel off a 2×2 block.
    if (
      hi >= 2 &&
      Math.abs(H[(hi - 1) * N + hi - 2]!) <
        tol * (Math.abs(H[(hi - 2) * N + hi - 2]!) + Math.abs(H[(hi - 1) * N + hi - 1]!) + 1e-30)
    ) {
      H[(hi - 1) * N + hi - 2] = 0;
      hi -= 2;
      continue;
    }
    // Find the lower bound of the active sub-block (largest lo with
    // H[lo, lo−1] effectively zero, defaulting to 0).
    let lo = 0;
    for (let k = hi; k > 0; k--) {
      if (
        Math.abs(H[k * N + k - 1]!) <
        tol * (Math.abs(H[(k - 1) * N + k - 1]!) + Math.abs(H[k * N + k]!) + 1e-30)
      ) {
        H[k * N + k - 1] = 0;
        lo = k;
        break;
      }
    }
    // Wilkinson shift from trailing 2×2.
    const a = H[(hi - 1) * N + hi - 1]!;
    const b = H[(hi - 1) * N + hi]!;
    const c = H[hi * N + hi - 1]!;
    const d = H[hi * N + hi]!;
    const sMid = (a + d) / 2;
    const det = a * d - b * c;
    const disc = sMid * sMid - det;
    let shift: number;
    if (disc >= 0) {
      const sq = Math.sqrt(disc);
      // Pick the eigenvalue of the 2x2 closer to d (the trailing entry).
      shift = sMid + (sMid >= d ? -sq : sq);
    } else {
      // Complex eigenvalues — single-real shift collapses to sMid.
      // (For real-eigenvalue matrices this branch is rare; for complex
      // pairs Francis double-shift is preferred, but our matrices
      // (EOM-CCSD on stable ref) have real spectra so this fallback
      // is good enough.)
      shift = sMid;
    }
    // Exceptional shift every 10 stagnant iterations.
    if (iter % 10 === 0) shift = (Math.abs(c) + Math.abs(b)) * (1 + Math.sin(iter));
    qrStep(H, N, lo, hi, shift, Q);
  }
  if (iter >= maxIter && hi > 0) {
    throw new Error(`eigGeneral: QR iteration did not converge within ${maxIter} steps (hi=${hi}).`);
  }
}

/** One shifted QR step on the Hessenberg sub-block H[lo..hi, lo..hi].
 *  When Q is non-null, the right-Givens rotations are also applied
 *  to Q's columns k, k+1 — accumulating the similarity transform. */
function qrStep(H: Float64Array, N: number, lo: number, hi: number, shift: number, Q: Float64Array | null): void {
  // Subtract shift.
  for (let i = lo; i <= hi; i++) {
    const d = i * N + i;
    H[d] = H[d]! - shift;
  }
  // Forward sweep: n−1 Givens rotations from the LEFT to upper-triangularize.
  // Save (c, s) for each rotation so we can apply them from the right.
  const cs = new Float64Array(hi - lo);
  const ss = new Float64Array(hi - lo);
  for (let k = lo; k < hi; k++) {
    const a = H[k * N + k]!;
    const b = H[(k + 1) * N + k]!;
    const r = Math.hypot(a, b);
    let c: number, s: number;
    if (r < 1e-30) {
      c = 1;
      s = 0;
    } else {
      c = a / r;
      s = b / r;
    }
    cs[k - lo] = c;
    ss[k - lo] = s;
    // Left-apply G^T to rows k and k+1, columns k..N-1.
    for (let j = k; j < N; j++) {
      const x = H[k * N + j]!;
      const y = H[(k + 1) * N + j]!;
      H[k * N + j] = c * x + s * y;
      H[(k + 1) * N + j] = -s * x + c * y;
    }
  }
  // Backward sweep: apply G from the RIGHT to columns k and k+1.
  // For Hessenberg form, the only nonzeros in those columns are rows 0..k+1.
  for (let k = lo; k < hi; k++) {
    const c = cs[k - lo]!;
    const s = ss[k - lo]!;
    const rowEnd = Math.min(k + 2, N);
    for (let i = 0; i < rowEnd; i++) {
      const x = H[i * N + k]!;
      const y = H[i * N + k + 1]!;
      H[i * N + k] = c * x + s * y;
      H[i * N + k + 1] = -s * x + c * y;
    }
    // Same right-Givens applied to Q's columns k and k+1, all rows.
    if (Q !== null) {
      for (let i = 0; i < N; i++) {
        const x = Q[i * N + k]!;
        const y = Q[i * N + k + 1]!;
        Q[i * N + k] = c * x + s * y;
        Q[i * N + k + 1] = -s * x + c * y;
      }
    }
  }
  // Re-add shift.
  for (let i = lo; i <= hi; i++) {
    const d = i * N + i;
    H[d] = H[d]! + shift;
  }
}

// ─────────────────────────────────────────────────────────────
// Step 3b — Back-substitute eigenvectors from the (quasi-)triangular
// form T = Q^T · M · Q.
//
// For a 1×1 block at position k (real eigenvalue λ = T[k,k]):
//   v_k = 1, v_{>k} = 0
//   For i from k-1 down to 0:
//     v_i = − (Σ_{j=i+1}^{k} T[i,j] · v_j) / (T[i,i] − λ)
//     If |T[i,i] − λ| is tiny (degenerate eigenvalue), v_i = 0
//     (picks one representative from the degenerate eigenspace).
//
// 2×2 complex pair: would emit one column with the real part and
// the next with the imaginary part. Not implemented for the EOM-CCSD
// use case (stable closed-shell reference ⇒ all-real spectrum); we
// emit zeros for those columns and return.
//
// Final eigenvectors of M are Q · v.
// ─────────────────────────────────────────────────────────────
function backSubstituteEigenvectors(
  T: Float64Array,
  Q: Float64Array,
  real: Float64Array,
  imag: Float64Array,
  N: number,
  tol: number,
): Float64Array {
  const vectors = new Float64Array(N * N);
  let k = 0;
  while (k < N) {
    if (Math.abs(imag[k]!) > tol) {
      // 2×2 complex block at (k, k+1) — skip for now. Leave columns
      // k and k+1 as zeros (the EOM-CCSD use case never hits this).
      k += 2;
      continue;
    }
    const lambda = real[k]!;
    // Y-space eigenvector v: solve (T − λI) v = 0 with v_k = 1.
    const v = new Float64Array(N);
    v[k] = 1;
    for (let i = k - 1; i >= 0; i--) {
      let s = 0;
      for (let j = i + 1; j <= k; j++) s += T[i * N + j]! * v[j]!;
      const denom = T[i * N + i]! - lambda;
      if (Math.abs(denom) < tol) {
        v[i] = 0;
      } else {
        v[i] = -s / denom;
      }
    }
    // Transform back to M-basis: vectors[:, k] = Q · v.
    for (let row = 0; row < N; row++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += Q[row * N + j]! * v[j]!;
      vectors[k * N + row] = s;
    }
    // Normalize to unit length.
    let norm = 0;
    for (let row = 0; row < N; row++) {
      const x = vectors[k * N + row]!;
      norm += x * x;
    }
    norm = Math.sqrt(norm);
    if (norm > tol) {
      for (let row = 0; row < N; row++) {
        const idx = k * N + row;
        vectors[idx] = vectors[idx]! / norm;
      }
    }
    k += 1;
  }
  return vectors;
}

// ─────────────────────────────────────────────────────────────
// Step 3 — Extract eigenvalues from quasi-triangular form.
// ─────────────────────────────────────────────────────────────
function extractEigenvalues(H: Float64Array, N: number, tol: number): EigGeneralResult {
  const real = new Float64Array(N);
  const imag = new Float64Array(N);
  let i = 0;
  while (i < N) {
    const isLast = i === N - 1;
    const subDiag = isLast ? 0 : H[(i + 1) * N + i]!;
    if (
      isLast ||
      Math.abs(subDiag) <
        tol * (Math.abs(H[i * N + i]!) + Math.abs(H[(i + 1) * N + i + 1]!) + 1e-30)
    ) {
      real[i] = H[i * N + i]!;
      imag[i] = 0;
      i += 1;
    } else {
      // 2×2 block at (i, i+1).
      const a = H[i * N + i]!;
      const b = H[i * N + i + 1]!;
      const c = H[(i + 1) * N + i]!;
      const d = H[(i + 1) * N + i + 1]!;
      const tr = a + d;
      const det = a * d - b * c;
      const disc = (tr * tr) / 4 - det;
      if (disc >= 0) {
        const sq = Math.sqrt(disc);
        real[i] = tr / 2 + sq;
        real[i + 1] = tr / 2 - sq;
        imag[i] = 0;
        imag[i + 1] = 0;
      } else {
        const sq = Math.sqrt(-disc);
        real[i] = tr / 2;
        real[i + 1] = tr / 2;
        imag[i] = sq;
        imag[i + 1] = -sq;
      }
      i += 2;
    }
  }
  return { real, imag };
}
