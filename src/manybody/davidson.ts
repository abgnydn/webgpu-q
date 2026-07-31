// ─────────────────────────────────────────────────────────────
// davidson.ts — block Davidson eigensolver for real, possibly
// non-symmetric matrices, accessed only via a matvec function.
//
// Why this exists: EOM-CCSD's σ-equation matrix M can be huge
// (e.g. CH₂O / HCN STO-3G hit dim 2660-3488), and the dense
// Hessenberg-QR path scales as O(N³). For k lowest roots we
// only need O(N · k · m) operations where m is the subspace
// dimension (~5-10·k). On the diagnostic CH₂O STO-3G case the
// dense path takes 15-30 min; Davidson targets minutes.
//
// Algorithm — block Davidson (Liu 1978) generalized to
// non-symmetric matrices (Hirao-Nakatsuji 1982 / Olsen-Jensen
// 1990 style). For each iteration:
//
//   1. Project M onto the current subspace V (orthonormal columns):
//        H̃ = V^T · (M · V)
//      H̃ is a small (m × m) dense matrix.
//   2. Diagonalize H̃ (use the existing dense general eigensolver).
//      Pick the k Ritz values closest to the target.
//   3. For each picked Ritz pair (λᵢ, cᵢ), form the approximate
//      eigenvector yᵢ = V · cᵢ and residual rᵢ = M·yᵢ − λᵢ·yᵢ.
//   4. If max ‖rᵢ‖ < tol → converged.
//   5. Otherwise precondition each residual: tᵢ = rᵢ / (Dⱼ − λᵢ)
//      where D is the diagonal of M. Orthogonalize each tᵢ against
//      the current V using modified Gram-Schmidt, and append.
//   6. If subspace size hits maxSubspace, restart from the current
//      Ritz vectors.
//
// Notes:
//   - For non-symmetric M, ‖r‖ is not a strict upper bound on the
//     eigenvalue error (unlike the symmetric case), but it is a
//     reliable convergence indicator in practice for EOM-style
//     matrices whose spectrum is well separated near the ground.
//   - The preconditioner D is the M-diagonal estimate; for
//     EOM-CCSD this is the orbital-energy gap (ε_a − ε_i for R_1,
//     ε_a + ε_b − ε_i − ε_j for R_2). Higher-order corrections
//     to the diagonal don't matter for convergence behavior —
//     this is the standard Davidson-Liu preconditioner.
//   - We track right eigenvectors only. Left eigenvectors for
//     biorthogonal observables (transition densities, transition
//     dipoles via R^L · μ · R^R) are a follow-up that runs the
//     same Davidson on M^T.
// ─────────────────────────────────────────────────────────────

import { eigGeneralWithVectors } from "./dense-eig-general.js";

export interface DavidsonResult {
  /** k smallest real eigenvalues, sorted ascending. */
  readonly energies: Float64Array;
  /** Right eigenvectors, row-major: vectors[k * dim + row]. Each
   *  is unit-normalized in the Euclidean inner product. */
  readonly vectors: Float64Array;
  /** Imaginary parts (should be at roundoff for stable EOM problems). */
  readonly imag: Float64Array;
  /** Per-root residual ‖M·y − λ·y‖ at convergence (or at maxIter). */
  readonly residuals: Float64Array;
  /** Number of outer Davidson iterations executed. */
  readonly iterations: number;
  /** True iff every root's residual < tol. */
  readonly converged: boolean;
  /** Number of matvec evaluations performed (rough cost proxy). */
  readonly matvecs: number;
}

export interface DavidsonOpts {
  /** # roots to find. */
  readonly k: number;
  /** Convergence tolerance on residual norm. Default 1e-7. */
  readonly tol?: number;
  /** Max outer Davidson iterations. Default 200. */
  readonly maxIter?: number;
  /** Subspace dimension cap before restart. Default max(8·k, 40). */
  readonly maxSubspace?: number;
  /** Initial guess vectors, row-major (each row length = dim).
   *  Length must be ≥ k. If omitted, uses the k smallest-diagonal
   *  unit vectors. */
  readonly initialGuesses?: Float64Array;
  /** Number of initial guesses provided (in case initialGuesses
   *  has more than k rows). Default: k. */
  readonly nInitial?: number;
  /** Filter accepted eigenvalues. Default: real and > 1e-8. */
  readonly filter?: (re: number, im: number) => boolean;
  /** Tolerance for "real" eigenvalue. Default 1e-6. */
  readonly imagTol?: number;
}

/**
 * Block Davidson eigensolver. Returns the k lowest real eigenvalues
 * (sorted ascending) and their right eigenvectors of an N×N real
 * matrix M accessed only via `matvec` (which computes M·v) and
 * `diagonal` (the M diagonal, used as the preconditioner).
 *
 * For non-symmetric M with all-real spectrum (the EOM-CCSD case
 * for stable closed-shell references), this is a standard
 * residual-based block Davidson with Davidson-Liu preconditioning
 * and modified Gram-Schmidt orthogonalization.
 */
export function davidson(
  dim: number,
  matvec: (v: Float64Array) => Float64Array,
  diagonal: Float64Array,
  opts: DavidsonOpts,
): DavidsonResult {
  const k = opts.k;
  if (k <= 0) throw new Error(`davidson: k must be > 0 (got ${k})`);
  if (k > dim) throw new Error(`davidson: k=${k} > dim=${dim}`);
  if (diagonal.length !== dim) {
    throw new Error(`davidson: diagonal length ${diagonal.length} != dim ${dim}`);
  }
  const tol = opts.tol ?? 1e-7;
  const maxIter = opts.maxIter ?? 200;
  const maxSubspace = opts.maxSubspace ?? Math.max(8 * k, 40);
  const imagTol = opts.imagTol ?? 1e-6;
  const filter = opts.filter ?? ((re, im) => Math.abs(im) < imagTol && re > 1e-8);

  if (maxSubspace < 2 * k) {
    throw new Error(`davidson: maxSubspace=${maxSubspace} must be ≥ 2·k=${2 * k}`);
  }
  if (maxSubspace > dim) {
    // Fall back: subspace can't exceed problem dimension.
    // (At that point dense diag is no worse than Davidson.)
  }
  const subspaceCap = Math.min(maxSubspace, dim);

  // ── Subspace storage. V[s][i] is the i-th component of the s-th
  // basis vector; AV[s][i] is M·V[s].
  const V: Float64Array[] = [];
  const AV: Float64Array[] = [];
  let matvecs = 0;

  // ── Initial guesses ───────────────────────────────────────
  if (opts.initialGuesses) {
    const nInit = opts.nInitial ?? k;
    for (let s = 0; s < nInit; s++) {
      const v = new Float64Array(dim);
      for (let i = 0; i < dim; i++) {
        v[i] = opts.initialGuesses[s * dim + i]!;
      }
      orthonormalizeAgainst(v, V);
      const norm2 = dotSelf(v);
      if (norm2 > 1e-20) {
        scaleInPlace(v, 1 / Math.sqrt(norm2));
        V.push(v);
        AV.push(matvec(v));
        matvecs++;
      }
    }
  } else {
    // Pick the k indices with smallest diagonal entries as the
    // initial unit vectors.
    const idx = sortedDiagonalIndices(diagonal, k);
    for (let s = 0; s < idx.length; s++) {
      const v = new Float64Array(dim);
      v[idx[s]!] = 1;
      orthonormalizeAgainst(v, V);
      const norm2 = dotSelf(v);
      if (norm2 > 1e-20) {
        scaleInPlace(v, 1 / Math.sqrt(norm2));
        V.push(v);
        AV.push(matvec(v));
        matvecs++;
      }
    }
  }
  if (V.length < k) {
    throw new Error(`davidson: only ${V.length} linearly independent initial guesses (need ${k})`);
  }

  // Working storage for the converged result.
  const lastEnergies = new Float64Array(k);
  const lastImag = new Float64Array(k);
  const lastVectors = new Float64Array(k * dim);
  const lastResiduals = new Float64Array(k);
  // Whether the Ritz block below ever wrote lastEnergies/lastResiduals. Both
  // start as zeros, so without this the "honest convergence check" after the
  // loop reads finalMaxRes = 0 < tol and reports all-zero energies as CONVERGED
  // when the subspace could not grow (the `if (!added) break` exit) before a
  // single Ritz step ran.
  let haveRitz = false;
  let converged = false;
  let iter = 0;

  for (iter = 0; iter < maxIter; iter++) {
    const m = V.length;
    // ── Build projected H̃ = V^T · AV (m × m, row-major). ─────
    const Htilde = new Float64Array(m * m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        Htilde[i * m + j] = dotProduct(V[i]!, AV[j]!);
      }
    }
    // ── Diagonalize the small projected problem. ──────────────
    const eig = eigGeneralWithVectors(Htilde, m);

    // Filter + sort: pick k smallest accepted eigenvalues.
    const accepted: Array<{ re: number; im: number; col: number }> = [];
    for (let p = 0; p < m; p++) {
      if (filter(eig.real[p]!, eig.imag[p]!)) {
        accepted.push({ re: eig.real[p]!, im: eig.imag[p]!, col: p });
      }
    }
    accepted.sort((a, b) => a.re - b.re);

    if (accepted.length < k) {
      // Not enough filter-passing Ritz values yet. DON'T drop the
      // filter — that would return wrong eigenvalues. Instead, grow
      // the subspace with the next-best filter-passing diagonal
      // direction we haven't used, OR a random direction if all
      // smallest-diagonal seeds are exhausted.
      let added = false;
      const used = new Set<number>();
      // Mark indices already strongly represented in V.
      for (let s = 0; s < V.length; s++) {
        let maxAbs = 0, maxIdx = -1;
        for (let i = 0; i < dim; i++) {
          const a = Math.abs(V[s]![i]!);
          if (a > maxAbs) { maxAbs = a; maxIdx = i; }
        }
        if (maxIdx >= 0) used.add(maxIdx);
      }
      // Sweep diagonal indices in ascending order, pick first unused.
      const order = sortedDiagonalIndices(diagonal, dim);
      for (let s = 0; s < order.length; s++) {
        const idx = order[s]!;
        if (used.has(idx)) continue;
        const v = new Float64Array(dim);
        v[idx] = 1;
        orthonormalizeAgainst(v, V);
        const n2 = dotSelf(v);
        if (n2 > 1e-20) {
          scaleInPlace(v, 1 / Math.sqrt(n2));
          V.push(v);
          AV.push(matvec(v));
          matvecs++;
          added = true;
          break;
        }
      }
      if (!added) break;  // can't grow further
      continue;
    }

    // ── Build Ritz vectors yᵢ = V · cᵢ and residuals rᵢ = AV · cᵢ − λᵢ · yᵢ.
    const ritz: Float64Array[] = [];
    const residuals: Float64Array[] = [];
    const resNorms = new Float64Array(k);
    let maxResNorm = 0;
    for (let p = 0; p < k; p++) {
      const { re: lam, col } = accepted[p]!;
      // Ritz vector yᵢ = Σ_s c[s] · V[s]
      const y = new Float64Array(dim);
      for (let s = 0; s < m; s++) {
        const c_s = eig.vectors[col * m + s]!;
        const Vs = V[s]!;
        for (let i = 0; i < dim; i++) y[i]! += c_s * Vs[i]!;
      }
      // Renormalize for numerical hygiene.
      const yNorm = Math.sqrt(dotSelf(y));
      if (yNorm > 1e-20) scaleInPlace(y, 1 / yNorm);

      // Residual rᵢ = M·yᵢ − λᵢ·yᵢ.
      // M·yᵢ = Σ_s c[s] · AV[s] — reuse the cached AV from the
      // subspace builds. V is orthonormal and c comes from the
      // small eig problem normalized to unit length, so y has
      // unit norm to roundoff and the c-basis identity holds.
      // Scale by 1/yNorm to match the renormalization above.
      const inv = yNorm > 1e-20 ? 1 / yNorm : 0;
      const My = new Float64Array(dim);
      for (let s = 0; s < m; s++) {
        const c_s = eig.vectors[col * m + s]! * inv;
        const AVs = AV[s]!;
        for (let i = 0; i < dim; i++) My[i]! += c_s * AVs[i]!;
      }
      const r = new Float64Array(dim);
      for (let i = 0; i < dim; i++) r[i] = My[i]! - lam * y[i]!;
      const rN = Math.sqrt(dotSelf(r));
      resNorms[p] = rN;
      if (rN > maxResNorm) maxResNorm = rN;
      ritz.push(y);
      residuals.push(r);
    }

    // Cache the current best estimate (in case we hit maxIter).
    for (let p = 0; p < k; p++) {
      haveRitz = true;
      lastEnergies[p] = accepted[p]!.re;
      lastImag[p] = accepted[p]!.im;
      lastResiduals[p] = resNorms[p]!;
      const y = ritz[p]!;
      for (let i = 0; i < dim; i++) lastVectors[p * dim + i] = y[i]!;
    }

    if (maxResNorm < tol) {
      converged = true;
      break;
    }

    // ── Restart if subspace is full. ──────────────────────────
    if (m + k > subspaceCap) {
      V.length = 0;
      AV.length = 0;
      // Re-seed from the current Ritz vectors (orthogonalize for hygiene).
      for (let p = 0; p < k; p++) {
        const y = new Float64Array(ritz[p]!);
        orthonormalizeAgainst(y, V);
        const n2 = dotSelf(y);
        if (n2 > 1e-20) {
          scaleInPlace(y, 1 / Math.sqrt(n2));
          V.push(y);
          AV.push(matvec(y));
          matvecs++;
        }
      }
    }

    // ── Add preconditioned residuals to the subspace. ────────
    for (let p = 0; p < k; p++) {
      if (resNorms[p]! < tol) continue;
      const lam = accepted[p]!.re;
      const r = residuals[p]!;
      const t = new Float64Array(dim);
      for (let i = 0; i < dim; i++) {
        const denom = lam - diagonal[i]!;
        // Guard against tiny denominator (Davidson breakdown).
        // Standard fix: if |denom| < eps, zero that component
        // (a different basis direction will pick it up later).
        t[i] = Math.abs(denom) > 1e-10 ? r[i]! / denom : 0;
      }
      // Modified Gram-Schmidt against V (twice for stability).
      orthonormalizeAgainst(t, V);
      orthonormalizeAgainst(t, V);
      const n2 = dotSelf(t);
      if (n2 > 1e-20) {
        scaleInPlace(t, 1 / Math.sqrt(n2));
        V.push(t);
        AV.push(matvec(t));
        matvecs++;
      }
    }

    if (V.length === m) {
      // No new directions added — subspace can't expand. Bail
      // with whatever we have. We may still be at or below tol
      // for every root; the honest converged flag is computed
      // from the actual residual norms below.
      break;
    }
  }

  // Honest convergence check: max residual norm vs tol, regardless
  // of which loop exit fired. This handles the case where the
  // preconditioned correction stalls (no new directions to add)
  // while every root's residual is already at or below tol.
  let finalMaxRes = 0;
  for (let p = 0; p < k; p++) {
    if (lastResiduals[p]! > finalMaxRes) finalMaxRes = lastResiduals[p]!;
  }
  // Only meaningful once a Ritz step has actually produced energies + residuals.
  if (!converged && haveRitz && finalMaxRes < tol) converged = true;

  return {
    energies: lastEnergies,
    imag: lastImag,
    vectors: lastVectors,
    residuals: lastResiduals,
    iterations: iter + 1,
    converged,
    matvecs,
  };
}

// ── Helpers ──────────────────────────────────────────────────

function dotProduct(a: Float64Array, b: Float64Array): number {
  let s = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) s += a[i]! * b[i]!;
  return s;
}

function dotSelf(a: Float64Array): number {
  let s = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    const x = a[i]!;
    s += x * x;
  }
  return s;
}

function scaleInPlace(a: Float64Array, s: number): void {
  const n = a.length;
  for (let i = 0; i < n; i++) a[i]! *= s;
}

/** Modified Gram-Schmidt: project v orthogonal to every vector in U.
 *  Modifies v in place. Does NOT renormalize. */
function orthonormalizeAgainst(v: Float64Array, U: Float64Array[]): void {
  for (let s = 0; s < U.length; s++) {
    const u = U[s]!;
    const c = dotProduct(v, u);
    if (c !== 0) {
      const n = v.length;
      for (let i = 0; i < n; i++) v[i]! -= c * u[i]!;
    }
  }
}

/** Return the indices of the k smallest entries in `d`. */
function sortedDiagonalIndices(d: Float64Array, k: number): Int32Array {
  const indexed: Array<{ v: number; i: number }> = [];
  for (let i = 0; i < d.length; i++) indexed.push({ v: d[i]!, i });
  indexed.sort((a, b) => a.v - b.v);
  const out = new Int32Array(Math.min(k, indexed.length));
  for (let s = 0; s < out.length; s++) out[s] = indexed[s]!.i;
  return out;
}
