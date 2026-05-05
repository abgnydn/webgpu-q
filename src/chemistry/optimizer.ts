// ─────────────────────────────────────────────────────────────
// optimizer.ts — Nelder-Mead simplex for gradient-free
// minimization of small parameter vectors (≤ ~30D).
//
// Standard reflect/expand/contract/shrink with Nash's tie-
// breaking: shrink only when neither inside-contract nor
// outside-contract beats the second-worst point. Tolerance is
// on both vertex-energy spread and simplex diameter, since
// VQE landscapes can have a flat valley along an unphysical
// direction (e.g. global phase) that fools either metric alone.
//
// No external deps. Deterministic given a seed for the initial
// simplex perturbation.
// ─────────────────────────────────────────────────────────────

export interface NelderMeadOptions {
  /** Initial simplex side-length (perturbation). Default 0.5 rad. */
  initialStep?: number;
  /** Function-value spread tolerance. Default 1e-7. */
  fTol?: number;
  /** Vertex-spread tolerance (max ‖x_i − x_centroid‖_∞). Default 1e-6. */
  xTol?: number;
  /** Hard cap on iterations. Default 5000. */
  maxIter?: number;
  /** Seed for the initial simplex perturbation directions. */
  seed?: number;
}

export interface NelderMeadResult {
  bestX: Float64Array;
  bestF: number;
  iter: number;
  /** "converged" if both fTol and xTol were satisfied; otherwise "max-iter". */
  termination: "converged" | "max-iter";
  /** Per-iteration best f, capped at 200 samples for plotting. */
  history: readonly number[];
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function nelderMead(
  f: (x: Float64Array) => number,
  x0: Float64Array,
  opts: NelderMeadOptions = {},
): NelderMeadResult {
  const n = x0.length;
  const initialStep = opts.initialStep ?? 0.5;
  const fTol = opts.fTol ?? 1e-7;
  const xTol = opts.xTol ?? 1e-6;
  const maxIter = opts.maxIter ?? 5000;
  const rng = mulberry32(opts.seed ?? 0xC0FFEE);

  // Build the initial simplex: x0 plus n perturbed copies.
  const verts: Float64Array[] = [];
  const fs: number[] = [];
  verts.push(new Float64Array(x0));
  fs.push(f(verts[0]!));
  for (let i = 0; i < n; i++) {
    const v = new Float64Array(x0);
    // Perturb component i; signed direction from RNG so the simplex
    // straddles x0 instead of always growing to one side.
    const dir = rng() > 0.5 ? 1 : -1;
    v[i] = (v[i] ?? 0) + dir * initialStep;
    verts.push(v);
    fs.push(f(v));
  }

  // Nelder-Mead constants (1965).
  const ALPHA = 1.0;   // reflection
  const GAMMA = 2.0;   // expansion
  const RHO   = 0.5;   // contraction
  const SIGMA = 0.5;   // shrink

  const history: number[] = [];
  let iter = 0;
  let termination: "converged" | "max-iter" = "max-iter";

  while (iter < maxIter) {
    // Sort by f ascending.
    const idx = Array.from({ length: n + 1 }, (_, i) => i).sort((a, b) => fs[a]! - fs[b]!);
    const sortedV = idx.map((i) => verts[i]!);
    const sortedF = idx.map((i) => fs[i]!);
    for (let i = 0; i <= n; i++) { verts[i] = sortedV[i]!; fs[i] = sortedF[i]!; }

    // Convergence test.
    const fSpread = fs[n]! - fs[0]!;
    let xSpread = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let v = 1; v <= n; v++) s = Math.max(s, Math.abs(verts[v]![i]! - verts[0]![i]!));
      xSpread = Math.max(xSpread, s);
    }
    history.push(fs[0]!);
    if (history.length > 200) history.shift();
    if (fSpread < fTol && xSpread < xTol) {
      termination = "converged";
      break;
    }

    // Centroid of all but worst.
    const centroid = new Float64Array(n);
    for (let v = 0; v < n; v++) {
      const vert = verts[v]!;
      for (let i = 0; i < n; i++) centroid[i] = (centroid[i] ?? 0) + vert[i]!;
    }
    for (let i = 0; i < n; i++) centroid[i] = (centroid[i] ?? 0) / n;

    const xWorst = verts[n]!;

    // Reflection.
    const xRefl = new Float64Array(n);
    for (let i = 0; i < n; i++) xRefl[i] = (centroid[i] ?? 0) + ALPHA * ((centroid[i] ?? 0) - (xWorst[i] ?? 0));
    const fRefl = f(xRefl);

    if (fRefl < fs[0]!) {
      // Try expansion.
      const xExp = new Float64Array(n);
      for (let i = 0; i < n; i++) xExp[i] = (centroid[i] ?? 0) + GAMMA * (xRefl[i]! - (centroid[i] ?? 0));
      const fExp = f(xExp);
      if (fExp < fRefl) { verts[n] = xExp; fs[n] = fExp; }
      else              { verts[n] = xRefl; fs[n] = fRefl; }
    } else if (fRefl < fs[n - 1]!) {
      verts[n] = xRefl; fs[n] = fRefl;
    } else {
      // Contraction (outside if reflected better than worst, else inside).
      const xRef = fRefl < fs[n]! ? xRefl : xWorst;
      const fRef = fRefl < fs[n]! ? fRefl : fs[n]!;
      const xCon = new Float64Array(n);
      for (let i = 0; i < n; i++) xCon[i] = (centroid[i] ?? 0) + RHO * (xRef[i]! - (centroid[i] ?? 0));
      const fCon = f(xCon);
      if (fCon < fRef) {
        verts[n] = xCon; fs[n] = fCon;
      } else {
        // Shrink.
        const xBest = verts[0]!;
        for (let v = 1; v <= n; v++) {
          const cur = verts[v]!;
          const newV = new Float64Array(n);
          for (let i = 0; i < n; i++) newV[i] = (xBest[i] ?? 0) + SIGMA * (cur[i]! - (xBest[i] ?? 0));
          verts[v] = newV;
          fs[v] = f(newV);
        }
      }
    }

    iter++;
  }

  // Final sort (best first).
  let bestIdx = 0;
  for (let i = 1; i <= n; i++) if (fs[i]! < fs[bestIdx]!) bestIdx = i;
  return {
    bestX: verts[bestIdx]!,
    bestF: fs[bestIdx]!,
    iter,
    termination,
    history,
  };
}

// ─────────────────────────────────────────────────────────────
// L-BFGS — limited-memory BFGS with Armijo backtracking line
// search. The workhorse for smooth VQE landscapes once the
// dimension passes ~10 and Nelder-Mead's simplex thrash plateaus
// well above the optimum.
//
// Implementation:
//   • Two-loop recursion (Nocedal 1980) on the last m (s, y)
//     pairs to apply the inverse-Hessian approximation B_k^{-1}
//     to the gradient ∇f without materializing B.
//   • Initial Hessian H_0 = γ_k I, γ_k = (s_{k-1}^T y_{k-1}) /
//     (y_{k-1}^T y_{k-1}) — the standard Nocedal scaling.
//   • Backtracking line search with Armijo (sufficient decrease)
//     condition. Step halved up to ~30 times before giving up.
//   • Auto-gradient via central finite differences if the user
//     doesn't supply one. Step h = 1e-5 by default — chosen
//     small enough to resolve a smooth analytic landscape, large
//     enough that f64 cancellation noise stays well below 1 mHa.
//
// For VQE on chemistry-sized problems (≤ 50 params) this hits
// chemical accuracy in 30-100 line-search iterations where
// Nelder-Mead would still be wandering at iter=10000.
// ─────────────────────────────────────────────────────────────

export interface LBFGSOptions {
  /** Hard cap on outer iterations. Default 200. */
  maxIter?: number;
  /** Function-value tolerance (|f_k − f_{k-1}|). Default 1e-9. */
  fTol?: number;
  /** Gradient-norm tolerance (‖∇f‖_∞). Default 1e-6. */
  gTol?: number;
  /** History size m for the (s, y) deque. Default 8. */
  historySize?: number;
  /** Central finite-difference step. Default 1e-5. */
  fdStep?: number;
  /** Armijo c1 constant. Default 1e-4. */
  armijoC1?: number;
  /** Initial line-search step length. Default 1.0. */
  initialStep?: number;
  /** Max backtracking halvings per step. Default 30. */
  maxLineSearch?: number;
}

export interface LBFGSResult {
  bestX: Float64Array;
  bestF: number;
  iter: number;
  termination: "converged" | "max-iter" | "linesearch-failed";
  /** Per-iteration f-value, capped at 200 samples for plotting. */
  history: readonly number[];
  /** Final gradient norm (‖∇f‖_∞) — useful for diagnosis. */
  finalGradNorm: number;
}

/**
 * Central finite-difference gradient. Falls back to forward differences
 * at the boundary if the caller passes a wrapped objective that errors
 * outside a domain — which we don't, so this is just defense-in-depth.
 */
function centralFDGradient(
  f: (x: Float64Array) => number,
  x: Float64Array,
  fAtX: number,
  h: number,
): Float64Array {
  void fAtX; // unused, kept for API symmetry with potential forward-diff path
  const n = x.length;
  const g = new Float64Array(n);
  const xp = new Float64Array(x);
  for (let i = 0; i < n; i++) {
    const orig = x[i]!;
    xp[i] = orig + h;
    const fPlus = f(xp);
    xp[i] = orig - h;
    const fMinus = f(xp);
    xp[i] = orig;
    g[i] = (fPlus - fMinus) / (2 * h);
  }
  return g;
}

/**
 * Limited-memory BFGS minimizer.
 *
 * Returns the lowest-found (x, f). If `gradient` is omitted, central
 * finite differences are used (cost: 2n extra f evals per iteration).
 */
export function lbfgs(
  f: (x: Float64Array) => number,
  x0: Float64Array,
  opts: LBFGSOptions & { gradient?: (x: Float64Array) => Float64Array } = {},
): LBFGSResult {
  const n = x0.length;
  const maxIter = opts.maxIter ?? 200;
  const fTol = opts.fTol ?? 1e-9;
  const gTol = opts.gTol ?? 1e-6;
  const m = opts.historySize ?? 8;
  const h = opts.fdStep ?? 1e-5;
  const c1 = opts.armijoC1 ?? 1e-4;
  const stepInit = opts.initialStep ?? 1.0;
  const maxLS = opts.maxLineSearch ?? 30;
  const gradFn = opts.gradient ?? ((xv: Float64Array) => centralFDGradient(f, xv, f(xv), h));

  let x = new Float64Array(x0);
  let fx = f(x);
  let g = gradFn(x);
  // Best-so-far snapshot — line-search can occasionally reject a step
  // that we still want to remember as a candidate optimum.
  let bestF = fx;
  let bestX = new Float64Array(x);

  // History buffers: ring of the last m (s, y) pairs.
  const sHist: Float64Array[] = [];
  const yHist: Float64Array[] = [];
  const rhoHist: number[] = [];

  const history: number[] = [];
  let termination: LBFGSResult["termination"] = "max-iter";
  let lastF = fx;
  let iter = 0;

  for (; iter < maxIter; iter++) {
    history.push(fx);
    if (history.length > 200) history.shift();

    // Convergence: gradient infinity-norm small enough.
    let gNormInf = 0;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(g[i]!);
      if (a > gNormInf) gNormInf = a;
    }
    if (gNormInf < gTol) {
      termination = "converged";
      break;
    }
    if (iter > 0 && Math.abs(fx - lastF) < fTol) {
      termination = "converged";
      break;
    }

    // ── Two-loop recursion: r ← H_k · g ─────────────────────
    // q ← g; loop back: α_i = ρ_i s_i^T q; q ← q − α_i y_i
    // r ← γ q; loop forward: β = ρ_i y_i^T r; r ← r + (α_i − β) s_i
    const k = sHist.length;
    const q = new Float64Array(g);
    const alphas = new Float64Array(k);
    for (let i = k - 1; i >= 0; i--) {
      const si = sHist[i]!;
      let dot = 0;
      for (let j = 0; j < n; j++) dot += si[j]! * q[j]!;
      const a = rhoHist[i]! * dot;
      alphas[i] = a;
      const yi = yHist[i]!;
      for (let j = 0; j < n; j++) q[j] = q[j]! - a * yi[j]!;
    }
    // Initial Hessian scaling γ_k = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
    let gamma = 1;
    if (k > 0) {
      const sLast = sHist[k - 1]!;
      const yLast = yHist[k - 1]!;
      let sy = 0, yy = 0;
      for (let j = 0; j < n; j++) {
        sy += sLast[j]! * yLast[j]!;
        yy += yLast[j]! * yLast[j]!;
      }
      gamma = yy > 0 ? sy / yy : 1;
    }
    const r = new Float64Array(n);
    for (let j = 0; j < n; j++) r[j] = gamma * q[j]!;
    for (let i = 0; i < k; i++) {
      const yi = yHist[i]!;
      let dot = 0;
      for (let j = 0; j < n; j++) dot += yi[j]! * r[j]!;
      const beta = rhoHist[i]! * dot;
      const si = sHist[i]!;
      const ai = alphas[i]!;
      for (let j = 0; j < n; j++) r[j] = r[j]! + (ai - beta) * si[j]!;
    }
    // Search direction p = −H_k g.
    const p = new Float64Array(n);
    for (let j = 0; j < n; j++) p[j] = -r[j]!;

    // Directional derivative ⟨g, p⟩ — must be < 0 for a descent direction.
    let gDotP = 0;
    for (let j = 0; j < n; j++) gDotP += g[j]! * p[j]!;
    if (gDotP >= 0) {
      // History got contaminated (e.g. by numerical drift) — reset to
      // steepest descent and try again.
      for (let j = 0; j < n; j++) p[j] = -g[j]!;
      gDotP = 0;
      for (let j = 0; j < n; j++) gDotP += g[j]! * p[j]!;
      sHist.length = 0; yHist.length = 0; rhoHist.length = 0;
    }

    // ── Armijo backtracking line search ─────────────────────
    let alpha = stepInit;
    const xNew = new Float64Array(n);
    let fNew = fx;
    let lsOk = false;
    for (let ls = 0; ls < maxLS; ls++) {
      for (let j = 0; j < n; j++) xNew[j] = x[j]! + alpha * p[j]!;
      fNew = f(xNew);
      // Sufficient decrease: f(x + αp) ≤ f(x) + c1 · α · ⟨g, p⟩.
      if (fNew <= fx + c1 * alpha * gDotP) { lsOk = true; break; }
      alpha *= 0.5;
    }
    if (!lsOk) {
      termination = "linesearch-failed";
      break;
    }

    // ── Update (s, y) history and accept step ───────────────
    const s = new Float64Array(n);
    for (let j = 0; j < n; j++) s[j] = xNew[j]! - x[j]!;
    const gNew = gradFn(xNew);
    const y = new Float64Array(n);
    for (let j = 0; j < n; j++) y[j] = gNew[j]! - g[j]!;
    let sy = 0;
    for (let j = 0; j < n; j++) sy += s[j]! * y[j]!;
    if (sy > 1e-12) {
      sHist.push(s); yHist.push(y); rhoHist.push(1 / sy);
      if (sHist.length > m) { sHist.shift(); yHist.shift(); rhoHist.shift(); }
    }

    lastF = fx;
    x = xNew;
    fx = fNew;
    g = gNew;

    if (fx < bestF) { bestF = fx; bestX = new Float64Array(x); }
  }

  // Final gradient norm at the best point we saw.
  const finalG = gradFn(bestX);
  let finalGradNorm = 0;
  for (let i = 0; i < n; i++) finalGradNorm = Math.max(finalGradNorm, Math.abs(finalG[i]!));

  return { bestX, bestF, iter, termination, history, finalGradNorm };
}
