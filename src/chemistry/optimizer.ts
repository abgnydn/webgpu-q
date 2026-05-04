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
