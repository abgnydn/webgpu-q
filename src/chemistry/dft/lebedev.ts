// ─────────────────────────────────────────────────────────────
// lebedev.ts — Lebedev-Laikov spherical quadrature points.
//
// Lebedev grids are octahedral-symmetric quadrature rules on the
// unit sphere that are exact for spherical harmonics up to a given
// algebraic order. They reach the same accuracy as a product
// θ × φ grid with substantially fewer points:
//   • Lebedev   50 ≈ Gauss-Legendre × uniform 12 × 24 ( =288 pts)
//     in algebraic exactness (both up to L = 11).
//   • Lebedev  110 → exact for L ≤ 17; ~2.6× fewer points than
//     288 with strictly better accuracy. Default for chemistry.
//   • Lebedev  302 → exact for L ≤ 29; for very-tight DFT runs.
//
// Construction: Lebedev (1976, 1995) and Lebedev-Laikov (1999)
// give the rules as orbits of the octahedral group Oh acting on
// generator points (a, b, c). Each orbit is one of 6 symmetry
// classes and contributes 6, 12, 8, 24, 24, or 48 points with a
// shared weight v. We tabulate (code, a, b, v) tuples per order
// and expand them via `genOh` below — this matches the original
// Fortran routine in Lebedev-Laikov.F (Christoph van Wuellen
// translation of Dmitri Laikov's C code; included verbatim in
// PySCF's `dft/LebedevGrid.py`).
//
// Weights in the tables sum to 1; `lebedev(order)` rescales them
// so weights sum to 4π — matching the Gauss-Legendre × uniform-φ
// path used by the rest of `grid.ts` (which has wTh·wPhi summing
// to 2 · 2π = 4π).
// ─────────────────────────────────────────────────────────────

/** A single Lebedev orbit specification. */
interface LebOrbit {
  /** Symmetry class: 0..5 — see comments in `genOh`. */
  readonly code: 0 | 1 | 2 | 3 | 4 | 5;
  /** First generator parameter (radians-free, on the unit sphere). */
  readonly a: number;
  /** Second generator parameter (only used by code = 5 for (a,b,c)). */
  readonly b: number;
  /** Shared quadrature weight for every point in this orbit (sum-to-1
   *  normalization; rescaled to 4π by `lebedev`). */
  readonly v: number;
}

/**
 * Expand a Lebedev orbit into its full set of points under the
 * octahedral group Oh. Mirrors `SphGenOh` in PySCF.
 *
 *   code 0:  (1, 0, 0) and permutations               →  6 points
 *   code 1:  (0, a, a) with a = 1/√2                  → 12 points
 *   code 2:  (a, a, a) with a = 1/√3                  →  8 points
 *   code 3:  (a, a, b), b = √(1 − 2a²)                → 24 points
 *   code 4:  (a, b, 0), b = √(1 − a²)                 → 24 points
 *   code 5:  (a, b, c), c = √(1 − a² − b²)            → 48 points
 */
function genOh(code: number, aIn: number, bIn: number, v: number): number[] {
  const out: number[] = [];
  const push = (x: number, y: number, z: number): void => { out.push(x, y, z, v); };
  if (code === 0) {
    const a = 1.0;
    push( a, 0, 0); push(-a, 0, 0);
    push( 0, a, 0); push( 0,-a, 0);
    push( 0, 0, a); push( 0, 0,-a);
  } else if (code === 1) {
    const a = Math.SQRT1_2;
    push( 0, a, a); push( 0,-a, a); push( 0, a,-a); push( 0,-a,-a);
    push( a, 0, a); push(-a, 0, a); push( a, 0,-a); push(-a, 0,-a);
    push( a, a, 0); push(-a, a, 0); push( a,-a, 0); push(-a,-a, 0);
  } else if (code === 2) {
    const a = Math.sqrt(1 / 3);
    push( a, a, a); push(-a, a, a); push( a,-a, a); push(-a,-a, a);
    push( a, a,-a); push(-a, a,-a); push( a,-a,-a); push(-a,-a,-a);
  } else if (code === 3) {
    const a = aIn;
    const b = Math.sqrt(1 - 2 * a * a);
    // (a, a, ±b) and 3-axis permutations × 8 sign patterns = 24.
    for (const sx of [1, -1]) for (const sy of [1, -1]) for (const sz of [1, -1]) {
      push(sx * a, sy * a, sz * b);
      push(sx * a, sy * b, sz * a);
      push(sx * b, sy * a, sz * a);
    }
  } else if (code === 4) {
    const a = aIn;
    const b = Math.sqrt(1 - a * a);
    for (const sx of [1, -1]) for (const sy of [1, -1]) {
      push(sx * a, sy * b, 0);
      push(sx * b, sy * a, 0);
      push(sx * a, 0, sy * b);
      push(sx * b, 0, sy * a);
      push(0, sx * a, sy * b);
      push(0, sx * b, sy * a);
    }
  } else if (code === 5) {
    const a = aIn, b = bIn;
    const c = Math.sqrt(1 - a * a - b * b);
    for (const sx of [1, -1]) for (const sy of [1, -1]) for (const sz of [1, -1]) {
      // Six independent permutations of (a, b, c) under sign assignments.
      push(sx * a, sy * b, sz * c);
      push(sx * a, sy * c, sz * b);
      push(sx * b, sy * a, sz * c);
      push(sx * b, sy * c, sz * a);
      push(sx * c, sy * a, sz * b);
      push(sx * c, sy * b, sz * a);
    }
  } else {
    throw new Error(`genOh: unknown code ${code}`);
  }
  return out;
}

// ── Tabulated orbit specs (from Lebedev-Laikov.F via PySCF). ──
// Weights are sum-to-1 normalized; `lebedev` rescales to 4π.

/** Lebedev order 50 — exact for spherical harmonics up to L = 11. */
const LEB_50: readonly LebOrbit[] = [
  { code: 0, a: 0,                         b: 0, v: 0.1269841269841270e-1 },
  { code: 1, a: 0,                         b: 0, v: 0.2257495590828924e-1 },
  { code: 2, a: 0,                         b: 0, v: 0.2109375000000000e-1 },
  { code: 3, a: 0.3015113445777636e+0,     b: 0, v: 0.2017333553791887e-1 },
];

/** Lebedev order 110 — exact for L ≤ 17. Default chemistry choice. */
const LEB_110: readonly LebOrbit[] = [
  { code: 0, a: 0,                         b: 0, v: 0.3828270494937162e-2 },
  { code: 2, a: 0,                         b: 0, v: 0.9793737512487512e-2 },
  { code: 3, a: 0.1851156353447362e+0,     b: 0, v: 0.8211737283191111e-2 },
  { code: 3, a: 0.6904210483822922e+0,     b: 0, v: 0.9942814891178103e-2 },
  { code: 3, a: 0.3956894730559419e+0,     b: 0, v: 0.9595471336070963e-2 },
  { code: 4, a: 0.4783690288121502e+0,     b: 0, v: 0.9694996361663028e-2 },
];

/** Lebedev order 302 — exact for L ≤ 29. Tight-DFT option. */
const LEB_302: readonly LebOrbit[] = [
  { code: 0, a: 0,                         b: 0,                         v: 0.8545911725128148e-3 },
  { code: 2, a: 0,                         b: 0,                         v: 0.3599119285025571e-2 },
  { code: 3, a: 0.3515640345570105e+0,     b: 0,                         v: 0.3449788424305883e-2 },
  { code: 3, a: 0.6566329410219612e+0,     b: 0,                         v: 0.3604822601419882e-2 },
  { code: 3, a: 0.4729054132581005e+0,     b: 0,                         v: 0.3576729661743367e-2 },
  { code: 3, a: 0.9618308522614784e-1,     b: 0,                         v: 0.2352101413689164e-2 },
  { code: 3, a: 0.2219645236294178e+0,     b: 0,                         v: 0.3108953122413675e-2 },
  { code: 3, a: 0.7011766416089545e+0,     b: 0,                         v: 0.3650045807677255e-2 },
  { code: 4, a: 0.2644152887060663e+0,     b: 0,                         v: 0.2982344963171804e-2 },
  { code: 4, a: 0.5718955891878961e+0,     b: 0,                         v: 0.3600820932216460e-2 },
  { code: 5, a: 0.2510034751770465e+0,     b: 0.8000727494073952e+0,     v: 0.3571540554273387e-2 },
  { code: 5, a: 0.1233548532583327e+0,     b: 0.4127724083168531e+0,     v: 0.3392312205006170e-2 },
];

const LEBEDEV_TABLES: Record<number, readonly LebOrbit[]> = {
  50:  LEB_50,
  110: LEB_110,
  302: LEB_302,
};

/** Supported Lebedev orders. */
export const LEBEDEV_AVAILABLE_ORDERS = [50, 110, 302] as const;
export type LebedevOrder = (typeof LEBEDEV_AVAILABLE_ORDERS)[number];

/** A flat Lebedev grid: x, y, z on the unit sphere with weights summing to 4π. */
export interface LebedevGrid {
  /** Cartesian unit-sphere coordinates. */
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly z: Float64Array;
  /** Quadrature weights (sum to 4π — full-sphere measure). */
  readonly w: Float64Array;
}

/**
 * Build a Lebedev grid for the requested order. Throws if the
 * order isn't tabulated.
 */
export function lebedev(order: LebedevOrder): LebedevGrid {
  const orbits = LEBEDEV_TABLES[order];
  if (!orbits) {
    throw new Error(
      `lebedev: order ${order} not tabulated. Available: ${LEBEDEV_AVAILABLE_ORDERS.join(", ")}.`,
    );
  }
  // Expand all orbits.
  const flat: number[] = [];
  for (const o of orbits) flat.push(...genOh(o.code, o.a, o.b, o.v));
  const nPoints = flat.length / 4;
  const x = new Float64Array(nPoints);
  const y = new Float64Array(nPoints);
  const z = new Float64Array(nPoints);
  const w = new Float64Array(nPoints);
  // Tabulated weights sum to 1; rescale to 4π so ∫ Y_00 dΩ = √(4π).
  const fourPi = 4 * Math.PI;
  for (let i = 0; i < nPoints; i++) {
    x[i] = flat[i * 4 + 0]!;
    y[i] = flat[i * 4 + 1]!;
    z[i] = flat[i * 4 + 2]!;
    w[i] = flat[i * 4 + 3]! * fourPi;
  }
  // Cross-check that the order matched the expected size.
  const expected = order;
  if (nPoints !== expected) {
    throw new Error(`lebedev: order ${order} expanded to ${nPoints} points (expected ${expected})`);
  }
  return { x, y, z, w };
}
