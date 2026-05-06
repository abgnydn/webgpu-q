// ─────────────────────────────────────────────────────────────
// integrals-cg.ts — Cartesian-Gaussian integrals over s and p
// shells via the McMurchie-Davidson (1978) Hermite-Gaussian
// expansion. Phase-C v3 stage 1 — adds Be 2p (and any other
// p-shell) support to the molecular-integral pipeline.
//
// Convention: a Cartesian Gaussian primitive at center A with
// exponent α and angular momentum tuple I = (i_x, i_y, i_z) is
//     G_I(r; A, α) = (x − A_x)^{i_x} (y − A_y)^{i_y} (z − A_z)^{i_z}
//                    · exp(−α |r − A|²)
// L = i_x + i_y + i_z is the total angular momentum (0 = s,
// 1 = p, etc.). A "Cartesian-Gaussian shell" (CGShell) bundles a
// contraction (alpha[], c[]), a center, and the angular tuple.
//
// All integrals reduce to one of:
//   • S(I, J)        = ⟨G_I(A) | G_J(B)⟩
//   • T(I, J)        = ⟨G_I(A) | −∇²/2 | G_J(B)⟩
//   • V(I, J; Z, C)  = ⟨G_I(A) | −Z/|r − C| | G_J(B)⟩
//   • ERI(I, J, K, L) = ∫∫ G_I(A) G_J(B) (1/r12) G_K(C) G_L(D)
//
// Each is built from the E-coefficient recurrence (1D Hermite
// expansion of a 1D Gaussian product) plus the Boys function.
//
// All primitive normalization is rolled into the contracted
// shell calls below (`S_shell_cg`, etc.), so callers pass the
// raw Pople d-coefficients as in `integrals.ts`.
// ─────────────────────────────────────────────────────────────

const SQRT_PI = Math.sqrt(Math.PI);

/** A Cartesian-Gaussian shell: contraction × center × angular momentum. */
export interface CGShell {
  readonly center: readonly [number, number, number];
  readonly alpha: readonly number[];
  readonly c: readonly number[];
  /** (i_x, i_y, i_z). For STO-3G we only need (0,0,0)=s and (1,0,0)/(0,1,0)/(0,0,1)=p. */
  readonly angular: readonly [number, number, number];
  readonly label?: string;
}

// ── Primitive normalization for a Cartesian-Gaussian ──────────
// N(α, I) = (2α/π)^(3/4) · √[(2α)^L / ((2i_x − 1)!! (2i_y − 1)!! (2i_z − 1)!!)]
// where (−1)!! = 1, 1!! = 1, 3!! = 3, etc.

function doubleFactOdd(n: number): number {
  // (2n − 1)!! for n = 0, 1, 2, 3, …
  // n=0 → 1, n=1 → 1, n=2 → 3, n=3 → 15, n=4 → 105
  if (n <= 0) return 1;
  let r = 1;
  for (let k = 1; k <= n; k++) r *= 2 * k - 1;
  return r;
}

/** Per-primitive normalization factor for a Cartesian Gaussian. */
function normCG(alpha: number, I: readonly [number, number, number]): number {
  const L = I[0] + I[1] + I[2];
  const dfx = doubleFactOdd(I[0]);
  const dfy = doubleFactOdd(I[1]);
  const dfz = doubleFactOdd(I[2]);
  const radial = Math.pow(2 * alpha / Math.PI, 0.75);
  // Angular part: √[(4α)^L / (df_x · df_y · df_z)]
  const angular = Math.sqrt(Math.pow(4 * alpha, L) / (dfx * dfy * dfz));
  return radial * angular;
}

// ── McMurchie-Davidson 1D E-coefficients ──────────────────────
//
// Expand (x − A_x)^i (x − B_x)^j exp(−α(x−A_x)²) exp(−β(x−B_x)²)
// as Σ_t E^{ij}_t · Λ_t(x; P_x, p), where Λ_t is the t-th Hermite
// Gaussian centered at the product center P_x = (αA_x + βB_x)/p.
//
// Recurrence (HJO eq. 9.5.6–9.5.8):
//   E(0, 0, 0) = K_AB := exp(−μ (A_x − B_x)²)
//   E(0, 0, t) = 0   for t ≠ 0
//   Increase j: E(i, j+1, t) = (1/(2p)) E(i, j, t−1)
//                            + (P_x − B_x) E(i, j, t)
//                            + (t+1) E(i, j, t+1)
//   Increase i: E(i+1, j, t) = (1/(2p)) E(i, j, t−1)
//                            + (P_x − A_x) E(i, j, t)
//                            + (t+1) E(i, j, t+1)
//
// Returned as a flat Float64Array of length (i+1)·(j+1)·(i+j+1)
// indexed by row-major [a, b, t].

function eCoefTable(
  iMax: number, jMax: number,
  pa: number, pb: number, p: number, K: number,
): Float64Array {
  // Flat storage: e[(a · (jMax+1) + b) * (iMax+jMax+1) + t]
  const tDim = iMax + jMax + 1;
  const tab = new Float64Array((iMax + 1) * (jMax + 1) * tDim);
  const idx = (a: number, b: number, t: number): number =>
    (a * (jMax + 1) + b) * tDim + t;
  tab[idx(0, 0, 0)] = K;

  const inv2p = 1 / (2 * p);

  // First grow j with i = 0.
  for (let b = 0; b < jMax; b++) {
    for (let t = 0; t <= b + 1; t++) {
      const left  = t > 0       ? tab[idx(0, b, t - 1)]! : 0;
      const mid   = t <= b      ? tab[idx(0, b, t)]!     : 0;
      const right = t + 1 <= b  ? tab[idx(0, b, t + 1)]! : 0;
      tab[idx(0, b + 1, t)] = inv2p * left + pb * mid + (t + 1) * right;
    }
  }
  // Then grow i for each fixed b.
  for (let a = 0; a < iMax; a++) {
    for (let b = 0; b <= jMax; b++) {
      for (let t = 0; t <= a + b + 1; t++) {
        const left  = t > 0           ? tab[idx(a, b, t - 1)]! : 0;
        const mid   = t <= a + b      ? tab[idx(a, b, t)]!     : 0;
        const right = t + 1 <= a + b  ? tab[idx(a, b, t + 1)]! : 0;
        tab[idx(a + 1, b, t)] = inv2p * left + pa * mid + (t + 1) * right;
      }
    }
  }
  return tab;
}

// ── Boys function F_n(t) ──────────────────────────────────────
//
// F_n(t) = ∫₀¹ s^(2n) exp(−t s²) ds.
//
// Strategy (stable across all t we see, t ∈ [0, ~50]):
//   • F_0 always from the closed form
//       F_0(t) = ½ √(π/t) · erf(√t),
//     using the Abramowitz-Stegun 7.1.26 erf (~1.5e-7 max error).
//     This matches the legacy s-only path's precision exactly.
//   • For n ≥ 1:
//       — t > 1: upward recurrence
//           F_n(t) = ((2n − 1) F_{n−1}(t) − e^{−t}) / (2t).
//         Stable because each step *divides* by 2t > 2.
//       — t ≤ 1: Taylor series Σ_k (−t)^k / (k!·(2n + 2k + 1)).
//         Terms decay fast; no cancellation.
//
// Why this matters: a naive Taylor for F_n at moderate t (say
// t = 22) accumulates intermediate sums of order e^t ≈ 10^9 that
// must cancel down to F_0 ≈ 0.19 — f64 leaves only ~10^-7 absolute
// precision in F_0, which the LiH cross-shell matrix elements
// translate into ~mHa errors in the final FCI energy.

const SQRT_PI_BOYS = Math.sqrt(Math.PI);

/** Abramowitz & Stegun 7.1.26 — same erf the legacy s-only path uses. */
function erfApprox(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;
  const tt = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * Math.exp(-ax * ax);
  return sign * y;
}

function boys0(t: number): number {
  if (t < 1e-12) return 1 - t / 3 + (t * t) / 10;
  const s = Math.sqrt(t);
  return 0.5 * SQRT_PI_BOYS / s * erfApprox(s);
}

function boysNTaylor(n: number, t: number): number {
  // Σ_k (−t)^k / (k! · (2n + 2k + 1)). Stable for t ≤ ~10 even at high
  // n. Beyond that the peak term magnitude grows like e^t / √(2πt) and
  // f64 cancellation limits relative precision.
  let sum = 0;
  let term = 1 / (2 * n + 1);
  // Cap iterations at 200 — peak occurs around k ≈ t, terms decay
  // rapidly past that. 200 is more than enough up to t ~ 30.
  for (let k = 0; k < 200; k++) {
    sum += term;
    term *= -t / (k + 1) * (2 * n + 2 * k + 1) / (2 * n + 2 * k + 3);
    if (Math.abs(term) < 1e-18 * Math.abs(sum)) break;
  }
  return sum;
}

/**
 * Compute F_0(t), F_1(t), …, F_nMax(t) and return them in a length-(nMax+1)
 * Float64Array. Stable across all (nMax, t) we use, including high-L
 * basis sets like cc-pVTZ where nMax can reach 12.
 *
 * Strategy:
 *   • t < 1e-12: limit value F_n(0) = 1/(2n+1).
 *   • t < (2 nMax − 1) / 2: Taylor for each F_n directly. The series
 *     Σ (−t)^k / (k! · (2n + 2k + 1)) converges in ~2t terms with peak
 *     magnitude ~e^t/√(2πt). f64 handles up to ~10 orders of
 *     cancellation, so this branch covers t up to ~10–12.
 *   • t ≥ (2 nMax − 1) / 2: upward recurrence from F_0(t). The error
 *     growth per step is (2n−1)/(2t) ≤ 1 over the whole n range, so
 *     the closed-form F_0 precision (≈ 1.5e-7) is preserved.
 *
 * The threshold (2·nMax − 1)/2 is exactly where upward recurrence stops
 * amplifying errors: it's the point at which (2n − 1)/(2t) = 1 at the
 * worst-case n = nMax. Below the threshold, Taylor's well-conditioned
 * series wins; above it, upward's closed-form anchor wins.
 */
export function boysAll(nMax: number, t: number): Float64Array {
  const out = new Float64Array(nMax + 1);
  if (t < 1e-12) {
    for (let n = 0; n <= nMax; n++) out[n] = 1 / (2 * n + 1);
    return out;
  }

  const tUpwardThreshold = (2 * nMax - 1) / 2;

  if (t < tUpwardThreshold) {
    for (let n = 0; n <= nMax; n++) out[n] = boysNTaylor(n, t);
    return out;
  }

  out[0] = boys0(t);
  if (nMax === 0) return out;

  const expMt = Math.exp(-t);
  for (let n = 1; n <= nMax; n++) {
    out[n] = ((2 * n - 1) * out[n - 1]! - expMt) / (2 * t);
  }
  return out;
}

// ── McMurchie-Davidson auxiliary R integrals ─────────────────
//
// R_{tuv}^n(p, R) is the auxiliary integral that appears in the
// nuclear-attraction and ERI evaluations. Recurrence
// (HJO eq. 9.9.18):
//   R_{0,0,0}^n(p, R) = (−2p)^n · F_n(p R²)
//   R_{t+1, u, v}^n = t · R_{t−1, u, v}^{n+1} + R_x · R_{t, u, v}^{n+1}
//   R_{t, u+1, v}^n = u · R_{t, u−1, v}^{n+1} + R_y · R_{t, u, v}^{n+1}
//   R_{t, u, v+1}^n = v · R_{t, u, v−1}^{n+1} + R_z · R_{t, u, v}^{n+1}
//
// We only ever need R^0_{tuv} for the final answer; the recursion
// proceeds by lowering n while building up the t/u/v indices.

/**
 * Return an array R[t][u][v] of length (tMax+1)·(uMax+1)·(vMax+1),
 * holding R^0_{tuv}(p, R), where R = (Rx, Ry, Rz).
 */
function rAuxTable(
  tMax: number, uMax: number, vMax: number,
  p: number,
  Rx: number, Ry: number, Rz: number,
): Float64Array {
  const nMax = tMax + uMax + vMax;
  const F = boysAll(nMax, p * (Rx * Rx + Ry * Ry + Rz * Rz));
  // We need R^n_{tuv} for all n, t, u, v in their respective ranges.
  // Storage: r[n * Tdim * Udim * Vdim + t * Udim * Vdim + u * Vdim + v]
  const Tdim = tMax + 1, Udim = uMax + 1, Vdim = vMax + 1;
  const r = new Float64Array((nMax + 1) * Tdim * Udim * Vdim);
  const idx = (n: number, t: number, u: number, v: number): number =>
    ((n * Tdim + t) * Udim + u) * Vdim + v;

  // Seed: R^n_{000} = (−2p)^n F_n.
  for (let n = 0; n <= nMax; n++) {
    r[idx(n, 0, 0, 0)] = Math.pow(-2 * p, n) * F[n]!;
  }
  // Recurrence: walk down n while extending t.
  // Loop order chosen so the values we read are always already filled.
  for (let n = nMax - 1; n >= 0; n--) {
    for (let t = 0; t <= tMax; t++) {
      for (let u = 0; u <= uMax; u++) {
        for (let v = 0; v <= vMax; v++) {
          if (t === 0 && u === 0 && v === 0) continue;
          let acc = 0;
          if (v > 0) {
            acc += Rz * r[idx(n + 1, t, u, v - 1)]!;
            if (v >= 2) acc += (v - 1) * r[idx(n + 1, t, u, v - 2)]!;
          } else if (u > 0) {
            acc += Ry * r[idx(n + 1, t, u - 1, v)]!;
            if (u >= 2) acc += (u - 1) * r[idx(n + 1, t, u - 2, v)]!;
          } else {  // t > 0
            acc += Rx * r[idx(n + 1, t - 1, u, v)]!;
            if (t >= 2) acc += (t - 1) * r[idx(n + 1, t - 2, u, v)]!;
          }
          r[idx(n, t, u, v)] = acc;
        }
      }
    }
  }

  // Slice out only the n=0 layer.
  const out = new Float64Array(Tdim * Udim * Vdim);
  for (let t = 0; t <= tMax; t++) {
    for (let u = 0; u <= uMax; u++) {
      for (let v = 0; v <= vMax; v++) {
        out[t * Udim * Vdim + u * Vdim + v] = r[idx(0, t, u, v)]!;
      }
    }
  }
  return out;
}

// ── Primitive overlap, kinetic, V, ERI ───────────────────────

interface PairData {
  /** Combined exponent p = α + β. */
  p: number;
  /** Reduced exponent μ = αβ/(α+β). */
  mu: number;
  /** Product center (P_x, P_y, P_z). */
  P: readonly [number, number, number];
  /** Per-axis 1D Gaussian product factor K^x = exp(−μ (A_x − B_x)²). */
  Kx: number;
  Ky: number;
  Kz: number;
  /** 1D E-coefficient tables (one per axis). */
  Ex: Float64Array;
  Ey: Float64Array;
  Ez: Float64Array;
  /** Max angular momentum used to build E (for index decoding). */
  jMaxX: number; jMaxY: number; jMaxZ: number;
  iMaxX: number; iMaxY: number; iMaxZ: number;
}

function buildPair(
  alpha: number, A: readonly [number, number, number], I: readonly [number, number, number],
  beta: number, B: readonly [number, number, number], J: readonly [number, number, number],
  iExtra = 0, jExtra = 0,
): PairData {
  const p = alpha + beta;
  const mu = alpha * beta / p;
  const Px = (alpha * A[0] + beta * B[0]) / p;
  const Py = (alpha * A[1] + beta * B[1]) / p;
  const Pz = (alpha * A[2] + beta * B[2]) / p;
  const Kx = Math.exp(-mu * (A[0] - B[0]) * (A[0] - B[0]));
  const Ky = Math.exp(-mu * (A[1] - B[1]) * (A[1] - B[1]));
  const Kz = Math.exp(-mu * (A[2] - B[2]) * (A[2] - B[2]));
  const iMaxX = I[0] + iExtra, jMaxX = J[0] + jExtra;
  const iMaxY = I[1] + iExtra, jMaxY = J[1] + jExtra;
  const iMaxZ = I[2] + iExtra, jMaxZ = J[2] + jExtra;
  return {
    p, mu, P: [Px, Py, Pz],
    Kx, Ky, Kz,
    Ex: eCoefTable(iMaxX, jMaxX, Px - A[0], Px - B[0], p, Kx),
    Ey: eCoefTable(iMaxY, jMaxY, Py - A[1], Py - B[1], p, Ky),
    Ez: eCoefTable(iMaxZ, jMaxZ, Pz - A[2], Pz - B[2], p, Kz),
    iMaxX, jMaxX, iMaxY, jMaxY, iMaxZ, jMaxZ,
  };
}

function eAt(tab: Float64Array, jMax: number, iMax: number, i: number, j: number, t: number): number {
  const tDim = iMax + jMax + 1;
  return tab[(i * (jMax + 1) + j) * tDim + t]!;
}

/** Primitive overlap ⟨G_I(A, α) | G_J(B, β)⟩. */
function primOverlap(
  alpha: number, A: readonly [number, number, number], I: readonly [number, number, number],
  beta: number, B: readonly [number, number, number], J: readonly [number, number, number],
): number {
  const pd = buildPair(alpha, A, I, beta, B, J);
  const ex = eAt(pd.Ex, pd.jMaxX, pd.iMaxX, I[0], J[0], 0);
  const ey = eAt(pd.Ey, pd.jMaxY, pd.iMaxY, I[1], J[1], 0);
  const ez = eAt(pd.Ez, pd.jMaxZ, pd.iMaxZ, I[2], J[2], 0);
  return ex * ey * ez * Math.pow(Math.PI / pd.p, 1.5);
}

/**
 * Primitive kinetic ⟨G_I(A, α) | −∇²/2 | G_J(B, β)⟩.
 *
 * Uses ∂²/∂B_k² G_J = 4β² G_{J+2_k} − 2β(2J_k + 1) G_J  +  J_k(J_k − 1) G_{J−2_k}
 * so T(I, J) = −½ Σ_k [4β² S(I, J+2_k) − 2β(2J_k + 1) S(I, J) + J_k(J_k − 1) S(I, J−2_k)].
 */
function primKinetic(
  alpha: number, A: readonly [number, number, number], I: readonly [number, number, number],
  beta: number, B: readonly [number, number, number], J: readonly [number, number, number],
): number {
  // We need overlaps at J, J ± 2_k. Build a single E-coefficient table
  // with j-extra = 2 so we can read every offset in O(1).
  const pd = buildPair(alpha, A, I, beta, B, J, 0, 2);
  const sFactor = Math.pow(Math.PI / pd.p, 1.5);
  const sBase = (Jx: number, Jy: number, Jz: number): number =>
    eAt(pd.Ex, pd.jMaxX, pd.iMaxX, I[0], Jx, 0)
    * eAt(pd.Ey, pd.jMaxY, pd.iMaxY, I[1], Jy, 0)
    * eAt(pd.Ez, pd.jMaxZ, pd.iMaxZ, I[2], Jz, 0)
    * sFactor;

  const J0 = sBase(J[0], J[1], J[2]);
  let acc = 0;
  for (let k = 0; k < 3; k++) {
    const Jk = J[k]!;
    const Jp2 = [J[0], J[1], J[2]];
    Jp2[k] = Jk + 2;
    const Jm2 = [J[0], J[1], J[2]];
    Jm2[k] = Jk - 2;
    const sP2 = sBase(Jp2[0]!, Jp2[1]!, Jp2[2]!);
    const sM2 = Jk >= 2 ? sBase(Jm2[0]!, Jm2[1]!, Jm2[2]!) : 0;
    acc += 4 * beta * beta * sP2 - 2 * beta * (2 * Jk + 1) * J0 + Jk * (Jk - 1) * sM2;
  }
  return -0.5 * acc;
}

/** Primitive nuclear attraction ⟨G_I | −Z/|r − C| | G_J⟩. */
function primNuclear(
  alpha: number, A: readonly [number, number, number], I: readonly [number, number, number],
  beta: number, B: readonly [number, number, number], J: readonly [number, number, number],
  Z: number, C: readonly [number, number, number],
): number {
  const pd = buildPair(alpha, A, I, beta, B, J);
  const tMax = I[0] + J[0];
  const uMax = I[1] + J[1];
  const vMax = I[2] + J[2];
  const Rt = rAuxTable(tMax, uMax, vMax, pd.p, pd.P[0] - C[0], pd.P[1] - C[1], pd.P[2] - C[2]);
  const Udim = uMax + 1, Vdim = vMax + 1;
  let sum = 0;
  for (let t = 0; t <= tMax; t++) {
    const ex = eAt(pd.Ex, pd.jMaxX, pd.iMaxX, I[0], J[0], t);
    if (ex === 0) continue;
    for (let u = 0; u <= uMax; u++) {
      const ey = eAt(pd.Ey, pd.jMaxY, pd.iMaxY, I[1], J[1], u);
      if (ey === 0) continue;
      for (let v = 0; v <= vMax; v++) {
        const ez = eAt(pd.Ez, pd.jMaxZ, pd.iMaxZ, I[2], J[2], v);
        if (ez === 0) continue;
        sum += ex * ey * ez * Rt[t * Udim * Vdim + u * Vdim + v]!;
      }
    }
  }
  return -Z * (2 * Math.PI / pd.p) * sum;
}

/** Primitive ERI (G_IA G_JB | G_KC G_LD). */
function primERI(
  alpha: number, A: readonly [number, number, number], I: readonly [number, number, number],
  beta: number, B: readonly [number, number, number], J: readonly [number, number, number],
  gamma: number, C: readonly [number, number, number], K: readonly [number, number, number],
  delta: number, D: readonly [number, number, number], L: readonly [number, number, number],
): number {
  const pd1 = buildPair(alpha, A, I, beta, B, J);
  const pd2 = buildPair(gamma, C, K, delta, D, L);
  const p = pd1.p, q = pd2.p;
  const alphaPair = (p * q) / (p + q);
  const Px = pd1.P[0], Py = pd1.P[1], Pz = pd1.P[2];
  const Qx = pd2.P[0], Qy = pd2.P[1], Qz = pd2.P[2];

  const tMax = I[0] + J[0], uMax = I[1] + J[1], vMax = I[2] + J[2];
  const tauMax = K[0] + L[0], nuMax = K[1] + L[1], phiMax = K[2] + L[2];
  const tDimSum = tMax + tauMax;
  const uDimSum = uMax + nuMax;
  const vDimSum = vMax + phiMax;
  // R aux for the *combined* Hermite indices: we need R^0_{t+τ, u+ν, v+φ}.
  const Rt = rAuxTable(tDimSum, uDimSum, vDimSum, alphaPair, Px - Qx, Py - Qy, Pz - Qz);
  const UdimR = uDimSum + 1, VdimR = vDimSum + 1;

  let sum = 0;
  for (let t = 0; t <= tMax; t++) {
    const ex1 = eAt(pd1.Ex, pd1.jMaxX, pd1.iMaxX, I[0], J[0], t);
    if (ex1 === 0) continue;
    for (let u = 0; u <= uMax; u++) {
      const ey1 = eAt(pd1.Ey, pd1.jMaxY, pd1.iMaxY, I[1], J[1], u);
      if (ey1 === 0) continue;
      for (let v = 0; v <= vMax; v++) {
        const ez1 = eAt(pd1.Ez, pd1.jMaxZ, pd1.iMaxZ, I[2], J[2], v);
        if (ez1 === 0) continue;
        for (let tau = 0; tau <= tauMax; tau++) {
          const ex2 = eAt(pd2.Ex, pd2.jMaxX, pd2.iMaxX, K[0], L[0], tau);
          if (ex2 === 0) continue;
          for (let nu = 0; nu <= nuMax; nu++) {
            const ey2 = eAt(pd2.Ey, pd2.jMaxY, pd2.iMaxY, K[1], L[1], nu);
            if (ey2 === 0) continue;
            for (let phi = 0; phi <= phiMax; phi++) {
              const ez2 = eAt(pd2.Ez, pd2.jMaxZ, pd2.iMaxZ, K[2], L[2], phi);
              if (ez2 === 0) continue;
              const sign = ((tau + nu + phi) & 1) ? -1 : 1;
              const r = Rt[(t + tau) * UdimR * VdimR + (u + nu) * VdimR + (v + phi)]!;
              sum += ex1 * ey1 * ez1 * ex2 * ey2 * ez2 * sign * r;
            }
          }
        }
      }
    }
  }
  return 2 * Math.pow(Math.PI, 2.5) / (p * q * Math.sqrt(p + q)) * sum;
}

// ── Contracted-shell integrals ───────────────────────────────

/** ⟨A | B⟩ overlap between two contracted Cartesian-Gaussian shells. */
export function S_cg(A: CGShell, B: CGShell): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ci = A.c[i]! * normCG(ai, A.angular);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const cj = B.c[j]! * normCG(bj, B.angular);
      s += ci * cj * primOverlap(ai, A.center, A.angular, bj, B.center, B.angular);
    }
  }
  return s;
}

/** Kinetic ⟨A | −∇²/2 | B⟩ between two contracted CG shells. */
export function T_cg(A: CGShell, B: CGShell): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ci = A.c[i]! * normCG(ai, A.angular);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const cj = B.c[j]! * normCG(bj, B.angular);
      s += ci * cj * primKinetic(ai, A.center, A.angular, bj, B.center, B.angular);
    }
  }
  return s;
}

/** Nuclear attraction ⟨A | −Z/|r − C| | B⟩ between two contracted CG shells. */
export function V_cg(
  A: CGShell, B: CGShell, Z: number, C: readonly [number, number, number],
): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ci = A.c[i]! * normCG(ai, A.angular);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const cj = B.c[j]! * normCG(bj, B.angular);
      s += ci * cj * primNuclear(ai, A.center, A.angular, bj, B.center, B.angular, Z, C);
    }
  }
  return s;
}

/** Two-electron repulsion (chemist notation): (A B | C D). */
export function ERI_cg(A: CGShell, B: CGShell, C: CGShell, D: CGShell): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!, ci = A.c[i]! * normCG(ai, A.angular);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!, cj = B.c[j]! * normCG(bj, B.angular);
      for (let k = 0; k < C.alpha.length; k++) {
        const ck = C.alpha[k]!, cck = C.c[k]! * normCG(ck, C.angular);
        for (let l = 0; l < D.alpha.length; l++) {
          const dl = D.alpha[l]!, cdl = D.c[l]! * normCG(dl, D.angular);
          s += ci * cj * cck * cdl * primERI(
            ai, A.center, A.angular,
            bj, B.center, B.angular,
            ck, C.center, C.angular,
            dl, D.center, D.angular,
          );
        }
      }
    }
  }
  return s;
}

/** Convenience: build a CGShell from a basis-set entry, center, and angular tuple. */
export function makeCGShell(
  basis: { readonly alpha: readonly number[]; readonly c: readonly number[] },
  center: readonly [number, number, number],
  angular: readonly [number, number, number],
  label?: string,
): CGShell {
  return { center, alpha: basis.alpha, c: basis.c, angular, label };
}

void SQRT_PI; // suppress unused warning; legibility only
