// ─────────────────────────────────────────────────────────────
// density.ts — evaluate basis functions and the AO-density on a
// molecular grid.
//
// φ_μ(r) for a CGShell with primitives (α_p, c_p) and angular
// momentum I = (i_x, i_y, i_z) is:
//   φ_μ(r) = poly(r − R; I) · Σ_p c_p · N(α_p, I) · exp(−α_p |r − R|²)
//
// where N(α, I) is the per-primitive Cartesian-Gaussian
// normalization (matching `normCG` in integrals-cg.ts).
//
// ρ(r) at each grid point follows from the AO density matrix D:
//   ρ(r) = Σ_μν D_μν · φ_μ(r) · φ_ν(r)
// We pre-evaluate φ_μ(r) once (n × nGrid floats), then assemble
// ρ via a row-by-row matvec each call.
// ─────────────────────────────────────────────────────────────

import type { CGShell } from "../integrals-cg.js";
import type { MolecularGrid } from "./grid.js";

/** Pre-computed basis values: φ_μ(r_p) stored row-major as phi[p · n + μ]. */
export interface BasisValuesOnGrid {
  /** n × nGrid, row-major. phi[p · n + μ] = φ_μ(r_p). */
  readonly phi: Float64Array;
  readonly n: number;
  readonly nGrid: number;
}

/** Pre-computed basis gradients: ∇φ_μ(r_p) row-major as phi[p · n · 3 + μ · 3 + d]. */
export interface BasisGradientsOnGrid {
  /** ∂φ_μ/∂x at r_p — phix[p · n + μ]. */
  readonly phix: Float64Array;
  readonly phiy: Float64Array;
  readonly phiz: Float64Array;
  readonly n: number;
  readonly nGrid: number;
}

/** Pre-computed basis Hessians: ∂²φ_μ/∂r_a∂r_b at r_p, six unique
 *  components by symmetry. Used by the GGA piece of the analytical
 *  DFT gradient (∂γ/∂R requires these second derivatives). */
export interface BasisHessianOnGrid {
  /** ∂²φ_μ/∂x² at r_p — phixx[p · n + μ]. */
  readonly phixx: Float64Array;
  readonly phiyy: Float64Array;
  readonly phizz: Float64Array;
  readonly phixy: Float64Array;
  readonly phixz: Float64Array;
  readonly phiyz: Float64Array;
  readonly n: number;
  readonly nGrid: number;
}

/**
 * Evaluate every basis function at every grid point. Cost:
 * O(nGrid · n · n_prim) — the dominant per-DFT-iter expense.
 */
export function evalBasisOnGrid(
  shells: readonly CGShell[],
  grid: MolecularGrid,
): BasisValuesOnGrid {
  const n = shells.length;
  const nGrid = grid.x.length;
  const phi = new Float64Array(nGrid * n);

  // Per-shell precomputed primitive normalizations.
  const cnorm: Float64Array[] = shells.map((s) => {
    const out = new Float64Array(s.alpha.length);
    for (let p = 0; p < s.alpha.length; p++) {
      out[p] = (s.c[p] ?? 0) * normCG(s.alpha[p]!, s.angular);
    }
    return out;
  });

  for (let p = 0; p < nGrid; p++) {
    const rx = grid.x[p]!;
    const ry = grid.y[p]!;
    const rz = grid.z[p]!;
    for (let mu = 0; mu < n; mu++) {
      const s = shells[mu]!;
      const dx = rx - s.center[0];
      const dy = ry - s.center[1];
      const dz = rz - s.center[2];
      const r2 = dx * dx + dy * dy + dz * dz;
      // Cartesian polynomial (x − A_x)^i_x · (y − A_y)^i_y · (z − A_z)^i_z.
      let poly = 1;
      for (let k = 0; k < s.angular[0]; k++) poly *= dx;
      for (let k = 0; k < s.angular[1]; k++) poly *= dy;
      for (let k = 0; k < s.angular[2]; k++) poly *= dz;
      // Sum over primitives.
      let sum = 0;
      const cn = cnorm[mu]!;
      for (let pp = 0; pp < s.alpha.length; pp++) {
        sum += cn[pp]! * Math.exp(-s.alpha[pp]! * r2);
      }
      phi[p * n + mu] = poly * sum;
    }
  }

  return { phi, n, nGrid };
}

/**
 * Compute ρ(r_p) at every grid point from the AO density matrix.
 * ρ_p = Σ_μν D_μν φ_μ(r_p) φ_ν(r_p).
 *
 * Vectorized as ρ_p = sum over μ of φ_μ(r_p) · (Dφ)_μ where (Dφ) is
 * an n × nGrid intermediate. Cost: O(n² · nGrid).
 */
export function evalDensityOnGrid(
  D: Float64Array,
  basis: BasisValuesOnGrid,
): Float64Array {
  const { phi, n, nGrid } = basis;
  const rho = new Float64Array(nGrid);
  // Process one grid point at a time to avoid the n × nGrid intermediate.
  // ρ_p = Σ_{μν} D_{μν} φ_μ(r_p) φ_ν(r_p)
  //     = Σ_μ φ_μ · (Σ_ν D_{μν} φ_ν)
  const Dphi = new Float64Array(n);
  for (let p = 0; p < nGrid; p++) {
    // Dphi[μ] = Σ_ν D_{μν} φ_ν(r_p)
    for (let mu = 0; mu < n; mu++) {
      let s = 0;
      for (let nu = 0; nu < n; nu++) {
        s += D[mu * n + nu]! * phi[p * n + nu]!;
      }
      Dphi[mu] = s;
    }
    let r = 0;
    for (let mu = 0; mu < n; mu++) r += phi[p * n + mu]! * Dphi[mu]!;
    rho[p] = r;
  }
  return rho;
}

/** Integrate any per-grid-point quantity using the precomputed weights. */
export function integrateOverGrid(values: Float64Array, grid: MolecularGrid): number {
  let s = 0;
  for (let p = 0; p < values.length; p++) s += grid.w[p]! * values[p]!;
  return s;
}

/**
 * Evaluate ∇φ_μ at every grid point. Same cost as evalBasisOnGrid (3
 * polynomial × Gaussian evals per shell). Foundation for any GGA
 * functional which needs ∇ρ.
 *
 * For a primitive Cartesian Gaussian
 *   φ_p(r) = N(α, I) · (x−A_x)^{i_x} (y−A_y)^{i_y} (z−A_z)^{i_z} · exp(−α |r−A|²),
 * the partial wrt x is
 *   ∂φ/∂x = N · [i_x · (x−A_x)^{i_x−1} − 2α · (x−A_x)^{i_x+1}]
 *           · (y−A_y)^{i_y} (z−A_z)^{i_z} · exp(−α r²).
 * The per-primitive normalization N(α, I) is unchanged across the
 * derivative (we differentiate the basis function, not relabel it),
 * so the same `cnorm` table used for φ values applies here.
 */
export function evalBasisGradOnGrid(
  shells: readonly CGShell[],
  grid: MolecularGrid,
): BasisGradientsOnGrid {
  const n = shells.length;
  const nGrid = grid.x.length;
  const phix = new Float64Array(nGrid * n);
  const phiy = new Float64Array(nGrid * n);
  const phiz = new Float64Array(nGrid * n);

  const cnorm: Float64Array[] = shells.map((s) => {
    const out = new Float64Array(s.alpha.length);
    for (let p = 0; p < s.alpha.length; p++) {
      out[p] = (s.c[p] ?? 0) * normCG(s.alpha[p]!, s.angular);
    }
    return out;
  });

  for (let p = 0; p < nGrid; p++) {
    const rx = grid.x[p]!;
    const ry = grid.y[p]!;
    const rz = grid.z[p]!;
    for (let mu = 0; mu < n; mu++) {
      const s = shells[mu]!;
      const dx = rx - s.center[0];
      const dy = ry - s.center[1];
      const dz = rz - s.center[2];
      const r2 = dx * dx + dy * dy + dz * dz;
      const ix = s.angular[0], iy = s.angular[1], iz = s.angular[2];
      // Cartesian polynomials at (i_x, i_y, i_z), i_x±1.
      const polyx = ix === 0 ? 1 : Math.pow(dx, ix);
      const polyy = iy === 0 ? 1 : Math.pow(dy, iy);
      const polyz = iz === 0 ? 1 : Math.pow(dz, iz);
      const polyxm = ix === 0 ? 0 : (ix === 1 ? 1 : Math.pow(dx, ix - 1));
      const polyym = iy === 0 ? 0 : (iy === 1 ? 1 : Math.pow(dy, iy - 1));
      const polyzm = iz === 0 ? 0 : (iz === 1 ? 1 : Math.pow(dz, iz - 1));
      const polyxp = Math.pow(dx, ix + 1);
      const polyyp = Math.pow(dy, iy + 1);
      const polyzp = Math.pow(dz, iz + 1);

      // Sum over primitives. ∂_x φ collects ix · poly_{ix−1} (Gauss-independent
      // term) plus −2α_p · poly_{ix+1} (Gauss-decay term, primitive-dependent).
      let gx = 0, gy = 0, gz = 0;
      const cn = cnorm[mu]!;
      for (let pp = 0; pp < s.alpha.length; pp++) {
        const a = s.alpha[pp]!;
        const c = cn[pp]!;
        const e = c * Math.exp(-a * r2);
        // ∂/∂x:  e · [ix · polyxm · polyy · polyz  −  2α · polyxp · polyy · polyz]
        gx += e * (ix * polyxm * polyy * polyz - 2 * a * polyxp * polyy * polyz);
        gy += e * (iy * polyx * polyym * polyz - 2 * a * polyx * polyyp * polyz);
        gz += e * (iz * polyx * polyy * polyzm - 2 * a * polyx * polyy * polyzp);
      }
      phix[p * n + mu] = gx;
      phiy[p * n + mu] = gy;
      phiz[p * n + mu] = gz;
    }
  }

  return { phix, phiy, phiz, n, nGrid };
}

/**
 * Evaluate the basis Hessian ∂²φ_μ/∂r_a∂r_b at every grid point.
 *
 * For a Cartesian-Gaussian primitive at center A with angular
 * tuple I = (i_x, i_y, i_z) and exponent α:
 *   ∂²/∂r_a∂r_b [poly_I · e^{-αr²}] = ... (shifted-L formulas)
 *
 * Diagonal (a = b):
 *   ∂²/∂r_a² = { i_a(i_a−1)·poly_{I-2_a}
 *              − 2α(2·i_a + 1)·poly_I
 *              + 4α²·poly_{I+2_a} } · e^{-αr²}
 *
 * Off-diagonal (a ≠ b):
 *   ∂²/∂r_b∂r_a = { i_a·i_b·poly_{I-1_a-1_b}
 *                 − 2α·i_a·poly_{I-1_a+1_b}
 *                 − 2α·i_b·poly_{I+1_a-1_b}
 *                 + 4α²·poly_{I+1_a+1_b} } · e^{-αr²}
 *
 * Cost: same O(nGrid · n · n_prim) shape as gradient evaluation,
 * with a few extra polynomial powers per (μ, p) to assemble the
 * 6 unique components.
 */
export function evalBasisHessianOnGrid(
  shells: readonly CGShell[],
  grid: MolecularGrid,
): BasisHessianOnGrid {
  const n = shells.length;
  const nGrid = grid.x.length;
  const phixx = new Float64Array(nGrid * n);
  const phiyy = new Float64Array(nGrid * n);
  const phizz = new Float64Array(nGrid * n);
  const phixy = new Float64Array(nGrid * n);
  const phixz = new Float64Array(nGrid * n);
  const phiyz = new Float64Array(nGrid * n);

  const cnorm: Float64Array[] = shells.map((s) => {
    const out = new Float64Array(s.alpha.length);
    for (let p = 0; p < s.alpha.length; p++) {
      out[p] = (s.c[p] ?? 0) * normCG(s.alpha[p]!, s.angular);
    }
    return out;
  });

  // Helper: dx^k with clamping for k < 0 (poly_{I-2_a} when I_a < 2 etc).
  const pw = (d: number, k: number): number => {
    if (k < 0) return 0;
    if (k === 0) return 1;
    if (k === 1) return d;
    return Math.pow(d, k);
  };

  for (let p = 0; p < nGrid; p++) {
    const rx = grid.x[p]!, ry = grid.y[p]!, rz = grid.z[p]!;
    for (let mu = 0; mu < n; mu++) {
      const s = shells[mu]!;
      const dx = rx - s.center[0];
      const dy = ry - s.center[1];
      const dz = rz - s.center[2];
      const r2 = dx * dx + dy * dy + dz * dz;
      const ix = s.angular[0], iy = s.angular[1], iz = s.angular[2];

      // Powers of dx, dy, dz at offsets {-2, -1, 0, +1, +2} from I.
      const xm2 = pw(dx, ix - 2);
      const xm1 = pw(dx, ix - 1);
      const x0  = pw(dx, ix);
      const xp1 = pw(dx, ix + 1);
      const xp2 = pw(dx, ix + 2);
      const ym2 = pw(dy, iy - 2);
      const ym1 = pw(dy, iy - 1);
      const y0  = pw(dy, iy);
      const yp1 = pw(dy, iy + 1);
      const yp2 = pw(dy, iy + 2);
      const zm2 = pw(dz, iz - 2);
      const zm1 = pw(dz, iz - 1);
      const z0  = pw(dz, iz);
      const zp1 = pw(dz, iz + 1);
      const zp2 = pw(dz, iz + 2);

      // Per-axis "diagonal" polynomial pieces (×y0·z0 etc applied below):
      //   xx-row needs xm2, x0, xp2 with y0·z0 multiplier.
      const polyI    = x0  * y0  * z0;             // poly_I
      const polyXm2  = xm2 * y0  * z0;             // poly_{I-2_x}
      const polyXp2  = xp2 * y0  * z0;             // poly_{I+2_x}
      const polyYm2  = x0  * ym2 * z0;
      const polyYp2  = x0  * yp2 * z0;
      const polyZm2  = x0  * y0  * zm2;
      const polyZp2  = x0  * y0  * zp2;
      // Off-diagonal pieces (xy, xz, yz):
      const polyXmYm = xm1 * ym1 * z0;             // poly_{I-1_x-1_y}
      const polyXmYp = xm1 * yp1 * z0;             // poly_{I-1_x+1_y}
      const polyXpYm = xp1 * ym1 * z0;             // poly_{I+1_x-1_y}
      const polyXpYp = xp1 * yp1 * z0;             // poly_{I+1_x+1_y}
      const polyXmZm = xm1 * y0  * zm1;
      const polyXmZp = xm1 * y0  * zp1;
      const polyXpZm = xp1 * y0  * zm1;
      const polyXpZp = xp1 * y0  * zp1;
      const polyYmZm = x0  * ym1 * zm1;
      const polyYmZp = x0  * ym1 * zp1;
      const polyYpZm = x0  * yp1 * zm1;
      const polyYpZp = x0  * yp1 * zp1;

      let hxx = 0, hyy = 0, hzz = 0, hxy = 0, hxz = 0, hyz = 0;
      const cn = cnorm[mu]!;
      for (let pp = 0; pp < s.alpha.length; pp++) {
        const a = s.alpha[pp]!;
        const e = cn[pp]! * Math.exp(-a * r2);
        // Diagonal:
        //   I_a(I_a-1)·poly_{I-2_a} − 2α(2·I_a + 1)·poly_I + 4α²·poly_{I+2_a}
        hxx += e * (ix * (ix - 1) * polyXm2
                  - 2 * a * (2 * ix + 1) * polyI
                  + 4 * a * a * polyXp2);
        hyy += e * (iy * (iy - 1) * polyYm2
                  - 2 * a * (2 * iy + 1) * polyI
                  + 4 * a * a * polyYp2);
        hzz += e * (iz * (iz - 1) * polyZm2
                  - 2 * a * (2 * iz + 1) * polyI
                  + 4 * a * a * polyZp2);
        // Off-diagonal:
        //   I_a·I_b·poly_{I-1_a-1_b}
        //   − 2α·I_a·poly_{I-1_a+1_b}
        //   − 2α·I_b·poly_{I+1_a-1_b}
        //   + 4α²·poly_{I+1_a+1_b}
        hxy += e * (ix * iy * polyXmYm
                  - 2 * a * ix * polyXmYp
                  - 2 * a * iy * polyXpYm
                  + 4 * a * a * polyXpYp);
        hxz += e * (ix * iz * polyXmZm
                  - 2 * a * ix * polyXmZp
                  - 2 * a * iz * polyXpZm
                  + 4 * a * a * polyXpZp);
        hyz += e * (iy * iz * polyYmZm
                  - 2 * a * iy * polyYmZp
                  - 2 * a * iz * polyYpZm
                  + 4 * a * a * polyYpZp);
      }
      const off = p * n + mu;
      phixx[off] = hxx; phiyy[off] = hyy; phizz[off] = hzz;
      phixy[off] = hxy; phixz[off] = hxz; phiyz[off] = hyz;
    }
  }

  return { phixx, phiyy, phizz, phixy, phixz, phiyz, n, nGrid };
}

/**
 * Combined evaluator: ρ, ∇ρ_x, ∇ρ_y, ∇ρ_z, γ = |∇ρ|² at every grid
 * point, in one O(n² · nGrid) pass.
 *
 *   ρ(r)     = Σ_{μν} D_{μν} φ_μ(r) φ_ν(r)
 *   ∇ρ(r)   = 2 Σ_{μν} D_{μν} φ_μ(r) ∇φ_ν(r)    (D symmetric)
 *
 * The factor of 2 comes from the symmetry of D (each μν pair is
 * counted twice).
 */
export function evalDensityAndGradient(
  D: Float64Array,
  basis: BasisValuesOnGrid,
  basisGrad: BasisGradientsOnGrid,
): { rho: Float64Array; gradX: Float64Array; gradY: Float64Array; gradZ: Float64Array; gamma: Float64Array } {
  const { phi, n, nGrid } = basis;
  if (basisGrad.n !== n || basisGrad.nGrid !== nGrid) {
    throw new Error("evalDensityAndGradient: basis/grid shape mismatch");
  }
  const rho = new Float64Array(nGrid);
  const gx = new Float64Array(nGrid);
  const gy = new Float64Array(nGrid);
  const gz = new Float64Array(nGrid);
  const gamma = new Float64Array(nGrid);
  const Dphi = new Float64Array(n);
  for (let p = 0; p < nGrid; p++) {
    const off = p * n;
    // (Dφ)_μ at this grid point.
    for (let mu = 0; mu < n; mu++) {
      let s = 0;
      for (let nu = 0; nu < n; nu++) {
        s += D[mu * n + nu]! * phi[off + nu]!;
      }
      Dphi[mu] = s;
    }
    let r = 0, rx = 0, ry = 0, rz = 0;
    for (let mu = 0; mu < n; mu++) {
      const phimu = phi[off + mu]!;
      r  += phimu * Dphi[mu]!;
      rx += basisGrad.phix[off + mu]! * Dphi[mu]!;
      ry += basisGrad.phiy[off + mu]! * Dphi[mu]!;
      rz += basisGrad.phiz[off + mu]! * Dphi[mu]!;
    }
    rho[p] = r;
    // factor of 2 from D symmetry: ∇ρ = 2 Σ φ ∇φ_ν · D
    gx[p] = 2 * rx;
    gy[p] = 2 * ry;
    gz[p] = 2 * rz;
    gamma[p] = gx[p]! * gx[p]! + gy[p]! * gy[p]! + gz[p]! * gz[p]!;
  }
  return { rho, gradX: gx, gradY: gy, gradZ: gz, gamma };
}

// ── Cartesian-Gaussian normalization (mirrors integrals-cg.ts) ──

function doubleFactOdd(n: number): number {
  if (n <= 0) return 1;
  let r = 1;
  for (let k = 1; k <= n; k++) r *= 2 * k - 1;
  return r;
}

function normCG(alpha: number, I: readonly [number, number, number]): number {
  const L = I[0] + I[1] + I[2];
  const dfx = doubleFactOdd(I[0]);
  const dfy = doubleFactOdd(I[1]);
  const dfz = doubleFactOdd(I[2]);
  const radial = Math.pow((2 * alpha) / Math.PI, 0.75);
  const angular = Math.sqrt(Math.pow(4 * alpha, L) / (dfx * dfy * dfz));
  return radial * angular;
}
