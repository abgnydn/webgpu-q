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
