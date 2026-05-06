// ─────────────────────────────────────────────────────────────
// functional.ts — closed-shell LDA exchange-correlation kernels.
//
// Slater (Dirac) exchange:
//   ε_x(ρ)  = − ¾ · (3/π)^(1/3) · ρ^(1/3)
//   v_x(ρ)  = (4/3) · ε_x(ρ) = − (3/π)^(1/3) · ρ^(1/3)
//
// VWN5 correlation (Vosko-Wilk-Nusair 1980, RPA-fit
// parameterization 5; closed-shell paramagnetic limit):
//   r_s = (3/(4π ρ))^(1/3),   x = √r_s,   X(x) = x² + b x + c,
//   Q = √(4c − b²)
//   ε_c(r_s) = (A/2) · {  ln(x²/X(x))
//                       + (2b/Q) · arctan(Q/(2x + b))
//                       − (b x_0 / X(x_0)) · [ ln((x − x_0)²/X(x))
//                                            + 2(b + 2 x_0)/Q · arctan(Q/(2x + b)) ] }
//   v_c(r_s) = ε_c(r_s) − (r_s/3) · dε_c/dr_s
//   Parameters: A = 0.0621814, x_0 = −0.10498, b = 3.72744, c = 12.9352
//
// Both kernels accept and return densities per spatial point. They
// take a Float64Array of ρ and write into pre-allocated ερ_xc and
// v_xc arrays — no allocation per grid point. Densities below
// EPS_RHO are clipped to zero (LDA is undefined at ρ = 0 but the
// limits are well-defined).
// ─────────────────────────────────────────────────────────────

const EPS_RHO = 1e-15;

/** Slater (Dirac) exchange. Pure power law in ρ — no correlation. */
const C_X = -0.75 * Math.pow(3 / Math.PI, 1 / 3);          // ≈ −0.7385588
const VX_PREFACTOR = -Math.pow(3 / Math.PI, 1 / 3);        // ≈ −0.9847450
const C_RS = Math.pow(3 / (4 * Math.PI), 1 / 3);            // r_s = C_RS · ρ^(−1/3)

/**
 * Evaluate ε_xc(ρ) and v_xc(ρ) at every grid point.
 * Closed-shell LDA: Slater exchange + VWN5 correlation.
 *
 * @param rho     ρ(r_p), nGrid floats.
 * @param epsXc   out: ε_xc(ρ_p), nGrid floats. Energy DENSITY per
 *                 unit ρ — total E_xc = ∫ ε_xc · ρ dr = Σ w_p ε_xc[p] ρ[p].
 * @param vXc     out: v_xc(ρ_p), nGrid floats. Used as F_μν += Σ w_p
 *                 v_xc[p] · φ_μ(r_p) φ_ν(r_p).
 */
export function evalLDA(
  rho: Float64Array,
  epsXc: Float64Array,
  vXc: Float64Array,
): void {
  const n = rho.length;
  for (let p = 0; p < n; p++) {
    const r = rho[p]!;
    if (r < EPS_RHO) {
      epsXc[p] = 0;
      vXc[p]   = 0;
      continue;
    }
    const r13 = Math.cbrt(r);
    // Exchange ─────────────────────────────────────────────────
    const eX = C_X * r13;
    const vX = VX_PREFACTOR * r13;
    // Correlation (VWN5) ───────────────────────────────────────
    const rs = C_RS / r13;
    const { ec, vc } = vwn5(rs);
    epsXc[p] = eX + ec;
    vXc[p]   = vX + vc;
  }
}

// ── VWN5 correlation kernel ─────────────────────────────────
// Closed-shell paramagnetic limit only; spin polarization is
// trivially zero everywhere for closed-shell SCF.
//
// Closed-form (Vosko-Wilk-Nusair 1980, eq. 4.4):
//   ε_c(r_s) = (A/2) · {
//       ln(x²/X(x))
//     + (2b/Q) · arctan(Q/(2x + b))
//     − (b x_0 / X(x_0)) · [
//           ln((x − x_0)²/X(x))
//         + 2(b + 2 x_0)/Q · arctan(Q/(2x + b))
//       ]
//   }
// with x = √r_s, X(x) = x² + b x + c.
//
// Derivative dε_c/dr_s used in v_c = ε_c − (r_s/3) · dε_c/dr_s
// is computed analytically here for stability across all r_s.

const VWN_A   = 0.0621814;
const VWN_X0  = -0.10498;
const VWN_B   = 3.72744;
const VWN_C   = 12.9352;
const VWN_Q   = Math.sqrt(4 * VWN_C - VWN_B * VWN_B);
const VWN_X_X0 = VWN_X0 * VWN_X0 + VWN_B * VWN_X0 + VWN_C;

function vwn5(rs: number): { ec: number; vc: number } {
  const x = Math.sqrt(rs);
  const X  = x * x + VWN_B * x + VWN_C;
  const xMx0 = x - VWN_X0;
  const atanArg = VWN_Q / (2 * x + VWN_B);
  const atan = Math.atan(atanArg);
  // ε_c
  const ec = (VWN_A / 2) * (
      Math.log((x * x) / X)
    + (2 * VWN_B / VWN_Q) * atan
    - (VWN_B * VWN_X0 / VWN_X_X0) * (
        Math.log((xMx0 * xMx0) / X)
        + (2 * (VWN_B + 2 * VWN_X0) / VWN_Q) * atan
      )
  );
  // dε_c/dx (analytical). Using chain rule:
  //   d ln(x²/X) /dx = 2/x − (2x + b)/X
  //   d arctan(Q/(2x+b)) /dx = − 2 Q / ((2x+b)² + Q²) = − 2Q / (4 X)
  //                           = − Q / (2 X)
  //   d ln((x−x_0)²/X) /dx = 2/(x − x_0) − (2x + b)/X
  const dlnxX = 2 / x - (2 * x + VWN_B) / X;
  const dArc  = -VWN_Q / (2 * X);
  const dlnXm = 2 / xMx0 - (2 * x + VWN_B) / X;
  const dEcDx = (VWN_A / 2) * (
      dlnxX
    + (2 * VWN_B / VWN_Q) * dArc
    - (VWN_B * VWN_X0 / VWN_X_X0) * (
        dlnXm
        + (2 * (VWN_B + 2 * VWN_X0) / VWN_Q) * dArc
      )
  );
  // dε_c/dr_s = (1 / (2x)) · dε_c/dx
  const dEcDrs = dEcDx / (2 * x);
  // v_c = ε_c − (r_s/3) · dε_c/dr_s
  const vc = ec - (rs / 3) * dEcDrs;
  return { ec, vc };
}

/** Re-export for cross-module sanity tests. */
export const VWN5_PARAMS = {
  A: VWN_A, x0: VWN_X0, b: VWN_B, c: VWN_C, Q: VWN_Q, Xx0: VWN_X_X0,
};
