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

// ─────────────────────────────────────────────────────────────
// B88 GGA exchange — Becke 1988.
// Closed-shell form via spin-decomposition with ρ_α = ρ_β = ρ/2:
//
//   F(u)       = u² / [1 + 6β u · arcsinh u]
//   ε_x^B88   = ε_x^LDA  −  2^(−1/3) β ρ^(1/3) · F(u),   u = 2^(1/3) σ
//   σ          = √γ / ρ^(4/3)        (closed-shell reduced gradient)
//
// We write the result as the GGA CORRECTION on top of Slater so the
// composite "B3LYP" mix can scale Slater independently from the GGA
// gradient correction (Becke 1988's a_x parameter).
//
// Outputs add into existing arrays scaled by `coef` so multiple
// functional pieces compose without intermediate buffers.
// ─────────────────────────────────────────────────────────────

const B88_BETA = 0.0042;
const POW_2_THIRD = Math.pow(2, 1 / 3);            // ≈ 1.25992
const POW_2_NEG_THIRD = Math.pow(2, -1 / 3);       // ≈ 0.79370

/**
 * Add B88 GGA gradient-correction to ε_x and its v_ρ, v_γ.
 *
 * Sign convention: the correction is NEGATIVE — adds to the LDA
 * exchange to lower the energy (matches B88 1988).
 *
 * @param rho     ρ at each grid point.
 * @param gamma   γ = |∇ρ|² at each grid point.
 * @param coef    Mixing coefficient (B3LYP uses 0.72).
 * @param epsXc   in/out: ε_xc per particle accumulator.
 * @param vRho    in/out: ∂(ρ ε_xc)/∂ρ accumulator.
 * @param vGamma  in/out: ∂(ρ ε_xc)/∂γ accumulator.
 */
export function addB88X(
  rho: Float64Array,
  gamma: Float64Array,
  coef: number,
  epsXc: Float64Array,
  vRho: Float64Array,
  vGamma: Float64Array,
): void {
  const n = rho.length;
  for (let p = 0; p < n; p++) {
    const r = rho[p]!;
    if (r < EPS_RHO) continue;
    const g = gamma[p]!;
    const r13 = Math.cbrt(r);
    const r43 = r * r13;            // ρ^(4/3)
    const sigma = Math.sqrt(Math.max(g, 0)) / r43;
    const u = POW_2_THIRD * sigma;
    // F(u) and F'(u) using arcsinh.
    const sh = Math.asinh(u);
    const root = Math.sqrt(1 + u * u);
    const D = 1 + 6 * B88_BETA * u * sh;
    const F = (u * u) / D;
    const Dp = 6 * B88_BETA * (sh + u / root);
    // F'(u) = (2u D − u² D') / D²
    const Fp = (2 * u * D - u * u * Dp) / (D * D);
    // Energy density (per particle) correction: -2^(-1/3) β ρ^(1/3) F(u)
    const dEps = -POW_2_NEG_THIRD * B88_BETA * r13 * F;
    // ρ · dEps gives the per-volume correction (let g(ρ, γ) = ρ · dEps_perParticle).
    // ∂g/∂ρ |_γ = -(4/3) · 2^(-1/3) β · ρ^(1/3) [ F(u) - u F'(u) ]
    //   (derivation in functional.ts header comment above)
    const dVRho = -(4 / 3) * POW_2_NEG_THIRD * B88_BETA * r13 * (F - u * Fp);
    // ∂g/∂γ |_ρ = -2^(-1/3) β ρ^(4/3) · F'(u) · u / (2 γ)
    //   well-defined for γ > 0; vanishes as γ → 0 because F'(u) ∝ u.
    let dVGamma = 0;
    if (g > EPS_RHO) {
      dVGamma = -POW_2_NEG_THIRD * B88_BETA * r43 * Fp * u / (2 * g);
    }
    epsXc[p]! += coef * dEps;
    vRho[p]!  += coef * dVRho;
    vGamma[p]! += coef * dVGamma;
  }
}

// ─────────────────────────────────────────────────────────────
// LYP GGA correlation — Lee-Yang-Parr 1988 closed-shell.
//
// Closed-shell collapse cross-checked against the canonical libxc
// Maple source (`maple/gga_exc/gga_c_lyp.mpl`) and verified by
// independent rederivation at z = 0:
//
//   ε_LYP^closed-shell(ρ, γ) per particle =
//     a·(t1 + ω·(t2 + t3 + t4 + t5 + t6))
//
// with libxc's reduced variables (rr = ρ^(−1/3), xt² = γ/ρ^(8/3)).
// At z = 0 (closed shell, xs0 = xs1 = 2^(1/3)·xt):
//
//   t1            = −1/h
//   t3            = −C_F                                (constant)
//   t2 + t4 + t5 + t6 in xt² collects to (3 + 7δ)·xt² / 72.
//
// (Derivation: t2 = (1+7δ)·xt²/72,  t4 = (45−δ)·xt²/144,
//  t5 = (δ−11)·xt²/144, t6 = −5·xt²/24, summed and reduced.)
//
// So:
//   ε_LYP^closed = −a/h − a·b·C_F·E/h
//                + a·b·E·(3 + 7δ)·γ / (72·h·ρ^(8/3))
//
// Per-volume f_LYP = ρ·ε:
//   f_LYP^closed(ρ, γ) =
//     −a·ρ/h
//     − a·b·C_F·E·ρ/h
//     + a·b·E·u^5·(3 + 7δ)·γ / (72·h)               (u = ρ^(−1/3))
// equivalently
//     + a·b·ω·(3 + 7δ)·ρ²·γ / 72                     (ω = E/(h·ρ^(11/3)))
//
// Using:
//   u(ρ) = ρ^(−1/3),  h(ρ) = 1 + d·u,  E(ρ) = exp(−c·u),
//   δ(ρ) = c·u + d·u/h,
//   C_F  = (3/10)·(3π²)^(2/3).
// LYP parameters (LYP 1988 eq. 24):
//   a = 0.04918, b = 0.132, c = 0.2533, d = 0.349.
//
// Sign / scale: the form has BOUNDED per-particle ε in the high-
// density limit (≈ −68 mHa as ρ → ∞), matches the LYP UEG curve at
// r_s ≈ 0.62 (ρ = 1, ε ≈ −47 mHa), and reproduces published BLYP /
// B3LYP STO-3G energies to within ~10 mHa.
//
// The previous LYP attempt that shipped a 30–240 mHa bug used a
// hand-collapsed Miehlich form whose γ-coefficient came out to
// (73 + 11δ)/144 — about 10× larger than the correct (3 + 7δ)/72,
// with the wrong δ-coefficient. The libxc cross-check is what
// caught that, plus the FD self-test below as a forward-going moat.
//
// Functional derivatives (per-volume, used in the GGA Fock build):
//   ∂f/∂γ   = +a·b·E·u^5·(3 + 7δ) / (72·h)         // strictly > 0
//   ∂f_A/∂ρ = −a/h − a·d·u/(3·h²)
//   ∂f_B/∂ρ = f_B·(δ+3)/(3ρ) = −a·b·C_F·(E/h)·(δ+3)/3
//   ∂f_C/∂ρ = f_C · [(δ − 5)/(3ρ) + 7·δ' / (3 + 7δ)]
//     δ'    = −δ/(3ρ) + d²·u²/(3ρ·h²)
//
// All three derivative pieces are validated point-wise against
// central-FD in `tests/chemistry/lyp.test.ts`.
// ─────────────────────────────────────────────────────────────

const LYP_A = 0.04918;
const LYP_B = 0.132;
const LYP_C = 0.2533;
const LYP_D = 0.349;
/** Thomas-Fermi prefactor (3/10)·(3π²)^(2/3). */
const C_F = 0.3 * Math.pow(3 * Math.PI * Math.PI, 2 / 3);

/**
 * Add LYP closed-shell correlation to ε_xc, v_ρ, v_γ.
 *
 * Sign convention: ε_LYP(γ = 0) is strictly negative (UEG limit).
 * v_γ is strictly positive everywhere (gradient locally suppresses
 * correlation). The point-wise sign of ε_LYP for nonzero γ can flip
 * positive at large reduced gradients — that's a documented LYP
 * feature, not a bug. Integrated over a closed-shell molecular
 * density, E_c^LYP < 0 (verified by hierarchy tests).
 *
 * @param coef  Mixing coefficient (B3LYP uses 0.81; pure LYP uses 1).
 */
export function addLYPC(
  rho: Float64Array,
  gamma: Float64Array,
  coef: number,
  epsXc: Float64Array,
  vRho: Float64Array,
  vGamma: Float64Array,
): void {
  const n = rho.length;
  for (let p = 0; p < n; p++) {
    const r = rho[p]!;
    if (r < EPS_RHO) continue;
    const g = gamma[p]!;
    const u = Math.pow(r, -1 / 3);                    // ρ^(−1/3)
    const h = 1 + LYP_D * u;
    const Eexp = Math.exp(-LYP_C * u);
    const r113 = Math.pow(r, 11 / 3);
    const omega = Eexp / (h * r113);
    const delta = LYP_C * u + (LYP_D * u) / h;
    const C3_7 = 3 + 7 * delta;

    // Per-volume f_LYP = f_A + f_B + f_C.
    //   f_A = −a·ρ/h
    //   f_B = −a·b·C_F·E·ρ/h         (=  −a·b·ω·C_F·ρ^(14/3))
    //   f_C = +a·b·ω·(3 + 7δ)·ρ²·γ/72  (libxc-canonical closed-shell γ)
    const fA = -LYP_A * r / h;
    const fB = -LYP_A * LYP_B * C_F * Eexp * r / h;
    const omegaR2 = omega * r * r;
    const fC = LYP_A * LYP_B * omegaR2 * C3_7 * g / 72;

    // Per-particle accumulator (E_xc = Σ w_p · ε_p · ρ_p).
    epsXc[p]! += coef * (fA + fB + fC) / r;

    // ∂f_A/∂ρ.
    const dA = -LYP_A / h - LYP_A * LYP_D * u / (3 * h * h);
    // ∂f_B/∂ρ via log-derivative: d ln(E·ρ/h)/dρ = (3 + δ)/(3ρ),
    // so ∂f_B/∂ρ = −a·b·C_F·(E/h)·(3 + δ)/3 (the ρ cancels cleanly).
    const dB = -LYP_A * LYP_B * C_F * (Eexp / h) * (3 + delta) / 3;
    // ∂f_C/∂ρ. Uses δ'(ρ) and d ln(ω·ρ²)/dρ = (δ − 5)/(3ρ).
    const deltaPrime = -delta / (3 * r) + (LYP_D * LYP_D * u * u) / (3 * r * h * h);
    const dC = fC * ((delta - 5) / (3 * r) + 7 * deltaPrime / C3_7);
    vRho[p]! += coef * (dA + dB + dC);

    // ∂f_C/∂γ — only f_C depends on γ.
    vGamma[p]! += coef * LYP_A * LYP_B * omegaR2 * C3_7 / 72;
  }
}
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
// Slater-only and VWN5-only "add" routines, for composing custom
// XC mixes (e.g. B3VWN5 needs 0.80 Slater + 0.72 B88 + 1.0 VWN5
// + 0.20 HF exchange).
// ─────────────────────────────────────────────────────────────

export function addSlaterX(
  rho: Float64Array,
  coef: number,
  epsXc: Float64Array,
  vRho: Float64Array,
): void {
  const n = rho.length;
  for (let p = 0; p < n; p++) {
    const r = rho[p]!;
    if (r < EPS_RHO) continue;
    const r13 = Math.cbrt(r);
    epsXc[p]! += coef * C_X * r13;
    vRho[p]!  += coef * VX_PREFACTOR * r13;
  }
}

export function addVWN5C(
  rho: Float64Array,
  coef: number,
  epsXc: Float64Array,
  vRho: Float64Array,
): void {
  const n = rho.length;
  for (let p = 0; p < n; p++) {
    const r = rho[p]!;
    if (r < EPS_RHO) continue;
    const r13 = Math.cbrt(r);
    const rs = C_RS / r13;
    const { ec, vc } = vwn5(rs);
    epsXc[p]! += coef * ec;
    vRho[p]!  += coef * vc;
  }
}

// ─────────────────────────────────────────────────────────────
// High-level dispatch: evaluate the requested XC functional and
// fill ε_xc, v_ρ, v_γ at every grid point.
// ─────────────────────────────────────────────────────────────

/**
 * Standard functional kinds.
 *
 * - `lda-svwn`: Slater exchange + VWN5 correlation (pure LDA).
 * - `bvwn5`:    Slater + B88 GGA exchange + VWN5 correlation.
 * - `blyp`:     Slater + B88 GGA exchange + LYP GGA correlation.
 *               The classic GGA most chemists call "BLYP".
 * - `b3vwn5`:   Becke3-style hybrid with VWN5-only correlation:
 *                 E_xc = 0.20·E_x^HF + 0.80·E_x^Slater
 *                      + 0.72·ΔE_x^B88 + E_c^VWN5.
 *               (B3-style hybrid that ships even before LYP.)
 * - `b3lyp5`:   The published B3LYP hybrid (Becke 1993) using
 *               VWN5 instead of VWN_RPA — what PySCF calls
 *               "B3LYP5" / "B3LYPV5":
 *                 E_xc = 0.20·E_x^HF + 0.80·E_x^Slater
 *                      + 0.72·ΔE_x^B88
 *                      + 0.81·E_c^LYP + 0.19·E_c^VWN5.
 */
export type FunctionalKind = "lda-svwn" | "bvwn5" | "blyp" | "b3vwn5" | "b3lyp5";

export interface FunctionalSpec {
  readonly kind: FunctionalKind;
}

/** Fraction of HF exact exchange this functional needs in the Fock matrix. */
export function hfExchangeMixOf(kind: FunctionalKind): number {
  return (kind === "b3vwn5" || kind === "b3lyp5") ? 0.20 : 0;
}

export function evalXC(
  kind: FunctionalKind,
  rho: Float64Array,
  gamma: Float64Array | null,
  epsXc: Float64Array,
  vRho: Float64Array,
  vGamma: Float64Array | null,
): void {
  epsXc.fill(0);
  vRho.fill(0);
  if (vGamma) vGamma.fill(0);
  switch (kind) {
    case "lda-svwn":
      addSlaterX(rho, 1, epsXc, vRho);
      addVWN5C(rho, 1, epsXc, vRho);
      return;
    case "bvwn5":
      if (!gamma || !vGamma) throw new Error("evalXC[bvwn5]: need gamma + vGamma");
      addSlaterX(rho, 1, epsXc, vRho);
      addB88X(rho, gamma, 1, epsXc, vRho, vGamma);
      addVWN5C(rho, 1, epsXc, vRho);
      return;
    case "blyp":
      if (!gamma || !vGamma) throw new Error("evalXC[blyp]: need gamma + vGamma");
      addSlaterX(rho, 1, epsXc, vRho);
      addB88X(rho, gamma, 1, epsXc, vRho, vGamma);
      addLYPC(rho, gamma, 1, epsXc, vRho, vGamma);
      return;
    case "b3vwn5":
      if (!gamma || !vGamma) throw new Error("evalXC[b3vwn5]: need gamma + vGamma");
      // 0.20 HF exchange handled in Fock build, not here.
      addSlaterX(rho, 0.80, epsXc, vRho);
      addB88X(rho, gamma, 0.72, epsXc, vRho, vGamma);
      addVWN5C(rho, 1, epsXc, vRho);
      return;
    case "b3lyp5":
      if (!gamma || !vGamma) throw new Error("evalXC[b3lyp5]: need gamma + vGamma");
      // Becke 1993 hybrid w/ a₀=0.20 HF (in Fock), a_x=0.72 ΔB88,
      // a_c=0.81 LYP, (1−a_c)=0.19 VWN5 (Stephens et al. 1994 with VWN5).
      addSlaterX(rho, 0.80, epsXc, vRho);
      addB88X(rho, gamma, 0.72, epsXc, vRho, vGamma);
      addLYPC(rho, gamma, 0.81, epsXc, vRho, vGamma);
      addVWN5C(rho, 0.19, epsXc, vRho);
      return;
  }
}
