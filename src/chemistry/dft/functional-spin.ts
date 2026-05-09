// ─────────────────────────────────────────────────────────────
// functional-spin.ts — spin-polarized LSDA (Slater exchange +
// VWN5 correlation). Tier 2 stage 15b.
//
// Rationale: closed-shell triplet TDDFT needs the spin-polarized
// XC kernel. Specifically,
//   f_xc^triplet = ½ (f_↑↑ − f_↑↓)
// where f_σσ' = ∂v_σ/∂ρ_σ' evaluated at ρ_↑ = ρ_↓ = ρ/2. Computing
// f_↑↑ and f_↑↓ separately requires a spin-polarized first-derivative
// (v_↑, v_↓) from spin-polarized ε_xc(ρ_↑, ρ_↓) — which the
// closed-shell `functional.ts` doesn't provide.
//
// Slater exchange (per volume):
//   e_x^LSDA(ρ_↑, ρ_↓) = −C_X · 2^(1/3) · (ρ_↑^(4/3) + ρ_↓^(4/3))
//   v_x_σ = ∂e_x/∂ρ_σ = −(4/3) · C_X · 2^(1/3) · ρ_σ^(1/3)
// At ρ_↑ = ρ_↓ = ρ/2 this collapses to the closed-shell v_x:
//   v_x_σ(closed) = −(4/3) · C_X · ρ^(1/3) = closed-shell v_x. ✓
//
// VWN5 correlation (Vosko-Wilk-Nusair 1980, full spin form):
//   ε_c(r_s, ζ) = ε_c^P(r_s)
//                + α_c(r_s) · f(ζ)/f''(0) · (1 − ζ⁴)
//                + (ε_c^F(r_s) − ε_c^P(r_s)) · f(ζ) · ζ⁴
// with three independent (A, x_0, b, c) parameter sets — paramagnetic
// (P), ferromagnetic (F), and the spin-stiffness α_c — and the
// interpolation function
//   f(ζ) = [(1+ζ)^(4/3) + (1−ζ)^(4/3) − 2] / (2^(4/3) − 2).
//
// All parameters from VWN1980 Table I (the same paramagnetic block
// already in functional.ts).
//
// Outputs are reference-agnostic: caller passes ρ_↑, ρ_↓ arrays of
// the same length; routine fills ε_xc, v_↑, v_↓ in pre-allocated
// arrays. ρ_σ values below EPS_RHO are clipped to zero like the
// closed-shell path.
// ─────────────────────────────────────────────────────────────

const EPS_RHO = 1e-15;

// ── Slater exchange spin-polarized prefactors ──────────────
// e_x^LSDA = SLATER_LSDA_E · (ρ_↑^(4/3) + ρ_↓^(4/3))   per volume
// v_x_σ   = SLATER_LSDA_V · ρ_σ^(1/3)                  per volume
// SLATER_LSDA_E = −(3/4)·(3/π)^(1/3) · 2^(1/3)
// SLATER_LSDA_V = (4/3) · SLATER_LSDA_E = −(3/π)^(1/3) · 2^(1/3)
// At ρ_↑ = ρ_↓ = ρ/2 the LSDA v_x_σ collapses to the closed-shell
// −(3/π)^(1/3)·ρ^(1/3) (= `VX_PREFACTOR · ρ^(1/3)` in functional.ts).
const C_X = -0.75 * Math.pow(3 / Math.PI, 1 / 3);            // ≈ −0.7385588
const SLATER_LSDA_E = C_X * Math.pow(2, 1 / 3);              // ≈ −0.9305258
const SLATER_LSDA_V = (4 / 3) * SLATER_LSDA_E;               // ≈ −1.2407011
const C_RS = Math.pow(3 / (4 * Math.PI), 1 / 3);              // r_s = C_RS · ρ^(−1/3)

// ── VWN5 parameters ─────────────────────────────────────────
// Paramagnetic (P) — ζ = 0.
const VWN_P = { A: 0.0621814,   x0: -0.10498,    b: 3.72744,  c: 12.9352  };
// Ferromagnetic (F) — ζ = 1.
const VWN_F = { A: 0.0310907,   x0: -0.32500,    b: 7.06042,  c: 18.0578  };
// Spin-stiffness α_c — Vosko's third parameter set.
const VWN_AC = { A: -1 / (6 * Math.PI * Math.PI), x0: -0.0047584, b: 1.13107, c: 13.0045 };

// f''(0) for the VWN spin-interpolation function.
const F_DD_0 = (8 / 9) / (Math.pow(2, 4 / 3) - 2);             // ≈ 1.7099

interface VwnDerivs {
  /** ε_c (per particle, Hartree). */
  readonly ec: number;
  /** dε_c/dr_s. */
  readonly decDrs: number;
}

/** Closed-form ε_c(r_s) for one of the three VWN parameter sets +
 *  its r_s derivative. Re-derived analytically (matches the closed-
 *  shell `functional.ts` expression at the P set). */
function vwn(set: typeof VWN_P, rs: number): VwnDerivs {
  const x = Math.sqrt(rs);
  const X = x * x + set.b * x + set.c;
  const xMx0 = x - set.x0;
  const Q = Math.sqrt(4 * set.c - set.b * set.b);
  const Xx0 = set.x0 * set.x0 + set.b * set.x0 + set.c;
  const atan = Math.atan(Q / (2 * x + set.b));
  const ec = (set.A / 2) * (
      Math.log((x * x) / X)
    + (2 * set.b / Q) * atan
    - (set.b * set.x0 / Xx0) * (
        Math.log((xMx0 * xMx0) / X)
        + (2 * (set.b + 2 * set.x0) / Q) * atan
      )
  );
  // dε_c/dx
  const dlnxX  = 2 / x - (2 * x + set.b) / X;
  const dArc   = -Q / (2 * X);
  const dlnXm  = 2 / xMx0 - (2 * x + set.b) / X;
  const dEcDx = (set.A / 2) * (
      dlnxX
    + (2 * set.b / Q) * dArc
    - (set.b * set.x0 / Xx0) * (
        dlnXm
      + (2 * (set.b + 2 * set.x0) / Q) * dArc
      )
  );
  // dε_c/dr_s = (1 / (2x)) · dε_c/dx.
  return { ec, decDrs: dEcDx / (2 * x) };
}

/** Spin-interpolation f(ζ) and its first derivative. */
function spinFOf(zeta: number): { f: number; fp: number } {
  const z = Math.max(-1, Math.min(1, zeta));
  const denom = Math.pow(2, 4 / 3) - 2;
  const onePlus = Math.pow(1 + z, 4 / 3);
  const oneMinus = Math.pow(1 - z, 4 / 3);
  const f = (onePlus + oneMinus - 2) / denom;
  // f'(ζ) = (4/3) · [(1+ζ)^(1/3) − (1−ζ)^(1/3)] / denom
  const fp = (4 / 3) * (Math.pow(1 + z, 1 / 3) - Math.pow(1 - z, 1 / 3)) / denom;
  return { f, fp };
}

/**
 * Closed-shell-or-spin-polarized LSDA ε_xc, v_↑, v_↓ at every grid
 * point (Slater exchange + VWN5 correlation).
 *
 * Output convention: `epsXc` is per-particle (E_xc = Σ w · ε · ρ,
 * ρ = ρ_↑ + ρ_↓). `vUp` and `vDn` are per-volume potentials =
 * ∂(ρ·ε_xc)/∂ρ_↑ and ∂(ρ·ε_xc)/∂ρ_↓ respectively — the same
 * convention `functional.ts` uses for the closed-shell `vRho`.
 *
 * Below EPS_RHO the grid point is clipped to zero, same as the
 * closed-shell path.
 */
export function evalXC_LSDA(
  rhoUp: Float64Array,
  rhoDn: Float64Array,
  epsXc: Float64Array,
  vUp: Float64Array,
  vDn: Float64Array,
): void {
  const n = rhoUp.length;
  if (rhoDn.length !== n) {
    throw new Error(`evalXC_LSDA: rhoUp / rhoDn length mismatch (${n} vs ${rhoDn.length})`);
  }
  for (let p = 0; p < n; p++) {
    const ru = rhoUp[p]!;
    const rd = rhoDn[p]!;
    const rho = ru + rd;
    if (rho < EPS_RHO) {
      epsXc[p] = 0; vUp[p] = 0; vDn[p] = 0;
      continue;
    }
    // ── Slater exchange (per-volume) ─────────────────────────
    // e_x = −C_X · 2^(1/3) · (ρ_↑^(4/3) + ρ_↓^(4/3))
    const ru13 = ru >= EPS_RHO ? Math.cbrt(ru) : 0;
    const rd13 = rd >= EPS_RHO ? Math.cbrt(rd) : 0;
    const ex = SLATER_LSDA_E * (ru * ru13 + rd * rd13);
    const vxUp = SLATER_LSDA_V * ru13;
    const vxDn = SLATER_LSDA_V * rd13;

    // ── VWN5 correlation ────────────────────────────────────
    const r13 = Math.cbrt(rho);
    const rs = C_RS / r13;
    const zeta = (ru - rd) / rho;
    const { ec: ecP, decDrs: decPDrs } = vwn(VWN_P, rs);
    const { ec: ecF, decDrs: decFDrs } = vwn(VWN_F, rs);
    const { ec: alphaC, decDrs: dAlphaDrs } = vwn(VWN_AC, rs);
    const { f, fp } = spinFOf(zeta);
    const z2 = zeta * zeta;
    const z4 = z2 * z2;

    // ε_c(r_s, ζ) = ε_c^P + α_c · f/f''(0) · (1−ζ⁴) + (ε_c^F − ε_c^P) · f · ζ⁴
    const ec = ecP + alphaC * (f / F_DD_0) * (1 - z4) + (ecF - ecP) * f * z4;

    // d ε_c / d r_s  (chain rule on the three VWN sets)
    const decDrs = decPDrs + dAlphaDrs * (f / F_DD_0) * (1 - z4)
                   + (decFDrs - decPDrs) * f * z4;
    // d ε_c / d ζ
    //   = α_c/f''(0) · [f' · (1−ζ⁴) − f · 4ζ³]
    //   + (ε_c^F − ε_c^P) · [f' · ζ⁴ + f · 4ζ³]
    const decDz = (alphaC / F_DD_0) * (fp * (1 - z4) - f * 4 * zeta * z2)
                + (ecF - ecP)         * (fp * z4       + f * 4 * zeta * z2);

    // r_s and ζ as functions of (ρ_↑, ρ_↓):
    //   r_s = C_RS · ρ^(−1/3)        ⇒ dr_s/dρ_σ = −r_s / (3ρ)
    //   ζ = (ρ_↑ − ρ_↓)/ρ            ⇒ dζ/dρ_↑ = (1 − ζ)/ρ,
    //                                  dζ/dρ_↓ = −(1 + ζ)/ρ
    // v_c_σ = ∂(ρ·ε_c)/∂ρ_σ = ε_c + ρ · ∂ε_c/∂ρ_σ.
    const drsDrho = -rs / (3 * rho);                // dr_s/dρ_σ identical for σ = ↑, ↓
    const dzetaDrhoUp = (1 - zeta) / rho;
    const dzetaDrhoDn = -(1 + zeta) / rho;
    const decDrhoUp = decDrs * drsDrho + decDz * dzetaDrhoUp;
    const decDrhoDn = decDrs * drsDrho + decDz * dzetaDrhoDn;
    const vcUp = ec + rho * decDrhoUp;
    const vcDn = ec + rho * decDrhoDn;

    // ── Compose ──────────────────────────────────────────────
    // ε_xc per-particle = (e_x^per-vol + ρ·ε_c) / ρ.
    epsXc[p] = (ex + rho * ec) / rho;
    vUp[p]   = vxUp + vcUp;
    vDn[p]   = vxDn + vcDn;
  }
}

// ─────────────────────────────────────────────────────────────
// B88 GGA exchange — spin-polarized (Becke 1988).
//
// e_x^B88_spin(per volume) = − β · Σ_σ ρ_σ^(4/3) · F(x_σ)
//   x_σ      = √γ_σσ / ρ_σ^(4/3)        (per-spin reduced gradient)
//   F(x)     = x² / (1 + 6β x · arcsinh x)
//   ε_x^LSDA = −C_X · 2^(1/3) · (ρ_↑^(4/3) + ρ_↓^(4/3))   (Slater piece;
//              evaluated separately in `evalXC_LSDA`)
//
// Sign convention: B88 is the GGA CORRECTION on top of LSDA Slater,
// negative everywhere (lowers exchange energy below LSDA). We emit
// it into the same per-particle ε_xc / v_ρσ / v_γσσ' accumulators
// that `evalXC_LSDA` writes to, scaled by `coef` so the composite
// hybrid mixes can dial the Becke gradient correction up or down.
//
// First derivatives:
//   v_x_σ(B88, per volume) = −β · (4/3) · ρ_σ^(1/3) · [F(x_σ) − x_σ·F'(x_σ)]
//   v_γ_σσ(B88, per volume) = −β · F'(x_σ) / (2·√γ_σσ)        (γ_σσ > 0)
//   v_γ_↑↓(B88, per volume) = 0                                (no γ_↑↓ dep)
//
// Closed-shell limit ρ_↑ = ρ_↓ = ρ/2, γ_↑↑ = γ_↓↓ = γ/4 collapses
// to the closed-shell `addB88X` formulas in `functional.ts`:
//   x_σ → 2^(1/3) σ = u                       ✓
//   ε_x^correction → −2^(−1/3) β ρ^(1/3) F(u)  ✓
//   v_x_σ → closed-shell v_x correction         ✓
//   (1/2)(v_γ_↑↑ + v_γ_↓↓) → closed-shell v_γ ✓
// (The (1/2) above is the closed-shell chain-rule factor:
// v_γ^cs = ∂F/∂γ with γ = γ_↑↑+2γ_↑↓+γ_↓↓.)
// ─────────────────────────────────────────────────────────────

const B88_BETA = 0.0042;

/**
 * Add spin-polarized B88 GGA gradient-correction to ε_xc, v_ρ↑, v_ρ↓,
 * v_γ↑↑, v_γ↓↓. The γ_↑↓ derivative is identically zero for B88
 * (the functional doesn't depend on the cross-spin gradient).
 *
 * Sign: this is a NEGATIVE correction on top of LSDA Slater.
 *
 * @param coef Mixing coefficient (B3-style hybrids use 0.72; pure B88 = 1).
 */
export function addB88X_spin(
  rhoUp: Float64Array,
  rhoDn: Float64Array,
  gammaUU: Float64Array,
  gammaDD: Float64Array,
  coef: number,
  epsXc: Float64Array,
  vRhoUp: Float64Array,
  vRhoDn: Float64Array,
  vGammaUU: Float64Array,
  vGammaDD: Float64Array,
): void {
  const n = rhoUp.length;
  for (let p = 0; p < n; p++) {
    const ru = rhoUp[p]!;
    const rd = rhoDn[p]!;
    const rho = ru + rd;
    if (rho < EPS_RHO) continue;

    // Per-spin pieces (independent — no spin coupling in B88).
    addB88OneSpin(ru, gammaUU[p]!, coef, p, rho, epsXc, vRhoUp, vGammaUU);
    addB88OneSpin(rd, gammaDD[p]!, coef, p, rho, epsXc, vRhoDn, vGammaDD);
    // v_γ↑↓ contribution is zero (B88 has no γ_↑↓ dependence).
  }
}

/** Per-spin σ B88 piece. Adds into ε_xc (per particle), v_ρσ, v_γσσ.
 *  The per-particle ε bookkeeping divides by total ρ (= ρ_↑ + ρ_↓). */
function addB88OneSpin(
  rhoSigma: number,
  gammaSS: number,
  coef: number,
  p: number,
  rhoTotal: number,
  epsXc: Float64Array,
  vRhoSigma: Float64Array,
  vGammaSS: Float64Array,
): void {
  if (rhoSigma < EPS_RHO) return;
  const r43 = Math.pow(rhoSigma, 4 / 3);
  const r13 = Math.cbrt(rhoSigma);
  const x = gammaSS > EPS_RHO ? Math.sqrt(gammaSS) / r43 : 0;
  // F(x) and F'(x) using arcsinh.
  const sh = Math.asinh(x);
  const root = Math.sqrt(1 + x * x);
  const D = 1 + 6 * B88_BETA * x * sh;
  const F = (x * x) / D;
  const Dp = 6 * B88_BETA * (sh + x / root);
  const Fp = (2 * x * D - x * x * Dp) / (D * D);

  // Per-volume e_x correction (per-particle ε divides by total ρ).
  const dEPerVol = -B88_BETA * r43 * F;
  epsXc[p]! += coef * dEPerVol / rhoTotal;

  // v_ρ_σ correction (per volume).
  const dVRhoSigma = -B88_BETA * (4 / 3) * r13 * (F - x * Fp);
  vRhoSigma[p]! += coef * dVRhoSigma;

  // v_γ_σσ correction (per volume), well-defined for γ > 0; → 0 as γ → 0.
  if (gammaSS > EPS_RHO) {
    const dVGamma = -B88_BETA * Fp / (2 * Math.sqrt(gammaSS));
    vGammaSS[p]! += coef * dVGamma;
  }
}

// ─────────────────────────────────────────────────────────────
// LYP correlation — spin-polarized (Lee-Yang-Parr 1988, Miehlich
// 1989 integrated-by-parts form).
//
// Per-volume:
//   f_LYP^spin = A_0(ρ_↑, ρ_↓)
//              + A_↑↑(ρ_↑, ρ_↓) · γ_↑↑
//              + A_↑↓(ρ_↑, ρ_↓) · γ_↑↓
//              + A_↓↓(ρ_↑, ρ_↓) · γ_↓↓
//
// LYP is LINEAR in (γ_↑↑, γ_↑↓, γ_↓↓), so γ-derivatives are exactly
// the A_X coefficients. The four A_X are closed-form functions of
// (ρ_↑, ρ_↓):
//
//   ρ = ρ_↑ + ρ_↓,  u = ρ^(−1/3),  h = 1 + d·u,
//   E = exp(−c·u),  ω = E / (h·ρ^(11/3)),  δ = c·u + d·u/h
//
//   A_0  = −4a · ρ_↑·ρ_↓ / (ρ·h)
//          − a·b·ω · ρ_↑·ρ_↓ · 2^(11/3) · C_F · (ρ_↑^(8/3) + ρ_↓^(8/3))
//   A_↑↑ = a·b·ω · { ρ_↑·ρ_↓ · [−1/9 + δ/3 + (δ−11)/9 · ρ_↑/ρ] + ρ_↓² }
//   A_↓↓ = a·b·ω · { ρ_↑·ρ_↓ · [−1/9 + δ/3 + (δ−11)/9 · ρ_↓/ρ] + ρ_↑² }
//   A_↑↓ = a·b·ω · { −ρ_↑·ρ_↓ · (47 − 7δ)/9 + (4/3)·ρ² }
//
// Closed-shell collapse (verified algebraically): at ρ_↑ = ρ_↓ = ρ/2,
//   f_LYP^spin → −a·ρ/h − a·b·E·C_F·ρ/h + a·b·ω·(3 + 7δ)·ρ²·γ/72
//   v_γ^cs (= (2·A_↑↑ + A_↑↓)/4 by chain rule) → a·b·ω·(3 + 7δ)·ρ²/72
// matching `addLYPC` in `functional.ts` to 1e-12.
//
// First derivatives:
//   v_γ_↑↑ = A_↑↑   (analytic — LYP linear in γ)
//   v_γ_↑↓ = A_↑↓
//   v_γ_↓↓ = A_↓↓
//   v_↑    = ∂(A_0 + A_↑↑·γ_↑↑ + A_↑↓·γ_↑↓ + A_↓↓·γ_↓↓)/∂ρ_↑
//          via central FD on the analytic energy density (cheaper
//          than re-deriving 8 G_i pieces × 2 ρ-variables = 16
//          analytic ρ-derivatives by hand). FD step h ≈ 1e-5·ρ.
// ─────────────────────────────────────────────────────────────

const LYP_A = 0.04918;
const LYP_B = 0.132;
const LYP_C = 0.2533;
const LYP_D = 0.349;
const C_F  = 0.3 * Math.pow(3 * Math.PI * Math.PI, 2 / 3);
const POW_2_11_3 = Math.pow(2, 11 / 3);

interface LypCoefs {
  /** Constant (γ-independent) piece. */
  readonly A0: number;
  /** Coefficient of γ_↑↑ in f_LYP^spin (= ∂f/∂γ_↑↑ exactly). */
  readonly Auu: number;
  /** Coefficient of γ_↑↓ in f_LYP^spin (= ∂f/∂γ_↑↓ exactly). */
  readonly Aud: number;
  /** Coefficient of γ_↓↓ in f_LYP^spin (= ∂f/∂γ_↓↓ exactly). */
  readonly Add: number;
}

/** Closed-form A_X(ρ_↑, ρ_↓) for spin-polarized LYP. Returns zeros
 *  if total density is below EPS_RHO. */
function lypCoefficients(ru: number, rd: number): LypCoefs {
  const rho = ru + rd;
  if (rho < EPS_RHO) return { A0: 0, Auu: 0, Aud: 0, Add: 0 };
  const u  = Math.pow(rho, -1 / 3);
  const h  = 1 + LYP_D * u;
  const E  = Math.exp(-LYP_C * u);
  const omega = E / (h * Math.pow(rho, 11 / 3));
  const delta = LYP_C * u + LYP_D * u / h;
  const ab_omega = LYP_A * LYP_B * omega;

  const ru_rd = ru * rd;
  // Guard ρ_σ^(8/3) for tiny ρ_σ (the spin-decomposed C_F piece);
  // any ρ_σ < EPS_RHO contributes zero.
  // ρ^(8/3) = ρ² · ρ^(2/3) = ρ² · cbrt(ρ²).
  const ru83 = ru >= EPS_RHO ? ru * ru * Math.cbrt(ru * ru) : 0;
  const rd83 = rd >= EPS_RHO ? rd * rd * Math.cbrt(rd * rd) : 0;

  const A0 = -4 * LYP_A * ru_rd / (rho * h)
           - ab_omega * ru_rd * POW_2_11_3 * C_F * (ru83 + rd83);

  const Auu = ab_omega * (
      ru_rd * (-1 / 9 + delta / 3 + (delta - 11) / 9 * (ru / rho))
    + rd * rd
  );
  const Add = ab_omega * (
      ru_rd * (-1 / 9 + delta / 3 + (delta - 11) / 9 * (rd / rho))
    + ru * ru
  );
  const Aud = ab_omega * (
    -ru_rd * (47 - 7 * delta) / 9
    + (4 / 3) * rho * rho
  );

  return { A0, Auu, Aud, Add };
}

/** LYP energy density (per volume) at one grid point. Linear in
 *  (γ_↑↑, γ_↑↓, γ_↓↓), with closed-form A_X coefficients. */
function lypEpsPerVolume(
  ru: number, rd: number, gUU: number, gUD: number, gDD: number,
): number {
  const c = lypCoefficients(ru, rd);
  return c.A0 + c.Auu * gUU + c.Aud * gUD + c.Add * gDD;
}

/**
 * Add spin-polarized LYP correlation to ε_xc, v_ρ↑, v_ρ↓, v_γ↑↑,
 * v_γ↑↓, v_γ↓↓. γ-derivatives are analytic (LYP linear in γ);
 * ρ-derivatives via central FD on the closed-form energy density.
 *
 * @param coef Mixing coefficient (BLYP uses 1, B3LYP5 uses 0.81).
 */
export function addLYPC_spin(
  rhoUp: Float64Array,
  rhoDn: Float64Array,
  gammaUU: Float64Array,
  gammaUD: Float64Array,
  gammaDD: Float64Array,
  coef: number,
  epsXc: Float64Array,
  vRhoUp: Float64Array,
  vRhoDn: Float64Array,
  vGammaUU: Float64Array,
  vGammaUD: Float64Array,
  vGammaDD: Float64Array,
): void {
  const n = rhoUp.length;
  for (let p = 0; p < n; p++) {
    const ru = rhoUp[p]!;
    const rd = rhoDn[p]!;
    const rho = ru + rd;
    if (rho < EPS_RHO) continue;
    const gUU = gammaUU[p]!;
    const gUD = gammaUD[p]!;
    const gDD = gammaDD[p]!;

    // Analytic energy density + γ derivatives.
    const c = lypCoefficients(ru, rd);
    const ePerVol = c.A0 + c.Auu * gUU + c.Aud * gUD + c.Add * gDD;
    epsXc[p]!    += coef * ePerVol / rho;
    vGammaUU[p]! += coef * c.Auu;
    vGammaUD[p]! += coef * c.Aud;
    vGammaDD[p]! += coef * c.Add;

    // ρ-derivatives: central FD on the full energy density.
    // Step h ~ ρ_σ · 1e-5; rare boundary case ρ_σ tiny is handled
    // via lypCoefficients returning zero at sub-EPS_RHO.
    const huu = Math.max(ru * 1e-5, 1e-9);
    const hdd = Math.max(rd * 1e-5, 1e-9);
    const fUp_p = lypEpsPerVolume(ru + huu, rd, gUU, gUD, gDD);
    const fUp_m = lypEpsPerVolume(Math.max(ru - huu, EPS_RHO), rd, gUU, gUD, gDD);
    const fDn_p = lypEpsPerVolume(ru, rd + hdd, gUU, gUD, gDD);
    const fDn_m = lypEpsPerVolume(ru, Math.max(rd - hdd, EPS_RHO), gUU, gUD, gDD);
    vRhoUp[p]! += coef * (fUp_p - fUp_m) / (2 * huu);
    vRhoDn[p]! += coef * (fDn_p - fDn_m) / (2 * hdd);
  }
}

// ─────────────────────────────────────────────────────────────
// Spin-polarized GGA dispatch.
//
// Mirrors `evalXC` (closed-shell) for the BVWN5 / B3VWN5 functional
// kinds — those that compose Slater + B88 + VWN5 (no LYP). The
// `b3vwn5` mix has 0.20 HF exchange (handled in the Fock matrix,
// not here) and 0.80 Slater + 0.72 B88 + 1.0 VWN5 in the kernel.
//
// LYP-bearing functionals (`blyp`, `b3lyp5`) need the spin-polarized
// LYP form (Miehlich 1989) — its own session. They throw here.
// ─────────────────────────────────────────────────────────────

/** Functional kinds that have a spin-polarized form implemented. */
export type SpinFunctionalKind = "lda-svwn" | "bvwn5" | "b3vwn5" | "blyp" | "b3lyp5";

/** True if the functional has a spin-polarized form available. */
export function hasSpinPolarizedForm(kind: string): kind is SpinFunctionalKind {
  return kind === "lda-svwn" || kind === "bvwn5" || kind === "b3vwn5"
      || kind === "blyp"     || kind === "b3lyp5";
}

export interface XCSpinOutputs {
  readonly epsXc: Float64Array;
  readonly vRhoUp: Float64Array;
  readonly vRhoDn: Float64Array;
  readonly vGammaUU: Float64Array;
  readonly vGammaUD: Float64Array;
  readonly vGammaDD: Float64Array;
}

/**
 * Evaluate the spin-polarized XC functional `kind` at every grid
 * point. ρ_↑, ρ_↓, γ_↑↑, γ_↑↓, γ_↓↓ come in as pre-built arrays.
 * Output arrays are pre-allocated by the caller and overwritten.
 *
 * Throws for LYP-bearing functionals (`blyp`, `b3lyp5`) — those need
 * the spin-polarized LYP form, deferred to stage 15b-LYP.
 */
export function evalXC_GGA_spin(
  kind: SpinFunctionalKind | "lda-svwn",
  rhoUp: Float64Array,
  rhoDn: Float64Array,
  gammaUU: Float64Array,
  gammaUD: Float64Array,
  gammaDD: Float64Array,
  out: XCSpinOutputs,
): void {
  out.epsXc.fill(0); out.vRhoUp.fill(0); out.vRhoDn.fill(0);
  out.vGammaUU.fill(0); out.vGammaUD.fill(0); out.vGammaDD.fill(0);
  // ── LSDA core (Slater + VWN5) — bundled together. ─────────
  // For LDA / BVWN5 / B3VWN5 we keep VWN5 at 1.0; for BLYP we
  // strip VWN5 entirely (BLYP correlation is just LYP); for B3LYP5
  // we scale VWN5 to 0.19. The Slater piece is kept at 1.0 here
  // and adjusted below to 0.80 for any B3-style hybrid.
  evalXC_LSDA(rhoUp, rhoDn, out.epsXc, out.vRhoUp, out.vRhoDn);
  if (kind === "lda-svwn") return;

  // ── Adjust Slater scale: B3-style hybrids → 0.80 (= 1 − 0.20 HF). ──
  const isB3 = kind === "b3vwn5" || kind === "b3lyp5";
  if (isB3) {
    addSlaterScaleAdjust(rhoUp, rhoDn, -0.20, out.epsXc, out.vRhoUp, out.vRhoDn);
  }

  // ── Adjust VWN5 correlation scale per functional. ─────────
  // BVWN5/BLYP: 1.0/0.0 (already 1.0; subtract 1.0 for BLYP).
  // B3VWN5:     1.0 (no change).
  // B3LYP5:     0.19 (subtract 0.81).
  if (kind === "blyp") {
    addVWN5ScaleAdjust(rhoUp, rhoDn, -1.0, out.epsXc, out.vRhoUp, out.vRhoDn);
  } else if (kind === "b3lyp5") {
    addVWN5ScaleAdjust(rhoUp, rhoDn, -0.81, out.epsXc, out.vRhoUp, out.vRhoDn);
  }

  // ── B88 GGA exchange correction. ──────────────────────────
  // Pure GGA: 1.00 B88. B3-style: 0.72 B88.
  const b88Coef = isB3 ? 0.72 : 1.0;
  addB88X_spin(rhoUp, rhoDn, gammaUU, gammaDD, b88Coef,
    out.epsXc, out.vRhoUp, out.vRhoDn, out.vGammaUU, out.vGammaDD);
  // B88 has no γ_↑↓ dependence; vGammaUD untouched here.

  // ── LYP GGA correlation (BLYP / B3LYP5 only). ─────────────
  if (kind === "blyp" || kind === "b3lyp5") {
    const lypCoef = kind === "blyp" ? 1.0 : 0.81;
    addLYPC_spin(rhoUp, rhoDn, gammaUU, gammaUD, gammaDD, lypCoef,
      out.epsXc, out.vRhoUp, out.vRhoDn,
      out.vGammaUU, out.vGammaUD, out.vGammaDD);
  } else {
    void gammaUD;        // BVWN5/B3VWN5: no γ_↑↓ dependence (B88 + VWN5 only).
  }
}

/** Adjust the LSDA VWN5 contribution by `coef` × (closed-shell VWN5
 *  contribution at ρ_↑ = ρ_↑, ρ_↓ = ρ_↓). Implemented by re-running
 *  the spin-polarized VWN5 piece in `evalXC_LSDA` with a -1 contrast.
 *  Used to scale VWN5 to 0/0.19 in BLYP/B3LYP5. */
function addVWN5ScaleAdjust(
  rhoUp: Float64Array,
  rhoDn: Float64Array,
  coef: number,
  epsXc: Float64Array,
  vRhoUp: Float64Array,
  vRhoDn: Float64Array,
): void {
  // Re-evaluate LSDA = Slater + VWN5, subtract the Slater piece, then
  // scale the residual (= VWN5 alone) by `coef`. This avoids exposing
  // a separate `addVWN5C_spin` while staying numerically consistent
  // with the closed-shell `addVWN5C` path.
  const n = rhoUp.length;
  const epsTmp = new Float64Array(n);
  const vUpTmp = new Float64Array(n);
  const vDnTmp = new Float64Array(n);
  evalXC_LSDA(rhoUp, rhoDn, epsTmp, vUpTmp, vDnTmp);
  // Strip the Slater piece (analytic).
  for (let p = 0; p < n; p++) {
    const ru = rhoUp[p]!;
    const rd = rhoDn[p]!;
    const rho = ru + rd;
    if (rho < EPS_RHO) continue;
    const ru13 = ru >= EPS_RHO ? Math.cbrt(ru) : 0;
    const rd13 = rd >= EPS_RHO ? Math.cbrt(rd) : 0;
    const exSlater = SLATER_LSDA_E * (ru * ru13 + rd * rd13);
    epsTmp[p] = epsTmp[p]! - exSlater / rho;
    vUpTmp[p] = vUpTmp[p]! - SLATER_LSDA_V * ru13;
    vDnTmp[p] = vDnTmp[p]! - SLATER_LSDA_V * rd13;
    // What remains is the VWN5 piece — apply the requested adjustment.
    epsXc[p]!  += coef * epsTmp[p]!;
    vRhoUp[p]! += coef * vUpTmp[p]!;
    vRhoDn[p]! += coef * vDnTmp[p]!;
  }
}

/** Adjust the LSDA Slater contribution by `coef` × (the closed-shell
 *  Slater LSDA contribution). Used to scale Slater down to 0.80 in
 *  B3-style hybrids without re-implementing the Slater piece.
 *  ε is per-particle (= e_x_per_volume / ρ_total). */
function addSlaterScaleAdjust(
  rhoUp: Float64Array,
  rhoDn: Float64Array,
  coef: number,
  epsXc: Float64Array,
  vRhoUp: Float64Array,
  vRhoDn: Float64Array,
): void {
  const n = rhoUp.length;
  for (let p = 0; p < n; p++) {
    const ru = rhoUp[p]!;
    const rd = rhoDn[p]!;
    const rho = ru + rd;
    if (rho < EPS_RHO) continue;
    const ru13 = ru >= EPS_RHO ? Math.cbrt(ru) : 0;
    const rd13 = rd >= EPS_RHO ? Math.cbrt(rd) : 0;
    const dEx = SLATER_LSDA_E * (ru * ru13 + rd * rd13);
    epsXc[p]! += coef * dEx / rho;
    vRhoUp[p]! += coef * SLATER_LSDA_V * ru13;
    vRhoDn[p]! += coef * SLATER_LSDA_V * rd13;
  }
}

// ─────────────────────────────────────────────────────────────
// Triplet LDA XC kernel for closed-shell TDDFT.
//
// Closed-shell TDDFT/TDA needs two spin-resolved kernels:
//
//   f_↑↑(ρ_↑, ρ_↓) ≡ ∂v_↑/∂ρ_↑     (same-spin response)
//   f_↑↓(ρ_↑, ρ_↓) ≡ ∂v_↑/∂ρ_↓     (opposite-spin response)
//
// evaluated at the closed-shell baseline ρ_↑ = ρ_↓ = ρ/2.
// The Casida singlet / triplet kernels are then
//
//   f_xc^singlet = f_↑↑ + f_↑↓        (already covered by the
//                                       closed-shell `evalXCKernelLDA`,
//                                       up to the factor-of-2
//                                       convention used in tda-dft.ts)
//   f_xc^triplet = f_↑↑ − f_↑↓
//
// To match the SAME factor-of-2 convention used in tda-dft.ts
// (the matrix build does `2 · K_xc`), we return half the triplet
// kernel here:
//
//   evalXCKernelLDA_triplet(ρ) = (f_↑↑ − f_↑↓) / 2,
//
// so that `2 · K_xc^triplet = ∫ ψ_ia · (f_↑↑ − f_↑↓) · ψ_jb dr`,
// the standard Casida triplet integrand.
//
// Method: central FD on the LSDA v_↑.
//   f_↑↑ ≈ (v_↑(ρ/2 + h, ρ/2)   − v_↑(ρ/2 − h, ρ/2))   / (2h)
//   f_↑↓ ≈ (v_↑(ρ/2,   ρ/2 + h) − v_↑(ρ/2,   ρ/2 − h)) / (2h)
//
// Step h = max(ρ · 1e-5, 1e-9) — same scale as the singlet kernel.
// Cost: 4 evalXC_LSDA calls (vs 2 for the singlet kernel). Trivial.
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
// Triplet GGA XC kernel for closed-shell TDDFT.
//
// Closed-shell triplet GGA kernel matrix element:
//   K^triplet_GGA[ia, jb] = ∫ w · {
//       fRR^t · ψ_ia · ψ_jb
//     + 2·fRG^t · (ψ_ia · α_jb + α_ia · ψ_jb)
//     + 4·fGG^t · α_ia · α_jb
//     + 2·vG^t · ∇ψ_ia · ∇ψ_jb
//   } dr
// where ψ_ia = φ_i^MO · φ_a^MO, α_ia = ∇ρ · ∇ψ_ia, and the four
// triplet kernel components are
//   fRR^t = (H_{ρ↑ρ↑} − H_{ρ↑ρ↓}) / 2
//   fRG^t = (H_{ρ↑γ↑↑} − H_{ρ↑γ↓↓}) / 4
//   fGG^t = (H_{γ↑↑γ↑↑} − H_{γ↑↑γ↓↓}) / 8
//   vG^t  = (2·v_γ↑↑ − v_γ↑↓) / 4
//
// (The 1/2, 1/4, 1/8, 1/4 factors are chosen so that the SAME
// integrand template `2·K_xc` used by the closed-shell singlet
// path lands exactly on the textbook triplet integrand. Derivation:
// `∂²f_xc/∂c_a∂c_b` on the |T_z=0⟩ symmetric variation
// δρ_↑ = +ψ/√2, δρ_↓ = −ψ/√2.)
//
// Method: central FD on the spin-polarized v_↑ and v_γ↑↑ at the
// closed-shell baseline ρ_↑=ρ_↓=ρ/2, γ_↑↑=γ_↑↓=γ_↓↓=γ/4.
//   ∂v_↑/∂ρ_↑     ⇒ H_{ρ↑ρ↑}         (perturb ρ_↑ alone)
//   ∂v_↑/∂ρ_↓     ⇒ H_{ρ↑ρ↓}         (perturb ρ_↓ alone)
//   ∂v_γ↑↑/∂ρ_↑   ⇒ H_{ρ↑γ↑↑}        (Maxwell: same as ∂v_↑/∂γ_↑↑)
//   ∂v_γ↑↑/∂ρ_↓   ⇒ H_{ρ↓γ↑↑} = H_{ρ↑γ↓↓} (closed-shell symmetry)
//   ∂v_γ↑↑/∂γ_↑↑  ⇒ H_{γ↑↑γ↑↑}       (perturb γ_↑↑ alone)
//   ∂v_γ↑↑/∂γ_↓↓  ⇒ H_{γ↑↑γ↓↓}       (perturb γ_↓↓ alone)
//
// Cost: 4 FD pairs × 1 evalXC_GGA_spin call each + 1 baseline eval = 9
// evaluations of the spin-polarized GGA. Each over the full grid; for
// H₂O Lebedev-110 this is ~16k points and ~200 ms total. Trivial.
// ─────────────────────────────────────────────────────────────

export interface XCKernelGGATriplet {
  /** (H_{ρ↑ρ↑} − H_{ρ↑ρ↓}) / 2     — pairs with ψ·ψ. */
  readonly fRR: Float64Array;
  /** (H_{ρ↑γ↑↑} − H_{ρ↑γ↓↓}) / 4   — pairs with [ψα + αψ] (× 2 in K_xc). */
  readonly fRG: Float64Array;
  /** (H_{γ↑↑γ↑↑} − H_{γ↑↑γ↓↓}) / 8 — pairs with α·α (× 4 in K_xc). */
  readonly fGG: Float64Array;
  /** (2·v_γ↑↑ − v_γ↑↓) / 4         — pairs with ∇ψ·∇ψ (× 2 in K_xc).
   *  For B88 v_γ↑↓ = 0, so this is just v_γ↑↑/2. */
  readonly vG: Float64Array;
}

/**
 * Evaluate the closed-shell triplet GGA XC kernel components at every
 * grid point. The four returned arrays plug directly into the same
 * integrand template the singlet GGA path uses, with `2·K_xc` giving
 * the textbook triplet matrix-element integrand.
 *
 * Only the BVWN5 / B3VWN5 functionals are wired through here (B88
 * exchange + VWN5 correlation). For BLYP / B3LYP5, the spin-polarized
 * LYP form is required (deferred to stage 15b-LYP).
 */
export function evalXCKernelGGA_triplet(
  kind: SpinFunctionalKind,
  rho: Float64Array,
  gamma: Float64Array,
): XCKernelGGATriplet {
  if (kind === "lda-svwn") {
    // Pure-LDA path: no GGA pieces. Return LDA triplet kernel in fRR
    // and zeros elsewhere — caller can use the same integrand template.
    const n = rho.length;
    return {
      fRR: evalXCKernelLDA_triplet(rho),
      fRG: new Float64Array(n),
      fGG: new Float64Array(n),
      vG:  new Float64Array(n),
    };
  }

  const n = rho.length;
  // Closed-shell baseline arrays.
  const half = new Float64Array(n);
  const quart = new Float64Array(n);
  const stepArr = new Float64Array(n);
  for (let p = 0; p < n; p++) {
    half[p]  = rho[p]! / 2;
    quart[p] = gamma[p]! / 4;
    // Step size: scale to the local density / gradient magnitude.
    const r = rho[p]!;
    stepArr[p] = r > EPS_RHO ? Math.max(r * 1e-5, 1e-9) : 0;
  }

  // Allocate scratch for the spin-polarized evaluator.
  const allocOuts = (): XCSpinOutputs => ({
    epsXc:    new Float64Array(n),
    vRhoUp:   new Float64Array(n),
    vRhoDn:   new Float64Array(n),
    vGammaUU: new Float64Array(n),
    vGammaUD: new Float64Array(n),
    vGammaDD: new Float64Array(n),
  });
  const baseline = allocOuts();
  const fdPlus   = allocOuts();
  const fdMinus  = allocOuts();

  // ── Baseline at closed-shell: needed for vG^t = (2 v_γ↑↑ − v_γ↑↓) / 4. ──
  evalXC_GGA_spin(kind, half, half, quart, quart, quart, baseline);

  // Working perturbed-density buffers (reused across the 4 FD pairs).
  const ru = new Float64Array(n);
  const rd = new Float64Array(n);
  const guu = new Float64Array(n);
  const gud = new Float64Array(n);
  const gdd = new Float64Array(n);

  // Output accumulators.
  const fRR = new Float64Array(n);
  const fRG = new Float64Array(n);
  const fGG = new Float64Array(n);
  const vG  = new Float64Array(n);

  // Fill baseline copies.
  const reset = (): void => {
    ru.set(half); rd.set(half);
    guu.set(quart); gud.set(quart); gdd.set(quart);
  };

  // ── FD pair 1: perturb ρ_↑. ──────────────────────────────────
  reset();
  for (let p = 0; p < n; p++) {
    const r = rho[p]!;
    if (r < EPS_RHO) continue;
    ru[p] = half[p]! + stepArr[p]!;
  }
  evalXC_GGA_spin(kind, ru, rd, guu, gud, gdd, fdPlus);
  reset();
  for (let p = 0; p < n; p++) {
    const r = rho[p]!;
    if (r < EPS_RHO) continue;
    ru[p] = Math.max(half[p]! - stepArr[p]!, EPS_RHO);
  }
  evalXC_GGA_spin(kind, ru, rd, guu, gud, gdd, fdMinus);
  for (let p = 0; p < n; p++) {
    const h = stepArr[p]!;
    if (h === 0) continue;
    const Hrr_uu  = (fdPlus.vRhoUp[p]!   - fdMinus.vRhoUp[p]!)   / (2 * h);
    const Hrg_uu  = (fdPlus.vGammaUU[p]! - fdMinus.vGammaUU[p]!) / (2 * h);
    fRR[p] = fRR[p]! + Hrr_uu / 2;     // (H_{ρ↑ρ↑} − H_{ρ↑ρ↓}) / 2 — first half.
    fRG[p] = fRG[p]! + Hrg_uu / 4;     // (H_{ρ↑γ↑↑} − H_{ρ↑γ↓↓}) / 4 — first half.
  }

  // ── FD pair 2: perturb ρ_↓. ──────────────────────────────────
  reset();
  for (let p = 0; p < n; p++) {
    const r = rho[p]!;
    if (r < EPS_RHO) continue;
    rd[p] = half[p]! + stepArr[p]!;
  }
  evalXC_GGA_spin(kind, ru, rd, guu, gud, gdd, fdPlus);
  reset();
  for (let p = 0; p < n; p++) {
    const r = rho[p]!;
    if (r < EPS_RHO) continue;
    rd[p] = Math.max(half[p]! - stepArr[p]!, EPS_RHO);
  }
  evalXC_GGA_spin(kind, ru, rd, guu, gud, gdd, fdMinus);
  for (let p = 0; p < n; p++) {
    const h = stepArr[p]!;
    if (h === 0) continue;
    const Hrr_ud  = (fdPlus.vRhoUp[p]!   - fdMinus.vRhoUp[p]!)   / (2 * h);
    const Hrg_ud  = (fdPlus.vGammaUU[p]! - fdMinus.vGammaUU[p]!) / (2 * h);
    fRR[p] = fRR[p]! - Hrr_ud / 2;     // subtract H_{ρ↑ρ↓} / 2.
    fRG[p] = fRG[p]! - Hrg_ud / 4;     // subtract H_{ρ↑γ↓↓} / 4 (= H_{ρ↓γ↑↑} by sym).
  }

  // ── FD pair 3: perturb γ_↑↑ (γ_↑↓, γ_↓↓ fixed at γ/4). ─────
  // γ-step: scale to γ/4 magnitude. Use a separate step array.
  const gStep = new Float64Array(n);
  for (let p = 0; p < n; p++) {
    const g = quart[p]!;
    gStep[p] = g > EPS_RHO ? Math.max(g * 1e-5, 1e-9) : 0;
  }
  reset();
  for (let p = 0; p < n; p++) {
    const h = gStep[p]!;
    if (h === 0) continue;
    guu[p] = quart[p]! + h;
  }
  evalXC_GGA_spin(kind, ru, rd, guu, gud, gdd, fdPlus);
  reset();
  for (let p = 0; p < n; p++) {
    const h = gStep[p]!;
    if (h === 0) continue;
    guu[p] = Math.max(quart[p]! - h, 0);
  }
  evalXC_GGA_spin(kind, ru, rd, guu, gud, gdd, fdMinus);
  for (let p = 0; p < n; p++) {
    const h = gStep[p]!;
    if (h === 0) continue;
    const Hgg_uu = (fdPlus.vGammaUU[p]! - fdMinus.vGammaUU[p]!) / (2 * h);
    fGG[p] = fGG[p]! + Hgg_uu / 8;
  }

  // ── FD pair 4: perturb γ_↓↓. ───────────────────────────────
  reset();
  for (let p = 0; p < n; p++) {
    const h = gStep[p]!;
    if (h === 0) continue;
    gdd[p] = quart[p]! + h;
  }
  evalXC_GGA_spin(kind, ru, rd, guu, gud, gdd, fdPlus);
  reset();
  for (let p = 0; p < n; p++) {
    const h = gStep[p]!;
    if (h === 0) continue;
    gdd[p] = Math.max(quart[p]! - h, 0);
  }
  evalXC_GGA_spin(kind, ru, rd, guu, gud, gdd, fdMinus);
  for (let p = 0; p < n; p++) {
    const h = gStep[p]!;
    if (h === 0) continue;
    const Hgg_ud = (fdPlus.vGammaUU[p]! - fdMinus.vGammaUU[p]!) / (2 * h);
    fGG[p] = fGG[p]! - Hgg_ud / 8;
  }

  // ── vG^t at the baseline. ───────────────────────────────────
  for (let p = 0; p < n; p++) {
    vG[p] = (2 * baseline.vGammaUU[p]! - baseline.vGammaUD[p]!) / 4;
  }

  return { fRR, fRG, fGG, vG };
}

/**
 * Evaluate the closed-shell triplet LDA XC kernel ½(f_↑↑ − f_↑↓)
 * at every grid point. Returns a fresh Float64Array of length nGrid.
 *
 * Convention: this returns HALF the textbook (f_↑↑ − f_↑↓), so the
 * matrix build in tda-dft.ts can use the SAME `2 · K_xc` factor it
 * uses for the singlet (cancels to ∫ ψ · (f_↑↑ − f_↑↓) · ψ dr).
 */
export function evalXCKernelLDA_triplet(rho: Float64Array): Float64Array {
  const n = rho.length;
  const fxc = new Float64Array(n);
  // Closed-shell baseline ρ_↑ = ρ_↓ = ρ/2 + step arrays.
  const rhoUp_p = new Float64Array(n);
  const rhoUp_m = new Float64Array(n);
  const rhoDn_p = new Float64Array(n);
  const rhoDn_m = new Float64Array(n);
  const hArr    = new Float64Array(n);

  for (let p = 0; p < n; p++) {
    const r = rho[p]!;
    if (r < 2 * EPS_RHO) { hArr[p] = 0; continue; }
    const half = r / 2;
    const h = Math.max(half * 1e-5, 1e-9);
    hArr[p] = h;
    rhoUp_p[p] = half + h;   rhoUp_m[p] = Math.max(half - h, EPS_RHO);
    rhoDn_p[p] = half + h;   rhoDn_m[p] = Math.max(half - h, EPS_RHO);
  }

  const epsTmp = new Float64Array(n);
  const vUp_pUp = new Float64Array(n);  // v_↑(ρ_↑+h, ρ_↓)
  const vDn_pUp = new Float64Array(n);
  const vUp_mUp = new Float64Array(n);  // v_↑(ρ_↑−h, ρ_↓)
  const vDn_mUp = new Float64Array(n);
  const vUp_pDn = new Float64Array(n);  // v_↑(ρ_↑, ρ_↓+h)
  const vDn_pDn = new Float64Array(n);
  const vUp_mDn = new Float64Array(n);  // v_↑(ρ_↑, ρ_↓−h)
  const vDn_mDn = new Float64Array(n);

  // Build the four perturbed (ρ_↑, ρ_↓) configurations on closed-shell ρ/2.
  const halfArr = new Float64Array(n);
  for (let p = 0; p < n; p++) halfArr[p] = rho[p]! / 2;

  // (ρ_↑+h, ρ_↓ = ρ/2)
  evalXC_LSDA(rhoUp_p, halfArr, epsTmp, vUp_pUp, vDn_pUp);
  // (ρ_↑−h, ρ_↓ = ρ/2)
  evalXC_LSDA(rhoUp_m, halfArr, epsTmp, vUp_mUp, vDn_mUp);
  // (ρ_↑ = ρ/2, ρ_↓+h)
  evalXC_LSDA(halfArr, rhoDn_p, epsTmp, vUp_pDn, vDn_pDn);
  // (ρ_↑ = ρ/2, ρ_↓−h)
  evalXC_LSDA(halfArr, rhoDn_m, epsTmp, vUp_mDn, vDn_mDn);

  for (let p = 0; p < n; p++) {
    const h = hArr[p]!;
    if (h === 0) continue;
    const fUU = (vUp_pUp[p]! - vUp_mUp[p]!) / (2 * h);
    const fUD = (vUp_pDn[p]! - vUp_mDn[p]!) / (2 * h);
    fxc[p] = 0.5 * (fUU - fUD);
  }
  return fxc;
}
