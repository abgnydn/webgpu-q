// Tier 2 stage 4: LYP closed-shell correlation kernel — bug-catcher.
//
// The previous LYP attempt shipped a sign-error-grade bug (30–240
// mHa below PySCF B3LYP). The defensive moat for this attempt is a
// pointwise FD gradient self-test: v_ρ analytic must match
// ∂(ρ·ε_LYP)/∂ρ via central-FD, and same for v_γ. If a coefficient
// flips sign or carries the wrong factor, this test fails before the
// energy hierarchy ones do and the diagnosis is local.
//
// Pass bars:
//   • |v_ρ_analytic − v_ρ_FD| ≤ 1e-6 across (ρ, γ) sample grid
//   • |v_γ_analytic − v_γ_FD| ≤ 1e-6 across same grid
//   • At γ = 0 and uniform ρ, ε_LYP per particle matches a hand-
//     derived UEG value to 1e-10.
//
// Sign convention recap: in the GGA Fock build the kernel returns
// ε_xc per particle and v_ρ = ∂(ρ·ε)/∂ρ, v_γ = ∂(ρ·ε)/∂γ — the
// per-volume potentials the buildVxcGGA loop expects.
import { describe, expect, test } from "vitest";
import { addLYPC } from "../../src/chemistry/dft/functional.js";

/** Evaluate LYP correlation at a single (ρ, γ) point. */
function lypAt(rho: number, gamma: number): {
  eps: number; vRho: number; vGamma: number; fPerVol: number;
} {
  const r = new Float64Array([rho]);
  const g = new Float64Array([gamma]);
  const eps = new Float64Array(1);
  const vR  = new Float64Array(1);
  const vG  = new Float64Array(1);
  addLYPC(r, g, 1, eps, vR, vG);
  return { eps: eps[0]!, vRho: vR[0]!, vGamma: vG[0]!, fPerVol: rho * eps[0]! };
}

describe("LYP correlation: analytic ↔ central-FD agreement", () => {
  // (ρ, γ) span four regimes:
  //   • low ρ, vanishing γ  (tail of a molecule)
  //   • moderate ρ, low γ   (uniform-ish bulk)
  //   • moderate ρ, high γ  (bond region)
  //   • high ρ, high γ      (core)
  // ρ stays well above EPS_RHO so the conditional clip doesn't bias FD.
  const points: Array<[number, number]> = [
    [0.01,  1e-6],
    [0.05,  1e-4],
    [0.10,  1e-3],
    [0.25,  0.01],
    [0.50,  0.10],
    [1.00,  1.00],
    [2.00,  4.00],
  ];

  for (const [rho0, g0] of points) {
    test(`v_ρ analytic ≈ central-FD at ρ=${rho0}, γ=${g0}`, () => {
      // Choose FD step that's fp-safe — smaller of |ρ|·1e-5 and 1e-7,
      // floored above 1e-9 so the round-off floor stays below the
      // 1e-6 pass bar.
      const h = Math.max(rho0 * 1e-5, 1e-8);
      const fp = lypAt(rho0 + h, g0).fPerVol;
      const fm = lypAt(rho0 - h, g0).fPerVol;
      const fdVRho = (fp - fm) / (2 * h);
      const an = lypAt(rho0, g0).vRho;
      expect(Math.abs(an - fdVRho)).toBeLessThan(Math.max(1e-6, Math.abs(an) * 1e-5));
    });

    test(`v_γ analytic ≈ central-FD at ρ=${rho0}, γ=${g0}`, () => {
      // FD step on γ; |γ|·1e-4 is plenty since v_γ is linear in γ.
      const hG = Math.max(g0 * 1e-4, 1e-9);
      const fp = lypAt(rho0, g0 + hG).fPerVol;
      const fm = lypAt(rho0, g0 - hG).fPerVol;
      const fdVGamma = (fp - fm) / (2 * hG);
      const an = lypAt(rho0, g0).vGamma;
      expect(Math.abs(an - fdVGamma)).toBeLessThan(Math.max(1e-7, Math.abs(an) * 1e-5));
    });
  }
});

describe("LYP correlation: uniform-electron-gas closed-form", () => {
  // At γ = 0, the MSSP closed-shell LYP energy density per particle
  // collapses to:
  //   ε_LYP^UEG(ρ) = −a/h − a·b·C_F·E/h
  //               = −(a/h)·(1 + b·C_F·exp(−c·ρ^(−1/3)))
  // where h = 1 + d·ρ^(−1/3). This form is BOUNDED in the high-
  // density limit (ρ → ∞ ⇒ ε → −(a + b·C_F) ≈ −68 mHa), matching
  // the published LYP UEG envelope. At ρ = 1 (r_s ≈ 0.62) it gives
  // ≈ −47 mHa per electron, consistent with the LYP UEG curve.
  const a = 0.04918, b = 0.132, c = 0.2533, d = 0.349;
  const C_F = 0.3 * Math.pow(3 * Math.PI * Math.PI, 2 / 3);

  for (const rho of [0.05, 0.5, 1.0, 5.0]) {
    test(`ρ = ${rho}: ε_LYP(γ=0) matches closed form`, () => {
      const u = Math.pow(rho, -1 / 3);
      const h = 1 + d * u;
      const expected = -(a / h) * (1 + b * C_F * Math.exp(-c * u));
      const got = lypAt(rho, 0).eps;
      expect(Math.abs(got - expected)).toBeLessThan(1e-10);
    });
  }
});

describe("LYP correlation: physical sign + scale", () => {
  // Note on point-wise sign: bare LYP per-particle is NOT
  // guaranteed negative at every (ρ, γ) — even at moderate reduced
  // gradients (s ≈ 1–4) the γ-coupling f_C term can dominate and
  // flip ε locally. The physical statement is that the integral
  // ∫ f_LYP dr over a realistic CLOSED-SHELL molecular density is
  // negative (verified by the energy-hierarchy tests on H₂ / H₂O /
  // BeH₂ in dft.test.ts). Here we check only the unconditional
  // facts the math gives us:
  //   • ε_LYP(γ = 0) < 0 — UEG limit is strictly negative.
  //   • v_γ > 0 — ω > 0 and 73 + 11δ > 0, so the analytic v_γ is
  //     strictly positive everywhere ρ > EPS_RHO.
  test("ε_LYP(γ = 0) < 0 across the ρ grid (UEG limit)", () => {
    for (const rho of [0.001, 0.01, 0.1, 1, 10, 100]) {
      expect(lypAt(rho, 0).eps).toBeLessThan(0);
    }
  });
  test("v_γ > 0 across the (ρ, γ) sample grid", () => {
    for (const rho of [0.01, 0.1, 1, 10]) {
      for (const g of [0, 0.01, 1, 100]) {
        expect(lypAt(rho, g).vGamma).toBeGreaterThan(0);
      }
    }
  });
});
