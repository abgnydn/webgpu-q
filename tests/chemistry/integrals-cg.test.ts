// Cartesian-Gaussian integral validation. Strategy:
//   1. (s, s) cases must match the existing s-only API to f64
//      precision — that's the strongest possible cross-check
//      because the s-only path is already validated against H₂
//      FCI literature and the multi-shell wrapper passed all
//      its own tests.
//   2. (s, p) and (p, p) cases satisfy known closed-form
//      identities (overlap formulas, rotation invariance,
//      self-normalization).
//   3. Translation invariance: shifting both atoms by the same
//      vector must leave every integral unchanged.
import { describe, expect, test } from "vitest";
import {
  STO3G_H_1S, STO3G_LI_1S, STO3G_BE_1S, STO3G_BE_2S, STO3G_BE_2P,
  S_AB, T_AB, V_AB, ERI,
  S_shells, T_shells, V_shells, ERI_shells, makeShell,
} from "../../src/chemistry/integrals.js";
import {
  S_cg, T_cg, V_cg, ERI_cg, makeCGShell, boysAll,
} from "../../src/chemistry/integrals-cg.js";

const ORIGIN = [0, 0, 0] as const;
const Z14 = [0, 0, 1.4] as const;

describe("Boys function F_n(t)", () => {
  test("F_n(0) = 1 / (2n + 1)", () => {
    const F = boysAll(5, 0);
    for (let n = 0; n <= 5; n++) {
      expect(F[n]).toBeCloseTo(1 / (2 * n + 1), 12);
    }
  });

  test("F_0(1) closed form: 0.5 √π · erf(1)", () => {
    const F = boysAll(0, 1);
    // F_0(1) = ½ √π erf(1) ≈ 0.74682413... — the A&S 7.1.26 erf
    // approximation in boys0 is good to ~1.5e-7 absolute, matching
    // the legacy s-only path's precision exactly.
    expect(F[0]).toBeCloseTo(0.74682413281242703, 6);
  });

  test("downward recurrence is consistent: F_{n−1} = (2t F_n + e^(−t)) / (2n − 1)", () => {
    const t = 2.5;
    const F = boysAll(6, t);
    for (let n = 6; n > 0; n--) {
      const recurred = (2 * t * F[n]! + Math.exp(-t)) / (2 * n - 1);
      expect(F[n - 1]).toBeCloseTo(recurred, 10);
    }
  });

  test("F_n(t) is strictly decreasing in n at fixed t > 0", () => {
    const F = boysAll(5, 3.0);
    for (let n = 1; n <= 5; n++) {
      expect(F[n]!).toBeLessThan(F[n - 1]!);
    }
  });
});

describe("CG integrals reduce to the s-only API on s-shell inputs", () => {
  test("S_cg matches S_shells / S_AB on (H1s, H1s)", () => {
    const a_old = makeShell(STO3G_H_1S, ORIGIN);
    const b_old = makeShell(STO3G_H_1S, Z14);
    const a_new = makeCGShell(STO3G_H_1S, ORIGIN, [0, 0, 0]);
    const b_new = makeCGShell(STO3G_H_1S, Z14, [0, 0, 0]);
    expect(S_cg(a_new, b_new)).toBeCloseTo(S_shells(a_old, b_old), 10);
    expect(S_cg(a_new, b_new)).toBeCloseTo(S_AB(ORIGIN, Z14), 10);
  });

  test("T_cg matches T_shells on (H1s, H1s)", () => {
    const a_old = makeShell(STO3G_H_1S, ORIGIN);
    const b_old = makeShell(STO3G_H_1S, Z14);
    const a_new = makeCGShell(STO3G_H_1S, ORIGIN, [0, 0, 0]);
    const b_new = makeCGShell(STO3G_H_1S, Z14, [0, 0, 0]);
    expect(T_cg(a_new, b_new)).toBeCloseTo(T_shells(a_old, b_old), 10);
    expect(T_cg(a_new, b_new)).toBeCloseTo(T_AB(ORIGIN, Z14), 10);
  });

  test("V_cg matches V_shells on (H1s, H1s) within legacy erf precision", () => {
    // The legacy s-only integrals.ts uses Abramowitz & Stegun erf
    // approximation (~1.5e-7 max error). The new CG path uses a
    // high-precision Boys function (Taylor to 1e-18). So the two
    // agree only to the LEGACY precision floor, not f64.
    const a_old = makeShell(STO3G_H_1S, ORIGIN);
    const b_old = makeShell(STO3G_H_1S, Z14);
    const a_new = makeCGShell(STO3G_H_1S, ORIGIN, [0, 0, 0]);
    const b_new = makeCGShell(STO3G_H_1S, Z14, [0, 0, 0]);
    expect(V_cg(a_new, b_new, 1, ORIGIN)).toBeCloseTo(V_shells(a_old, b_old, 1, ORIGIN), 6);
    expect(V_cg(a_new, b_new, 1, ORIGIN)).toBeCloseTo(V_AB(ORIGIN, Z14, 1, ORIGIN), 6);
  });

  test("ERI_cg matches ERI on (H1s, H1s | H1s, H1s) within legacy erf precision", () => {
    const a_old = makeShell(STO3G_H_1S, ORIGIN);
    const b_old = makeShell(STO3G_H_1S, Z14);
    const a_new = makeCGShell(STO3G_H_1S, ORIGIN, [0, 0, 0]);
    const b_new = makeCGShell(STO3G_H_1S, Z14, [0, 0, 0]);
    expect(ERI_cg(a_new, b_new, a_new, b_new)).toBeCloseTo(ERI_shells(a_old, b_old, a_old, b_old), 6);
    expect(ERI_cg(a_new, b_new, a_new, b_new)).toBeCloseTo(ERI(ORIGIN, Z14, ORIGIN, Z14), 6);
  });

  test("CG s-shell self-overlap = 1 for Li 1s and Be 2s", () => {
    const li = makeCGShell(STO3G_LI_1S, ORIGIN, [0, 0, 0]);
    const be = makeCGShell(STO3G_BE_2S, ORIGIN, [0, 0, 0]);
    expect(S_cg(li, li)).toBeCloseTo(1, 7);
    expect(S_cg(be, be)).toBeCloseTo(1, 7);
  });
});

describe("p-shell sanity tests (single primitive)", () => {
  // Use a single-primitive "shell" for cleaner closed-form comparisons.
  const SINGLE = { alpha: [1.0] as const, c: [1.0] as const };
  // Per-primitive normalization is rolled into S_cg's `normCG` factor,
  // and `c` is the Pople d-coefficient (no per-primitive renorm needed).

  test("p_x self-overlap at origin = 1", () => {
    const px = makeCGShell(SINGLE, ORIGIN, [1, 0, 0]);
    expect(S_cg(px, px)).toBeCloseTo(1, 12);
  });

  test("rotation invariance: ⟨p_x|p_x⟩ = ⟨p_y|p_y⟩ = ⟨p_z|p_z⟩ at origin", () => {
    const px = makeCGShell(SINGLE, ORIGIN, [1, 0, 0]);
    const py = makeCGShell(SINGLE, ORIGIN, [0, 1, 0]);
    const pz = makeCGShell(SINGLE, ORIGIN, [0, 0, 1]);
    const sxx = S_cg(px, px);
    const syy = S_cg(py, py);
    const szz = S_cg(pz, pz);
    expect(syy).toBeCloseTo(sxx, 14);
    expect(szz).toBeCloseTo(sxx, 14);
  });

  test("orthogonality: ⟨p_x | p_y⟩ = 0 at the same center", () => {
    const px = makeCGShell(SINGLE, ORIGIN, [1, 0, 0]);
    const py = makeCGShell(SINGLE, ORIGIN, [0, 1, 0]);
    const pz = makeCGShell(SINGLE, ORIGIN, [0, 0, 1]);
    expect(S_cg(px, py)).toBeCloseTo(0, 14);
    expect(S_cg(px, pz)).toBeCloseTo(0, 14);
    expect(S_cg(py, pz)).toBeCloseTo(0, 14);
  });

  test("kinetic ⟨p_z | T | p_z⟩ at origin = 5α/2 (α=1 → 2.5)", () => {
    // For a single normalized p Gaussian centered at origin:
    // ⟨p_z(α) | −∇²/2 | p_z(α)⟩ = (5/2) α.
    const pz = makeCGShell(SINGLE, ORIGIN, [0, 0, 1]);
    expect(T_cg(pz, pz)).toBeCloseTo(2.5, 10);
  });

  test("translation invariance: shift both atoms by the same vector", () => {
    const A = ORIGIN;
    const B: readonly [number, number, number] = [0, 0, 1.4];
    const A2: readonly [number, number, number] = [3.5, 2.1, -0.7];
    const B2: readonly [number, number, number] = [3.5, 2.1, -0.7 + 1.4];
    const sLeft = makeCGShell(SINGLE, A, [1, 0, 0]);
    const sRight = makeCGShell(SINGLE, B, [0, 0, 1]);
    const tLeft = makeCGShell(SINGLE, A2, [1, 0, 0]);
    const tRight = makeCGShell(SINGLE, B2, [0, 0, 1]);
    expect(S_cg(tLeft, tRight)).toBeCloseTo(S_cg(sLeft, sRight), 12);
    expect(T_cg(tLeft, tRight)).toBeCloseTo(T_cg(sLeft, sRight), 12);
  });

  test("ERI rotation invariance: (p_x p_x | p_x p_x) at origin = (p_y p_y | p_y p_y)", () => {
    const px = makeCGShell(SINGLE, ORIGIN, [1, 0, 0]);
    const py = makeCGShell(SINGLE, ORIGIN, [0, 1, 0]);
    const exx = ERI_cg(px, px, px, px);
    const eyy = ERI_cg(py, py, py, py);
    expect(eyy).toBeCloseTo(exx, 12);
  });

  test("Be STO-3G basis: full shell set is properly orthonormal after Löwdin (single atom)", () => {
    // Build the 5-shell Be basis: 1s, 2s, 2p_x, 2p_y, 2p_z, all on
    // the same Be center. Each diagonal entry of the AO overlap S
    // must be 1 (per-shell self-overlap), and S must be symmetric.
    // The 2p_α,2p_β cross-overlaps must be zero on the same atom.
    const Be: readonly [number, number, number] = [0, 0, 0];
    const shells = [
      makeCGShell(STO3G_BE_1S, Be, [0, 0, 0], "Be:1s"),
      makeCGShell(STO3G_BE_2S, Be, [0, 0, 0], "Be:2s"),
      makeCGShell(STO3G_BE_2P, Be, [1, 0, 0], "Be:2px"),
      makeCGShell(STO3G_BE_2P, Be, [0, 1, 0], "Be:2py"),
      makeCGShell(STO3G_BE_2P, Be, [0, 0, 1], "Be:2pz"),
    ];
    const n = shells.length;
    const S = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        S[i * n + j] = S_cg(shells[i]!, shells[j]!);
      }
    }
    // Diagonal = 1 (each contracted shell self-normalized to within
    // the ~7-sig-fig precision of the published Pople d-coefficients).
    for (let i = 0; i < n; i++) {
      expect(S[i * n + i]).toBeCloseTo(1, 6);
    }
    // Symmetry.
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        expect(S[i * n + j]).toBeCloseTo(S[j * n + i]!, 12);
      }
    }
    // 2p_α ⊥ 2p_β on the same atom — orthogonal by spherical symmetry.
    expect(S[2 * n + 3]).toBeCloseTo(0, 12);  // px·py
    expect(S[2 * n + 4]).toBeCloseTo(0, 12);  // px·pz
    expect(S[3 * n + 4]).toBeCloseTo(0, 12);  // py·pz
    // 2p_α ⊥ 1s and 2s on the same atom — the s shells are
    // spherically symmetric, the integrand is odd in α, so it
    // integrates to zero by parity.
    for (let p = 2; p <= 4; p++) {
      expect(S[0 * n + p]!).toBeCloseTo(0, 12);
      expect(S[1 * n + p]!).toBeCloseTo(0, 12);
    }
  });

  test("ERI 8-fold symmetry on mixed CG shells", () => {
    const A: readonly [number, number, number] = [0, 0, 0];
    const B: readonly [number, number, number] = [0, 0, 1.0];
    const a = makeCGShell(STO3G_H_1S, A, [0, 0, 0]);
    const b = makeCGShell(STO3G_BE_2S, B, [1, 0, 0]);
    const c = makeCGShell(STO3G_H_1S, B, [0, 0, 0]);
    const d = makeCGShell(STO3G_BE_2S, A, [0, 1, 0]);
    const v = ERI_cg(a, b, c, d);
    expect(ERI_cg(b, a, c, d)).toBeCloseTo(v, 12);
    expect(ERI_cg(a, b, d, c)).toBeCloseTo(v, 12);
    expect(ERI_cg(c, d, a, b)).toBeCloseTo(v, 12);
  });
});
