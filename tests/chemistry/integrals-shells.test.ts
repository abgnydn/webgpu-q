// Sanity tests for the multi-shell integral API added in Phase C.
// Confirms the Li 1s / Li 2s / H 1s shells are correctly normalized
// and that the shell-based contractions agree with the existing
// hard-coded H 1s contractions on the H test cases.
import { describe, expect, test } from "vitest";
import {
  STO3G_H_1S, STO3G_LI_1S, STO3G_LI_2S,
  S_AB, T_AB, V_AB, ERI,
  S_shells, T_shells, V_shells, ERI_shells,
  makeShell,
} from "../../src/chemistry/integrals.js";

const ORIGIN = [0, 0, 0] as const;
const Z_AXIS_1 = [0, 0, 1] as const;

describe("Shell-based integrals — H 1s consistency", () => {
  test("S_shells(H1s, H1s) at same center = 1 (normalization)", () => {
    const h1 = makeShell(STO3G_H_1S, ORIGIN, "H:1s@0");
    expect(S_shells(h1, h1)).toBeCloseTo(1, 7);
  });

  test("S_shells equals S_AB on the cross-overlap term", () => {
    const A = makeShell(STO3G_H_1S, ORIGIN);
    const B = makeShell(STO3G_H_1S, Z_AXIS_1);
    expect(S_shells(A, B)).toBeCloseTo(S_AB(ORIGIN, Z_AXIS_1), 12);
  });

  test("T_shells matches T_AB for H 1s", () => {
    const A = makeShell(STO3G_H_1S, ORIGIN);
    const B = makeShell(STO3G_H_1S, Z_AXIS_1);
    expect(T_shells(A, B)).toBeCloseTo(T_AB(ORIGIN, Z_AXIS_1), 12);
  });

  test("V_shells matches V_AB for H 1s", () => {
    const A = makeShell(STO3G_H_1S, ORIGIN);
    const B = makeShell(STO3G_H_1S, Z_AXIS_1);
    expect(V_shells(A, B, 1, ORIGIN)).toBeCloseTo(V_AB(ORIGIN, Z_AXIS_1, 1, ORIGIN), 12);
  });

  test("ERI_shells matches ERI on (H,H|H,H)", () => {
    const A = makeShell(STO3G_H_1S, ORIGIN);
    const B = makeShell(STO3G_H_1S, Z_AXIS_1);
    expect(ERI_shells(A, B, A, B)).toBeCloseTo(ERI(ORIGIN, Z_AXIS_1, ORIGIN, Z_AXIS_1), 12);
  });
});

describe("Li STO-3G shell sanity", () => {
  test("Li 1s has unit self-overlap", () => {
    const li1s = makeShell(STO3G_LI_1S, ORIGIN, "Li:1s");
    // STO-3G coefficients are published to ~8 sig figs, so normalization
    // is good to ~1e-8 — not better.
    expect(S_shells(li1s, li1s)).toBeCloseTo(1, 7);
  });

  test("Li 2s has unit self-overlap", () => {
    const li2s = makeShell(STO3G_LI_2S, ORIGIN, "Li:2s");
    expect(S_shells(li2s, li2s)).toBeCloseTo(1, 7);
  });

  test("Li 1s and Li 2s are NOT orthogonal in STO-3G (Löwdin job)", () => {
    // The published STO-3G Li 1s/2s contractions are NOT mutually
    // orthogonal — that's why every multi-orbital calculation runs a
    // Löwdin S^{-1/2} step. Cross-overlap is small but nonzero (~0.24).
    const li1s = makeShell(STO3G_LI_1S, ORIGIN);
    const li2s = makeShell(STO3G_LI_2S, ORIGIN);
    const s12 = S_shells(li1s, li2s);
    expect(Math.abs(s12)).toBeGreaterThan(0.01);
    expect(Math.abs(s12)).toBeLessThan(0.5);
  });

  test("Li nuclear-attraction self-energy ⟨1s| -3/r |1s⟩ < 0 and large", () => {
    const li1s = makeShell(STO3G_LI_1S, ORIGIN);
    const v = V_shells(li1s, li1s, 3, ORIGIN);
    expect(v).toBeLessThan(0);
    // Order of magnitude: |⟨1s|-Z/r|1s⟩| ~ Z² for hydrogenic, so ~9.
    expect(Math.abs(v)).toBeGreaterThan(3);
  });

  test("Li 1s kinetic energy is large and positive", () => {
    const li1s = makeShell(STO3G_LI_1S, ORIGIN);
    const t = T_shells(li1s, li1s);
    expect(t).toBeGreaterThan(0);
    // Hydrogenic ⟨1s|T|1s⟩ scales as Z²/2; for Li, Z=3 → ~4.5.
    expect(t).toBeGreaterThan(2);
  });
});
