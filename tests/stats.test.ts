// ─────────────────────────────────────────────────────────────
// stats.test.ts — exercise the trimmed-statistics helpers.
//
// Stats code is the foundation under every research claim we make
// in RESEARCH.md; bugs here silently corrupt every experiment row.
// Tight coverage matters.
// ─────────────────────────────────────────────────────────────

import { describe, it, expect } from "vitest";
import { percentile, stats } from "../experiments/lib/stats.js";

describe("percentile", () => {
  it("exact on trivial inputs", () => {
    const s = [1, 2, 3, 4, 5];
    expect(percentile(s, 0)).toBe(1);
    expect(percentile(s, 100)).toBe(5);
    expect(percentile(s, 50)).toBe(3);
  });

  it("interpolates between bracketing samples (type-7)", () => {
    // 4 points, p=50 sits between index 1 and 2.
    const s = [10, 20, 30, 40];
    expect(percentile(s, 50)).toBeCloseTo(25, 10);
    expect(percentile(s, 25)).toBeCloseTo(17.5, 10);
    expect(percentile(s, 75)).toBeCloseTo(32.5, 10);
  });

  it("returns NaN on empty sample", () => {
    expect(percentile([], 50)).toBeNaN();
  });
});

describe("stats", () => {
  it("reports median, mean, std on a simple sample", () => {
    const s = stats([1, 2, 3, 4, 5]);
    expect(s.n).toBe(5);
    expect(s.median).toBe(3);
    expect(s.mean).toBeCloseTo(3, 10);
    expect(s.std).toBeCloseTo(Math.sqrt(2.5), 5); // sample std
    expect(s.min).toBe(1);
    expect(s.max).toBe(5);
  });

  it("flags a noisy distribution", () => {
    // std/median = 1 → definitely noisy.
    const s = stats([1, 1, 1, 1, 10, 10, 10, 10]);
    expect(s.noisy).toBe(true);
  });

  it("does not flag a tight distribution", () => {
    // std/median ≈ 0.003 — dead quiet.
    const s = stats([1.000, 1.001, 0.999, 1.002, 0.998, 1.001]);
    expect(s.noisy).toBe(false);
  });

  it("throws on empty input", () => {
    expect(() => stats([])).toThrowError();
  });

  it("is unsorted-input-safe", () => {
    const a = stats([5, 1, 3, 2, 4]);
    const b = stats([1, 2, 3, 4, 5]);
    expect(a.median).toBe(b.median);
    expect(a.std).toBeCloseTo(b.std, 10);
  });
});
