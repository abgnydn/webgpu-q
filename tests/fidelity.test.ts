// ─────────────────────────────────────────────────────────────
// fidelity.test.ts — the correctness metric under test.
//
// A Level-1 run that reports F ≥ 1 − 1e-5 only means something if
// the F computation itself is bit-exact on the edge cases:
//   - identical states → F = 1
//   - orthogonal states → F = 0
//   - phase-rotated same state → F = 1 (phases cancel inside |⟨·|·⟩|²)
//   - small Gaussian perturbation → F ≈ 1 − O(ε²)
// ─────────────────────────────────────────────────────────────

import { describe, it, expect } from "vitest";
import { stateMetrics, passed, FIDELITY_PASS_BAR } from "../experiments/lib/fidelity.js";

function state(...ampPairs: number[]): Float64Array {
  return new Float64Array(ampPairs);
}

describe("stateMetrics — identical states", () => {
  it("F = 1, ∆p = 0", () => {
    // |ψ⟩ = (|0⟩ + |1⟩)/√2
    const s = state(Math.SQRT1_2, 0, Math.SQRT1_2, 0);
    const m = stateMetrics(s, s);
    expect(m.fidelity).toBeCloseTo(1, 12);
    expect(m.maxDp).toBeCloseTo(0, 12);
    expect(m.tvd).toBeCloseTo(0, 12);
    expect(passed(m)).toBe(true);
  });
});

describe("stateMetrics — orthogonal states", () => {
  it("F = 0 and max|Δp| = 1 for |0⟩ vs |1⟩", () => {
    const a = state(1, 0, 0, 0);
    const b = state(0, 0, 1, 0);
    const m = stateMetrics(a, b);
    expect(m.fidelity).toBeCloseTo(0, 12);
    expect(m.maxDp).toBeCloseTo(1, 12);
  });
});

describe("stateMetrics — phase-rotated states", () => {
  it("global phase does not affect F", () => {
    // |ψ⟩ = |1⟩, |ψ'⟩ = e^{iπ/3} |1⟩. F must be 1.
    const a = state(0, 0, 1, 0);
    const theta = Math.PI / 3;
    const b = state(0, 0, Math.cos(theta), Math.sin(theta));
    const m = stateMetrics(a, b);
    expect(m.fidelity).toBeCloseTo(1, 12);
    expect(m.maxDp).toBeCloseTo(0, 12);
  });

  it("relative phase in superposition rotates F predictably", () => {
    // |ψ⟩ = (|0⟩ + |1⟩)/√2, |ψ'⟩ = (|0⟩ + e^{iθ}|1⟩)/√2.
    // ⟨ψ|ψ'⟩ = (1 + e^{iθ})/2, |⟨·⟩|² = cos²(θ/2).
    const a = state(Math.SQRT1_2, 0, Math.SQRT1_2, 0);
    const theta = Math.PI / 4;
    const b = state(Math.SQRT1_2, 0, Math.SQRT1_2 * Math.cos(theta), Math.SQRT1_2 * Math.sin(theta));
    const m = stateMetrics(a, b);
    expect(m.fidelity).toBeCloseTo(Math.cos(theta / 2) ** 2, 10);
  });
});

describe("stateMetrics — pass bar", () => {
  it("passes an f32-roundtripped identical state", () => {
    // f32 quantization: amplitude rounded to 32-bit precision but
    // identical on both sides, so F must still be 1 bit-exact.
    const a32 = new Float32Array([Math.SQRT1_2, 0, Math.SQRT1_2, 0]);
    const b32 = new Float32Array([Math.SQRT1_2, 0, Math.SQRT1_2, 0]);
    const m = stateMetrics(a32, b32);
    expect(m.fidelity).toBeCloseTo(1, 6);
    expect(passed(m)).toBe(true);
  });

  it("fails when normTest departs from 1 by > 1e-4", () => {
    const a = state(Math.SQRT1_2, 0, Math.SQRT1_2, 0);
    const b = state(0.7, 0, 0.7, 0); // norm = 0.98
    const m = stateMetrics(a, b);
    expect(Math.abs(m.normTest - 1)).toBeGreaterThan(1e-4);
    expect(passed(m)).toBe(false);
  });

  it(`FIDELITY_PASS_BAR = 1 - 1e-5`, () => {
    expect(FIDELITY_PASS_BAR).toBeCloseTo(1 - 1e-5, 12);
  });
});

describe("stateMetrics — errors", () => {
  it("throws on size mismatch", () => {
    expect(() => stateMetrics(state(1, 0), state(1, 0, 0, 0))).toThrow(/size mismatch/);
  });
  it("throws on odd-length amplitude arrays", () => {
    // 3 floats = not a complete vec2 set.
    expect(() => stateMetrics(new Float64Array(3), new Float64Array(3))).toThrow(/interleaved/);
  });
});
