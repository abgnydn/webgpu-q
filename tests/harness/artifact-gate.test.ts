import { describe, it, expect } from "vitest";
import type { Artifact, ArtifactMeta } from "../../experiments/lib/runner.js";
import { stateMetrics, passed, FIDELITY_PASS_BAR } from "../../experiments/lib/fidelity.js";

function state(...ampPairs: number[]): Float64Array {
  return new Float64Array(ampPairs);
}

function needsTimingNote(meta: ArtifactMeta): boolean {
  return meta.warmup !== 5 || meta.trials !== 20;
}

describe("artifact gate harness", () => {
  it("identical vectors -> F=1, passed true", () => {
    const psi = state(Math.SQRT1_2, 0, Math.SQRT1_2, 0);
    const m = stateMetrics(psi, psi);
    expect(m.fidelity).toBeCloseTo(1, 12);
    expect(m.maxDp).toBeCloseTo(0, 12);
    expect(passed(m)).toBe(true);
  });

  it("phase-flipped vector -> maxDp=0 but F<1-bar, passed false", () => {
    // |ψ⟩ = (|0⟩ + i|1⟩)/√2  vs  |ψ'⟩ = (|0⟩ - i|1⟩)/√2.
    // Probabilities match exactly; only the relative phase differs.
    const ref = state(Math.SQRT1_2, 0, 0, Math.SQRT1_2);
    const test = state(Math.SQRT1_2, 0, 0, -Math.SQRT1_2);
    const m = stateMetrics(ref, test);
    expect(m.maxDp).toBeCloseTo(0, 12);
    expect(m.fidelity).toBeLessThan(FIDELITY_PASS_BAR);
    expect(passed(m)).toBe(false);
  });

  it("slightly scaled vector passes fidelity but is rejected by norm gate", () => {
    const ref = state(Math.SQRT1_2, 0, Math.SQRT1_2, 0);
    const scale = 1.0001; // |s² - 1| ≈ 2e-4 ≥ 1e-4 norm gate
    const test = state(Math.SQRT1_2 * scale, 0, Math.SQRT1_2 * scale, 0);
    const m = stateMetrics(ref, test);
    expect(m.fidelity).toBeGreaterThanOrEqual(FIDELITY_PASS_BAR);
    expect(Math.abs(m.normTest - 1)).toBeGreaterThanOrEqual(1e-4);
    expect(passed(m)).toBe(false);
  });

  it("needsTimingNote is true unless warmup==5 and trials==20", () => {
    expect(needsTimingNote({ warmup: 0, trials: 1 } as ArtifactMeta)).toBe(true);
    expect(needsTimingNote({ warmup: 5, trials: 20 } as ArtifactMeta)).toBe(false);
  });
});

// Type-only compile-time check that the imported Artifact shape is usable.
const _typeCheck: Artifact<{ value: number }> = {
  meta: {
    protocol: "harness-gate",
    hypothesis: "type check",
    passBar: "n/a",
    seed: "seed-a",
    warmup: 5,
    trials: 20,
  },
  env: {
    gitSha: "test",
    timestamp: new Date().toISOString(),
    userAgent: "test",
    platform: "test",
    adapter: { vendor: "", architecture: "", device: "", description: "" },
    limits: {
      maxBufferSize: 0,
      maxStorageBufferBindingSize: 0,
      maxComputeWorkgroupsPerDimension: 0,
      maxComputeInvocationsPerWorkgroup: 0,
      maxComputeWorkgroupStorageSize: 0,
    },
    hardwareConcurrency: 1,
    devicePixelRatio: 1,
  },
  rows: [{ value: 1 }],
  status: "pass",
  diagnosis: "ok",
};
void _typeCheck;
