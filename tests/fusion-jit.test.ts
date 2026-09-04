import { describe, expect, test } from "vitest";
import { emitFusedChainWgsl } from "../src/fusion-jit.js";
import { FUSED_CHAIN_MAX_K } from "../src/fusion-max-k.js";

describe("emitFusedChainWgsl", () => {
  test("k=1 emits exactly one gate block and no runtime for-loop", () => {
    const wgsl = emitFusedChainWgsl(1);
    expect((wgsl.match(/let gate\d/g) || []).length).toBe(1);
    expect(wgsl).not.toContain("for (");
  });

  test("k=2 emits two ordered gate blocks", () => {
    const wgsl = emitFusedChainWgsl(2);
    expect((wgsl.match(/let gate\d/g) || []).length).toBe(2);
    expect(wgsl.indexOf("let gate0")).toBeLessThan(wgsl.indexOf("let gate1"));
  });

  test("emission is deterministic across calls", () => {
    expect(emitFusedChainWgsl(3)).toBe(emitFusedChainWgsl(3));
  });

  test("k > FUSED_CHAIN_MAX_K throws", () => {
    expect(() => emitFusedChainWgsl(FUSED_CHAIN_MAX_K + 1)).toThrow(
      `emitFusedChainWgsl: k=${FUSED_CHAIN_MAX_K + 1} out of [1, ${FUSED_CHAIN_MAX_K}]`,
    );
  });
});
