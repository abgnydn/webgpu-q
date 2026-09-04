import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import init from "../../wasm-gemm/pkg/wasm_gemm.js";
import { dgemmWasm } from "../../src/chemistry/wasm-gemm.js";
import { matmul, zeros } from "../../src/linalg.js";

describe("wasm-gemm seam", () => {
  it("dgemmWasm is integrable and matches TS matmul for a real 4x4", async () => {
    const m = 4;
    const n = 4;
    const k = 4;

    // Deterministic row-major real matrices (no Math.random).
    const a = new Float64Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
    ]);
    const b = new Float64Array([
      17, 18, 19, 20,
      21, 22, 23, 24,
      25, 26, 27, 28,
      29, 30, 31, 32,
    ]);

    // Reference via existing complex matmul with imaginary parts zeroed.
    const A = zeros(m, k);
    const B = zeros(k, n);
    for (let i = 0; i < a.length; i++) A.data[i * 2] = a[i]!;
    for (let i = 0; i < b.length; i++) B.data[i * 2] = b[i]!;
    const Cref = matmul(A, B);

    try {
      const wasmPath = new URL("../../wasm-gemm/pkg/wasm_gemm_bg.wasm", import.meta.url);
      await init({ module_or_path: readFileSync(wasmPath) });
      const c = new Float64Array(m * n);
      const Cwasm = await dgemmWasm(m, n, k, a, b, c);
      let maxDiff = 0;
      for (let i = 0; i < m * n; i++) {
        const diff = Math.abs(Cwasm[i]! - Cref.data[i * 2]!);
        if (diff > maxDiff) maxDiff = diff;
      }
      expect(maxDiff).toBeLessThan(1e-12);
    } catch (e) {
      throw new Error(`WASM dgemm failed to load or execute: ${(e as Error).message}`);
    }
  });
});
