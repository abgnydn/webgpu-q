import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import init from "../../wasm-eri/pkg/wasm_eri.js";
import { buildERIWasm } from "../../src/chemistry/wasm-eri.js";
import { STO3G_H_1S } from "../../src/chemistry/integrals.js";
import { ERI_cg, makeCGShell } from "../../src/chemistry/integrals-cg.js";

const ORIGIN = [0, 0, 0] as const;
const Z14 = [0, 0, 1.4] as const;

describe("wasm-eri seam", () => {
  test("buildERIWasm matches ERI_cg on H2 STO-3G s-shells", async () => {
    try {
      const wasmPath = new URL("../../wasm-eri/pkg/wasm_eri_bg.wasm", import.meta.url);
      await init({ module_or_path: readFileSync(wasmPath) });
      const a = makeCGShell(STO3G_H_1S, ORIGIN, [0, 0, 0]);
      const b = makeCGShell(STO3G_H_1S, Z14, [0, 0, 0]);
      const quads = [a, b];
      const eri = await buildERIWasm(quads);
      expect(eri.length).toBe(16);
      let maxDiff = 0;
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++)
          for (let k = 0; k < 2; k++)
            for (let l = 0; l < 2; l++) {
              const got = eri[((i * 2 + j) * 2 + k) * 2 + l]!;
              const ref = ERI_cg(quads[i]!, quads[j]!, quads[k]!, quads[l]!);
              const diff = Math.abs(got - ref);
              if (diff > maxDiff) maxDiff = diff;
            }
      expect(maxDiff).toBeLessThan(1e-12);
    } catch (e) {
      throw new Error(`WASM ERI failed to load or execute: ${(e as Error).message}`);
    }
  });
});
