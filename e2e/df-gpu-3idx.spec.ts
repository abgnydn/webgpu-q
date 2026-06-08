import { test, expect } from "@playwright/test";

// WebGPU integral build #3: the 3-index tensor V[μν,P] = (μν|P) on the GPU vs
// the f64 WASM reference. The new piece vs the metric is the BRA PAIR (μ,ν) — a
// product of two Gaussians on different centers. s-only here (validates the
// bra-pair Gaussian-product geometry + the 3-index dispatch and V layout); p/d
// and the d-function R-memory work are later increments.

test.describe("WebGPU 3-index tensor (f32) vs WASM (f64)", () => {
  test("s-only V[μν,P] — H atoms STO-3G", async ({ page }) => {
    test.setTimeout(60 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[gpu3idx]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const result = await page.evaluate(async () => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[gpu3idx] ${s}`); };
      const [
        { moleculeToShellsNuclei },
        { generateAutoAux, buildV3idxCPU },
        { buildV3idxGPU_sOnly },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);
      const atoms = [
        { symbol: "H", pos: [0, 0, 0] }, { symbol: "H", pos: [1.4, 0, 0] },
        { symbol: "H", pos: [0.2, 1.1, 0] }, { symbol: "H", pos: [1.6, 1.0, 0.7] },
      ];
      const { shells } = moleculeToShellsNuclei(atoms as never, "sto-3g");
      const aux = generateAutoAux(shells, 0) as Array<{ angular: [number, number, number] }>;
      const maxL = Math.max(...aux.map((s) => s.angular[0] + s.angular[1] + s.angular[2]));
      if (maxL > 0) return { ok: false, error: `aux not s-only (maxL=${maxL})` };

      const cpu = await buildV3idxCPU(shells as never, aux as never); // Float64Array
      const gpu = await buildV3idxGPU_sOnly(shells as never, aux as never); // Float32Array
      let maxAbs = 0, maxRel = 0, maxMag = 0;
      for (let i = 0; i < cpu.length; i++) {
        const c = cpu[i] as number, g = gpu[i] as number;
        const ab = Math.abs(g - c);
        if (ab > maxAbs) maxAbs = ab;
        if (Math.abs(c) > 1e-6 && ab / Math.abs(c) > maxRel) maxRel = ab / Math.abs(c);
        if (Math.abs(c) > maxMag) maxMag = Math.abs(c);
      }
      log(`n=${shells.length} n_aux=${aux.length} V=${cpu.length}: max|Δ|=${maxAbs.toExponential(3)}, max rel=${maxRel.toExponential(3)}, max|V|=${maxMag.toFixed(3)}`);
      return { ok: true, n: shells.length, nAux: aux.length, maxAbs, maxRel, maxMag };
    });

    console.log(`\n[gpu-3idx] ${JSON.stringify(result)}\n`);
    expect(result.ok, result.error ?? "").toBe(true);
    expect(result.maxRel!).toBeLessThan(1e-4);
  });
});
