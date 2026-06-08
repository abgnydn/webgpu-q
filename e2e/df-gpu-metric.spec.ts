import { test, expect } from "@playwright/test";

// WebGPU integral build — increment #1 validation: the 2-index DF metric (P|Q)
// computed in a WGSL f32 compute shader vs the f64 WASM reference.
//
// This answers the make-or-break for the whole GPU-integral effort: does f32
// *evaluation* (Boys function via an erf approximation, single precision) hold
// chemical accuracy? s-only system (H atoms, STO-3G, auto-aux level 0) so the
// metric is a closed form — the minimal real integral that exercises the GPU
// Boys/erf path. Reports the f32-vs-f64 error; a green run greenlights p/d + 3-index.

test.describe("WebGPU 2-index metric (f32) vs WASM (f64)", () => {
  test("s-only metric matches the f64 reference within f32 tolerance", async ({ page }) => {
    test.setTimeout(60 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[gpumetric]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const result = await page.evaluate(async () => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[gpumetric] ${s}`); };
      const [
        { moleculeToShellsNuclei },
        { generateAutoAux, buildMetric2idxCPU },
        { buildMetric2idxGPU_sOnly },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);

      // 6 H atoms (STO-3G → s only) at varied separations for non-trivial off-diagonals.
      const atoms = [
        { symbol: "H", pos: [0, 0, 0] }, { symbol: "H", pos: [1.4, 0, 0] },
        { symbol: "H", pos: [0, 1.1, 0] }, { symbol: "H", pos: [1.4, 1.1, 0.6] },
        { symbol: "H", pos: [0.7, 0.5, 1.9] }, { symbol: "H", pos: [2.3, 0.2, 1.0] },
      ];
      const { shells } = moleculeToShellsNuclei(atoms as never, "sto-3g");
      const aux = generateAutoAux(shells, 0); // s-only aux
      const nonS = aux.filter((s) => s.angular[0] || s.angular[1] || s.angular[2]).length;
      log(`n_aux=${aux.length}, non-s shells=${nonS}`);
      if (nonS > 0) return { ok: false, error: `aux not s-only (${nonS} non-s)` };

      const cpu = await buildMetric2idxCPU(aux);   // Float64Array
      const gpu = await buildMetric2idxGPU_sOnly(aux); // Float32Array

      let maxAbs = 0, maxRel = 0, maxMag = 0;
      for (let i = 0; i < cpu.length; i++) {
        const c = cpu[i]!, g = gpu[i]!;
        const ab = Math.abs(g - c);
        if (ab > maxAbs) maxAbs = ab;
        if (Math.abs(c) > 1e-6 && ab / Math.abs(c) > maxRel) maxRel = ab / Math.abs(c);
        if (Math.abs(c) > maxMag) maxMag = Math.abs(c);
      }
      log(`nAux=${aux.length}: max|Δ|=${maxAbs.toExponential(3)}, max rel=${maxRel.toExponential(3)}, max|M|=${maxMag.toFixed(3)}`);
      return { ok: true, nAux: aux.length, maxAbs, maxRel, maxMag };
    });

    console.log(`\n[gpu-metric] result:`, JSON.stringify(result), "\n");
    expect(result.ok, result.error ?? "").toBe(true);
    // f32 evaluation tolerance: erf approx ~1.5e-7 + f32 arithmetic → expect
    // relative error comfortably below 1e-4 (well inside chemical accuracy once
    // folded into energies, per the step-0 storage probe).
    expect(result.maxRel!).toBeLessThan(1e-4);
  });
});
