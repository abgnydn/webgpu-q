import { test, expect } from "@playwright/test";

// WebGPU integral build validation: the 2-index DF metric (P|Q) in a WGSL f32
// compute shader vs the f64 WASM reference.
//   #1: s-only (closed form) — answers the f32-evaluation question.
//   #2: s/p/d (full McMurchie–Davidson: Hermite E-coefs + R-tensor recurrence).
// A green run greenlights the 3-index tensor next.

interface MetricCase { atoms: { symbol: string; pos: number[] }[]; basis: string; auxLevel: number; }

async function runMetric(page: import("@playwright/test").Page, c: MetricCase): Promise<{
  ok: boolean; error?: string; nAux?: number; maxAbs?: number; maxRel?: number; maxMag?: number; maxL?: number;
}> {
  return await page.evaluate(async (cfg: MetricCase) => {
    const log = (s: string): void => { console.log(`[gpumetric] ${s}`); };
    const [
      { moleculeToShellsNuclei },
      { generateAutoAux, buildMetric2idxCPU },
      { buildMetric2idxGPU },
    ] = await Promise.all([
      import("/src/chemistry/atoms.ts" as string),
      import("/src/chemistry/df-aux.ts" as string),
      import("/src/chemistry/df-gpu.ts" as string),
    ]);
    const { shells } = moleculeToShellsNuclei(cfg.atoms as never, cfg.basis);
    const aux = generateAutoAux(shells, cfg.auxLevel) as Array<{ angular: [number, number, number] }>;
    const maxL = Math.max(...aux.map((s) => s.angular[0] + s.angular[1] + s.angular[2]));
    const cpu = await buildMetric2idxCPU(aux);
    const gpu = await buildMetric2idxGPU(aux);
    let maxAbs = 0, maxRel = 0, maxMag = 0;
    for (let i = 0; i < cpu.length; i++) {
      const cc = cpu[i] as number, gg = gpu[i] as number;
      const ab = Math.abs(gg - cc);
      if (ab > maxAbs) maxAbs = ab;
      if (Math.abs(cc) > 1e-6 && ab / Math.abs(cc) > maxRel) maxRel = ab / Math.abs(cc);
      if (Math.abs(cc) > maxMag) maxMag = Math.abs(cc);
    }
    log(`nAux=${aux.length} maxL=${maxL}: max|Δ|=${maxAbs.toExponential(3)}, max rel=${maxRel.toExponential(3)}, max|M|=${maxMag.toFixed(2)}`);
    return { ok: true, nAux: aux.length, maxAbs, maxRel, maxMag, maxL };
  }, c);
}

test.describe("WebGPU 2-index metric (f32) vs WASM (f64)", () => {
  test("#1 s-only — H atoms STO-3G", async ({ page }) => {
    test.setTimeout(60 * 1000);
    page.on("console", (m) => { if (m.text().startsWith("[gpumetric]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    const r = await runMetric(page, {
      atoms: [
        { symbol: "H", pos: [0, 0, 0] }, { symbol: "H", pos: [1.4, 0, 0] },
        { symbol: "H", pos: [0, 1.1, 0] }, { symbol: "H", pos: [1.4, 1.1, 0.6] },
        { symbol: "H", pos: [0.7, 0.5, 1.9] }, { symbol: "H", pos: [2.3, 0.2, 1.0] },
      ], basis: "sto-3g", auxLevel: 0,
    });
    console.log(`\n[gpu-metric s] ${JSON.stringify(r)}\n`);
    expect(r.ok, r.error ?? "").toBe(true);
    expect(r.maxL).toBe(0);
    expect(r.maxRel!).toBeLessThan(1e-4);
  });

  test("#2 s/p/d — H2O cc-pVDZ", async ({ page }) => {
    test.setTimeout(90 * 1000);
    page.on("console", (m) => { if (m.text().startsWith("[gpumetric]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    const half = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(half), z = 0.9572 * Math.cos(half);
    const r = await runMetric(page, {
      atoms: [
        { symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] },
      ], basis: "cc-pvdz", auxLevel: 0,
    });
    console.log(`\n[gpu-metric spd] ${JSON.stringify(r)}\n`);
    expect(r.ok, r.error ?? "").toBe(true);
    expect(r.maxL).toBeGreaterThanOrEqual(2); // exercises p AND d
    expect(r.maxRel!).toBeLessThan(1e-3);      // f32 with angular-momentum recurrences
  });
});
