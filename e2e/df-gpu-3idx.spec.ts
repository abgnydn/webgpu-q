import { test, expect } from "@playwright/test";

// WebGPU integral build: the 3-index tensor V[μν,P]=(μν|P) on the GPU vs f64 WASM.
//   #3  s-only  (bra-pair geometry + dispatch + V layout)
//   #3b s/p     (two-center bra E-coefficient recurrence + R-tensor)
//   #3c s/p/d   (also probes whether the 7^4 R-tensor private array is viable)

interface Case { atoms: { symbol: string; pos: number[] }[]; basis: string; sOnly: boolean; }

async function run(page: import("@playwright/test").Page, c: Case): Promise<{
  ok: boolean; error?: string; n?: number; nAux?: number; maxL?: number; maxRel?: number; maxMag?: number;
}> {
  return await page.evaluate(async (cfg: Case) => {
    const log = (s: string): void => { console.log(`[gpu3idx] ${s}`); };
    const [
      { moleculeToShellsNuclei },
      { generateAutoAux, buildV3idxCPU },
      { buildV3idxGPU, buildV3idxGPU_sOnly },
    ] = await Promise.all([
      import("/src/chemistry/atoms.ts" as string),
      import("/src/chemistry/df-aux.ts" as string),
      import("/src/chemistry/df-gpu.ts" as string),
    ]);
    const { shells } = moleculeToShellsNuclei(cfg.atoms as never, cfg.basis);
    const aux = generateAutoAux(shells, 0) as Array<{ angular: [number, number, number] }>;
    const maxL = Math.max(
      ...aux.map((s) => s.angular[0] + s.angular[1] + s.angular[2]),
      ...(shells as Array<{ angular: [number, number, number] }>).map((s) => s.angular[0] + s.angular[1] + s.angular[2]),
    );
    const cpu = await buildV3idxCPU(shells as never, aux as never);
    const gpu = cfg.sOnly
      ? await buildV3idxGPU_sOnly(shells as never, aux as never)
      : await buildV3idxGPU(shells as never, aux as never);
    let maxAbs = 0, maxRel = 0, maxMag = 0;
    for (let i = 0; i < cpu.length; i++) {
      const cc = cpu[i] as number, gg = gpu[i] as number;
      const ab = Math.abs(gg - cc);
      if (ab > maxAbs) maxAbs = ab;
      if (Math.abs(cc) > 1e-6 && ab / Math.abs(cc) > maxRel) maxRel = ab / Math.abs(cc);
      if (Math.abs(cc) > maxMag) maxMag = Math.abs(cc);
    }
    log(`n=${shells.length} n_aux=${aux.length} maxL=${maxL}: max|Δ|=${maxAbs.toExponential(3)}, max rel=${maxRel.toExponential(3)}, max|V|=${maxMag.toFixed(2)}`);
    return { ok: true, n: shells.length, nAux: aux.length, maxL, maxRel, maxMag };
  }, c);
}

const H2O = (() => {
  const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
  return [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }];
})();

test.describe("WebGPU 3-index tensor (f32) vs WASM (f64)", () => {
  test("#3 s-only — H atoms STO-3G", async ({ page }) => {
    test.setTimeout(60 * 1000);
    page.on("console", (m) => { if (m.text().startsWith("[gpu3idx]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    const r = await run(page, {
      atoms: [
        { symbol: "H", pos: [0, 0, 0] }, { symbol: "H", pos: [1.4, 0, 0] },
        { symbol: "H", pos: [0.2, 1.1, 0] }, { symbol: "H", pos: [1.6, 1.0, 0.7] },
      ], basis: "sto-3g", sOnly: true,
    });
    console.log(`\n[gpu-3idx s] ${JSON.stringify(r)}\n`);
    expect(r.ok, r.error ?? "").toBe(true);
    expect(r.maxL).toBe(0);
    expect(r.maxRel!).toBeLessThan(1e-4);
  });

  test("#3b s/p — H2O STO-3G", async ({ page }) => {
    test.setTimeout(90 * 1000);
    page.on("console", (m) => { if (m.text().startsWith("[gpu3idx]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    const r = await run(page, { atoms: H2O, basis: "sto-3g", sOnly: false });
    console.log(`\n[gpu-3idx sp] ${JSON.stringify(r)}\n`);
    expect(r.ok, r.error ?? "").toBe(true);
    expect(r.maxL).toBe(1);
    expect(r.maxRel!).toBeLessThan(1e-3);
  });

  test("#3c s/p/d — H2O cc-pVDZ (probes 7^4 R scratch)", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("console", (m) => { if (m.text().startsWith("[gpu3idx]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    const r = await run(page, { atoms: H2O, basis: "cc-pvdz", sOnly: false });
    console.log(`\n[gpu-3idx spd] ${JSON.stringify(r)}\n`);
    expect(r.ok, r.error ?? "").toBe(true);
    expect(r.maxL).toBeGreaterThanOrEqual(2);
    expect(r.maxRel!).toBeLessThan(1e-3);
  });
});
