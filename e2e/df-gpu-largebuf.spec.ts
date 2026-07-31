import { test, expect } from "@playwright/test";

// Regression for the requiredLimits fix: buildV3idxGPU must handle a V output
// larger than the default 128 MB storage-buffer cap. Naphthalene cc-pVDZ low-aux
// (s/p/d) V is ~130 MB — before the fix the GPU device was created with default
// limits and this threw (validation), forcing a WASM fallback exactly on the
// large molecules where GPU help matters most. After the fix the GPU carries it.
// V-build only (no SCF), so this is fast.

const NAPHTHALENE: { symbol: string; pos: number[] }[] = [
  { symbol: "C", pos: [0.0, 0.716, 0.0] }, { symbol: "C", pos: [0.0, -0.716, 0.0] },
  { symbol: "C", pos: [1.241, 1.403, 0.0] }, { symbol: "C", pos: [1.241, -1.403, 0.0] },
  { symbol: "C", pos: [-1.241, 1.403, 0.0] }, { symbol: "C", pos: [-1.241, -1.403, 0.0] },
  { symbol: "C", pos: [2.433, 0.710, 0.0] }, { symbol: "C", pos: [2.433, -0.710, 0.0] },
  { symbol: "C", pos: [-2.433, 0.710, 0.0] }, { symbol: "C", pos: [-2.433, -0.710, 0.0] },
  { symbol: "H", pos: [1.235, 2.489, 0.0] }, { symbol: "H", pos: [1.235, -2.489, 0.0] },
  { symbol: "H", pos: [-1.235, 2.489, 0.0] }, { symbol: "H", pos: [-1.235, -2.489, 0.0] },
  { symbol: "H", pos: [3.376, 1.247, 0.0] }, { symbol: "H", pos: [3.376, -1.247, 0.0] },
  { symbol: "H", pos: [-3.376, 1.247, 0.0] }, { symbol: "H", pos: [-3.376, -1.247, 0.0] },
];

test.describe("GPU 3-index build — V output beyond the 128 MB default cap", () => {
  test("naphthalene cc-pVDZ low-aux — GPU builds the ~130 MB V and matches WASM", async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[buf]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async (atoms: { symbol: string; pos: number[] }[]) => {
      const log = (s: string): void => { console.log(`[buf] ${s}`); };
      const [{ moleculeToShellsNuclei }, { generateAutoAux, buildV3idxCPU }, { buildV3idxGPU }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);
      const { shells } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const n = shells.length;
      // The hybrid feeds the GPU the s/p/d (L≤2) columns of the extraL=1 aux —
      // that's the ~130 MB V output that exceeded the default cap. Reproduce it.
      const aux1 = generateAutoAux(shells, 1);
      const auxLow = aux1.filter((s: { angular: number[] }) => (s.angular[0]! + s.angular[1]! + s.angular[2]!) <= 2);
      const outBytes = n * n * auxLow.length * 4;
      const outMB = outBytes / 1024 / 1024;

      const Vg = await buildV3idxGPU(shells as never, auxLow as never); // must NOT throw now
      const Vc = await buildV3idxCPU(shells as never, auxLow as never);
      // Magnitude-gated rel error on the SIGNIFICANT integrals (|c| > 1e-3·max);
      // relative error on near-zero elements is meaningless f32 dust.
      let mag = 0; for (let i = 0; i < Vc.length; i++) if (Math.abs(Vc[i]!) > mag) mag = Math.abs(Vc[i]!);
      const cut = mag * 1e-3;
      let maxRel = 0, maxAbs = 0;
      for (let i = 0; i < Vc.length; i++) { const c = Vc[i]!, g = Vg[i]!; const ab = Math.abs(g - c); if (ab > maxAbs) maxAbs = ab; if (Math.abs(c) > cut) maxRel = Math.max(maxRel, ab / Math.abs(c)); }
      log(`naphthalene n=${n}, nAuxLow=${auxLow.length}/${aux1.length}: GPU V output = ${outMB.toFixed(0)} MB (> 128 default), GPU len=${Vg.length}, maxAbs=${maxAbs.toExponential(2)}, maxRel(sig)=${maxRel.toExponential(2)}`);
      return { n, nAuxLow: auxLow.length, outMB, gpuLen: Vg.length, refLen: Vc.length, maxRel, maxAbs };
    }, NAPHTHALENE);
console.log(`\n[df-gpu-largebuf] ${JSON.stringify(r)}\n`);
    expect(r.outMB).toBeGreaterThan(128);        // genuinely past the default cap
    expect(r.gpuLen).toBe(r.refLen);             // GPU produced the full tensor (didn't throw)
    expect(r.maxRel).toBeLessThan(5e-3);         // and it's correct (f32 3-index ~1e-4..1e-3)
  });
});
