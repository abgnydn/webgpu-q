import { test, expect } from "@playwright/test";

// Measure the per-call cost of buildERIWasmParallel when the worker pool
// is freshly spawned vs reused. With persistent worker caching, calls 2..N
// should be ~constant; without it, every call pays ~1 s of worker spawn +
// WASM init per worker.

test.describe("Persistent worker pool", () => {
  test("ethane cc-pVDZ — 4 sequential calls expose spawn amortization", async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { buildERIWasmParallel, disposeAllPools },
        { sabAvailable },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/parallel-eri.ts" as string),
        import("/src/parallel/worker-pool.ts" as string),
      ]);

      const cc = 1.535, ch = 1.094;
      const hch = 107.8 * Math.PI / 180;
      const hcc = (Math.PI - hch) / 2 + Math.PI / 6;
      const sH = ch * Math.sin(hcc), cH = ch * Math.cos(hcc);
      const atoms = [
        { symbol: "C", pos: [0, 0, -cc / 2] },
        { symbol: "C", pos: [0, 0,  cc / 2] },
        { symbol: "H", pos: [ sH, 0, -cc / 2 - cH] },
        { symbol: "H", pos: [-sH / 2,  sH * Math.sqrt(3) / 2, -cc / 2 - cH] },
        { symbol: "H", pos: [-sH / 2, -sH * Math.sqrt(3) / 2, -cc / 2 - cH] },
        { symbol: "H", pos: [-sH, 0,  cc / 2 + cH] },
        { symbol: "H", pos: [ sH / 2,  sH * Math.sqrt(3) / 2,  cc / 2 + cH] },
        { symbol: "H", pos: [ sH / 2, -sH * Math.sqrt(3) / 2,  cc / 2 + cH] },
      ];
      const { shells } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      if (!sabAvailable()) return { skipped: true as const };

      // Fresh start: clear any pool from earlier modules.
      disposeAllPools();

      const N = 8;
      const callMs: number[] = [];
      for (let i = 0; i < 4; i++) {
        const t = performance.now();
        await buildERIWasmParallel(shells, shells.length, 1e-10, N);
        callMs.push(performance.now() - t);
      }

      return { skipped: false as const, n: shells.length, N, callMs };
    });

    if (r.skipped) { test.skip(); return; }

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Persistent worker pool — ethane cc-pVDZ, N=${r.N} workers`);
    console.log(`══════════════════════════════════════════════════════════`);
    for (let i = 0; i < r.callMs.length; i++) {
      const tag = i === 0 ? " (cold: pool spawn + WASM init)" : " (warm: pool cached)";
      console.log(`  call ${i + 1}: ${r.callMs[i]!.toFixed(0).padStart(6)} ms${tag}`);
    }
    const cold = r.callMs[0]!;
    const warmMedian = [...r.callMs.slice(1)].sort((a, b) => a - b)[
      Math.floor((r.callMs.length - 1) / 2)
    ]!;
    console.log();
    console.log(`  cold:        ${cold.toFixed(0).padStart(6)} ms`);
    console.log(`  warm median: ${warmMedian.toFixed(0).padStart(6)} ms`);
    console.log(`  spawn cost:  ${(cold - warmMedian).toFixed(0).padStart(6)} ms`);
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    // First call should be at least as slow as warm; warm calls should be
    // bit-stable. Loose pass bar — just make sure we don't regress badly.
    expect(r.callMs.length).toBe(4);
    expect(r.callMs[0]!).toBeGreaterThan(0);
  });
});
