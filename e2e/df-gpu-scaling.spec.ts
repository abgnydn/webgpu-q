import { test, expect } from "@playwright/test";

// WebGPU integral build — what drives the GPU win? Measured finding: NOT system
// size, but angular momentum (compute intensity per pair). On an H2 lattice
// (cc-pVDZ, s/p only) the GPU is ~0.8x — it LOSES — and the ratio is roughly flat
// in n (0.66x→0.82x): light s/p pairs leave too little per-thread work to amortize
// dispatch overhead. The GPU only wins where pairs are heavy: d-functions (benzene
// cc-pVDZ is 1.19x, see df-gpu-benchmark). This characterizes the s/p regime so
// the regime boundary is documented, not assumed. Single-shot per size.

function h2Lattice(M: number): { symbol: string; pos: number[] }[] {
  const S = 12.0, B = 0.74, side = Math.ceil(Math.cbrt(M));
  const a: { symbol: string; pos: number[] }[] = [];
  let placed = 0;
  for (let i = 0; i < side && placed < M; i++)
    for (let j = 0; j < side && placed < M; j++)
      for (let k = 0; k < side && placed < M; k++) {
        a.push({ symbol: "H", pos: [i * S, j * S, k * S - B / 2] });
        a.push({ symbol: "H", pos: [i * S, j * S, k * S + B / 2] });
        placed++;
      }
  return a;
}

test.describe("WebGPU 3-index build — speedup vs system size", () => {
  test("GPU/WASM V-build ratio across an H2 lattice (cc-pVDZ)", async ({ page }) => {
    test.setTimeout(8 * 60 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[scale]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const rows = await page.evaluate(async (lattices: { symbol: string; pos: number[] }[][]) => {
      const log = (s: string): void => { console.log(`[scale] ${s}`); };
      const [
        { moleculeToShellsNuclei },
        { generateAutoAux, buildV3idxCPU },
        { buildV3idxGPU },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);
      const out: { n: number; nAux: number; wasmMs: number; gpuMs: number; ratio: number }[] = [];
      for (const atoms of lattices) {
        const { shells } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
        const aux = generateAutoAux(shells, 0);
        // warm both paths once (wasm jit / gpu pipeline), then time once.
        await buildV3idxCPU(shells as never, aux as never);
        await buildV3idxGPU(shells as never, aux as never);
        const t0 = performance.now(); await buildV3idxCPU(shells as never, aux as never); const wasmMs = performance.now() - t0;
        const t1 = performance.now(); await buildV3idxGPU(shells as never, aux as never); const gpuMs = performance.now() - t1;
        const r = { n: shells.length, nAux: aux.length, wasmMs, gpuMs, ratio: wasmMs / gpuMs };
        log(`n=${r.n} n_aux=${r.nAux}: WASM ${(wasmMs / 1000).toFixed(2)}s, GPU ${(gpuMs / 1000).toFixed(2)}s → ${r.ratio.toFixed(2)}x`);
        out.push(r);
      }
      return out;
    }, [h2Lattice(4), h2Lattice(8), h2Lattice(14), h2Lattice(22)]);
console.log(`\n[scaling] ${JSON.stringify(rows)}\n`);
    // Characterize the s/p regime: GPU does NOT win here (light pairs), but the
    // ratio is STABLE in n (parallelism holds the line; it doesn't collapse).
    // The win is angular-momentum-driven — benzene s/p/d is 1.19x elsewhere.
    const largest = rows[rows.length - 1]!;
    expect(largest.ratio).toBeGreaterThan(0.5); // doesn't collapse with size
    expect(largest.ratio).toBeLessThan(1.1);    // s/p stays GPU-unfavorable (documents the regime)
  });

  test("d-regime: GPU/WASM V-build ratio across a C-atom lattice (cc-pVDZ, s/p/d)", async ({ page }) => {
    test.setTimeout(10 * 60 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[scaleD]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    // Lattice of well-separated C atoms — cc-pVDZ has d on C, so this is the
    // heavy (GPU-favorable) regime, scalable with trivial geometry. V-build
    // timing only (no SCF), so open-shell C is irrelevant.
    const cLattice = (M: number): { symbol: string; pos: number[] }[] => {
      const S = 12.0, side = Math.ceil(Math.cbrt(M)); const a: { symbol: string; pos: number[] }[] = []; let p = 0;
      for (let i = 0; i < side && p < M; i++) for (let j = 0; j < side && p < M; j++) for (let k = 0; k < side && p < M; k++) { a.push({ symbol: "C", pos: [i * S, j * S, k * S] }); p++; }
      return a;
    };

    const rows = await page.evaluate(async (lattices: { symbol: string; pos: number[] }[][]) => {
      const log = (s: string): void => { console.log(`[scaleD] ${s}`); };
      const [{ moleculeToShellsNuclei }, { generateAutoAux, buildV3idxCPU }, { buildV3idxGPU }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);
      const out: { n: number; nAux: number; wasmMs: number; gpuMs: number; ratio: number }[] = [];
      for (const atoms of lattices) {
        const { shells } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
        const aux = generateAutoAux(shells, 0);
        await buildV3idxCPU(shells as never, aux as never); await buildV3idxGPU(shells as never, aux as never);
        const t0 = performance.now(); await buildV3idxCPU(shells as never, aux as never); const wasmMs = performance.now() - t0;
        const t1 = performance.now(); await buildV3idxGPU(shells as never, aux as never); const gpuMs = performance.now() - t1;
        const r = { n: shells.length, nAux: aux.length, wasmMs, gpuMs, ratio: wasmMs / gpuMs };
        log(`n=${r.n} n_aux=${r.nAux}: WASM ${(wasmMs / 1000).toFixed(2)}s, GPU ${(gpuMs / 1000).toFixed(2)}s → ${r.ratio.toFixed(2)}x`);
        out.push(r);
      }
      return out;
    }, [cLattice(2), cLattice(6), cLattice(12)]);
console.log(`\n[scaling-d] ${JSON.stringify(rows)}\n`);
    const largest = rows[rows.length - 1]!;
    expect(largest.ratio).toBeGreaterThan(1.0); // d-regime: GPU wins
  });

  test("end-to-end DF build: buildDFAuto vs WASM streaming scales (C lattice, d)", async ({ page }) => {
    test.setTimeout(10 * 60 * 1000);
    page.on("console", (m) => { if (m.text().startsWith("[e2e]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    const cLattice = (M: number): { symbol: string; pos: number[] }[] => {
      const S = 12.0, side = Math.ceil(Math.cbrt(M)); const a: { symbol: string; pos: number[] }[] = []; let p = 0;
      for (let i = 0; i < side && p < M; i++) for (let j = 0; j < side && p < M; j++) for (let k = 0; k < side && p < M; k++) { a.push({ symbol: "C", pos: [i * S, j * S, k * S] }); p++; }
      return a;
    };
    const rows = await page.evaluate(async (lattices: { symbol: string; pos: number[] }[][]) => {
      const log = (s: string): void => { console.log(`[e2e] ${s}`); };
      const [{ moleculeToShellsNuclei }, { generateAutoAux, buildAuxBasisDFStreaming }, { buildDFAuto }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);
      const out: { n: number; gpuMs: number; wasmMs: number; ratio: number; path: string }[] = [];
      for (const atoms of lattices) {
        const { shells } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
        const aux = generateAutoAux(shells, 0);
        await buildDFAuto(shells as never, aux as never); await buildAuxBasisDFStreaming(shells as never, aux as never);
        const t0 = performance.now(); const a = await buildDFAuto(shells as never, aux as never); const gpuMs = performance.now() - t0;
        const t1 = performance.now(); await buildAuxBasisDFStreaming(shells as never, aux as never); const wasmMs = performance.now() - t1;
        const r = { n: shells.length, gpuMs, wasmMs, ratio: wasmMs / gpuMs, path: a.path };
        log(`n=${r.n} (${r.path}): DF build GPU-auto ${(gpuMs / 1000).toFixed(2)}s vs WASM ${(wasmMs / 1000).toFixed(2)}s → ${r.ratio.toFixed(2)}x`);
        out.push(r);
      }
      return out;
    }, [cLattice(6), cLattice(12)]);
    console.log(`\n[scaling-d-e2e] ${JSON.stringify(rows)}\n`);
    const largest = rows[rows.length - 1]!;
    expect(largest.path).toBe("gpu");
    expect(largest.ratio).toBeGreaterThan(1.0); // end-to-end DF build wins in the d-regime
  });
});
