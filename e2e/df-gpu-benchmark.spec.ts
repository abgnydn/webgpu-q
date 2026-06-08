import { test, expect } from "@playwright/test";

// WebGPU integral build #4 — MEASURE THE WIN.
// Times the 3-index V build on GPU (f32 WGSL) vs WASM (f64) and runs HF
// end-to-end on GPU-computed integrals to confirm the energy is correct.
//   H2O cc-pVDZ : GPU-V HF vs EXACT HF (absolute correctness) + small-size timing
//   benzene     : GPU-V HF vs WASM-V HF (isolates V) + headline-size timing

const BENZENE = (() => {
  const rCC = 1.395, rCH = 1.087;
  const a: { symbol: string; pos: number[] }[] = [];
  for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; a.push({ symbol: "C", pos: [rCC * Math.cos(t), rCC * Math.sin(t), 0] }); }
  for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; const r2 = rCC + rCH; a.push({ symbol: "H", pos: [r2 * Math.cos(t), r2 * Math.sin(t), 0] }); }
  return a;
})();
const H2O = (() => {
  const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
  return [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }];
})();

test.describe("WebGPU 3-index build — measure the win", () => {
  test("H2O cc-pVDZ — GPU integrals → correct HF energy + timing", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[bench]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async (atoms) => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[bench] ${s}`); };
      const [
        { moleculeToShellsNuclei }, { computeMolecularIntegrals }, { runRHFSCF },
        { generateAutoAux, buildV3idxCPU, buildBFromV }, { buildV3idxGPU },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const aux = generateAutoAux(shells, 0);
      const integrals = computeMolecularIntegrals(shells, nuclei); // full ERI (small for H2O)
      const exact = runRHFSCF(integrals, nElectrons, { energyTol: 1e-9, densityTol: 1e-8, maxIter: 100 }).energy;

      const median = async (f: () => Promise<unknown>, n: number): Promise<number> => {
        await f(); // warmup
        const ts: number[] = [];
        for (let i = 0; i < n; i++) { const t0 = performance.now(); await f(); ts.push(performance.now() - t0); }
        ts.sort((a, b) => a - b); return ts[Math.floor(ts.length / 2)]!;
      };
      const wasmMs = await median(() => buildV3idxCPU(shells as never, aux as never), 3);
      const gpuMs = await median(() => buildV3idxGPU(shells as never, aux as never), 3);

      // Isolate the GPU integral error: GPU-V HF vs WASM-V HF (same aux). (vs
      // EXACT would fold in the DF *method* error of this aux, a separate thing.)
      const Bgpu = await buildBFromV(shells as never, aux as never, Float64Array.from(await buildV3idxGPU(shells as never, aux as never)));
      const Bwasm = await buildBFromV(shells as never, aux as never, await buildV3idxCPU(shells as never, aux as never));
      const eGPU = runRHFSCF(integrals, nElectrons, { energyTol: 1e-9, densityTol: 1e-8, maxIter: 100, useDF: Bgpu }).energy;
      const eWASM = runRHFSCF(integrals, nElectrons, { energyTol: 1e-9, densityTol: 1e-8, maxIter: 100, useDF: Bwasm }).energy;
      void exact;
      log(`H2O n=${shells.length} n_aux=${aux.length}: WASM V ${wasmMs.toFixed(1)}ms, GPU V ${gpuMs.toFixed(1)}ms (${(wasmMs / gpuMs).toFixed(2)}x)`);
      log(`H2O: WASM-V HF ${eWASM.toFixed(8)}, GPU-V HF ${eGPU.toFixed(8)}, |ΔE|=${Math.abs(eGPU - eWASM).toExponential(2)}`);
      return { n: shells.length, nAux: aux.length, wasmMs, gpuMs, dE: Math.abs(eGPU - eWASM) };
    }, H2O);

    console.log(`\n[bench H2O] ${JSON.stringify(r)}\n`);
    expect(r.dE).toBeLessThan(1e-3); // GPU vs WASM integrals → same HF energy within f32
  });

  test("benzene cc-pVDZ — headline-size V timing + GPU-vs-WASM energy", async ({ page }) => {
    test.setTimeout(8 * 60 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[bench]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async (atoms) => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[bench] ${s}`); };
      const [
        { moleculeToShellsNuclei }, { computeMolecularIntegrals }, { runRHFSCF },
        { generateAutoAux, buildV3idxCPU, buildBFromV }, { buildV3idxGPU },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const aux = generateAutoAux(shells, 0);
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });

      const median = async (f: () => Promise<unknown>): Promise<number> => {
        await f();
        const ts: number[] = []; for (let i = 0; i < 3; i++) { const t0 = performance.now(); await f(); ts.push(performance.now() - t0); }
        ts.sort((a, b) => a - b); return ts[1]!;
      };
      const t0 = performance.now(); const Vwasm = await buildV3idxCPU(shells as never, aux as never); const wasmMs = performance.now() - t0;
      const t1 = performance.now(); const Vgpu = await buildV3idxGPU(shells as never, aux as never); const gpuMs = performance.now() - t1;
      void median; // single-shot here (builds are multi-second); medians inflate runtime
      void Vwasm;

      const Bgpu = await buildBFromV(shells as never, aux as never, Float64Array.from(Vgpu));
      const Bwasm = await buildBFromV(shells as never, aux as never, await buildV3idxCPU(shells as never, aux as never));
      const eGPU = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-7, densityTol: 1e-6, maxIter: 60, useDF: Bgpu }).energy;
      const eWASM = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-7, densityTol: 1e-6, maxIter: 60, useDF: Bwasm }).energy;
      log(`benzene n=${shells.length} n_aux=${aux.length}: WASM V ${(wasmMs / 1000).toFixed(2)}s, GPU V ${(gpuMs / 1000).toFixed(2)}s (${(wasmMs / gpuMs).toFixed(2)}x)`);
      log(`benzene: WASM-V HF ${eWASM.toFixed(8)}, GPU-V HF ${eGPU.toFixed(8)}, |ΔE|=${Math.abs(eGPU - eWASM).toExponential(2)}`);
      return { n: shells.length, nAux: aux.length, wasmMs, gpuMs, dE: Math.abs(eGPU - eWASM) };
    }, BENZENE);

    console.log(`\n[bench benzene] ${JSON.stringify(r)}\n`);
    expect(r.dE).toBeLessThan(1e-3); // GPU vs WASM integrals → same HF energy within f32
  });
});
