import { test, expect } from "@playwright/test";

// Fully-GPU DF-HF — whole-loop wall-clock vs the WASM DF-HF loop. df-hf-gpu-full
// proves the GPU loop is CORRECT; this measures whether it's FASTER end-to-end.
//
// Honest expectation: the GPU integral build wins in the d-regime (df-gpu-benchmark
// is 1.1-1.35x), but the per-iteration GPU JK pays a readback sync each SCF step,
// which on a small system (benzene, n=114) can be dwarfed by the already-optimized
// WASM JK worker pool. So this records the ratio WHATEVER it is — a per-iter-JK
// negative on small n is real data, not a failure. The build-side win is the
// established result; this characterizes the loop side.

test.describe("Fully-GPU DF-HF — whole-loop speed vs WASM", () => {
  test("benzene cc-pVDZ — GPU loop vs WASM loop wall-clock", async ({ page }) => {
    test.setTimeout(180 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[bench]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[bench] ${s}`); };
      const [
        { moleculeToShellsNuclei }, { computeMolecularIntegrals }, { runRHFSCF, runRHFSCFAsync },
        { generateAutoAux, buildAuxBasisDFStreaming }, { buildDFAuto }, { makeGpuDFJK },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
        import("/src/chemistry/jk-df-gpu.ts" as string),
      ]);
      const rCC = 1.395, rCH = 1.087; const atoms: { symbol: string; pos: number[] }[] = [];
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; atoms.push({ symbol: "C", pos: [rCC * Math.cos(t), rCC * Math.sin(t), 0] }); }
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; const r2 = rCC + rCH; atoms.push({ symbol: "H", pos: [r2 * Math.cos(t), r2 * Math.sin(t), 0] }); }
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const aux = generateAutoAux(shells, 0);

      // f32-appropriate tolerances so both paths run the same SCF target.
      const scfOpts = { useDIIS: true, energyTol: 1e-5, densityTol: 1e-4, maxIter: 60 };

      // WASM whole loop: build DF (WASM streaming) + SCF (WASM JK).
      await buildAuxBasisDFStreaming(shells as never, aux as never); // warm
      const tw0 = performance.now();
      const wasmDF = await buildAuxBasisDFStreaming(shells as never, aux as never);
      const eWASM = runRHFSCF(integrals, nElectrons, { ...scfOpts, useDF: wasmDF });
      const wasmMs = performance.now() - tw0;

      // GPU whole loop: build DF (GPU integrals) + SCF (GPU JK every iter).
      await buildDFAuto(shells as never, aux as never); // warm pipelines
      const tg0 = performance.now();
      const auto = await buildDFAuto(shells as never, aux as never);
      const gpuJK = await makeGpuDFJK(auto.df);
      const customJKBuilder = async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
        const { J, K } = await gpuJK.jk(D);
        const Jf = new Float64Array(J.length), Kf = new Float64Array(K.length);
        for (let i = 0; i < J.length; i++) { Jf[i] = J[i]!; Kf[i] = K[i]!; }
        return { J: Jf, K: Kf };
      };
      const eGPU = await runRHFSCFAsync(integrals, nElectrons, { ...scfOpts, parallel: 1, customJKBuilder });
      const gpuMs = performance.now() - tg0;
      gpuJK.dispose();

      const ratio = wasmMs / gpuMs;
      log(`benzene n=${shells.length} n_aux=${auto.df.nAux}: WASM loop ${(wasmMs / 1000).toFixed(2)}s (${eWASM.energy.toFixed(6)}), ` +
          `GPU loop ${(gpuMs / 1000).toFixed(2)}s (${eGPU.energy.toFixed(6)}, ${eGPU.iter}it) → ${ratio.toFixed(2)}x`);
      return { ratio, dE: Math.abs(eGPU.energy - eWASM.energy), path: auto.path, gpuConverged: eGPU.converged };
    });

    console.log(`\n[df-hf-gpu-bench] ${JSON.stringify(r)}\n`);
    // Correctness gates (always). The ratio is reported, not asserted — the
    // build-side GPU win is established; the loop-side number on small n is
    // characterization (per-iter JK readback can lose to the WASM pool).
    expect(r.path).toBe("gpu");
    expect(r.gpuConverged).toBe(true);
    expect(r.dE).toBeLessThan(1.594e-3); // both loops agree to chemical accuracy
  });
});
