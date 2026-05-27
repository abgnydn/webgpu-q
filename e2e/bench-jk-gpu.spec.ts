import { test, expect } from "@playwright/test";

// Bench the WebGPU JK build vs the WASM JK build. Measures both raw
// per-iter speed and the f32-vs-f64 precision delta.

test.describe("WGSL JK build (WebGPU)", () => {
  test("ethane cc-pVDZ — WASM vs GPU JK build, per-iter wall time", async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildGWasmParallel },
        { buildGGpu, disposeJKGpu },
        { initGPU },
        { sabAvailable },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/parallel/parallel-buildG-wasm.ts" as string),
        import("/src/chemistry/jk-gpu.ts" as string),
        import("/src/quantum.ts" as string),
        import("/src/parallel/worker-pool.ts" as string),
      ]);

      const device = await initGPU();
      if (!device) return { skipped: true as const, reason: "no GPU" };
      if (!sabAvailable()) return { skipped: true as const, reason: "no SAB" };

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
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      const integrals = computeMolecularIntegrals(shells, nuclei);
      const n = integrals.n;
      const eri = integrals.eri_AO;

      const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true });
      const D = hf.D;

      const N = 8;

      // Warmup both paths.
      await buildGWasmParallel(D, eri, n, N);
      await buildGWasmParallel(D, eri, n, N);
      await buildGGpu(device, D, eri, n);
      await buildGGpu(device, D, eri, n);

      const trials = 5;
      const wasmMs: number[] = [];
      const gpuMs: number[] = [];
      for (let t = 0; t < trials; t++) {
        const t0 = performance.now();
        await buildGWasmParallel(D, eri, n, N);
        wasmMs.push(performance.now() - t0);
      }
      for (let t = 0; t < trials; t++) {
        const t0 = performance.now();
        await buildGGpu(device, D, eri, n);
        gpuMs.push(performance.now() - t0);
      }

      // Precision comparison.
      const Gwasm = await buildGWasmParallel(D, eri, n, N);
      const Ggpu = await buildGGpu(device, D, eri, n);
      let maxAbs = 0, maxRel = 0;
      for (let i = 0; i < Gwasm.length; i++) {
        const d = Math.abs(Gwasm[i]! - Ggpu[i]!);
        const refMag = Math.max(Math.abs(Gwasm[i]!), 1e-300);
        if (d > maxAbs) maxAbs = d;
        const rel = d / refMag;
        if (rel > maxRel && refMag > 1e-6) maxRel = rel;
      }

      disposeJKGpu();
      return {
        skipped: false as const, n, wasmMs, gpuMs, maxAbs, maxRel,
      };
    });

    if (r.skipped) {
      console.log(`(skipped: ${r.reason})`);
      test.skip();
      return;
    }

    const median = (xs: number[]): number => {
      const s = [...xs].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)]!;
    };

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`JK build per-iter wall — ethane cc-pVDZ (n=${r.n}), 5 trials`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`WASM JK (par=8): median ${median(r.wasmMs).toFixed(2).padStart(7)} ms   trials [${r.wasmMs.map((x) => x.toFixed(1)).join(", ")}] ms`);
    console.log(`GPU JK (WGSL):   median ${median(r.gpuMs).toFixed(2).padStart(7)} ms   trials [${r.gpuMs.map((x) => x.toFixed(1)).join(", ")}] ms`);
    console.log(`speedup: ${(median(r.wasmMs) / median(r.gpuMs)).toFixed(2)}×`);
    console.log(`max |G_WASM - G_GPU|: ${r.maxAbs.toExponential(2)} Ha   (f32 precision: expect ~1e-5 abs)`);
    console.log(`max relative error:   ${r.maxRel.toExponential(2)}    (expect ~1e-6 from f32)`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.maxAbs).toBeLessThan(0.01);  // 10 mHa is generous for f32
  });

  test("benzene cc-pVDZ — WASM vs GPU JK build, per-iter wall time", async ({ page }) => {
    test.setTimeout(15 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCFAsync },
        { buildGWasmParallel },
        { buildGGpu, disposeJKGpu },
        { buildERIWasmParallel },
        { initGPU },
        { sabAvailable },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/parallel/parallel-buildG-wasm.ts" as string),
        import("/src/chemistry/jk-gpu.ts" as string),
        import("/src/chemistry/parallel-eri.ts" as string),
        import("/src/quantum.ts" as string),
        import("/src/parallel/worker-pool.ts" as string),
      ]);

      const device = await initGPU();
      if (!device) return { skipped: true as const, reason: "no GPU" };
      if (!sabAvailable()) return { skipped: true as const, reason: "no SAB" };

      const rCC = 1.395, rCH = 1.087;
      const atoms: Array<{ symbol: string; pos: readonly [number, number, number] }> = [];
      for (let i = 0; i < 6; i++) {
        const θ = i * Math.PI / 3;
        atoms.push({ symbol: "C", pos: [rCC * Math.cos(θ), rCC * Math.sin(θ), 0] });
      }
      for (let i = 0; i < 6; i++) {
        const θ = i * Math.PI / 3;
        const r2 = rCC + rCH;
        atoms.push({ symbol: "H", pos: [r2 * Math.cos(θ), r2 * Math.sin(θ), 0] });
      }
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      // ERI via WASM-parallel (fast path).
      const eri = await buildERIWasmParallel(shells, shells.length, 1e-10, 8);
      const integrals = computeMolecularIntegrals(shells, nuclei);
      (integrals as unknown as { eri_AO: Float64Array }).eri_AO = eri;
      const n = integrals.n;

      // Converge HF once to get D.
      const hf = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, parallel: 8, useWasmJK: true,
      });
      const D = hf.D;

      const N = 8;

      // Warmup both.
      await buildGWasmParallel(D, eri, n, N);
      await buildGWasmParallel(D, eri, n, N);
      await buildGGpu(device, D, eri, n);
      await buildGGpu(device, D, eri, n);

      const trials = 3;
      const wasmMs: number[] = [];
      const gpuMs: number[] = [];
      for (let t = 0; t < trials; t++) {
        const t0 = performance.now();
        await buildGWasmParallel(D, eri, n, N);
        wasmMs.push(performance.now() - t0);
      }
      for (let t = 0; t < trials; t++) {
        const t0 = performance.now();
        await buildGGpu(device, D, eri, n);
        gpuMs.push(performance.now() - t0);
      }

      // Precision comparison.
      const Gwasm = await buildGWasmParallel(D, eri, n, N);
      const Ggpu = await buildGGpu(device, D, eri, n);
      let maxAbs = 0, maxRel = 0;
      for (let i = 0; i < Gwasm.length; i++) {
        const d = Math.abs(Gwasm[i]! - Ggpu[i]!);
        const refMag = Math.max(Math.abs(Gwasm[i]!), 1e-300);
        if (d > maxAbs) maxAbs = d;
        const rel = d / refMag;
        if (rel > maxRel && refMag > 1e-6) maxRel = rel;
      }

      disposeJKGpu();
      return {
        skipped: false as const, n, wasmMs, gpuMs, maxAbs, maxRel,
      };
    });

    if (r.skipped) {
      console.log(`(skipped: ${r.reason})`);
      test.skip();
      return;
    }

    const median = (xs: number[]): number => {
      const s = [...xs].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)]!;
    };

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`JK build per-iter wall — benzene cc-pVDZ (n=${r.n}), 3 trials`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`WASM JK (par=8): median ${median(r.wasmMs).toFixed(1).padStart(7)} ms   trials [${r.wasmMs.map((x) => x.toFixed(0)).join(", ")}] ms`);
    console.log(`GPU JK (WGSL):   median ${median(r.gpuMs).toFixed(1).padStart(7)} ms   trials [${r.gpuMs.map((x) => x.toFixed(0)).join(", ")}] ms`);
    console.log(`speedup: ${(median(r.wasmMs) / median(r.gpuMs)).toFixed(2)}×`);
    console.log(`max |G_WASM - G_GPU|: ${r.maxAbs.toExponential(2)} Ha`);
    console.log(`max relative error:   ${r.maxRel.toExponential(2)}`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.maxAbs).toBeLessThan(0.1);  // 100 mHa generous for f32 at n=120
  });
});
