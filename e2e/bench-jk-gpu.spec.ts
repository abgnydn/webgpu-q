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

    if (r.skipped) { console.log(`(skipped: ${r.reason})`);
      test.skip();
      return;
    }

    const median = (xs: number[]): number => {
      const s = [...xs].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)]!;
    };

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`JK build per-iter wall — ethane cc-pVDZ (n=${r.n}), 5 trials`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`WASM JK (par=8): median ${median(r.wasmMs).toFixed(2).padStart(7)} ms   trials [${r.wasmMs.map((x) => x.toFixed(1)).join(", ")}] ms`);
    console.log(`GPU JK (WGSL):   median ${median(r.gpuMs).toFixed(2).padStart(7)} ms   trials [${r.gpuMs.map((x) => x.toFixed(1)).join(", ")}] ms`);
    console.log(`speedup: ${(median(r.wasmMs) / median(r.gpuMs)).toFixed(2)}×`);
    console.log(`max |G_WASM - G_GPU|: ${r.maxAbs.toExponential(2)} Ha   (f32 precision: expect ~1e-5 abs)`);
    console.log(`max relative error:   ${r.maxRel.toExponential(2)}    (expect ~1e-6 from f32)`);
    console.log(`══════════════════════════════════════════════════════════\n`);
     

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

    if (r.skipped) { console.log(`(skipped: ${r.reason})`);
      test.skip();
      return;
    }

    const median = (xs: number[]): number => {
      const s = [...xs].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)]!;
    };

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`JK build per-iter wall — benzene cc-pVDZ (n=${r.n}), 3 trials`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`WASM JK (par=8): median ${median(r.wasmMs).toFixed(1).padStart(7)} ms   trials [${r.wasmMs.map((x) => x.toFixed(0)).join(", ")}] ms`);
    console.log(`GPU JK (WGSL):   median ${median(r.gpuMs).toFixed(1).padStart(7)} ms   trials [${r.gpuMs.map((x) => x.toFixed(0)).join(", ")}] ms`);
    console.log(`speedup: ${(median(r.wasmMs) / median(r.gpuMs)).toFixed(2)}×`);
    console.log(`max |G_WASM - G_GPU|: ${r.maxAbs.toExponential(2)} Ha`);
    console.log(`max relative error:   ${r.maxRel.toExponential(2)}`);
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    expect(r.maxAbs).toBeLessThan(0.1);  // 100 mHa generous for f32 at n=120
  });

  test("benzene cc-pVDZ — full HF SCF with WGSL JK: convergence + energy", async ({ page }) => {
    test.setTimeout(15 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCFAsync },
        { buildERIWasmParallel },
        { initGPU },
        { sabAvailable },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
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

      const tEri = performance.now();
      const eri = await buildERIWasmParallel(shells, shells.length, 1e-10, 8);
      const eriMs = performance.now() - tEri;

      const integrals = computeMolecularIntegrals(shells, nuclei);
      (integrals as unknown as { eri_AO: Float64Array }).eri_AO = eri;

      // Two runs: WASM JK reference, then WGSL JK.
      const HFopts = { useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 100 } as const;

      const tWasm = performance.now();
      const hfWasm = await runRHFSCFAsync(integrals, nElectrons, {
        ...HFopts, parallel: 8, useWasmJK: true,
      });
      const wasmMs = performance.now() - tWasm;

      const tGpu = performance.now();
      const hfGpu = await runRHFSCFAsync(integrals, nElectrons, {
        ...HFopts, parallel: 8, useWgpuJK: device,
      });
      const gpuMs = performance.now() - tGpu;

      return {
        skipped: false as const,
        n: integrals.n, eriMs, wasmMs, gpuMs,
        eWasm: hfWasm.energy, iWasm: hfWasm.iter, cWasm: hfWasm.converged,
        eGpu: hfGpu.energy, iGpu: hfGpu.iter, cGpu: hfGpu.converged,
      };
    });

    if (r.skipped) { test.skip(); return; }

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Benzene cc-pVDZ HF SCF — WASM JK vs WGSL JK (n=${r.n})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`ERI build (WASM ×8 parallel):  ${(r.eriMs / 1000).toFixed(2)} s`);
    console.log();
    console.log(`HF SCF (WASM JK):  ${(r.wasmMs / 1000).toFixed(2)} s  iters=${r.iWasm}  E=${r.eWasm.toFixed(8)} Ha  converged=${r.cWasm}`);
    console.log(`HF SCF (WGSL JK):  ${(r.gpuMs / 1000).toFixed(2)} s  iters=${r.iGpu}  E=${r.eGpu.toFixed(8)} Ha  converged=${r.cGpu}`);
    console.log();
    console.log(`HF SCF speedup: ${(r.wasmMs / r.gpuMs).toFixed(2)}×`);
    console.log(`Energy delta:   ${(r.eWasm - r.eGpu).toExponential(2)} Ha (WASM minus WGSL)`);
    console.log();
    console.log(`Total cold→converged with WASM:  ${((r.eriMs + r.wasmMs) / 1000).toFixed(2)} s`);
    console.log(`Total cold→converged with WGSL:  ${((r.eriMs + r.gpuMs) / 1000).toFixed(2)} s`);
    console.log(`══════════════════════════════════════════════════════════\n`);
     
  });

  test("benzene cc-pVDZ — WGSL JK at LOOSE tolerance (1e-3) for fast geom-scan", async ({ page }) => {
    test.setTimeout(15 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCFAsync },
        { buildERIWasmParallel },
        { initGPU },
        { sabAvailable },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
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

      const eri = await buildERIWasmParallel(shells, shells.length, 1e-10, 8);
      const integrals = computeMolecularIntegrals(shells, nuclei);
      (integrals as unknown as { eri_AO: Float64Array }).eri_AO = eri;

      // Three tolerance levels to find the f32 convergence sweet spot.
      const looseOpts = { useDIIS: true, energyTol: 1e-3, densityTol: 1e-3, maxIter: 100 } as const;
      const medOpts = { useDIIS: true, energyTol: 1e-4, densityTol: 1e-4, maxIter: 100 } as const;
      const tightOpts = { useDIIS: true, energyTol: 1e-5, densityTol: 1e-4, maxIter: 100 } as const;

      const tLoose = performance.now();
      const hfLoose = await runRHFSCFAsync(integrals, nElectrons, {
        ...looseOpts, parallel: 8, useWgpuJK: device,
      });
      const looseMs = performance.now() - tLoose;

      const tMed = performance.now();
      const hfMed = await runRHFSCFAsync(integrals, nElectrons, {
        ...medOpts, parallel: 8, useWgpuJK: device,
      });
      const medMs = performance.now() - tMed;

      const tTight = performance.now();
      const hfTight = await runRHFSCFAsync(integrals, nElectrons, {
        ...tightOpts, parallel: 8, useWgpuJK: device,
      });
      const tightMs = performance.now() - tTight;

      // Reference WASM run at tight tolerance.
      const tRef = performance.now();
      const hfRef = await runRHFSCFAsync(integrals, nElectrons, {
        ...tightOpts, parallel: 8, useWasmJK: true,
      });
      const refMs = performance.now() - tRef;

      return {
        skipped: false as const, n: integrals.n,
        loose: { ms: looseMs, e: hfLoose.energy, iter: hfLoose.iter, cnv: hfLoose.converged },
        med:   { ms: medMs,   e: hfMed.energy,   iter: hfMed.iter,   cnv: hfMed.converged   },
        tight: { ms: tightMs, e: hfTight.energy, iter: hfTight.iter, cnv: hfTight.converged },
        ref:   { ms: refMs,   e: hfRef.energy,   iter: hfRef.iter,   cnv: hfRef.converged   },
      };
    });

    if (r.skipped) { test.skip(); return; }

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`WGSL JK convergence vs tolerance — benzene cc-pVDZ (n=${r.n})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`WGSL JK (tol=1e-3):  ${(r.loose.ms / 1000).toFixed(2).padStart(7)} s  iters=${r.loose.iter}  E=${r.loose.e.toFixed(6)}  cnv=${r.loose.cnv}`);
    console.log(`WGSL JK (tol=1e-4):  ${(r.med.ms / 1000).toFixed(2).padStart(7)} s  iters=${r.med.iter}  E=${r.med.e.toFixed(6)}  cnv=${r.med.cnv}`);
    console.log(`WGSL JK (tol=1e-5):  ${(r.tight.ms / 1000).toFixed(2).padStart(7)} s  iters=${r.tight.iter}  E=${r.tight.e.toFixed(6)}  cnv=${r.tight.cnv}`);
    console.log(`WASM JK (tol=1e-5):  ${(r.ref.ms / 1000).toFixed(2).padStart(7)} s  iters=${r.ref.iter}  E=${r.ref.e.toFixed(6)}  cnv=${r.ref.cnv}  (reference)`);
    console.log();
    console.log(`Energy errors vs WASM reference:`);
    console.log(`  loose: ${(r.loose.e - r.ref.e).toExponential(2)} Ha`);
    console.log(`  med:   ${(r.med.e - r.ref.e).toExponential(2)} Ha`);
    console.log(`  tight: ${(r.tight.e - r.ref.e).toExponential(2)} Ha`);
    console.log();
    if (r.loose.cnv) { console.log(`Speedup at tol=1e-3 (loose): ${(r.ref.ms / r.loose.ms).toFixed(2)}× vs WASM tight`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
     
  });
});
