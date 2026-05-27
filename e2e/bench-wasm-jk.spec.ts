import { test, expect } from "@playwright/test";

// Benchmark the WASM JK build vs the TS JK build, holding everything
// else (ERI, density matrix, worker pool size) equal.

test.describe("WASM JK build", () => {
  test("ethane cc-pVDZ — TS vs WASM JK, per-iter wall time", async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildGParallel },
        { buildGWasmParallel },
        { sabAvailable },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/parallel/parallel-buildG.ts" as string),
        import("/src/parallel/parallel-buildG-wasm.ts" as string),
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
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      if (!sabAvailable()) return { skipped: true as const };

      const integrals = computeMolecularIntegrals(shells, nuclei);
      const n = integrals.n;
      const eri = integrals.eri_AO;

      // Use the converged density from a sync HF run as the bench D matrix.
      const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true });
      const D = hf.D;

      const N = 8;

      // Warmup: 2 calls each path.
      await buildGParallel(D, eri, n, N);
      await buildGParallel(D, eri, n, N);
      await buildGWasmParallel(D, eri, n, N);
      await buildGWasmParallel(D, eri, n, N);

      const trials = 5;
      const tsMs: number[] = [];
      const wasmMs: number[] = [];
      for (let t = 0; t < trials; t++) {
        const t0 = performance.now();
        await buildGParallel(D, eri, n, N);
        tsMs.push(performance.now() - t0);
      }
      for (let t = 0; t < trials; t++) {
        const t0 = performance.now();
        await buildGWasmParallel(D, eri, n, N);
        wasmMs.push(performance.now() - t0);
      }

      // Verify match on the most recent G outputs.
      const Gts = await buildGParallel(D, eri, n, N);
      const Gwasm = await buildGWasmParallel(D, eri, n, N);
      let maxDelta = 0;
      for (let i = 0; i < Gts.length; i++) {
        const d = Math.abs(Gts[i]! - Gwasm[i]!);
        if (d > maxDelta) maxDelta = d;
      }

      return {
        skipped: false as const,
        n, tsMs, wasmMs, maxDelta,
      };
    });

    if (r.skipped) { test.skip(); return; }

    const median = (xs: number[]): number => {
      const s = [...xs].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)]!;
    };

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`JK build per-iter wall time — ethane cc-pVDZ (n=${r.n}), 5 trials`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`TS-parallel=8:   median ${median(r.tsMs).toFixed(1).padStart(7)} ms   trials [${r.tsMs.map((x) => x.toFixed(0)).join(", ")}] ms`);
    console.log(`WASM-parallel=8: median ${median(r.wasmMs).toFixed(1).padStart(7)} ms   trials [${r.wasmMs.map((x) => x.toFixed(0)).join(", ")}] ms`);
    console.log(`speedup: ${(median(r.tsMs) / median(r.wasmMs)).toFixed(2)}×`);
    console.log(`max |G_TS - G_WASM|: ${r.maxDelta.toExponential(2)} Ha`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.maxDelta).toBeLessThan(1e-10);
  });

  test("benzene cc-pVDZ — TS vs WASM JK, per-iter wall time", async ({ page }) => {
    test.setTimeout(20 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCFAsync },
        { buildGParallel },
        { buildGWasmParallel },
        { buildERIWasmParallel },
        { sabAvailable },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/parallel/parallel-buildG.ts" as string),
        import("/src/parallel/parallel-buildG-wasm.ts" as string),
        import("/src/chemistry/parallel-eri.ts" as string),
        import("/src/parallel/worker-pool.ts" as string),
      ]);

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

      if (!sabAvailable()) return { skipped: true as const };

      // ERI via WASM parallel (fast path).
      const tEri = performance.now();
      const eri = await buildERIWasmParallel(shells, shells.length, 1e-10, 8);
      const eriMs = performance.now() - tEri;

      // Synthesize a MolecularIntegrals via the TS path (S, h, X, Vnn)
      // then patch in WASM-built eri.
      const integrals = computeMolecularIntegrals(shells, nuclei);
      (integrals as unknown as { eri_AO: Float64Array }).eri_AO = eri;
      const n = integrals.n;

      // Run HF once to get a converged D for the bench.
      const hf = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, parallel: 8, useWasmJK: true,
      });
      const D = hf.D;

      const N = 8;

      // Warmup.
      await buildGParallel(D, eri, n, N);
      await buildGParallel(D, eri, n, N);
      await buildGWasmParallel(D, eri, n, N);
      await buildGWasmParallel(D, eri, n, N);

      const trials = 3;
      const tsMs: number[] = [];
      const wasmMs: number[] = [];
      for (let t = 0; t < trials; t++) {
        const t0 = performance.now();
        await buildGParallel(D, eri, n, N);
        tsMs.push(performance.now() - t0);
      }
      for (let t = 0; t < trials; t++) {
        const t0 = performance.now();
        await buildGWasmParallel(D, eri, n, N);
        wasmMs.push(performance.now() - t0);
      }

      // Verify match.
      const Gts = await buildGParallel(D, eri, n, N);
      const Gwasm = await buildGWasmParallel(D, eri, n, N);
      let maxDelta = 0;
      for (let i = 0; i < Gts.length; i++) {
        const d = Math.abs(Gts[i]! - Gwasm[i]!);
        if (d > maxDelta) maxDelta = d;
      }

      // Also measure full HF with WASM JK (to confirm SCF still converges).
      const tFullHF = performance.now();
      const hf2 = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, parallel: 8, useWasmJK: true,
      });
      const fullHFms = performance.now() - tFullHF;

      return {
        skipped: false as const,
        n, eriMs, tsMs, wasmMs, maxDelta,
        fullHFms, energy: hf2.energy, iters: hf2.iter,
      };
    });

    if (r.skipped) { test.skip(); return; }

    const median = (xs: number[]): number => {
      const s = [...xs].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)]!;
    };

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`JK build per-iter wall time — benzene cc-pVDZ (n=${r.n})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`TS-parallel=8:   median ${(median(r.tsMs) / 1000).toFixed(2).padStart(7)} s   trials [${r.tsMs.map((x) => x.toFixed(0)).join(", ")}] ms`);
    console.log(`WASM-parallel=8: median ${(median(r.wasmMs) / 1000).toFixed(2).padStart(7)} s   trials [${r.wasmMs.map((x) => x.toFixed(0)).join(", ")}] ms`);
    console.log(`speedup: ${(median(r.tsMs) / median(r.wasmMs)).toFixed(2)}×`);
    console.log(`max |G_TS - G_WASM|: ${r.maxDelta.toExponential(2)} Ha`);
    console.log();
    console.log(`Full HF (parallel=8, useWasmJK=true): ${(r.fullHFms / 1000).toFixed(2)} s`);
    console.log(`  iters: ${r.iters}   E = ${r.energy.toFixed(8)} Ha`);
    console.log(`  ERI build: ${(r.eriMs / 1000).toFixed(2)} s`);
    console.log(`  total cold-shell → converged: ${((r.eriMs + r.fullHFms) / 1000).toFixed(2)} s`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.maxDelta).toBeLessThan(1e-10);
    expect(Math.abs(r.energy - (-230.72273482))).toBeLessThan(1e-6);
  });
});
