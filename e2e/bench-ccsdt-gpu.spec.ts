import { test, expect } from "@playwright/test";

// Bench CCSD(T) GPU kernel with proper warmup + 20 trials.
// Closes critique gap #4 — replaces the headline single-run "39×"
// number with median + p10/p90/std on H₂O cc-pVDZ, M2 Pro.
//
// Calls runCCSDT_GPU in a real browser tab; each call returns
// `gpuSeconds` (time inside the GPU kernel, post-warmup + sync).
// We aggregate the per-call gpuSeconds and compute stats.

test.describe("CCSD(T) GPU kernel — proper trials benchmark", () => {
  test("H₂O cc-pVDZ — 5 warmup + 20 trials, report median/p10/p90/std", async ({ page }) => {
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const results = await page.evaluate(async () => {
      const [
        { initGPU },
        { runRHFSCF },
        { runCCSD },
        { runCCSDT },
        { runCCSDT_GPU },
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
      ] = await Promise.all([
        import("/src/quantum.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/ccsd.ts" as string),
        import("/src/chemistry/ccsd-t.ts" as string),
        import("/src/chemistry/ccsd-t-gpu.ts" as string),
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
      ]);

      // H₂O cc-pVDZ — the headline benchmark molecule.
      const half = (104.52 / 2) * Math.PI / 180;
      const xH = 0.9572 * Math.sin(half);
      const zH = 0.9572 * Math.cos(half);
      const atoms = [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ xH, 0, zH] },
        { symbol: "H", pos: [-xH, 0, zH] },
      ] as const;
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei, { spherical: true });
      const hf = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, densityTol: 1e-7, maxIter: 200,
      });
      const ccsd = runCCSD(hf, integrals, { tol: 1e-9, maxIter: 100 });

      let device: GPUDevice;
      try {
        device = await initGPU();
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }

      const WARMUP = 5;
      const TRIALS = 20;

      // CPU reference — one shot, expensive (~3 minutes on M2 Pro).
      // Run only once to get the absolute speedup magnitude.
      const tCpu0 = performance.now();
      const cpuRes = runCCSDT(ccsd, hf, integrals);
      const cpuSeconds = (performance.now() - tCpu0) / 1000;

      // GPU warmup.
      let lastGpuRes: { tripleCorrection: number; gpuSeconds: number } | null = null;
      for (let i = 0; i < WARMUP; i++) {
        lastGpuRes = await runCCSDT_GPU(ccsd, hf, integrals, device);
      }

      // 20 trials, record gpuSeconds per call.
      const samples: number[] = [];
      for (let i = 0; i < TRIALS; i++) {
        const r = await runCCSDT_GPU(ccsd, hf, integrals, device);
        samples.push(r.gpuSeconds);
        lastGpuRes = r;
      }

      // Stats.
      const sorted = [...samples].sort((a, b) => a - b);
      const at = (q: number): number => sorted[Math.floor(q * (sorted.length - 1))]!;
      const median = at(0.5);
      const p10 = at(0.1);
      const p90 = at(0.9);
      const p99 = at(0.99);
      const min = sorted[0]!;
      const max = sorted[sorted.length - 1]!;
      const mean = samples.reduce((s, x) => s + x, 0) / samples.length;
      const variance = samples.reduce((s, x) => s + (x - mean) ** 2, 0) / samples.length;
      const std = Math.sqrt(variance);
      const iqr = at(0.75) - at(0.25);
      const stdOverMedian = std / median;

      return {
        n: integrals.n,
        NSO: 2 * integrals.n,
        NOCC: 2 * hf.nOccupied,
        NVIRT: 2 * (integrals.n - hf.nOccupied),
        cpuSeconds,
        cpuET: cpuRes.tripleCorrection,
        gpuET: lastGpuRes!.tripleCorrection,
        delta: Math.abs(lastGpuRes!.tripleCorrection - cpuRes.tripleCorrection),
        samples,
        gpuStats: { median, p10, p90, p99, min, max, mean, std, iqr, stdOverMedian },
        speedupMedian: cpuSeconds / median,
        speedupP10: cpuSeconds / p10,
        speedupP90: cpuSeconds / p90,
        adapterInfo: (await navigator.gpu.requestAdapter())?.info ?? null,
        ua: navigator.userAgent,
      };
    });

    if ("error" in results) {
      test.skip(true, results.error as string);
      return;
    }

     console.log("\n══════════════════════════════════════════════════════════");
    console.log("CCSD(T) GPU kernel — 5 warmup + 20 trials");
    console.log("══════════════════════════════════════════════════════════");
    console.log(`Molecule: H₂O cc-pVDZ (n=${results.n}, NOCC=${results.NOCC}, NVIRT=${results.NVIRT})`);
    console.log(`Hardware: ${results.adapterInfo?.vendor ?? "?"} / ${results.adapterInfo?.architecture ?? "?"}`);
    console.log();
    console.log(`CPU (single run): ${results.cpuSeconds.toFixed(2)} s   E_(T) = ${results.cpuET.toFixed(8)} Ha`);
    console.log();
    console.log(`GPU per-call gpuSeconds across 20 trials:`);
    const s = results.gpuStats;
    console.log(`  median: ${s.median.toFixed(3)} s`);
    console.log(`  p10:    ${s.p10.toFixed(3)} s`);
    console.log(`  p90:    ${s.p90.toFixed(3)} s`);
    console.log(`  p99:    ${s.p99.toFixed(3)} s`);
    console.log(`  min:    ${s.min.toFixed(3)} s`);
    console.log(`  max:    ${s.max.toFixed(3)} s`);
    console.log(`  mean:   ${s.mean.toFixed(3)} s`);
    console.log(`  std:    ${s.std.toFixed(3)} s   (std/median = ${(s.stdOverMedian * 100).toFixed(1)}%)`);
    console.log(`  IQR:    ${s.iqr.toFixed(3)} s`);
    console.log();
    console.log(`Speedup vs CPU TypeScript:`);
    console.log(`  at median: ${results.speedupMedian.toFixed(1)}×`);
    console.log(`  at p10:    ${results.speedupP10.toFixed(1)}× (best case)`);
    console.log(`  at p90:    ${results.speedupP90.toFixed(1)}× (worst case)`);
    console.log();
    console.log(`Correctness: |GPU − CPU| = ${results.delta.toExponential(2)} Ha (pass bar: ≤ 5e-6 Ha)`);
    if (s.stdOverMedian > 0.1) { console.log(`⚠  noisy: std/median = ${(s.stdOverMedian * 100).toFixed(1)}% > 10% — treat as range, not point estimate`);
    }
    console.log("══════════════════════════════════════════════════════════\n");
     

    // Correctness gate.
    expect(results.delta).toBeLessThan(5e-6);
    // Speedup sanity (any positive direction at median).
    expect(results.speedupMedian).toBeGreaterThan(1);
  });
});
