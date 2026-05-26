import { test, expect } from "@playwright/test";

// Bench parallel HF buildG vs single-threaded, end-to-end in a real
// browser (so SharedArrayBuffer + crossOriginIsolated are live). Runs
// HF/cc-pVDZ on H₂O (n = 25 basis functions), 5 trials per config.
// Prints median + p10/p90 + per-iteration breakdown.
//
// This is the "open critique gap #5" closer: we shipped runRHFSCFAsync
// with parallel=N but never measured. Now we measure.

test.describe("Parallel HF buildG benchmark", () => {
  test("H₂O cc-pVDZ — runRHFSCFAsync(parallel=N) vs sync", async ({ page }) => {
    // Navigate to molecule.html so the page is cross-origin-isolated
    // (its index.html ships the COOP/COEP meta via the Vite dev headers).
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    // crossOriginIsolated must be true for SAB / Worker pool to engage.
    const isolated = await page.evaluate(() => self.crossOriginIsolated);
    expect(isolated).toBe(true);

    const results = await page.evaluate(async () => {
      const [{ runRHFSCF, runRHFSCFAsync }, { moleculeToShellsNuclei },
             { computeMolecularIntegrals }] = await Promise.all([
        import("/src/chemistry/hf-scf.ts"),
        import("/src/chemistry/atoms.ts"),
        import("/src/chemistry/cg-molecular.ts"),
      ]);

      // H₂O in cc-pVDZ — n = 25 basis functions, real workload.
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
      const integrals = computeMolecularIntegrals(shells, nuclei);

      const N_TRIALS = 5;
      const HFopts = {
        useDIIS: true, energyTol: 1e-9, densityTol: 1e-7, maxIter: 200,
      } as const;

      async function timeOnce(fn: () => unknown | Promise<unknown>): Promise<number> {
        const t0 = performance.now();
        await fn();
        return performance.now() - t0;
      }

      function stats(arr: number[]): { median: number; p10: number; p90: number; min: number; max: number } {
        const sorted = [...arr].sort((a, b) => a - b);
        const at = (q: number): number => sorted[Math.floor(q * (sorted.length - 1))]!;
        return { median: at(0.5), p10: at(0.1), p90: at(0.9), min: sorted[0]!, max: sorted[sorted.length - 1]! };
      }

      // Warmup: run sync HF once to amortize JIT.
      runRHFSCF(integrals, nElectrons, HFopts);

      // Sync trials.
      const syncTimes: number[] = [];
      let syncEnergy = 0;
      for (let i = 0; i < N_TRIALS; i++) {
        syncTimes.push(await timeOnce(() => {
          syncEnergy = runRHFSCF(integrals, nElectrons, HFopts).energy;
        }));
      }

      // Parallel trials, varying worker counts.
      const parallelResults: Record<string, { times: number[]; energy: number }> = {};
      for (const N of [2, 4, 8] as const) {
        const times: number[] = [];
        let energy = 0;
        // Warmup the worker pool once (spawn cost amortized across trials).
        await runRHFSCFAsync(integrals, nElectrons, { ...HFopts, parallel: N });
        for (let i = 0; i < N_TRIALS; i++) {
          times.push(await timeOnce(async () => {
            energy = (await runRHFSCFAsync(integrals, nElectrons,
              { ...HFopts, parallel: N })).energy;
          }));
        }
        parallelResults[`parallel=${N}`] = { times, energy };
      }

      return {
        n: integrals.n,
        hwConcurrency: navigator.hardwareConcurrency,
        crossOriginIsolated: self.crossOriginIsolated,
        sync: { stats: stats(syncTimes), energy: syncEnergy, times: syncTimes },
        parallel: Object.fromEntries(Object.entries(parallelResults).map(
          ([k, v]) => [k, { stats: stats(v.times), energy: v.energy, times: v.times }],
        )),
      };
    });

    /* eslint-disable no-console */
    console.log("\n──────────────────────────────────────────────────────────");
    console.log("Parallel HF buildG benchmark — H₂O / cc-pVDZ");
    console.log(`n = ${results.n} basis functions, hardwareConcurrency = ${results.hwConcurrency}, COI = ${results.crossOriginIsolated}`);
    console.log("──────────────────────────────────────────────────────────");
    const fmt = (ms: number): string => `${ms.toFixed(0)} ms`;
    console.log(`sync                : median ${fmt(results.sync.stats.median)}  p10 ${fmt(results.sync.stats.p10)}  p90 ${fmt(results.sync.stats.p90)}`);
    for (const [name, entry] of Object.entries(results.parallel)) {
      const speedup = (results.sync.stats.median / entry.stats.median).toFixed(2);
      console.log(`${name.padEnd(20)}: median ${fmt(entry.stats.median)}  p10 ${fmt(entry.stats.p10)}  p90 ${fmt(entry.stats.p90)}  → ${speedup}× vs sync`);
    }
    console.log("──────────────────────────────────────────────────────────");
    console.log(`Energies match: sync = ${results.sync.energy.toFixed(10)}, parallel = ${Object.values(results.parallel)[0]!.energy.toFixed(10)}`);
    console.log(`Energy max |Δ|: ${Math.max(...Object.values(results.parallel).map(p => Math.abs(p.energy - results.sync.energy))).toExponential(2)} Ha`);
    /* eslint-enable no-console */

    // Energy correctness: parallel path must match sync to numerical noise.
    for (const entry of Object.values(results.parallel)) {
      expect(Math.abs(entry.energy - results.sync.energy)).toBeLessThan(1e-8);
    }

    // No hard assertion on speedup direction — record the number honestly.
    // (Worker overhead can dominate for n=25 cc-pVDZ; that's a known
    // finding. If it loses, the bench should still report the loss.)
  });
});
