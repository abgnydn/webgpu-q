import { test, expect } from "@playwright/test";

// Bench parallel HF buildG vs single-threaded, end-to-end in a real
// browser (so SharedArrayBuffer + crossOriginIsolated are live). Runs
// HF/cc-pVDZ on H₂O (n = 25 basis functions), 5 trials per config.
// Prints median + p10/p90 + per-iteration breakdown.
//
// This is the "open critique gap #5" closer: we shipped runRHFSCFAsync
// with parallel=N but never measured. Now we measure.

// Shared bench helper — run sync vs parallel=N for any molecule.
async function benchMolecule(
  page: import("@playwright/test").Page,
  atoms: ReadonlyArray<{ symbol: string; pos: readonly [number, number, number] }>,
  label: string,
  nTrials = 5,
  parallelConfigs: readonly number[] = [2, 4, 8],
): Promise<unknown> {
  await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
  const isolated = await page.evaluate(() => self.crossOriginIsolated);
  expect(isolated).toBe(true);

  return page.evaluate(async ({ atomsIn, labelIn, trialsIn, parallelConfigsIn }) => {
    const [{ runRHFSCF, runRHFSCFAsync }, { moleculeToShellsNuclei },
           { computeMolecularIntegrals }] = await Promise.all([
      import("/src/chemistry/hf-scf.ts" as string),
      import("/src/chemistry/atoms.ts" as string),
      import("/src/chemistry/cg-molecular.ts" as string),
    ]);
    const tEriStart = performance.now();
    const { shells, nuclei, nElectrons } =
      moleculeToShellsNuclei(atomsIn as never, "cc-pvdz");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const eriBuildMs = performance.now() - tEriStart;

    const N_TRIALS = trialsIn;
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

    runRHFSCF(integrals, nElectrons, HFopts); // warmup

    const syncTimes: number[] = [];
    let syncEnergy = 0;
    for (let i = 0; i < N_TRIALS; i++) {
      syncTimes.push(await timeOnce(() => {
        syncEnergy = runRHFSCF(integrals, nElectrons, HFopts).energy;
      }));
    }

    const parallelResults: Record<string, { times: number[]; energy: number }> = {};
    for (const N of parallelConfigsIn) {
      const times: number[] = [];
      let energy = 0;
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
      label: labelIn, n: integrals.n,
      hwConcurrency: navigator.hardwareConcurrency,
      eriBuildMs,
      sync: { stats: stats(syncTimes), energy: syncEnergy, times: syncTimes },
      parallel: Object.fromEntries(Object.entries(parallelResults).map(
        ([k, v]) => [k, { stats: stats(v.times), energy: v.energy, times: v.times }],
      )),
    };
  }, { atomsIn: atoms as never, labelIn: label, trialsIn: nTrials, parallelConfigsIn: parallelConfigs as number[] });
}

function logBench(r: {
  label: string; n: number; hwConcurrency: number; eriBuildMs?: number;
  sync: { stats: { median: number; p10: number; p90: number }; energy: number };
  parallel: Record<string, { stats: { median: number; p10: number; p90: number }; energy: number }>;
}): void {
  /* eslint-disable no-console */
  console.log("\n──────────────────────────────────────────────────────────");
  console.log(`Parallel HF buildG — ${r.label} (n = ${r.n} basis functions, HC = ${r.hwConcurrency})`);
  if (r.eriBuildMs !== undefined) {
    console.log(`ERI build (one-time): ${(r.eriBuildMs / 1000).toFixed(1)} s`);
  }
  console.log("──────────────────────────────────────────────────────────");
  const fmt = (ms: number): string => `${ms.toFixed(0).padStart(6)} ms`;
  console.log(`sync                : median ${fmt(r.sync.stats.median)}  p10 ${fmt(r.sync.stats.p10)}  p90 ${fmt(r.sync.stats.p90)}`);
  for (const [name, entry] of Object.entries(r.parallel)) {
    const speedup = (r.sync.stats.median / entry.stats.median).toFixed(2);
    console.log(`${name.padEnd(20)}: median ${fmt(entry.stats.median)}  p10 ${fmt(entry.stats.p10)}  p90 ${fmt(entry.stats.p90)}  → ${speedup.padStart(5)}× vs sync`);
  }
  const maxDelta = Math.max(...Object.values(r.parallel).map(p => Math.abs(p.energy - r.sync.energy)));
  console.log(`Energy: sync = ${r.sync.energy.toFixed(8)}, parallel max |Δ| = ${maxDelta.toExponential(2)} Ha`);
  /* eslint-enable no-console */
}

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
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
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

  test("Ethane C₂H₆ cc-pVDZ — bigger molecule (n ≈ 58), expect bigger speedup", async ({ page }) => {
    // Standard ethane geometry, C-C 1.535 Å, C-H 1.094 Å, HCH 107.8°.
    // In cc-pVDZ: C (3s, 2p, 1d) × 2 + H (2s, 1p) × 6 = 2·14 + 6·5 = 58 fns.
    // JK build cost grows as n⁴ → ~30× more work per iter than H₂O cc-pVDZ.
    const cc = 1.535;
    const ch = 1.094;
    const hch = 107.8 * Math.PI / 180;
    const hcc = (Math.PI - hch) / 2 + Math.PI / 6; // approximate staggered geometry
    const sH = ch * Math.sin(hcc);
    const cH = ch * Math.cos(hcc);
    // Place C atoms along z-axis; arrange 3 H's around each C in a staggered ring.
    const atoms = [
      { symbol: "C", pos: [0, 0, -cc / 2] as const },
      { symbol: "C", pos: [0, 0,  cc / 2] as const },
      // H's on lower C
      { symbol: "H", pos: [ sH,             0,    -cc / 2 - cH] as const },
      { symbol: "H", pos: [-sH / 2,  sH * Math.sqrt(3) / 2, -cc / 2 - cH] as const },
      { symbol: "H", pos: [-sH / 2, -sH * Math.sqrt(3) / 2, -cc / 2 - cH] as const },
      // H's on upper C, staggered by 60°
      { symbol: "H", pos: [-sH,             0,     cc / 2 + cH] as const },
      { symbol: "H", pos: [ sH / 2,  sH * Math.sqrt(3) / 2,  cc / 2 + cH] as const },
      { symbol: "H", pos: [ sH / 2, -sH * Math.sqrt(3) / 2,  cc / 2 + cH] as const },
    ] as const;
    const results = await benchMolecule(page, atoms, "Ethane C₂H₆ cc-pVDZ") as {
      label: string; n: number; hwConcurrency: number;
      sync: { stats: { median: number; p10: number; p90: number }; energy: number };
      parallel: Record<string, { stats: { median: number; p10: number; p90: number }; energy: number }>;
    };
    logBench(results);

    // Energy correctness.
    for (const entry of Object.values(results.parallel)) {
      expect(Math.abs(entry.energy - results.sync.energy)).toBeLessThan(1e-8);
    }
  });

  test("Furan C₄H₄O cc-pVDZ — n=90, intermediate scaling point", async ({ page }) => {
    // Standard furan geometry (rough planar pentagon, O at one corner).
    // C-C 1.43 Å, C=C 1.36 Å, C-O 1.36 Å, C-H 1.08 Å. Ring in xy-plane.
    // In cc-pVDZ: 4 × 14 (C) + 1 × 14 (O) + 4 × 5 (H) = 90 basis functions.
    // Intermediate between ethane (n=60) and benzene (n=114).
    //
    // Approximate ring geometry: 5-membered ring; place atoms at
    // pentagonal vertices around the centroid, then push H's outward.
    const ringR = 1.18; // pentagon circumradius for the C-C/C-O ≈ 1.39 Å typical bond
    const hR = ringR + 1.08; // C-H 1.08 Å outward
    const atoms: Array<{ symbol: string; pos: readonly [number, number, number] }> = [];
    // O at top (angle 90°), then 4 C's around it.
    const ringAngles = [Math.PI / 2, Math.PI / 2 + 2 * Math.PI / 5,
                        Math.PI / 2 + 4 * Math.PI / 5, Math.PI / 2 - 4 * Math.PI / 5,
                        Math.PI / 2 - 2 * Math.PI / 5];
    const symbols = ["O", "C", "C", "C", "C"];
    for (let i = 0; i < 5; i++) {
      atoms.push({
        symbol: symbols[i]!,
        pos: [ringR * Math.cos(ringAngles[i]!), ringR * Math.sin(ringAngles[i]!), 0],
      });
    }
    // H on each of the 4 C atoms (not O), pointing outward.
    for (let i = 1; i < 5; i++) {
      const a = ringAngles[i]!;
      atoms.push({
        symbol: "H",
        pos: [hR * Math.cos(a), hR * Math.sin(a), 0],
      });
    }

    test.setTimeout(20 * 60 * 1000); // 20 min — fits if ERI < 10 min.

    const results = await benchMolecule(page, atoms, "Furan C₄H₄O cc-pVDZ", 1, [8]) as {
      label: string; n: number; hwConcurrency: number; eriBuildMs: number;
      sync: { stats: { median: number; p10: number; p90: number }; energy: number };
      parallel: Record<string, { stats: { median: number; p10: number; p90: number }; energy: number }>;
    };
    logBench(results);
    // Cartesian-d cc-pVDZ: 4×15 (C) + 1×15 (O) + 4×5 (H) = 95 basis fns.
    // (Spherical-d would give 4×14 + 14 + 4×5 = 90; webgpu-q defaults
    // to Cartesian, so 95 is correct.)
    expect(results.n).toBe(95);
    for (const entry of Object.values(results.parallel)) {
      expect(Math.abs(entry.energy - results.sync.energy)).toBeLessThan(1e-8);
    }
  });

  test.skip("Benzene C₆H₆ cc-pVDZ — n=120 (Cartesian-d), the canonical small-organic bench", async ({ page }) => {
    // SKIPPED 2026-05-26: Attempted at 20 min and 30 min Playwright
    // timeouts; both exceeded. ERI build alone at n=114 cc-pVDZ in
    // pure TypeScript takes ≈ 20-25 min, dominating wall-time. This is
    // the practical ceiling of the current implementation for browser-
    // tab benching. The ethane cc-pVDZ (n=60) result already establishes
    // the scaling trend (1.47× at n=25 → 3.0× at n=60); at parallel=8
    // on 12-thread M2 Pro the parallel-efficiency plateau (3/8 = 37%)
    // means extrapolation to n=114 predicts only ~3.5-4×, marginal.
    // To bench benzene properly we'd need: (a) faster ERI build (vectorize
    // Obara-Saika contractions), (b) shell-pair Schwarz screening, or
    // (c) density fitting. All deferred to follow-up work.
    //
    // Standard D6h benzene: C-C 1.395 Å, C-H 1.087 Å, ring in xy-plane.
    // In cc-pVDZ: 6 × 14 (C) + 6 × 5 (H) = 114 basis functions.
    // ERI tensor: 114⁴ × 8 B ≈ 1.35 GB — feasible on M2 Pro with 16 GB.
    const rCC = 1.395;
    const rCH = 1.087;
    const atoms: Array<{ symbol: string; pos: readonly [number, number, number] }> = [];
    for (let i = 0; i < 6; i++) {
      const θ = i * Math.PI / 3;
      atoms.push({ symbol: "C", pos: [rCC * Math.cos(θ), rCC * Math.sin(θ), 0] });
    }
    for (let i = 0; i < 6; i++) {
      const θ = i * Math.PI / 3;
      const r = rCC + rCH;
      atoms.push({ symbol: "H", pos: [r * Math.cos(θ), r * Math.sin(θ), 0] });
    }

    test.setTimeout(30 * 60 * 1000); // 30 min cap — ERI build dominates at n=114

    // 1 trial per config, single parallel=8 — minimal cost to get a data
    // point at n=114. Don't need variance characterization here; just
    // need to know whether the speedup curve keeps climbing past ethane.
    const results = await benchMolecule(page, atoms, "Benzene C₆H₆ cc-pVDZ", 1, [8]) as {
      label: string; n: number; hwConcurrency: number; eriBuildMs: number;
      sync: { stats: { median: number; p10: number; p90: number }; energy: number };
      parallel: Record<string, { stats: { median: number; p10: number; p90: number }; energy: number }>;
    };
    logBench(results);

    expect(results.n).toBe(114);
    // Energy correctness across parallel=N.
    for (const entry of Object.values(results.parallel)) {
      expect(Math.abs(entry.energy - results.sync.energy)).toBeLessThan(1e-8);
    }
  });
});
