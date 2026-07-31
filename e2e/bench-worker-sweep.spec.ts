import { test, expect } from "@playwright/test";

// Worker-count sweep on JK_DF — is 8 the right number on this hardware,
// or are we memory-bandwidth limited (fewer workers → less DRAM
// contention) or core-limited (more workers → more compute)?

test.describe("JK_DF worker-count sweep", () => {
  test("benzene cc-pVDZ — sweep poolSize 2..16", async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDFCholesky, generateAutoAux },
        { preloadWasmJK },
        { buildJK_DF_Parallel },
        { sabAvailable },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
        import("/src/parallel/parallel-jk-df.ts" as string),
        import("/src/parallel/worker-pool.ts" as string),
      ]);
      if (!sabAvailable()) return { skipped: true as const };
      await preloadWasmJK();

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
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const auxShells = generateAutoAux(shells, 1);
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);

      const hf = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-4, maxIter: 10, useDF: df,
      });
      const D = hf.D;

      const counts = [2, 4, 6, 8, 10, 12, 16];
      const results: Array<{ N: number; medianMs: number; trialsMs: number[] }> = [];
      // Per-count warmup + 5 trials.
      for (const N of counts) {
        await buildJK_DF_Parallel(df, D, N);  // warmup
        const trials: number[] = [];
        for (let t = 0; t < 5; t++) {
          const t0 = performance.now();
          await buildJK_DF_Parallel(df, D, N);
          trials.push(performance.now() - t0);
        }
        const sorted = [...trials].sort((a, b) => a - b);
        results.push({ N, medianMs: sorted[Math.floor(sorted.length / 2)]!, trialsMs: trials });
      }
      return { skipped: false as const, n: df.n, nAux: df.nAux, results };
    });

    if (r.skipped) { test.skip(); return; }

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Worker-count sweep — benzene cc-pVDZ (n=${r.n}, n_aux=${r.nAux})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`workers | median ms | trials`);
    console.log(`--------+-----------+----------------------`);
    for (const row of r.results) { console.log(`${String(row.N).padStart(7)} | ${row.medianMs.toFixed(0).padStart(9)} | [${row.trialsMs.map((x) => x.toFixed(0)).join(", ")}]`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    expect(r.results.length).toBeGreaterThan(0);
  });
});
