import { test, expect } from "@playwright/test";

// Parallel buildJK_DF vs serial WASM build_jk_df.

test.describe("Parallel buildJK_DF", () => {
  test("benzene cc-pVDZ — parallel vs serial buildJK_DF", async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDFCholesky, generateAutoAux },
        { preloadWasmJK, buildJK_DF },
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

      // Need a D matrix — get one from a quick HF.
      const hf = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-4, maxIter: 10, useDF: df,
      });
      const D = hf.D;

      // Warmup.
      buildJK_DF(df, D);
      await buildJK_DF_Parallel(df, D, 8);

      // Bench serial (WASM).
      const trials = 3;
      const serialMs: number[] = [];
      for (let t = 0; t < trials; t++) {
        const t0 = performance.now();
        buildJK_DF(df, D);
        serialMs.push(performance.now() - t0);
      }
      const parMs: number[] = [];
      for (let t = 0; t < trials; t++) {
        const t0 = performance.now();
        await buildJK_DF_Parallel(df, D, 8);
        parMs.push(performance.now() - t0);
      }

      // Verify match.
      const ref = buildJK_DF(df, D);
      const par = await buildJK_DF_Parallel(df, D, 8);
      let maxDJ = 0, maxDK = 0;
      for (let i = 0; i < ref.J.length; i++) {
        const dJ = Math.abs(ref.J[i]! - par.J[i]!);
        const dK = Math.abs(ref.K[i]! - par.K[i]!);
        if (dJ > maxDJ) maxDJ = dJ;
        if (dK > maxDK) maxDK = dK;
      }

      return {
        skipped: false as const,
        n: df.n, nAux: df.nAux,
        serialMs, parMs, maxDJ, maxDK,
      };
    });

    if (r.skipped) { test.skip(); return; }

    const median = (xs: number[]): number => {
      const s = [...xs].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)]!;
    };

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Parallel buildJK_DF — benzene cc-pVDZ (n=${r.n}, n_aux=${r.nAux})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Serial WASM:  median ${median(r.serialMs).toFixed(0).padStart(6)} ms   trials [${r.serialMs.map((x) => x.toFixed(0)).join(", ")}] ms`);
    console.log(`Parallel ×8:  median ${median(r.parMs).toFixed(0).padStart(6)} ms   trials [${r.parMs.map((x) => x.toFixed(0)).join(", ")}] ms`);
    console.log(`speedup: ${(median(r.serialMs) / median(r.parMs)).toFixed(2)}×`);
    console.log(`max |J_serial − J_parallel|: ${r.maxDJ.toExponential(2)}`);
    console.log(`max |K_serial − K_parallel|: ${r.maxDK.toExponential(2)}`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.maxDJ).toBeLessThan(1e-8);
    expect(r.maxDK).toBeLessThan(1e-8);
  });
});
