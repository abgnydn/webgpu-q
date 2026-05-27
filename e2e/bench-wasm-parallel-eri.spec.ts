import { test, expect } from "@playwright/test";

// Bench the WASM × Workers compound. Compares three paths for the same
// shells:
//   A. TS single-thread     — the original sequential baseline
//   B. WASM single-thread   — buildERIWasm()  (~4.7× over A on benzene)
//   C. WASM + parallel=N    — buildERIWasmParallel()  (target: ~2-3× over B)
//
// All three must agree element-wise to ≤ 1e-10 Ha.

test.describe("WASM × Workers ERI compound", () => {
  test("ethane cc-pVDZ — TS vs WASM vs WASM+parallel", async ({ page }) => {
    test.setTimeout(10 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { buildERIWasm },
        { buildERIWasmParallel },
        { sabAvailable },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/wasm-eri.ts" as string),
        import("/src/chemistry/parallel-eri.ts" as string),
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
      const { shells, nuclei } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      if (!sabAvailable()) {
        return { skipped: true as const, reason: "SAB unavailable" };
      }

      // Warm WASM.
      await buildERIWasm(shells, 1e-10);
      await buildERIWasmParallel(shells, shells.length, 1e-10, 2);

      // A: TS single-thread (also gives us reference ERI tensor + S/h/X).
      const tTs = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const tsMs = performance.now() - tTs;
      const eriTs = integrals.eri_AO;

      // B: WASM single-thread.
      const tWasm = performance.now();
      const eriWasm = await buildERIWasm(shells, 1e-10);
      const wasmMs = performance.now() - tWasm;

      // C: WASM + parallel=N for N in {2, 4, 8}.
      const wasmParResults: Array<{ N: number; ms: number; maxDelta: number }> = [];
      for (const N of [2, 4, 8]) {
        // Warmup at this N.
        await buildERIWasmParallel(shells, shells.length, 1e-10, N);
        const t = performance.now();
        const eriPar = await buildERIWasmParallel(shells, shells.length, 1e-10, N);
        const ms = performance.now() - t;
        let maxD = 0;
        for (let i = 0; i < eriTs.length; i++) {
          const d = Math.abs(eriTs[i]! - eriPar[i]!);
          if (d > maxD) maxD = d;
        }
        wasmParResults.push({ N, ms, maxDelta: maxD });
      }

      // Verify single-thread WASM matches TS.
      let maxDeltaWasm = 0;
      for (let i = 0; i < eriTs.length; i++) {
        const d = Math.abs(eriTs[i]! - eriWasm[i]!);
        if (d > maxDeltaWasm) maxDeltaWasm = d;
      }

      return {
        skipped: false as const,
        n: integrals.n,
        tsMs,
        wasmMs, maxDeltaWasm,
        wasmParResults,
      };
    });

    if (r.skipped) {
      console.log(`(skipped: ${r.reason})`);
      test.skip();
      return;
    }

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Ethane cc-pVDZ ERI build — TS vs WASM vs WASM+parallel`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Molecule:  ethane cc-pVDZ, n=${r.n} basis functions`);
    console.log();
    console.log(`A. TS single-thread:    ${r.tsMs.toFixed(0).padStart(7)} ms   (baseline)`);
    console.log(`B. WASM single-thread:  ${r.wasmMs.toFixed(0).padStart(7)} ms   → ${(r.tsMs / r.wasmMs).toFixed(2)}× vs A   max|Δ|=${r.maxDeltaWasm.toExponential(2)}`);
    for (const c of r.wasmParResults) {
      const vsA = r.tsMs / c.ms;
      const vsB = r.wasmMs / c.ms;
      console.log(`C. WASM × parallel=${c.N}:  ${c.ms.toFixed(0).padStart(7)} ms   → ${vsA.toFixed(2)}× vs A, ${vsB.toFixed(2)}× vs B   max|Δ|=${c.maxDelta.toExponential(2)}`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.maxDeltaWasm).toBeLessThan(1e-10);
    for (const c of r.wasmParResults) {
      expect(c.maxDelta).toBeLessThan(1e-10);
    }
  });

  test("benzene cc-pVDZ — WASM single-thread vs WASM+parallel", async ({ page }) => {
    test.setTimeout(25 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { buildERIWasm },
        { buildERIWasmParallel },
        { sabAvailable },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/wasm-eri.ts" as string),
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
      const { shells } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      if (!sabAvailable()) {
        return { skipped: true as const, reason: "SAB unavailable" };
      }

      // Warm WASM (both paths).
      await buildERIWasm(shells, 1e-10);

      // B: WASM single-thread.
      const tWasm = performance.now();
      const eriWasm = await buildERIWasm(shells, 1e-10);
      const wasmMs = performance.now() - tWasm;

      // C: WASM + parallel=N for N in {4, 8}.
      const parResults: Array<{ N: number; ms: number; maxDelta: number }> = [];
      for (const N of [4, 8]) {
        const tPar = performance.now();
        const eriPar = await buildERIWasmParallel(shells, shells.length, 1e-10, N);
        const parMs = performance.now() - tPar;
        let maxD = 0;
        for (let i = 0; i < eriWasm.length; i++) {
          const d = Math.abs(eriWasm[i]! - eriPar[i]!);
          if (d > maxD) maxD = d;
        }
        parResults.push({ N, ms: parMs, maxDelta: maxD });
      }

      return {
        skipped: false as const,
        n: shells.length,
        wasmMs, parResults,
      };
    });

    if (r.skipped) {
      console.log(`(skipped: ${r.reason})`);
      test.skip();
      return;
    }

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Benzene cc-pVDZ ERI build — WASM single vs WASM+parallel`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Molecule:  benzene cc-pVDZ, n=${r.n} basis functions`);
    console.log();
    console.log(`B. WASM single-thread:   ${(r.wasmMs / 1000).toFixed(1).padStart(7)} s`);
    for (const c of r.parResults) {
      console.log(`C. WASM × parallel=${c.N}:    ${(c.ms / 1000).toFixed(1).padStart(7)} s   → ${(r.wasmMs / c.ms).toFixed(2)}× vs B   max|Δ|=${c.maxDelta.toExponential(2)}`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    for (const c of r.parResults) {
      expect(c.maxDelta).toBeLessThan(1e-10);
    }
  });
});
