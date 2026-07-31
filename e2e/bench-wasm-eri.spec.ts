import { test, expect } from "@playwright/test";

// Bench the Rust+WASM ERI kernel vs the sequential TypeScript ERI build.
// Goal: measure the native-compile-vs-JIT delta on the n⁴ ERI inner
// loop. Same algorithm, same Schwarz screening, same 8-fold symmetry —
// the only difference is the language the inner loop runs in.

test.describe("WASM ERI kernel", () => {
  test("benzene cc-pVDZ — full HF wall-time with WASM ERI vs TS ERI", async ({ page }) => {
    test.setTimeout(20 * 60 * 1000); // 20 min — WASM path should easily fit
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF, runRHFSCFAsync },
        { buildERIWasm },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/wasm-eri.ts" as string),
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

      // Path A: WASM ERI + parallel=8 HF.
      const tWasmEri = performance.now();
      const eriWasm = await buildERIWasm(shells, 1e-10);
      const wasmEriMs = performance.now() - tWasmEri;

      // We still need the rest of computeMolecularIntegrals — S, h, X, Vnn, etc.
      // The trick: build the full integrals (with TS ERI) but only TIME the ERI part.
      // To avoid double-paying ERI cost, we build via WASM and patch the field.
      // Simpler proxy: use TS ERI cost as baseline; replace eri_AO in the struct.

      // Build everything-except-ERI by calling computeMolecularIntegrals
      // and overwriting eri_AO after. This costs us the TS ERI build once
      // (as the baseline measurement), then HF runs use the WASM ERI.
      const tBaselineEri = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const tsEriMs = performance.now() - tBaselineEri; // approx, includes S/h/X too

      // Verify match.
      let maxDelta = 0;
      for (let i = 0; i < eriWasm.length; i++) {
        const d = Math.abs(eriWasm[i]! - integrals.eri_AO[i]!);
        if (d > maxDelta) maxDelta = d;
      }

      // Patch integrals to use WASM ERI for HF.
      (integrals as unknown as { eri_AO: Float64Array }).eri_AO = eriWasm;

      const HFopts = {
        useDIIS: true, energyTol: 1e-9, densityTol: 1e-7, maxIter: 200,
      } as const;
      // Warmup HF.
      runRHFSCF(integrals, nElectrons, HFopts);
      // Measure sync HF on WASM-ERI integrals.
      const tHfSync = performance.now();
      const hfSync = runRHFSCF(integrals, nElectrons, HFopts);
      const hfSyncMs = performance.now() - tHfSync;
      // Measure parallel=8 HF on the same integrals.
      const tHfPar = performance.now();
      const hfPar = await runRHFSCFAsync(integrals, nElectrons, { ...HFopts, parallel: 8 });
      const hfParMs = performance.now() - tHfPar;

      return {
        n: integrals.n,
        wasmEriMs, tsEriMs, maxDelta,
        hfSyncMs, hfParMs,
        eSync: hfSync.energy, ePar: hfPar.energy,
      };
    });

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Benzene cc-pVDZ — end-to-end HF wall time with WASM ERI`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Molecule:  benzene cc-pVDZ, n=${r.n} basis functions`);
    console.log();
    console.log(`ERI build:`);
    console.log(`  TypeScript (baseline):   ${(r.tsEriMs / 1000).toFixed(1).padStart(6)} s`);
    console.log(`  Rust+WASM:               ${(r.wasmEriMs / 1000).toFixed(1).padStart(6)} s   → ${(r.tsEriMs / r.wasmEriMs).toFixed(2)}× vs TS`);
    console.log(`  max |Δ|:                 ${r.maxDelta.toExponential(2)} Ha`);
    console.log();
    console.log(`HF SCF (using WASM-built ERI):`);
    console.log(`  sync:                    ${(r.hfSyncMs / 1000).toFixed(1).padStart(6)} s   E = ${r.eSync.toFixed(8)} Ha`);
    console.log(`  parallel=8:              ${(r.hfParMs / 1000).toFixed(1).padStart(6)} s   E = ${r.ePar.toFixed(8)} Ha   speedup ${(r.hfSyncMs / r.hfParMs).toFixed(2)}×`);
    console.log();
    console.log(`Total wall time (HF complete from cold shells):`);
    console.log(`  TS path:                 ${((r.tsEriMs + r.hfSyncMs) / 1000).toFixed(1).padStart(6)} s`);
    console.log(`  WASM + parallel HF:      ${((r.wasmEriMs + r.hfParMs) / 1000).toFixed(1).padStart(6)} s   → ${((r.tsEriMs + r.hfSyncMs) / (r.wasmEriMs + r.hfParMs)).toFixed(2)}× faster than TS-only`);
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    expect(r.maxDelta).toBeLessThan(1e-10);
    expect(Math.abs(r.eSync - r.ePar)).toBeLessThan(1e-8);
  });

  test("ethane cc-pVDZ — TypeScript sequential vs Rust+WASM", async ({ page }) => {
    test.setTimeout(10 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { buildERIWasm },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/wasm-eri.ts" as string),
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

      // TS sequential reference.
      const tTs = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const tsMs = performance.now() - tTs;
      const eriTs = integrals.eri_AO;

      // WASM warmup (first call loads + initializes wasm).
      await buildERIWasm(shells, 1e-10);

      // WASM measurement.
      const tWasm = performance.now();
      const eriWasm = await buildERIWasm(shells, 1e-10);
      const wasmMs = performance.now() - tWasm;

      // Verify bit-equality (or as close as the two paths get).
      let maxDelta = 0;
      let maxRel = 0;
      for (let i = 0; i < eriTs.length; i++) {
        const d = Math.abs(eriTs[i]! - eriWasm[i]!);
        if (d > maxDelta) maxDelta = d;
        const refMag = Math.max(Math.abs(eriTs[i]!), 1e-300);
        const rel = d / refMag;
        if (rel > maxRel && refMag > 1e-10) maxRel = rel;
      }
      return { n: integrals.n, tsMs, wasmMs, maxDelta, maxRel };
    });

     console.log(`\n── WASM ERI vs TypeScript ERI — ethane cc-pVDZ (n=${r.n}) ──`);
    console.log(`TypeScript sequential:  ${r.tsMs.toFixed(0).padStart(7)} ms`);
    console.log(`Rust+WASM sequential:   ${r.wasmMs.toFixed(0).padStart(7)} ms  → ${(r.tsMs / r.wasmMs).toFixed(2)}× vs TS`);
    console.log(`max |Δ|: ${r.maxDelta.toExponential(2)} Ha    max relative diff: ${r.maxRel.toExponential(2)}`);
     

    // Bit-level match: WASM and TS implement the same algorithm; expect ≤ 1e-10.
    expect(r.maxDelta).toBeLessThan(1e-10);
  });
});
