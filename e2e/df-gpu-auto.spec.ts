import { test, expect } from "@playwright/test";

// buildDFAuto: pick the GPU integral path in the d-regime (where it wins), WASM
// otherwise — and return a DF tensor that gives the right HF energy either way.

test.describe("buildDFAuto — GPU/WASM path selection", () => {
  test("benzene cc-pVDZ picks GPU and gives the same HF energy as WASM", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[auto]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[auto] ${s}`); };
      const [
        { moleculeToShellsNuclei }, { computeMolecularIntegrals }, { runRHFSCF },
        { generateAutoAux, buildAuxBasisDFStreaming }, { buildDFAuto },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);
      const rCC = 1.395, rCH = 1.087; const atoms: { symbol: string; pos: number[] }[] = [];
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; atoms.push({ symbol: "C", pos: [rCC * Math.cos(t), rCC * Math.sin(t), 0] }); }
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; const r2 = rCC + rCH; atoms.push({ symbol: "H", pos: [r2 * Math.cos(t), r2 * Math.sin(t), 0] }); }
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const aux = generateAutoAux(shells, 0);

      // End-to-end DF-build timing (V + projection), GPU-auto vs WASM streaming.
      await buildDFAuto(shells as never, aux as never); await buildAuxBasisDFStreaming(shells as never, aux as never); // warm
      const tg0 = performance.now(); const auto = await buildDFAuto(shells as never, aux as never); const gpuMs = performance.now() - tg0;
      const tw0 = performance.now(); const wasm = await buildAuxBasisDFStreaming(shells as never, aux as never); const wasmMs = performance.now() - tw0;
      const eAuto = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-7, densityTol: 1e-6, maxIter: 60, useDF: auto.df }).energy;
      const eWASM = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-7, densityTol: 1e-6, maxIter: 60, useDF: wasm }).energy;
      log(`benzene n=${shells.length}: path=${auto.path}, DF build GPU-auto ${(gpuMs / 1000).toFixed(2)}s vs WASM ${(wasmMs / 1000).toFixed(2)}s (${(wasmMs / gpuMs).toFixed(2)}x)`);
      log(`benzene: auto HF ${eAuto.toFixed(8)}, wasm HF ${eWASM.toFixed(8)}, |ΔE|=${Math.abs(eAuto - eWASM).toExponential(2)}`);
      return { path: auto.path, dE: Math.abs(eAuto - eWASM), gpuMs, wasmMs };
    });

    console.log(`\n[auto benzene] ${JSON.stringify(r)}\n`);
    expect(r.path).toBe("gpu");          // d-regime, n=120 → GPU
    expect(r.dE).toBeLessThan(1e-3);     // same HF energy as WASM
  });

  test("H2O STO-3G (s/p) picks WASM", async ({ page }) => {
    test.setTimeout(60 * 1000);
    page.on("console", (m) => { if (m.text().startsWith("[auto]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    const r = await page.evaluate(async () => {
      const [{ moleculeToShellsNuclei }, { generateAutoAux }, { buildDFAuto }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);
      const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
      const { shells } = moleculeToShellsNuclei([{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }] as never, "sto-3g");
      const aux = generateAutoAux(shells, 0);
      const auto = await buildDFAuto(shells as never, aux as never);
      return { path: auto.path };
    });
    console.log(`\n[auto h2o-sto3g] ${JSON.stringify(r)}\n`);
    expect(r.path).toBe("wasm");         // s/p regime → WASM
  });
});
