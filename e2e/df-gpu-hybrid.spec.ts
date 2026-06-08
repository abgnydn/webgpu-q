import { test, expect } from "@playwright/test";

// Hybrid 3-index build (buildV3idxHybrid): GPU does the s/p/d-aux columns (f32),
// WASM does the f-aux columns (f64). The question this answers: does the f32
// low-aux block degrade the chemistry-grade accuracy that the extraL=1 aux basis
// buys? i.e. can the GPU carry the bulk of the integral work AND stay chemically
// accurate, instead of being stuck at the d-only level-0 (~30 mHa) screening tier?

test.describe("Hybrid GPU/WASM 3-index DF build — accuracy", () => {
  test("H2O cc-pVDZ extraL=1 — hybrid V (f32 low + f64 f-aux) stays chemistry-grade", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[hyb]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[hyb] ${s}`); };
      const [
        { moleculeToShellsNuclei }, { computeMolecularIntegrals }, { runRHFSCF },
        { generateAutoAux, buildV3idxCPU, buildBFromV, buildAuxBasisDFStreaming }, { buildV3idxHybrid },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
      ]);
      const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(
        [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }] as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const scf = { useDIIS: true, energyTol: 1e-9, densityTol: 1e-7, maxIter: 200 };
      const eExact = runRHFSCF(integrals, nElectrons, scf).energy;

      const aux = generateAutoAux(shells, 1); // extraL=1 → has f-aux
      const nF = aux.filter((s: { angular: number[] }) => (s.angular[0]! + s.angular[1]! + s.angular[2]!) > 2).length;

      // Pure f64 DF (the chemistry-grade baseline).
      const dfW = await buildAuxBasisDFStreaming(shells as never, aux as never);
      const eW = runRHFSCF(integrals, nElectrons, { ...scf, useDF: dfW }).energy;

      // Hybrid: GPU(f32) low-aux columns + WASM(f64) f-aux columns.
      const { V, auxOrdered } = await buildV3idxHybrid(shells as never, aux as never);
      // Cross-check the merged V against a full f64 WASM V (same column order).
      const Vref = await buildV3idxCPU(shells as never, auxOrdered as never);
      let vMaxRel = 0, vMag = 0;
      for (let i = 0; i < Vref.length; i++) { const c = Vref[i]!, g = V[i]!; if (Math.abs(c) > vMag) vMag = Math.abs(c); if (Math.abs(c) > 1e-6) vMaxRel = Math.max(vMaxRel, Math.abs(g - c) / Math.abs(c)); }
      const dfH = await buildBFromV(shells as never, auxOrdered as never, V);
      const eH = runRHFSCF(integrals, nElectrons, { ...scf, useDF: dfH }).energy;

      log(`H2O n=${shells.length}: ${nF} f-aux of ${aux.length}. exact ${eExact.toFixed(8)}`);
      log(`  pure-f64 DF ${eW.toFixed(8)} |Δexact|=${Math.abs(eW - eExact).toExponential(2)}`);
      log(`  hybrid   DF ${eH.toFixed(8)} |Δexact|=${Math.abs(eH - eExact).toExponential(2)}, V maxRel(f32 low)=${vMaxRel.toExponential(2)}`);
      return { nF, dWexact: Math.abs(eW - eExact), dHexact: Math.abs(eH - eExact), dHW: Math.abs(eH - eW), vMaxRel };
    });

    console.log(`\n[df-gpu-hybrid] ${JSON.stringify(r)}\n`);
    expect(r.nF).toBeGreaterThan(0);                 // extraL=1 really produced f-aux
    expect(r.dWexact).toBeLessThan(1.594e-3);        // pure f64 DF is chemistry-grade
    // The headline: hybrid (GPU f32 bulk + f64 f-aux) is ALSO chemistry-grade —
    // the f32 low-aux block doesn't blow the chemical-accuracy budget.
    expect(r.dHexact).toBeLessThan(1.594e-3);
    expect(r.dHW).toBeLessThan(1.0e-3);              // hybrid tracks the pure-f64 DF closely
  });
});
