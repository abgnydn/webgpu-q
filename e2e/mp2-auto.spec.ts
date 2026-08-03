import { test, expect } from "@playwright/test";

// runMP2Auto — correlation energy on top of the size-gated HF, reusing the DF
// B-tensor. DF-MP2 is the standard way to get MP2 on large molecules: the SAME
// B-tensor builds the HF reference AND the (ia|jb) integrals, so the O(n⁴) ERI is
// never materialized. Validates that DF-MP2 (f64 and hybrid GPU) agrees with the
// exact 4-index MP2 to chemical accuracy on H2O, and that the DF path skips the ERI.

test.describe("runMP2Auto — RHF-MP2 with size-gated exact/DF", () => {
  test("H2O cc-pVDZ — exact vs f64-DF vs hybrid-GPU MP2 correlation agree", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[mp2]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const log = (s: string): void => { console.log(`[mp2] ${s}`); };
      const [{ moleculeToShellsNuclei }, { runMP2Auto }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/rhf-auto.ts" as string),
      ]);
      const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(
        [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }] as never, "cc-pvdz");

      const exact = await runMP2Auto(shells as never, nuclei as never, nElectrons, { force: "exact" });
      const dfW = await runMP2Auto(shells as never, nuclei as never, nElectrons, { force: "df" });
      const dfG = await runMP2Auto(shells as never, nuclei as never, nElectrons, { force: "df", fast: true });

      log(`H2O: exact E_corr=${exact.correlationEnergy.toFixed(8)} (tot ${exact.totalEnergy.toFixed(8)}) [${exact.provenance.method}/${exact.provenance.engine}]`);
      log(`  f64-DF E_corr=${dfW.correlationEnergy.toFixed(8)} |Δ|=${Math.abs(dfW.correlationEnergy - exact.correlationEnergy).toExponential(2)} [${dfW.provenance.engine}]`);
      log(`  hybrid E_corr=${dfG.correlationEnergy.toFixed(8)} |Δ|=${Math.abs(dfG.correlationEnergy - exact.correlationEnergy).toExponential(2)} [${dfG.provenance.engine}]`);
      return {
        exactProv: exact.provenance, dfWProv: dfW.provenance, dfGProv: dfG.provenance,
        eCorrExact: exact.correlationEnergy,
        dW: Math.abs(dfW.correlationEnergy - exact.correlationEnergy),
        dG: Math.abs(dfG.correlationEnergy - exact.correlationEnergy),
        eriExact: exact.integrals.eri_AO.length, eriDF: dfW.integrals.eri_AO.length,
      };
    });
console.log(`\n[mp2-auto] ${JSON.stringify(r)}\n`);
    const CHEM = 1.594e-3;
    expect(r.exactProv.method).toBe("exact-eri");
    expect(r.dfWProv).toMatchObject({ method: "density-fitting", engine: "wasm" });
    expect(r.dfGProv).toMatchObject({ method: "density-fitting", engine: "gpu+wasm" });
    expect(r.eriExact).toBeGreaterThan(0);
    expect(r.eriDF).toBe(0);                       // DF-MP2 never builds the 4-index ERI
    // H2O cc-pVDZ MP2 correlation ≈ -0.20 Ha; DF reproduces it to chemical accuracy.
    expect(r.eCorrExact).toBeLessThan(-0.15);
    expect(r.eCorrExact).toBeGreaterThan(-0.25);
    expect(r.dW).toBeLessThan(CHEM);               // f64 DF-MP2 chemistry-grade
    expect(r.dG).toBeLessThan(CHEM);               // hybrid GPU DF-MP2 too
  });

  test("OH radical cc-pVDZ — DF-UMP2 (open-shell) agrees with exact UMP2", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[ump2]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const log = (s: string): void => { console.log(`[ump2] ${s}`); };
      const [{ moleculeToShellsNuclei }, { runUMP2Auto }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/rhf-auto.ts" as string),
      ]);
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(
        [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [0, 0, 0.97] }] as never, "cc-pvdz");
      const nAlpha = (nElectrons + 1) / 2, nBeta = (nElectrons - 1) / 2;

      const exact = await runUMP2Auto(shells as never, nuclei as never, nAlpha, nBeta, { force: "exact" });
      const dfW = await runUMP2Auto(shells as never, nuclei as never, nAlpha, nBeta, { force: "df" });
      const dfG = await runUMP2Auto(shells as never, nuclei as never, nAlpha, nBeta, { force: "df", fast: true });

      log(`OH•: exact E_corr=${exact.correlationEnergy.toFixed(8)} (tot ${exact.totalEnergy.toFixed(8)}) [${exact.provenance.method}]`);
      log(`  f64-DF E_corr=${dfW.correlationEnergy.toFixed(8)} |Δ|=${Math.abs(dfW.correlationEnergy - exact.correlationEnergy).toExponential(2)} [${dfW.provenance.engine}]`);
      log(`  hybrid E_corr=${dfG.correlationEnergy.toFixed(8)} |Δ|=${Math.abs(dfG.correlationEnergy - exact.correlationEnergy).toExponential(2)} [${dfG.provenance.engine}]`);
      return {
        exactM: exact.provenance.method, dfWEng: dfW.provenance.engine, dfGEng: dfG.provenance.engine,
        eCorrExact: exact.correlationEnergy,
        dW: Math.abs(dfW.correlationEnergy - exact.correlationEnergy),
        dG: Math.abs(dfG.correlationEnergy - exact.correlationEnergy),
        eriDF: dfW.integrals.eri_AO.length,
      };
    });
console.log(`\n[ump2-auto] ${JSON.stringify(r)}\n`);
    const CHEM = 1.594e-3;
    expect(r.exactM).toBe("exact-eri");
    expect(r.dfWEng).toBe("wasm");
    expect(r.dfGEng).toBe("gpu+wasm");
    expect(r.eriDF).toBe(0);                       // DF-UMP2 never builds the ERI
    expect(r.eCorrExact).toBeLessThan(-0.15);      // OH UMP2 corr ≈ -0.2 Ha
    expect(r.eCorrExact).toBeGreaterThan(-0.30);
    expect(r.dW).toBeLessThan(CHEM);               // DF-UMP2 reproduces exact UMP2 to chemical accuracy
    expect(r.dG).toBeLessThan(CHEM);
  });
});
