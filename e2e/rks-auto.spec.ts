import { test, expect } from "@playwright/test";

// runRKSAuto — size-gated exact/DF Kohn-Sham DFT. DFT is "~90% of real chemistry"
// and a pure functional needs only the cheap DF Coulomb J (no exchange K), so the
// size-gated DF machinery extends to DFT naturally. Validates that DFT-via-DF
// agrees with exact-ERI DFT to chemical accuracy — for a PURE functional (LDA,
// J-only) and a HYBRID (B3LYP5, which also exercises the DF K path) — while the
// DF path skips the O(n⁴) ERI and provenance names what ran.

test.describe("runRKSAuto — DFT with size-gated exact/DF", () => {
  for (const fn of ["lda-svwn", "b3lyp5"] as const) {
    test(`H2O cc-pVDZ ${fn} — exact vs f64-DF vs hybrid-GPU-DF agree`, async ({ page }) => {
      test.setTimeout(120 * 1000);
      page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
      page.on("console", (m) => { if (m.text().startsWith("[rks]")) console.log(m.text()); });
      await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

      const r = await page.evaluate(async (functional: string) => {
        const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[rks] ${s}`); };
        const [{ moleculeToShellsNuclei }, { runRKSAuto }] = await Promise.all([
          import("/src/chemistry/atoms.ts" as string),
          import("/src/chemistry/rhf-auto.ts" as string),
        ]);
        const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
        const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(
          [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }] as never, "cc-pvdz");
        const symbols = ["O", "H", "H"];
        const o = { functional: functional as never };
        const exact = await runRKSAuto(shells as never, nuclei as never, nElectrons, symbols as never, { ...o, force: "exact" });
        const dfW = await runRKSAuto(shells as never, nuclei as never, nElectrons, symbols as never, { ...o, force: "df" });
        const dfG = await runRKSAuto(shells as never, nuclei as never, nElectrons, symbols as never, { ...o, force: "df", fast: true });

        log(`H2O ${functional}: exact ${exact.rks.energy.toFixed(8)} [${exact.provenance.method}]`);
        log(`  f64-DF ${dfW.rks.energy.toFixed(8)} [${dfW.provenance.engine}] |Δexact|=${Math.abs(dfW.rks.energy - exact.rks.energy).toExponential(2)}`);
        log(`  hybrid ${dfG.rks.energy.toFixed(8)} [${dfG.provenance.engine}] |Δexact|=${Math.abs(dfG.rks.energy - exact.rks.energy).toExponential(2)}`);
        return {
          exactM: exact.provenance.method, dfWProv: dfW.provenance, dfGProv: dfG.provenance,
          dW: Math.abs(dfW.rks.energy - exact.rks.energy),
          dG: Math.abs(dfG.rks.energy - exact.rks.energy),
          eriExact: exact.integrals.eri_AO.length, eriDF: dfW.integrals.eri_AO.length,
          conv: exact.rks.converged && dfW.rks.converged && dfG.rks.converged,
        };
      }, fn);

      console.log(`\n[rks-auto ${fn}] ${JSON.stringify(r)}\n`);
      const CHEM = 1.594e-3;
      expect(r.exactM).toBe("exact-eri");
      expect(r.dfWProv).toMatchObject({ method: "density-fitting", engine: "wasm" });
      expect(r.dfGProv).toMatchObject({ method: "density-fitting", engine: "gpu+wasm" });
      expect(r.eriExact).toBeGreaterThan(0);
      expect(r.eriDF).toBe(0);            // DF DFT skips the 4-index ERI
      expect(r.conv).toBe(true);
      expect(r.dW).toBeLessThan(CHEM);    // f64 DF DFT is chemistry-grade
      expect(r.dG).toBeLessThan(CHEM);    // hybrid GPU DF DFT too
    });
  }
});
