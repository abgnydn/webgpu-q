import { test, expect } from "@playwright/test";

// runUHFAuto — open-shell UHF with size-gated exact/DF. Radicals/doublets/O2 are a
// big slice of real chemistry and were previously off-limits to the DF/GPU path
// (runRHFAuto throws on odd electrons). Validates that UHF-via-DF agrees with
// exact-ERI UHF to chemical accuracy on the OH radical (doublet), for both the
// f64 WASM DF and the hybrid GPU/WASM DF, while the DF path skips the 4-index ERI.

test.describe("runUHFAuto — open-shell UHF with size-gated exact/DF", () => {
  test("OH radical cc-pVDZ (doublet) — exact vs f64-DF vs hybrid-GPU agree", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[uhf]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[uhf] ${s}`); };
      const [{ moleculeToShellsNuclei }, { runUHFAuto }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/rhf-auto.ts" as string),
      ]);
      // OH: 9 electrons → doublet (nα=5, nβ=4). O has d in cc-pVDZ.
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(
        [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [0, 0, 0.97] }] as never, "cc-pvdz");
      const nAlpha = (nElectrons + 1) / 2, nBeta = (nElectrons - 1) / 2;

      const exact = await runUHFAuto(shells as never, nuclei as never, nAlpha, nBeta, { force: "exact" });
      const dfW = await runUHFAuto(shells as never, nuclei as never, nAlpha, nBeta, { force: "df" });
      const dfG = await runUHFAuto(shells as never, nuclei as never, nAlpha, nBeta, { force: "df", fast: true });

      log(`OH n=${shells.length}, ${nElectrons}e⁻ (${nAlpha}α/${nBeta}β): exact ${exact.uhf.energy.toFixed(8)} [${exact.provenance.method}], ⟨S²⟩=${exact.uhf.s2.toFixed(4)} (exact ${exact.uhf.s2Exact})`);
      log(`  f64-DF ${dfW.uhf.energy.toFixed(8)} [${dfW.provenance.engine}] |Δexact|=${Math.abs(dfW.uhf.energy - exact.uhf.energy).toExponential(2)}`);
      log(`  hybrid ${dfG.uhf.energy.toFixed(8)} [${dfG.provenance.engine}] |Δexact|=${Math.abs(dfG.uhf.energy - exact.uhf.energy).toExponential(2)}`);
      return {
        exactProv: exact.provenance, dfWProv: dfW.provenance, dfGProv: dfG.provenance,
        dW: Math.abs(dfW.uhf.energy - exact.uhf.energy),
        dG: Math.abs(dfG.uhf.energy - exact.uhf.energy),
        eriExact: exact.integrals.eri_AO.length, eriDF: dfW.integrals.eri_AO.length,
        conv: exact.uhf.converged && dfW.uhf.converged && dfG.uhf.converged,
        s2: exact.uhf.s2, energy: exact.uhf.energy,
      };
    });

    console.log(`\n[uhf-auto] ${JSON.stringify(r)}\n`);
    const CHEM = 1.594e-3;
    expect(r.exactProv.method).toBe("exact-eri");
    expect(r.dfWProv).toMatchObject({ method: "density-fitting", engine: "wasm", precision: "f64" });
    expect(r.dfGProv).toMatchObject({ method: "density-fitting", engine: "gpu+wasm" });
    expect(r.eriExact).toBeGreaterThan(0);
    expect(r.eriDF).toBe(0);                 // DF UHF skips the 4-index ERI
    expect(r.conv).toBe(true);
    expect(r.dW).toBeLessThan(CHEM);         // f64 DF UHF chemistry-grade
    expect(r.dG).toBeLessThan(CHEM);         // hybrid GPU DF UHF too
    expect(r.s2).toBeGreaterThan(0.74);      // doublet S(S+1)=0.75; small contamination ok
    expect(r.s2).toBeLessThan(0.80);
    expect(r.energy).toBeLessThan(-75);      // OH UHF/cc-pVDZ ≈ -75.4 Ha
    expect(r.energy).toBeGreaterThan(-76);
  });
});
