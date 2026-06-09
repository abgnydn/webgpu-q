import { test, expect } from "@playwright/test";

// runRHFAuto — the size-gated "do the right thing" RHF entry point. Two probes:
//   1. agreement on a SMALL d-containing molecule (H2O cc-pVDZ, n=25, O has d) —
//      exact ERI is cheap there, so it's the gold standard the two DF paths
//      (f64 WASM, f32 GPU) must land on, while provenance names what ran.
//   2. the LARGE-system auto path (benzene cc-pVDZ, n=114 > 80) — picks DF
//      automatically and SKIPS the O(n⁴) ERI (the whole point), so it's cheap.
// We deliberately never force exact ERI on benzene: that 1.3 GB build is exactly
// what the size gate exists to avoid.

test.describe("runRHFAuto — size-gated exact/DF with provenance", () => {
  test("H2O cc-pVDZ — exact vs f64-DF vs hybrid-GPU-DF agree, provenance honest", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[auto]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[auto] ${s}`); };
      const [{ moleculeToShellsNuclei }, { runRHFAuto }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/rhf-auto.ts" as string),
      ]);
      const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(
        [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }] as never, "cc-pvdz");
      const n = shells.length;

      const exact = await runRHFAuto(shells as never, nuclei as never, nElectrons, { force: "exact" });
      const dfW = await runRHFAuto(shells as never, nuclei as never, nElectrons, { force: "df" });
      const dfG = await runRHFAuto(shells as never, nuclei as never, nElectrons, { force: "df", fast: true });

      log(`H2O n=${n}: exact ${exact.hf.energy.toFixed(8)} [${exact.provenance.method}/${exact.provenance.engine}/${exact.provenance.precision}]`);
      log(`  f64-DF  ${dfW.hf.energy.toFixed(8)} [${dfW.provenance.engine}/${dfW.provenance.precision}, nAux=${dfW.provenance.nAux}] |Δexact|=${Math.abs(dfW.hf.energy - exact.hf.energy).toExponential(2)}`);
      log(`  hybrid  ${dfG.hf.energy.toFixed(8)} [${dfG.provenance.engine}/${dfG.provenance.precision}, nAux=${dfG.provenance.nAux}] |Δexact|=${Math.abs(dfG.hf.energy - exact.hf.energy).toExponential(2)}`);

      return {
        exactProv: exact.provenance, dfWProv: dfW.provenance, dfGProv: dfG.provenance,
        dWexact: Math.abs(dfW.hf.energy - exact.hf.energy),
        dGexact: Math.abs(dfG.hf.energy - exact.hf.energy),
        dHW: Math.abs(dfG.hf.energy - dfW.hf.energy),
        eriLenExact: exact.integrals.eri_AO.length, eriLenDF: dfW.integrals.eri_AO.length,
      };
    });

    console.log(`\n[rhf-auto] ${JSON.stringify(r)}\n`);

    // Provenance is honest about what ran (the core contract of runRHFAuto).
    expect(r.exactProv.method).toBe("exact-eri");
    expect(r.exactProv.precision).toBe("f64");
    expect(r.dfWProv).toMatchObject({ method: "density-fitting", engine: "wasm", precision: "f64" });
    expect(r.dfGProv).toMatchObject({ method: "density-fitting", engine: "gpu+wasm", precision: "mixed" });
    expect(r.dfWProv.nAux).toBeGreaterThan(0);

    // DF skips the O(n⁴) ERI; exact builds it. (The structural payoff of DF.)
    expect(r.eriLenExact).toBeGreaterThan(0);
    expect(r.eriLenDF).toBe(0);

    // BOTH DF paths are now chemistry-grade vs exact:
    //  - f64 DF (extraL=1, with f-aux): ~0.19 mHa.
    //  - hybrid GPU/WASM DF (GPU f32 s/p/d-aux + f64 f-aux + f64 JK): ~0.19 mHa
    //    too — the f32 low-aux block costs only ~8 µHa, so GPU acceleration no
    //    longer means giving up chemical accuracy (it used to be ~30 mHa).
    expect(r.dWexact).toBeLessThan(1.594e-3); // f64 DF chemistry-grade (measured 0.19 mHa)
    expect(r.dGexact).toBeLessThan(1.594e-3); // hybrid GPU DF ALSO chemistry-grade
    expect(r.dHW).toBeLessThan(1.0e-3);       // hybrid tracks the pure-f64 DF closely
  });

  test("benzene cc-pVDZ (n=114 > 80) — auto picks DF and skips the ERI", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[big]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    const r = await page.evaluate(async () => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[big] ${s}`); };
      const [{ moleculeToShellsNuclei }, { runRHFAuto }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/rhf-auto.ts" as string),
      ]);
      const rCC = 1.395, rCH = 1.087; const atoms: { symbol: string; pos: number[] }[] = [];
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; atoms.push({ symbol: "C", pos: [rCC * Math.cos(t), rCC * Math.sin(t), 0] }); }
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; const r2 = rCC + rCH; atoms.push({ symbol: "H", pos: [r2 * Math.cos(t), r2 * Math.sin(t), 0] }); }
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const auto = await runRHFAuto(shells as never, nuclei as never, nElectrons); // default → size gate
      log(`benzene n=${shells.length}: auto ${auto.hf.energy.toFixed(8)} [${auto.provenance.method}/${auto.provenance.engine}/${auto.provenance.precision}, nAux=${auto.provenance.nAux}], converged=${auto.hf.converged}, ERI len=${auto.integrals.eri_AO.length}`);
      return { n: shells.length, prov: auto.provenance, energy: auto.hf.energy, converged: auto.hf.converged, eriLen: auto.integrals.eri_AO.length };
    });
    console.log(`\n[rhf-auto big] ${JSON.stringify(r)}\n`);
    expect(r.n).toBeGreaterThan(80);
    expect(r.prov.method).toBe("density-fitting"); // large → DF automatically
    expect(r.eriLen).toBe(0);                        // O(n⁴) ERI never built
    expect(r.converged).toBe(true);
    // extraL=1 f64 DF reproduces the true benzene RHF/cc-pVDZ energy (lit.
    // ≈ -230.722 Ha; the old level-0 aux gave -230.6617, ~60 mHa too high).
    expect(Math.abs(r.energy - (-230.722))).toBeLessThan(5e-3);
  });

  test("size gating: small picks exact, large picks DF (default exactMaxN=80)", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("console", (m) => { if (m.text().startsWith("[gate]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    const r = await page.evaluate(async () => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[gate] ${s}`); };
      const [{ moleculeToShellsNuclei }, { runRHFAuto }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/rhf-auto.ts" as string),
      ]);
      // H2O cc-pVDZ: n=24 (≤80) → exact, gold standard small molecule.
      const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
      const w = moleculeToShellsNuclei([{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }] as never, "cc-pvdz");
      const small = await runRHFAuto(w.shells as never, w.nuclei as never, w.nElectrons);
      log(`H2O n=${w.shells.length}: ${small.provenance.method}/${small.provenance.engine}`);
      return { smallN: w.shells.length, smallMethod: small.provenance.method, smallConv: small.hf.converged };
    });
    console.log(`\n[rhf-auto gate] ${JSON.stringify(r)}\n`);
    expect(r.smallN).toBeLessThanOrEqual(80);
    expect(r.smallMethod).toBe("exact-eri"); // small → gold standard
    expect(r.smallConv).toBe(true);
  });
});
