import { test, expect } from "@playwright/test";

// DF-vs-exact accuracy ladder — addresses the "chemistry-grade validated on ONE
// small molecule (H₂O)" critique. Checks that DF (f64 WASM) and the hybrid GPU
// build stay within chemical accuracy (1.594 mHa) of the exact 4-index ERI as the
// system grows, across closed-shell molecules with d-functions where the exact
// ERI is still feasible in a tab: H₂O (n=25) → CH₂O (n=38) → C₂H₄ (n=48).
//
// HONEST SCOPE: this is a small SIZE ladder, not a statistical benchmark suite
// (GMTKN55/W4-11). It shows DF accuracy holds and the trend across size + d-count,
// not a full error distribution. Naphthalene-scale precision still needs an
// external (PySCF) reference — the exact ERI is uncomputable in a tab there.

const MOLS: { name: string; basis: string; atoms: { symbol: string; pos: number[] }[] }[] = [
  {
    name: "H2O", basis: "cc-pvdz",
    atoms: (() => { const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
      return [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }]; })(),
  },
  {
    name: "CH2O", basis: "cc-pvdz",
    atoms: [
      { symbol: "C", pos: [0, 0, 0] }, { symbol: "O", pos: [0, 0, 1.208] },
      { symbol: "H", pos: [0.943, 0, -0.588] }, { symbol: "H", pos: [-0.943, 0, -0.588] },
    ],
  },
  {
    name: "C2H4", basis: "cc-pvdz",
    atoms: [
      { symbol: "C", pos: [0, 0, 0.667] }, { symbol: "C", pos: [0, 0, -0.667] },
      { symbol: "H", pos: [0.923, 0, 1.230] }, { symbol: "H", pos: [-0.923, 0, 1.230] },
      { symbol: "H", pos: [0.923, 0, -1.230] }, { symbol: "H", pos: [-0.923, 0, -1.230] },
    ],
  },
];

test.describe("DF accuracy ladder — DF stays chemistry-grade as size grows", () => {
  for (const m of MOLS) {
    test(`${m.name} cc-pVDZ — f64-DF and hybrid-GPU within chemical accuracy of exact`, async ({ page }) => {
      test.setTimeout(120 * 1000);
      page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
      page.on("console", (mm) => { if (mm.text().startsWith("[lad]")) console.log(mm.text()); });
      await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

      const r = await page.evaluate(async (mol) => {
        const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[lad] ${s}`); };
        const [{ moleculeToShellsNuclei }, { runRHFAuto }] = await Promise.all([
          import("/src/chemistry/atoms.ts" as string),
          import("/src/chemistry/rhf-auto.ts" as string),
        ]);
        const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(mol.atoms as never, mol.basis as never);
        const exact = await runRHFAuto(shells as never, nuclei as never, nElectrons, { force: "exact" });
        const dfW = await runRHFAuto(shells as never, nuclei as never, nElectrons, { force: "df" });
        const dfG = await runRHFAuto(shells as never, nuclei as never, nElectrons, { force: "df", fast: true });
        const dW = Math.abs(dfW.hf.energy - exact.hf.energy), dG = Math.abs(dfG.hf.energy - exact.hf.energy);
        log(`${mol.name} n=${shells.length}: exact ${exact.hf.energy.toFixed(8)} | f64-DF Δ=${dW.toExponential(2)} | hybrid Δ=${dG.toExponential(2)} [${dfG.provenance.engine}]`);
        return { n: shells.length, dW, dG, gEng: dfG.provenance.engine,
          eriExact: exact.integrals.eri_AO.length, eriDF: dfW.integrals.eri_AO.length };
      }, m);

      console.log(`\n[df-ladder ${m.name}] ${JSON.stringify(r)}\n`);
      const CHEM = 1.594e-3;
      expect(r.eriExact).toBeGreaterThan(0);   // exact really built the 4-index ERI
      expect(r.eriDF).toBe(0);                  // DF really skipped it
      expect(r.dW).toBeLessThan(CHEM);          // f64 DF chemistry-grade at this size
      expect(r.dG).toBeLessThan(CHEM);          // hybrid GPU DF chemistry-grade at this size
      expect(r.gEng).toBe("gpu+wasm");          // hybrid engaged (all have d-functions)
    });
  }
});
