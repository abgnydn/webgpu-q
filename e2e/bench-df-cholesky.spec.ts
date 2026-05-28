import { test, expect } from "@playwright/test";

// Pivoted-Cholesky variant of aux-DF. Naturally handles rank-deficient
// auto-aux on multi-heavy-atom systems without eigendecomp regularization
// dropping critical modes.

test.describe("aux-DF via pivoted Cholesky", () => {
  test("H₂O cc-pVDZ — Cholesky vs eigendecomp HF energy", async ({ page }) => {
    test.setTimeout(2 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDF, buildAuxBasisDFCholesky, generateAutoAux },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
      ]);

      const half = (104.52 / 2) * Math.PI / 180;
      const xH = 0.9572 * Math.sin(half) / 0.529177;
      const zH = 0.9572 * Math.cos(half) / 0.529177;
      const atoms = [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ xH, 0, zH] },
        { symbol: "H", pos: [-xH, 0, zH] },
      ] as const;
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const eDirect = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
      }).energy;

      const results: Array<{
        extraL: number; nAuxIn: number;
        eigE: number; eigCnv: boolean; eigR: number;
        choE: number; choCnv: boolean; choR: number;
      }> = [];

      for (const extraL of [0, 1, 2]) {
        const auxShells = generateAutoAux(shells, extraL);

        const dfEig = await buildAuxBasisDF(shells, auxShells, 1e-10);
        const hfEig = runRHFSCF(integrals, nElectrons, {
          useDIIS: true, energyTol: 1e-9, maxIter: 100, useDF: dfEig,
        });

        const dfCho = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);
        const hfCho = runRHFSCF(integrals, nElectrons, {
          useDIIS: true, energyTol: 1e-9, maxIter: 100, useDF: dfCho,
        });

        results.push({
          extraL, nAuxIn: auxShells.length,
          eigE: hfEig.energy, eigCnv: hfEig.converged, eigR: dfEig.nAux,
          choE: hfCho.energy, choCnv: hfCho.converged, choR: dfCho.nAux,
        });
      }
      return { eDirect, results };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`H₂O cc-pVDZ — pivoted Cholesky vs eigendecomp`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Direct ERI HF:  E = ${r.eDirect.toFixed(8)} Ha`);
    console.log();
    for (const x of r.results) {
      const eigErr = x.eigE - r.eDirect;
      const choErr = x.choE - r.eDirect;
      console.log(`extraL=${x.extraL}, n_aux_in=${String(x.nAuxIn).padStart(3)}:`);
      console.log(`  eigendecomp: rank=${String(x.eigR).padStart(3)}  E=${x.eigE.toFixed(6).padStart(13)}  cnv=${x.eigCnv}  err=${eigErr.toExponential(2)}`);
      console.log(`  Cholesky:    rank=${String(x.choR).padStart(3)}  E=${x.choE.toFixed(6).padStart(13)}  cnv=${x.choCnv}  err=${choErr.toExponential(2)}`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.results.every((x) => Number.isFinite(x.choE))).toBe(true);
  });

  test("benzene cc-pVDZ — Cholesky aux-DF should fix the SCF blowup", async ({ page }) => {
    test.setTimeout(15 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDFCholesky, generateAutoAux },
        { preloadWasmJK },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
      ]);
      await preloadWasmJK();  // unlock WASM build_jk_df path

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

      const tDF = performance.now();
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);
      const dfMs = performance.now() - tDF;

      const tHF = performance.now();
      const hf = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-7, densityTol: 1e-6, maxIter: 100,
        useDF: df,
      });
      const hfMs = performance.now() - tHF;

      return {
        n: integrals.n, nAuxIn: auxShells.length, nAuxKept: df.nAux,
        dfMs, hfMs,
        energy: hf.energy, iter: hf.iter, converged: hf.converged,
      };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Benzene cc-pVDZ aux-DF via Cholesky (n=${r.n}, n_aux=${r.nAuxIn} → rank ${r.nAuxKept})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`B-tensor build (Cholesky path):  ${(r.dfMs / 1000).toFixed(2)} s`);
    console.log(`HF SCF (${r.iter} iters):         ${(r.hfMs / 1000).toFixed(2)} s   converged=${r.converged}`);
    console.log(`Energy:                          ${r.energy.toFixed(8)} Ha`);
    const directE = -230.72273482;
    console.log(`Error vs WASM-direct reference:  ${(r.energy - directE).toExponential(2)} Ha`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(Number.isFinite(r.energy)).toBe(true);
  });
});
