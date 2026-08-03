import { test, expect } from "@playwright/test";

// GPU DF-JK (buildJK_DF_GPU) vs the WASM buildJK_DF: the per-SCF-iteration Fock
// contraction from the DF B-tensor, on the GPU. Validates J and K (f32) against
// the f64 reference on a converged RHF density.

test.describe("GPU DF-JK (f32) vs WASM buildJK_DF (f64)", () => {
  test("benzene cc-pVDZ — J and K match the reference", async ({ page }) => {
    test.setTimeout(120 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[jkdf]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const log = (s: string): void => { console.log(`[jkdf] ${s}`); };
      const [
        { moleculeToShellsNuclei }, { computeMolecularIntegrals }, { runRHFSCF },
        { generateAutoAux, buildAuxBasisDFStreaming }, { buildJK_DF }, { buildJK_DF_GPU },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
        import("/src/chemistry/jk-df-gpu.ts" as string),
      ]);
      const rCC = 1.395, rCH = 1.087; const atoms: { symbol: string; pos: number[] }[] = [];
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; atoms.push({ symbol: "C", pos: [rCC * Math.cos(t), rCC * Math.sin(t), 0] }); }
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; const r2 = rCC + rCH; atoms.push({ symbol: "H", pos: [r2 * Math.cos(t), r2 * Math.sin(t), 0] }); }
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const aux = generateAutoAux(shells, 0);
      const df = await buildAuxBasisDFStreaming(shells as never, aux as never);
      const D = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-7, densityTol: 1e-6, maxIter: 60, useDF: df }).D;

      const cpu = buildJK_DF(df, D);
      const gpu = await buildJK_DF_GPU(df, D);
      const rel = (a: Float64Array, b: Float32Array): { mAbs: number; mRel: number; mMag: number } => {
        let mAbs = 0, mRel = 0, mMag = 0;
        for (let i = 0; i < a.length; i++) { const c = a[i]!, g = b[i]!; const ab = Math.abs(g - c); if (ab > mAbs) mAbs = ab; if (Math.abs(c) > 1e-6 && ab / Math.abs(c) > mRel) mRel = ab / Math.abs(c); if (Math.abs(c) > mMag) mMag = Math.abs(c); }
        return { mAbs, mRel, mMag };
      };
      const jr = rel(cpu.J, gpu.J); const kr = rel(cpu.K, gpu.K);
      log(`benzene n=${shells.length} n_aux=${df.nAux}: J max rel ${jr.mRel.toExponential(2)} (|J|max ${jr.mMag.toFixed(2)}), K max rel ${kr.mRel.toExponential(2)} (|K|max ${kr.mMag.toFixed(2)})`);
      return { jRel: jr.mRel, kRel: kr.mRel };
    });
console.log(`\n[jk-df-gpu] ${JSON.stringify(r)}\n`);
    expect(r.jRel).toBeLessThan(1e-3);
    expect(r.kRel).toBeLessThan(1e-3);
  });
});
