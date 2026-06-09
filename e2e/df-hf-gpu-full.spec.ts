import { test, expect } from "@playwright/test";

// Fully-GPU DF-HF: GPU builds the DF tensor (buildDFAuto → GPU integrals) AND
// does J/K on the GPU every SCF iteration (makeGpuDFJK), so the whole Hartree–
// Fock loop runs on the GPU from a URL — no full 4-index ERI, ever. Validates the
// converged energy against the WASM DF-HF reference.

test.describe("Fully-GPU DF-HF SCF", () => {
  test("benzene cc-pVDZ — GPU integrals + GPU JK loop matches WASM DF-HF", async ({ page }) => {
    test.setTimeout(180 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[fullgpu]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[fullgpu] ${s}`); };
      const [
        { moleculeToShellsNuclei }, { computeMolecularIntegrals }, { runRHFSCF, runRHFSCFAsync },
        { generateAutoAux, buildAuxBasisDFStreaming }, { buildDFAuto }, { makeGpuDFJK },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df-gpu.ts" as string),
        import("/src/chemistry/jk-df-gpu.ts" as string),
      ]);
      const rCC = 1.395, rCH = 1.087; const atoms: { symbol: string; pos: number[] }[] = [];
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; atoms.push({ symbol: "C", pos: [rCC * Math.cos(t), rCC * Math.sin(t), 0] }); }
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; const r2 = rCC + rCH; atoms.push({ symbol: "H", pos: [r2 * Math.cos(t), r2 * Math.sin(t), 0] }); }
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const aux = generateAutoAux(shells, 0);

      // WASM reference: same DF tensor, WASM JK.
      const wasmDF = await buildAuxBasisDFStreaming(shells as never, aux as never);
      const eWASM = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-7, densityTol: 1e-6, maxIter: 60, useDF: wasmDF }).energy;

      // Fully GPU: GPU integrals for B (buildDFAuto), GPU JK every iteration.
      const auto = await buildDFAuto(shells as never, aux as never);
      const gpuJK = await makeGpuDFJK(auto.df);
      let iters = 0;
      const customJKBuilder = async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
        iters++;
        const { J, K } = await gpuJK.jk(D);
        const Jf = new Float64Array(J.length), Kf = new Float64Array(K.length);
        for (let i = 0; i < J.length; i++) { Jf[i] = J[i]!; Kf[i] = K[i]!; }
        return { J: Jf, K: Kf };
      };
      // f32 JK has a ~6e-4 noise floor, so the SCF converges to ~f32 precision,
      // not 1e-7 — use f32-appropriate tolerances (still far inside chemical acc).
      const t0 = performance.now();
      const sw = await runRHFSCFAsync(integrals, nElectrons, { useDIIS: true, energyTol: 1e-5, densityTol: 1e-4, maxIter: 60, parallel: 1, customJKBuilder });
      const ms = performance.now() - t0;
      gpuJK.dispose();

      log(`benzene: DF path=${auto.path}, fully-GPU HF ${sw.energy.toFixed(8)} (${iters} iters, ${(ms / 1000).toFixed(1)}s), WASM HF ${eWASM.toFixed(8)}, |ΔE|=${Math.abs(sw.energy - eWASM).toExponential(2)}`);
      return { path: auto.path, converged: sw.converged, dE: Math.abs(sw.energy - eWASM) };
    });

    console.log(`\n[fully-gpu-hf] ${JSON.stringify(r)}\n`);
    expect(r.converged).toBe(true);     // converges at f32-appropriate tolerances
    expect(r.path).toBe("gpu");         // d-regime → GPU integrals
    // The meaningful bar: fully-GPU HF is correct to chemical accuracy. (f32 JK +
    // looser-than-f64 SCF convergence land it at ~4e-4 Ha, ≪ 1.594 mHa.)
    expect(r.dE).toBeLessThan(1.594e-3);
  });
});
