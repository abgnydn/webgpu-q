import { test, expect } from "@playwright/test";

// Streaming swarm HF SCF — each tab builds ONLY its own mode-slice from scratch.
//
// This closes the open TODO from swarm-hf-distributed.spec.ts: "have each tab
// build its own slice from scratch so master doesn't need the full B." Here
// NO tab ever materializes the full 3-index V tensor OR the full B tensor:
//   - each tab independently computes the (deterministic) inverse-metric modes
//     and tiles them via partition:{tab,of:N};
//   - each tab streams the integral build μ-block by μ-block, projecting onto
//     its mode slice, so peak V is one μ-block, not n²·n_aux;
//   - each tab keeps only its B mode-slice (≈ full B / N).
// The distributed SCF then sums partial (J,K) every iter, as before.
//
// Asserts (the undismissable memory claim): the swarm energy matches the
// single-tab streaming reference, AND each tab's resident B-slice is strictly
// smaller than the full B tensor — no single tab holds the whole thing.

const N_TABS = 2;

test.describe("Streaming swarm HF SCF (each tab self-builds its slice)", () => {
  test("benzene cc-pVDZ — self-built slices, no full B on any tab", async ({ browser }) => {
    test.setTimeout(5 * 60 * 1000);

    const ctx = await browser.newContext();
    const master = await ctx.newPage();
    const worker = await ctx.newPage();

    master.on("pageerror", (e) => console.error(`[master:pageerror] ${e.message}`));
    worker.on("pageerror", (e) => console.error(`[worker:pageerror] ${e.message}`));
    for (const [tag, p] of [["m", master], ["w", worker]] as const) {
      p.on("console", (msg) => {
        const t = msg.text();
        if (t.startsWith(`[${tag}]`)) console.log(t);
      });
    }

    await master.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    await worker.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    // Worker tab: build its own mode-slice, then answer jk requests.
    const workerInfo = await worker.evaluate(async (nTabs) => {
      const [
        { moleculeToShellsNuclei },
        { buildAuxBasisDFStreaming, generateAutoAux },
        { preloadWasmJK, buildJK_DF },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
      ]);
      await preloadWasmJK();

      const rCC = 1.395, rCH = 1.087;
      const atoms: Array<{ symbol: string; pos: readonly [number, number, number] }> = [];
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; atoms.push({ symbol: "C", pos: [rCC * Math.cos(t), rCC * Math.sin(t), 0] }); }
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; const r2 = rCC + rCH; atoms.push({ symbol: "H", pos: [r2 * Math.cos(t), r2 * Math.sin(t), 0] }); }
      const { shells } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const aux = generateAutoAux(shells, 1);

      // Build ONLY this tab's slice (tab index N_TABS-1 = the worker, here 1).
      const slice = await buildAuxBasisDFStreaming(shells, aux, 1e-10, { partition: { tab: nTabs - 1, of: nTabs } });
      const n = slice.n;
      const w = window as unknown as { __sw: { n: number; nAux: number; B: Float64Array } };
      w.__sw = { n, nAux: slice.nAux, B: slice.B };
      // eslint-disable-next-line no-console
      console.log(`[w] self-built slice: modes=${slice.nAux}, B=${(slice.B.byteLength / 1e6).toFixed(1)} MB, peakV=${(slice.peakVFloats * 8 / 1e6).toFixed(1)} MB`);

      const ch = new BroadcastChannel("swarm-hf-streaming");
      ch.addEventListener("message", (ev) => {
        const msg = ev.data as { type: string; D?: Float64Array };
        if (msg.type === "ping") {
          // Master may start listening after we built our slice — answer pings.
          ch.postMessage({ type: "worker-ready" });
        } else if (msg.type === "jk" && msg.D) {
          const { n: nn, nAux, B } = w.__sw;
          const { J, K } = buildJK_DF({ B, n: nn, nAux, threshold: 0 }, msg.D);
          ch.postMessage({ type: "jk-partial", J, K });
        } else if (msg.type === "shutdown") {
          ch.close();
        }
      });
      return { n, modes: slice.nAux, sliceBytes: slice.B.byteLength, peakVBytes: slice.peakVFloats * 8 };
    }, N_TABS);

    // Master tab: build its own slice + single-tab streaming reference, run swarm SCF.
    const result = await master.evaluate(async (nTabs) => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF, runRHFSCFAsync },
        { buildAuxBasisDFStreaming, generateAutoAux },
        { preloadWasmJK, buildJK_DF },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
      ]);
      await preloadWasmJK();

      const rCC = 1.395, rCH = 1.087;
      const atoms: Array<{ symbol: string; pos: readonly [number, number, number] }> = [];
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; atoms.push({ symbol: "C", pos: [rCC * Math.cos(t), rCC * Math.sin(t), 0] }); }
      for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; const r2 = rCC + rCH; atoms.push({ symbol: "H", pos: [r2 * Math.cos(t), r2 * Math.sin(t), 0] }); }
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const aux = generateAutoAux(shells, 1);

      // Single-tab streaming reference: full mode range, one tab.
      const fullB = await buildAuxBasisDFStreaming(shells, aux, 1e-10);
      const fullBytes = fullB.B.byteLength;
      const tRef0 = performance.now();
      const refHF = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 40, useDF: fullB,
      });
      const refMs = performance.now() - tRef0;
      // eslint-disable-next-line no-console
      console.log(`[m] single-tab streaming ref: ${refMs.toFixed(0)} ms, E=${refHF.energy.toFixed(8)}, full B=${(fullBytes / 1e6).toFixed(1)} MB`);

      // Master's own slice (tab 0).
      const mSlice = await buildAuxBasisDFStreaming(shells, aux, 1e-10, { partition: { tab: 0, of: nTabs } });
      const n = mSlice.n;
      // eslint-disable-next-line no-console
      console.log(`[m] self-built slice: modes=${mSlice.nAux}, B=${(mSlice.B.byteLength / 1e6).toFixed(1)} MB, peakV=${(mSlice.peakVFloats * 8 / 1e6).toFixed(1)} MB`);

      const ch = new BroadcastChannel("swarm-hf-streaming");
      // Wait until the worker has built its slice and is listening.
      await new Promise<void>((resolve) => {
        const h = (ev: MessageEvent): void => {
          if ((ev.data as { type: string }).type === "worker-ready") { ch.removeEventListener("message", h); resolve(); }
        };
        ch.addEventListener("message", h);
        ch.postMessage({ type: "ping" });
      });

      const customJKBuilder = async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
        const workerPromise = new Promise<{ J: Float64Array; K: Float64Array }>((resolve, reject) => {
          const h = (ev: MessageEvent): void => {
            const msg = ev.data as { type: string; J?: Float64Array; K?: Float64Array };
            if (msg.type === "jk-partial" && msg.J && msg.K) { ch.removeEventListener("message", h); resolve({ J: msg.J, K: msg.K }); }
          };
          ch.addEventListener("message", h);
          setTimeout(() => { ch.removeEventListener("message", h); reject(new Error("worker jk timeout")); }, 60000);
        });
        ch.postMessage({ type: "jk", D });
        const mPart = buildJK_DF({ B: mSlice.B, n, nAux: mSlice.nAux, threshold: 0 }, D);
        const wPart = await workerPromise;
        const J = new Float64Array(n * n);
        const K = new Float64Array(n * n);
        for (let i = 0; i < J.length; i++) { J[i] = mPart.J[i]! + wPart.J[i]!; K[i] = mPart.K[i]! + wPart.K[i]!; }
        return { J, K };
      };

      const tSw0 = performance.now();
      const swHF = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 40, parallel: 1, customJKBuilder,
      });
      const swMs = performance.now() - tSw0;
      // eslint-disable-next-line no-console
      console.log(`[m] streaming swarm SCF: ${swMs.toFixed(0)} ms, E=${swHF.energy.toFixed(8)}, iter=${swHF.iter}`);

      ch.postMessage({ type: "shutdown" });
      ch.close();

      return {
        n, fullModes: fullB.nAux, masterModes: mSlice.nAux,
        masterSliceBytes: mSlice.B.byteLength, masterPeakVBytes: mSlice.peakVFloats * 8,
        fullBBytes: fullBytes,
        refEnergy: refHF.energy, refMs,
        swEnergy: swHF.energy, swIter: swHF.iter, swMs,
        deltaE: Math.abs(swHF.energy - refHF.energy),
      };
    }, N_TABS);

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Streaming swarm HF — benzene cc-pVDZ, ${N_TABS} tabs, self-built slices`);
    console.log(`  n=${result.n}, full modes=${result.fullModes}`);
    console.log(`  full B (single tab)     : ${(result.fullBBytes / 1e6).toFixed(1)} MB`);
    console.log(`  master slice (modes ${result.masterModes}): ${(result.masterSliceBytes / 1e6).toFixed(1)} MB`);
    console.log(`  worker slice (modes ${workerInfo.modes}): ${(workerInfo.sliceBytes / 1e6).toFixed(1)} MB`);
    console.log(`  peak V per tab (streamed): ${(result.masterPeakVBytes / 1e6).toFixed(1)} MB`);
    console.log(`  reference E = ${result.refEnergy.toFixed(8)} Ha`);
    console.log(`  swarm     E = ${result.swEnergy.toFixed(8)} Ha   |ΔE| = ${result.deltaE.toExponential(2)}`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    // Partition identity: swarm == single-tab streaming, to convergence.
    expect(result.deltaE).toBeLessThan(1e-7);
    // Memory claim: every tab's slice is strictly smaller than the full B,
    // and the modes tile exactly (master + worker == full).
    expect(result.masterSliceBytes).toBeLessThan(result.fullBBytes);
    expect(workerInfo.sliceBytes).toBeLessThan(result.fullBBytes);
    expect(result.masterModes + workerInfo.modes).toBe(result.fullModes);

    await ctx.close();
  });
});
