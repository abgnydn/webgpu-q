import { test, expect } from "@playwright/test";

// First-real-step swarm chemistry: distribute the JK_DF build across
// two browser tabs via BroadcastChannel. Master computes the full
// reference (J, K), splits the B-tensor by aux index P, ships one
// half to the worker, then each tab computes its partial JK. Master
// sums the partials and verifies match against the reference.
//
// This is the algorithmic + transport proof in one. Once this lands,
// the remaining work is just wrapping the SCF iter loop around it.

test.describe("Distributed JK_DF across tabs via BroadcastChannel", () => {
  test("benzene cc-pVDZ — 2-tab JK partial sum = single-tab reference", async ({ browser }) => {
    test.setTimeout(2 * 60 * 1000);

    const ctx = await browser.newContext();
    const master = await ctx.newPage();
    const worker = await ctx.newPage();

    master.on("pageerror", (e) => console.error(`[master:pageerror] ${e.message}`));
    worker.on("pageerror", (e) => console.error(`[worker:pageerror] ${e.message}`));
    master.on("console", (msg) => {
      const t = msg.text();
      if (t.startsWith("[m]") || t.startsWith("[swarm]")) console.log(t);
    });
    worker.on("console", (msg) => {
      const t = msg.text();
      if (t.startsWith("[w]") || t.startsWith("[swarm]")) console.log(t);
    });

    await master.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    await worker.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    // Worker setup: listen for B-slice + D, compute partial JK, post back.
    await worker.evaluate(async () => {
      const [{ preloadWasmJK, buildJK_DF }] = await Promise.all([
        import("/src/chemistry/df.ts" as string),
      ]);
      await preloadWasmJK();
      const ch = new BroadcastChannel("swarm-jk-test");
       console.log("[w] ready, listening on swarm-jk-test");

      const w = window as unknown as { __swarmJK: { n: number; nAuxLocal: number; B: Float64Array } | null };
      w.__swarmJK = null;

      ch.addEventListener("message", (ev) => {
        const msg = ev.data as { type: string; n?: number; nAuxLocal?: number; B?: Float64Array; D?: Float64Array };
        if (msg.type === "B-slice" && msg.B && typeof msg.n === "number" && typeof msg.nAuxLocal === "number") {
          w.__swarmJK = { n: msg.n, nAuxLocal: msg.nAuxLocal, B: msg.B };
           console.log(`[w] received B-slice: ${(msg.B.byteLength / 1e6).toFixed(1)} MB, n_aux_local=${msg.nAuxLocal}`);
          ch.postMessage({ type: "B-slice-ack" });
        } else if (msg.type === "jk" && msg.D && w.__swarmJK) {
          const { n, nAuxLocal, B } = w.__swarmJK;
          const localDF = { B, n, nAux: nAuxLocal, threshold: 0 };
          const t0 = performance.now();
          const { J, K } = buildJK_DF(localDF, msg.D);
          const durMs = performance.now() - t0;
           console.log(`[w] JK partial computed in ${durMs.toFixed(1)} ms`);
          ch.postMessage({ type: "jk-partial", J, K, durMs });
        }
      });
    });

    // Master orchestration.
    const result = await master.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDFCholesky, generateAutoAux },
        { preloadWasmJK, buildJK_DF },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
      ]);
      await preloadWasmJK();

      // Benzene cc-pVDZ.
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
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);
      const hf = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-4, maxIter: 10, useDF: df,
      });
      const D = hf.D;
      const n = df.n;
      const nAux = df.nAux;
      const B = df.B;

      // Reference: full JK on the full B.
      const ref = buildJK_DF(df, D);

       console.log(`[m] benzene n=${n}, n_aux=${nAux}, B size=${(B.byteLength / 1e6).toFixed(1)} MB`);

      // Partition by P: master takes [0..nAux/2), worker takes [nAux/2..nAux).
      const pMid = Math.floor(nAux / 2);
      const buildSlice = (pStart: number, pEnd: number): { B: Float64Array; nAuxLocal: number } => {
        const nAuxLocal = pEnd - pStart;
        const Blocal = new Float64Array(n * n * nAuxLocal);
        for (let mu = 0; mu < n; mu++) {
          for (let nu = 0; nu < n; nu++) {
            const baseFull = (mu * n + nu) * nAux;
            const baseLoc = (mu * n + nu) * nAuxLocal;
            for (let p = 0; p < nAuxLocal; p++) {
              Blocal[baseLoc + p] = B[baseFull + (pStart + p)]!;
            }
          }
        }
        return { B: Blocal, nAuxLocal };
      };
      const masterSlice = buildSlice(0, pMid);
      const workerSlice = buildSlice(pMid, nAux);

      const ch = new BroadcastChannel("swarm-jk-test");
      // Send B-slice to worker, wait for ack
      const ackOk = await new Promise<boolean>((resolve, reject) => {
        const handler = (ev: MessageEvent): void => {
          const msg = ev.data as { type: string };
          if (msg.type === "B-slice-ack") {
            ch.removeEventListener("message", handler);
            resolve(true);
          }
        };
        ch.addEventListener("message", handler);
        ch.postMessage({ type: "B-slice", n, nAuxLocal: workerSlice.nAuxLocal, B: workerSlice.B });
        // 30 s ack timeout
        setTimeout(() => { ch.removeEventListener("message", handler); reject(new Error("worker ack timeout")); }, 30000);
      });
       console.log(`[m] B-slice sent (${(workerSlice.B.byteLength / 1e6).toFixed(1)} MB to worker), ack=${ackOk}`);

      // Now request JK with D
      const partialPromise = new Promise<{ J: Float64Array; K: Float64Array; durMs: number }>((resolve, reject) => {
        const handler = (ev: MessageEvent): void => {
          const msg = ev.data as { type: string; J?: Float64Array; K?: Float64Array; durMs?: number };
          if (msg.type === "jk-partial" && msg.J && msg.K && typeof msg.durMs === "number") {
            ch.removeEventListener("message", handler);
            resolve({ J: msg.J, K: msg.K, durMs: msg.durMs });
          }
        };
        ch.addEventListener("message", handler);
        ch.postMessage({ type: "jk", D });
        setTimeout(() => { ch.removeEventListener("message", handler); reject(new Error("worker jk timeout")); }, 30000);
      });
      // While worker computes its half, master computes its own half
      const masterDF = { B: masterSlice.B, n, nAux: masterSlice.nAuxLocal, threshold: 0 };
      const tM0 = performance.now();
      const masterPartial = buildJK_DF(masterDF, D);
      const masterDurMs = performance.now() - tM0;
      const workerPartial = await partialPromise;

      // Sum partials
      const Jsum = new Float64Array(n * n);
      const Ksum = new Float64Array(n * n);
      for (let i = 0; i < Jsum.length; i++) {
        Jsum[i]! = masterPartial.J[i]! + workerPartial.J[i]!;
        Ksum[i]! = masterPartial.K[i]! + workerPartial.K[i]!;
      }

      let maxDJ = 0, maxDK = 0, refMaxJ = 0, refMaxK = 0;
      for (let i = 0; i < Jsum.length; i++) {
        maxDJ = Math.max(maxDJ, Math.abs(Jsum[i]! - ref.J[i]!));
        maxDK = Math.max(maxDK, Math.abs(Ksum[i]! - ref.K[i]!));
        refMaxJ = Math.max(refMaxJ, Math.abs(ref.J[i]!));
        refMaxK = Math.max(refMaxK, Math.abs(ref.K[i]!));
      }
      ch.close();
      return {
        n, nAux,
        masterDurMs, workerDurMs: workerPartial.durMs,
        masterBytes: masterSlice.B.byteLength,
        workerBytes: workerSlice.B.byteLength,
        refMaxJ, refMaxK, maxDJ, maxDK,
        relDJ: maxDJ / Math.max(refMaxJ, 1e-30),
        relDK: maxDK / Math.max(refMaxK, 1e-30),
      };
    });

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Swarm JK_DF — benzene cc-pVDZ across 2 tabs (n=${result.n}, n_aux=${result.nAux})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Master B-slice:  ${(result.masterBytes / 1e6).toFixed(1)} MB   compute ${result.masterDurMs.toFixed(1)} ms`);
    console.log(`Worker B-slice:  ${(result.workerBytes / 1e6).toFixed(1)} MB   compute ${result.workerDurMs.toFixed(1)} ms`);
    console.log();
    console.log(`Reference ‖J‖_max = ${result.refMaxJ.toFixed(4)}, ‖K‖_max = ${result.refMaxK.toFixed(4)}`);
    console.log(`max |ΔJ| = ${result.maxDJ.toExponential(2)}  (rel ${result.relDJ.toExponential(2)})`);
    console.log(`max |ΔK| = ${result.maxDK.toExponential(2)}  (rel ${result.relDK.toExponential(2)})`);
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    expect(result.relDJ).toBeLessThan(1e-12);
    expect(result.relDK).toBeLessThan(1e-12);

    await ctx.close();
  });
});
