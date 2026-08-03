import { test, expect } from "@playwright/test";
import { writeSwarmArtifact } from "./lib/swarm-artifact";

// Full distributed HF SCF across 2 browser tabs via BroadcastChannel.
//
// Master builds the benzene B-tensor, partitions by aux index P, ships
// one half to the worker tab. Then runs `runRHFSCFAsync` with a custom
// JK builder that distributes the per-iter JK build across both tabs.
// Every SCF iter:
//   - master sends D via BroadcastChannel
//   - master computes its own partial (J, K)
//   - worker computes its partial (J, K) and posts back
//   - master sums → returns (J, K) to the SCF loop
//
// Convergence + energy must match the single-tab reference. This is
// the full swarm-HF flow proof — what's left for anthracene-in-browser
// is just (a) scale to N tabs, (b) have each tab build its own slice
// from scratch so master doesn't need the full B.

test.describe("Full distributed HF SCF across tabs", () => {
  test("benzene cc-pVDZ — distributed-SCF energy matches single-tab reference", async ({ browser }) => {
    test.setTimeout(3 * 60 * 1000);

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

    // Worker: stay alive, handle every "jk" message until shutdown.
    await worker.evaluate(async () => {
      const [{ preloadWasmJK, buildJK_DF }] = await Promise.all([
        import("/src/chemistry/df.ts" as string),
      ]);
      await preloadWasmJK();
      const ch = new BroadcastChannel("swarm-hf-test");
      const w = window as unknown as { __swarmHF: { n: number; nAuxLocal: number; B: Float64Array } | null };
      w.__swarmHF = null;
       console.log("[w] listener ready");

      ch.addEventListener("message", (ev) => {
        const msg = ev.data as { type: string; n?: number; nAuxLocal?: number; B?: Float64Array; D?: Float64Array };
        if (msg.type === "B-slice" && msg.B && typeof msg.n === "number" && typeof msg.nAuxLocal === "number") {
          w.__swarmHF = { n: msg.n, nAuxLocal: msg.nAuxLocal, B: msg.B };
           console.log(`[w] B-slice stored: ${(msg.B.byteLength / 1e6).toFixed(1)} MB, n_aux_local=${msg.nAuxLocal}`);
          ch.postMessage({ type: "B-slice-ack" });
        } else if (msg.type === "jk" && msg.D && w.__swarmHF) {
          const { n, nAuxLocal, B } = w.__swarmHF;
          const localDF = { B, n, nAux: nAuxLocal, threshold: 0 };
          const { J, K } = buildJK_DF(localDF, msg.D);
          ch.postMessage({ type: "jk-partial", J, K });
        } else if (msg.type === "shutdown") {
          ch.close();
          w.__swarmHF = null;
        }
      });
    });

    // Master: build B, partition, ship half to worker, run runRHFSCFAsync
    // with a customJKBuilder that distributes per-iter JK.
    const result = await master.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF, runRHFSCFAsync },
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

      const n = df.n;
      const nAux = df.nAux;
      const B = df.B;

      // Reference SCF — single tab.
      const tRef0 = performance.now();
      const refHF = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 40, useDF: df,
      });
      const refMs = performance.now() - tRef0;
       console.log(`[m] reference HF SCF: ${refMs.toFixed(0)} ms, E = ${refHF.energy.toFixed(8)}, iter=${refHF.iter}`);

      // Partition by P.
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

      const ch = new BroadcastChannel("swarm-hf-test");

      // Ship worker slice + await ack
      await new Promise<void>((resolve, reject) => {
        const handler = (ev: MessageEvent): void => {
          const msg = ev.data as { type: string };
          if (msg.type === "B-slice-ack") {
            ch.removeEventListener("message", handler);
            resolve();
          }
        };
        ch.addEventListener("message", handler);
        ch.postMessage({ type: "B-slice", n, nAuxLocal: workerSlice.nAuxLocal, B: workerSlice.B });
        setTimeout(() => { ch.removeEventListener("message", handler); reject(new Error("worker ack timeout")); }, 30000);
      });
       console.log("[m] worker slice acked");

      // Distributed JK builder: master + worker each compute partial,
      // master sums.
      const iterDiag: Array<{ iter: number; ms: number; maxJ: number; maxK: number; anyNaN: boolean; maxD: number }> = [];
      let iterCount = 0;
      const customJKBuilder = async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
        iterCount++;
        const t0 = performance.now();
        let maxD = 0;
        for (let i = 0; i < D.length; i++) {
          const a = Math.abs(D[i]!); if (a > maxD) maxD = a;
        }
        // Set up worker reply listener
        const workerPromise = new Promise<{ J: Float64Array; K: Float64Array }>((resolve, reject) => {
          const handler = (ev: MessageEvent): void => {
            const msg = ev.data as { type: string; J?: Float64Array; K?: Float64Array };
            if (msg.type === "jk-partial" && msg.J && msg.K) {
              ch.removeEventListener("message", handler);
              resolve({ J: msg.J, K: msg.K });
            }
          };
          ch.addEventListener("message", handler);
          setTimeout(() => { ch.removeEventListener("message", handler); reject(new Error(`worker jk timeout (iter ${iterCount})`)); }, 60000);
        });
        // Request worker JK
        ch.postMessage({ type: "jk", D });
        // Compute master's own partial
        const masterDF = { B: masterSlice.B, n, nAux: masterSlice.nAuxLocal, threshold: 0 };
        const masterPartial = buildJK_DF(masterDF, D);
        const workerPartial = await workerPromise;
        // Sum
        const J = new Float64Array(n * n);
        const K = new Float64Array(n * n);
        let maxJ = 0, maxK = 0, anyNaN = false;
        for (let i = 0; i < J.length; i++) {
          J[i] = masterPartial.J[i]! + workerPartial.J[i]!;
          K[i] = masterPartial.K[i]! + workerPartial.K[i]!;
          if (!Number.isFinite(J[i]!) || !Number.isFinite(K[i]!)) anyNaN = true;
          const a = Math.abs(J[i]!); if (a > maxJ) maxJ = a;
          const b = Math.abs(K[i]!); if (b > maxK) maxK = b;
        }
        iterDiag.push({ iter: iterCount, ms: performance.now() - t0, maxJ, maxK, anyNaN, maxD });
        return { J, K };
      };

      const tSw0 = performance.now();
      const swHF = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 40,
        parallel: 1,         // force the async loop body (customJKBuilder is only honored there)
        customJKBuilder,
      });
      const swMs = performance.now() - tSw0;
       console.log(`[m] swarm HF SCF: ${swMs.toFixed(0)} ms, E = ${swHF.energy.toFixed(8)}, iter=${swHF.iter}`);

      ch.postMessage({ type: "shutdown" });
      ch.close();

      return {
        n, nAux,
        refEnergy: refHF.energy, refIter: refHF.iter, refMs,
        swEnergy: swHF.energy, swIter: swHF.iter, swMs,
        // Real SCF convergence flags, not "agreed with the reference". The
        // artifact's passBar is "converged === true", so it must carry the
        // actual SCF state — see the assertion note at the artifact write.
        swConverged: swHF.converged, refConverged: refHF.converged,
        deltaE: Math.abs(swHF.energy - refHF.energy),
        iterDiag,
      };
    });

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Swarm HF SCF — benzene cc-pVDZ across 2 tabs (n=${result.n}, n_aux=${result.nAux})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Reference (1-tab):  ${result.refMs.toFixed(0).padStart(6)} ms   iter=${result.refIter}   E = ${result.refEnergy.toFixed(8)} Ha`);
    console.log(`Swarm     (2-tab):  ${result.swMs.toFixed(0).padStart(6)} ms   iter=${result.swIter}   E = ${result.swEnergy.toFixed(8)} Ha`);
    console.log(`|ΔE| = ${result.deltaE.toExponential(2)} Ha`);
    console.log();
    console.log(`Per-iter diag (first 10):`);
    console.log(`iter | ms  | ‖D‖_max  | ‖J‖_max     | ‖K‖_max     | NaN?`);
    for (const d of result.iterDiag.slice(0, 10)) { console.log(
        ` ${String(d.iter).padStart(3)} | ${d.ms.toFixed(0).padStart(3)} | ` +
        `${d.maxD.toExponential(2)} | ${d.maxJ.toExponential(3)} | ${d.maxK.toExponential(3)} | ${d.anyNaN ? "YES" : "no"}`,
      );
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    // Write the artifact BEFORE asserting, so a failing run still produces
    // evidence with status "fail" — this repo's honest-negative rule. Previously
    // the expect() ran first and `converged` was derived from the same deltaE the
    // expect had just guaranteed, making the field a hardcoded true that no run
    // could ever falsify against passBar "converged === true".
    await writeSwarmArtifact(
      master,
      { molecule: "benzene", formula: "C6H6", basis: "cc-pVDZ", nTabs: 2, innerPool: 1 },
      {
        ...result,
        energy: result.swEnergy,
        iter: result.swIter,
        converged: result.swConverged && result.refConverged,
      },
    );

    expect(result.swConverged, "swarm SCF must actually converge").toBe(true);
    expect(result.refConverged, "reference SCF must actually converge").toBe(true);
    expect(result.deltaE).toBeLessThan(1e-7);  // energies match to convergence tolerance

    await ctx.close();
  });
});
