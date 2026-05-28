import { test, expect } from "@playwright/test";

// Swarm HF SCF scaled to N tabs on naphthalene cc-pVDZ.
// Each worker tab gets a distinct aux-index slice of B (assigned by
// the Playwright orchestrator at setup time, no BroadcastChannel
// handshake). Each iter: master broadcasts D, every worker computes
// its partial (J, K) in parallel, master sums + reduces.

const N_WORKERS_TOTAL = 4;  // 1 master + 3 worker tabs

test.describe(`Swarm HF SCF scaled to ${N_WORKERS_TOTAL} tabs`, () => {
  test(`naphthalene cc-pVDZ — ${N_WORKERS_TOTAL}-tab swarm matches single-tab reference`, async ({ browser }) => {
    test.setTimeout(5 * 60 * 1000);

    const ctx = await browser.newContext();
    const pages = await Promise.all(
      Array.from({ length: N_WORKERS_TOTAL }, () => ctx.newPage()),
    );
    pages.forEach((p, i) => {
      p.on("pageerror", (e) => console.error(`[t${i}:pageerror] ${e.message}`));
    });

    await Promise.all(pages.map((p) =>
      p.goto("/molecule.html", { waitUntil: "domcontentloaded" })));

    // Set up workers (tabs 1..N-1). Each worker stores its slice idx
    // and waits for B-slice + jk requests.
    for (let i = 1; i < N_WORKERS_TOTAL; i++) {
      const worker = pages[i]!;
      await worker.evaluate(async (sliceIdx: number) => {
        const [{ preloadWasmJK, buildJK_DF }] = await Promise.all([
          import("/src/chemistry/df.ts" as string),
        ]);
        await preloadWasmJK();
        const ch = new BroadcastChannel("swarm-hf-n");
        const w = window as unknown as { __slice: { idx: number; B: Float64Array; n: number; nAuxLocal: number } | null };
        w.__slice = null;
        ch.addEventListener("message", (ev) => {
          const msg = ev.data as { type: string; sliceIdx?: number; B?: Float64Array; n?: number; nAuxLocal?: number; D?: Float64Array };
          if (msg.type === "B-slice" && msg.sliceIdx === sliceIdx && msg.B && typeof msg.n === "number" && typeof msg.nAuxLocal === "number") {
            w.__slice = { idx: sliceIdx, B: msg.B, n: msg.n, nAuxLocal: msg.nAuxLocal };
            ch.postMessage({ type: "B-slice-ack", sliceIdx });
          } else if (msg.type === "jk" && msg.D && w.__slice) {
            const { B, n, nAuxLocal } = w.__slice;
            const localDF = { B, n, nAux: nAuxLocal, threshold: 0 };
            const { J, K } = buildJK_DF(localDF, msg.D);
            ch.postMessage({ type: "jk-partial", sliceIdx, J, K });
          } else if (msg.type === "shutdown") { ch.close(); w.__slice = null; }
        });
      }, i);
    }

    // Master: drives the full distributed SCF.
    const result = await pages[0]!.evaluate(async ({ nTabs }) => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCFAsync },
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

      // Naphthalene cc-pVDZ — same geometry as bench-naphthalene.
      const ATOMS = [
        { symbol: "C" as const, pos: [-2.4225, -0.7176, 0] as const },
        { symbol: "C" as const, pos: [-1.2451, -1.4032, 0] as const },
        { symbol: "C" as const, pos: [ 0.0,    -0.7176, 0] as const },
        { symbol: "C" as const, pos: [ 0.0,     0.7176, 0] as const },
        { symbol: "C" as const, pos: [-1.2451,  1.4032, 0] as const },
        { symbol: "C" as const, pos: [-2.4225,  0.7176, 0] as const },
        { symbol: "C" as const, pos: [ 1.2451, -1.4032, 0] as const },
        { symbol: "C" as const, pos: [ 2.4225, -0.7176, 0] as const },
        { symbol: "C" as const, pos: [ 2.4225,  0.7176, 0] as const },
        { symbol: "C" as const, pos: [ 1.2451,  1.4032, 0] as const },
        ...([[-2.4225, -0.7176], [-1.2451, -1.4032], [-1.2451, 1.4032], [-2.4225, 0.7176],
             [1.2451, -1.4032], [2.4225, -0.7176], [2.4225, 0.7176], [1.2451, 1.4032]] as const)
          .map((c) => {
            const cx: number = c[0]; const cy: number = c[1];
            const rcX = cx < 0 ? -1.245 : 1.245;
            const dx = cx - rcX;
            const len = Math.sqrt(dx * dx + cy * cy);
            return { symbol: "H" as const, pos: [cx + 1.09 * dx / len, cy + 1.09 * cy / len, 0] as const };
          }),
      ];
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(ATOMS as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const auxShells = generateAutoAux(shells, 1);
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);
      const n = df.n, nAux = df.nAux, B = df.B;

      // Reference SCF using the optimized single-tab parallel path.
      const tR0 = performance.now();
      const refHF = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 40,
        useDF: df, parallel: 8,
      });
      const refMs = performance.now() - tR0;

      // Partition by aux index P into nTabs ranges.
      const step = Math.ceil(nAux / nTabs);
      const ranges: Array<[number, number]> = [];
      for (let i = 0; i < nTabs; i++) {
        const s = i * step;
        const e = Math.min(s + step, nAux);
        if (e > s) ranges.push([s, e]);
      }
      // Build slices.
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
      const slices = ranges.map(([s, e]) => buildSlice(s, e));
      const masterSlice = slices[0]!;

      const ch = new BroadcastChannel("swarm-hf-n");
      // Send slices 1..nTabs-1 to workers, wait for acks.
      const ackPromises = slices.slice(1).map((_, idx) => new Promise<void>((resolve, reject) => {
        const sliceIdx = idx + 1;
        const handler = (ev: MessageEvent): void => {
          const msg = ev.data as { type: string; sliceIdx?: number };
          if (msg.type === "B-slice-ack" && msg.sliceIdx === sliceIdx) {
            ch.removeEventListener("message", handler);
            resolve();
          }
        };
        ch.addEventListener("message", handler);
        setTimeout(() => { ch.removeEventListener("message", handler); reject(new Error(`worker ${sliceIdx} ack timeout`)); }, 30000);
      }));
      for (let i = 1; i < slices.length; i++) {
        ch.postMessage({ type: "B-slice", sliceIdx: i, n, nAuxLocal: slices[i]!.nAuxLocal, B: slices[i]!.B });
      }
      await Promise.all(ackPromises);

      // Per-iter distributed JK.
      let iterCount = 0;
      const customJKBuilder = async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
        iterCount++;
        const nNeeded = slices.length - 1;
        const partials = new Map<number, { J: Float64Array; K: Float64Array }>();
        const allReceived = new Promise<void>((resolve, reject) => {
          const handler = (ev: MessageEvent): void => {
            const msg = ev.data as { type: string; sliceIdx?: number; J?: Float64Array; K?: Float64Array };
            if (msg.type === "jk-partial" && typeof msg.sliceIdx === "number" && msg.J && msg.K) {
              partials.set(msg.sliceIdx, { J: msg.J, K: msg.K });
              if (partials.size === nNeeded) {
                ch.removeEventListener("message", handler);
                resolve();
              }
            }
          };
          ch.addEventListener("message", handler);
          setTimeout(() => { ch.removeEventListener("message", handler); reject(new Error(`gather timeout iter ${iterCount}`)); }, 60000);
        });
        ch.postMessage({ type: "jk", D });
        const masterDF = { B: masterSlice.B, n, nAux: masterSlice.nAuxLocal, threshold: 0 };
        const masterPart = buildJK_DF(masterDF, D);
        await allReceived;
        const J = new Float64Array(n * n);
        const K = new Float64Array(n * n);
        for (let i = 0; i < J.length; i++) {
          J[i] = masterPart.J[i]!;
          K[i] = masterPart.K[i]!;
        }
        for (const { J: Jp, K: Kp } of partials.values()) {
          for (let i = 0; i < J.length; i++) {
            J[i]! += Jp[i]!;
            K[i]! += Kp[i]!;
          }
        }
        return { J, K };
      };

      const tS0 = performance.now();
      const swHF = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 40,
        customJKBuilder,
      });
      const swMs = performance.now() - tS0;
      ch.postMessage({ type: "shutdown" });
      ch.close();
      return {
        n, nAux,
        refEnergy: refHF.energy, refIter: refHF.iter, refMs,
        swEnergy: swHF.energy, swIter: swHF.iter, swMs,
        deltaE: Math.abs(swHF.energy - refHF.energy),
        slicesPerTab: slices.map((s) => s.nAuxLocal),
        masterSliceMB: masterSlice.B.byteLength / 1e6,
      };
    }, { nTabs: N_WORKERS_TOTAL });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Swarm HF SCF — naphthalene cc-pVDZ across ${N_WORKERS_TOTAL} tabs (n=${result.n}, n_aux=${result.nAux})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Slices per tab (aux entries): [${result.slicesPerTab.join(", ")}]`);
    console.log(`Each tab's B-slice: ~${result.masterSliceMB.toFixed(0)} MB (master); full B = ${(result.nAux * result.n * result.n * 8 / 1e6).toFixed(0)} MB`);
    console.log();
    console.log(`Reference (1-tab parallel ×8):  ${result.refMs.toFixed(0).padStart(6)} ms   iter=${result.refIter}   E = ${result.refEnergy.toFixed(8)} Ha`);
    console.log(`Swarm     (${N_WORKERS_TOTAL}-tab serial):       ${result.swMs.toFixed(0).padStart(6)} ms   iter=${result.swIter}   E = ${result.swEnergy.toFixed(8)} Ha`);
    console.log(`|ΔE| = ${result.deltaE.toExponential(2)} Ha`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(result.deltaE).toBeLessThan(1e-7);

    await ctx.close();
  });
});
