import { test, expect } from "@playwright/test";

// Anthracene C₁₄H₁₀ cc-pVDZ swarm HF SCF — the actual memory-wall
// stress test. n=274; B ≈ 900 MB. Single-tab earlier attempts OOM'd
// at the Chrome SAB ceiling.
//
// With 4-tab swarm: each worker holds 225 MB B-slice (well below the
// per-tab cap). Master still builds the full B once before partitioning,
// so master peaks at ~1.8 GB during the V+B build phase — that's the
// risk: if Chrome's SAB ceiling on this machine is below ~2 GB, the
// master will OOM before reaching the SCF loop. If it survives, this
// is the first browser-tab anthracene HF in this repo.

const N_TABS = 4;

test.describe(`Swarm anthracene HF SCF — ${N_TABS} tabs`, () => {
  test(`anthracene cc-pVDZ — ${N_TABS}-tab swarm (memory-wall stress)`, async ({ browser }) => {
    test.setTimeout(10 * 60 * 1000);

    const ctx = await browser.newContext();
    const pages = await Promise.all(
      Array.from({ length: N_TABS }, () => ctx.newPage()),
    );
    pages.forEach((p, i) => {
      p.on("pageerror", (e) => console.error(`[t${i}:pageerror] ${e.message}`));
    });

    await Promise.all(pages.map((p) =>
      p.goto("/molecule.html", { waitUntil: "domcontentloaded" })));

    // Worker setup (tabs 1..N-1).
    for (let i = 1; i < N_TABS; i++) {
      const worker = pages[i]!;
      await worker.evaluate(async (sliceIdx: number) => {
        const [{ preloadWasmJK, buildJK_DF }] = await Promise.all([
          import("/src/chemistry/df.ts" as string),
        ]);
        await preloadWasmJK();
        const ch = new BroadcastChannel("swarm-anth");
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

    // Master.
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

      // Anthracene C₁₄H₁₀ — planar, three fused benzene rings.
      const A = 1.40;
      const H = A * Math.sqrt(3) / 2;
      const CH = 1.09;
      const C_xy: ReadonlyArray<readonly [number, number]> = [
        [-3 * A,        0],         [-2.5 * A,     -H],         [-1.5 * A,     -H],
        [-1.5 * A,      H],         [-2.5 * A,      H],         [-0.5 * A,     -H],
        [ 0.5 * A,     -H],         [ 0.5 * A,      H],         [-0.5 * A,      H],
        [ 1.5 * A,     -H],         [ 2.5 * A,     -H],         [ 3 * A,        0],
        [ 2.5 * A,      H],         [ 1.5 * A,      H],
      ];
      const carbons = C_xy.map((p) => ({ symbol: "C" as const, pos: [p[0], p[1], 0] as const }));
      const BRIDGE = new Set([2, 3, 6, 7]);
      const hydrogens: Array<{ symbol: "H"; pos: readonly [number, number, number] }> = [];
      for (let i = 0; i < C_xy.length; i++) {
        if (BRIDGE.has(i)) continue;
        const cx = C_xy[i]![0];
        const cy = C_xy[i]![1];
        const rad = Math.hypot(cx, cy);
        hydrogens.push({ symbol: "H" as const, pos: [cx + cx / rad * CH, cy + cy / rad * CH, 0] as const });
      }
      const ATOMS = [...carbons, ...hydrogens];
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(ATOMS as never, "cc-pvdz");

      const tInt0 = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const integMs = performance.now() - tInt0;

      const auxShells = generateAutoAux(shells, 1);
      // eslint-disable-next-line no-console
      console.log(`[m] anthracene n=${integrals.n}, n_aux_in=${auxShells.length}, building DF B (this is the memory peak)...`);

      const tDF0 = performance.now();
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);
      const dfMs = performance.now() - tDF0;
      const n = df.n, nAux = df.nAux;
      const B = df.B;

      // eslint-disable-next-line no-console
      console.log(`[m] DF built: n=${n}, n_aux=${nAux}, B=${(B.byteLength / 1e6).toFixed(0)} MB, in ${(dfMs / 1000).toFixed(2)} s`);

      // Partition.
      const step = Math.ceil(nAux / nTabs);
      const ranges: Array<[number, number]> = [];
      for (let i = 0; i < nTabs; i++) {
        const s = i * step;
        const e = Math.min(s + step, nAux);
        if (e > s) ranges.push([s, e]);
      }
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

      const ch = new BroadcastChannel("swarm-anth");
      // Stream-send to workers and release full B asap.
      const ackPromises: Array<Promise<void>> = [];
      for (let i = 1; i < ranges.length; i++) {
        const range = ranges[i]!;
        ackPromises.push(new Promise<void>((resolve, reject) => {
          const handler = (ev: MessageEvent): void => {
            const msg = ev.data as { type: string; sliceIdx?: number };
            if (msg.type === "B-slice-ack" && msg.sliceIdx === i) {
              ch.removeEventListener("message", handler);
              resolve();
            }
          };
          ch.addEventListener("message", handler);
          setTimeout(() => { ch.removeEventListener("message", handler); reject(new Error(`worker ${i} ack timeout`)); }, 60000);
        }));
        const slice = buildSlice(range[0], range[1]);
        ch.postMessage({ type: "B-slice", sliceIdx: i, n, nAuxLocal: slice.nAuxLocal, B: slice.B });
        // slice goes out of scope after the postMessage clone is queued;
        // local reference dropped immediately
      }
      // Build master's slice (kept around for SCF)
      const masterSlice = buildSlice(ranges[0]![0], ranges[0]![1]);
      // Release master's full B copy — it's no longer needed
      const sentB = df.B;
      (df as { B: Float64Array }).B = new Float64Array(0);
      void sentB;  // explicit reference dropped
      await Promise.all(ackPromises);
      // eslint-disable-next-line no-console
      console.log(`[m] all worker slices acked; master holds only its own ${(masterSlice.B.byteLength / 1e6).toFixed(0)} MB slice`);

      let iterCount = 0;
      const customJKBuilder = async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
        iterCount++;
        const nNeeded = ranges.length - 1;
        const partials = new Map<number, { J: Float64Array; K: Float64Array }>();
        const allReceived = new Promise<void>((resolve, reject) => {
          const handler = (ev: MessageEvent): void => {
            const msg = ev.data as { type: string; sliceIdx?: number; J?: Float64Array; K?: Float64Array };
            if (msg.type === "jk-partial" && typeof msg.sliceIdx === "number" && msg.J && msg.K) {
              partials.set(msg.sliceIdx, { J: msg.J, K: msg.K });
              if (partials.size === nNeeded) { ch.removeEventListener("message", handler); resolve(); }
            }
          };
          ch.addEventListener("message", handler);
          setTimeout(() => { ch.removeEventListener("message", handler); reject(new Error(`gather timeout iter ${iterCount}`)); }, 120000);
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
        useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 80,
        levelShift: 1.0,   // anthracene has low-lying virtuals; lift
                           // them to stabilize the initial iters
        customJKBuilder,
      });
      const swMs = performance.now() - tS0;
      ch.postMessage({ type: "shutdown" });
      ch.close();
      return {
        n, nAux, integMs, dfMs, swMs,
        energy: swHF.energy, iter: swHF.iter, converged: swHF.converged,
        slicesPerTab: ranges.map(([s, e]) => e - s),
        masterSliceMB: masterSlice.B.byteLength / 1e6,
      };
    }, { nTabs: N_TABS });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Swarm HF SCF — anthracene C₁₄H₁₀ cc-pVDZ across ${N_TABS} tabs`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`n_orb = ${result.n}    n_aux = ${result.nAux}`);
    console.log(`Slice sizes (aux entries): [${result.slicesPerTab.join(", ")}]`);
    console.log(`Per-tab B-slice: ~${result.masterSliceMB.toFixed(0)} MB`);
    console.log(`Full B-tensor:    ${(result.n * result.n * result.nAux * 8 / 1e9).toFixed(2)} GB total`);
    console.log();
    console.log(`Integrals:     ${(result.integMs / 1000).toFixed(2).padStart(7)} s`);
    console.log(`Aux-DF build:  ${(result.dfMs / 1000).toFixed(2).padStart(7)} s`);
    console.log(`Swarm SCF (${result.iter} iters): ${(result.swMs / 1000).toFixed(2).padStart(7)} s  converged=${result.converged}`);
    console.log();
    console.log(`E = ${result.energy.toFixed(8)} Ha`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(Number.isFinite(result.energy)).toBe(true);
    expect(result.converged).toBe(true);

    await ctx.close();
  });
});
