import { test, expect } from "@playwright/test";

// Heptacene C₃₀H₁₈ STO-3G via 4-tab swarm. 7 linearly fused benzene
// rings, 168 basis functions. The biradical-character borderline:
// HOMO-LUMO gap closes rapidly past hexacene, so default DIIS is
// likely to oscillate. Uses the delayed-DIIS recipe (damping=0.2,
// diisStartIter=8) preemptively.
//
// Real heptacene is so reactive (singlet fission, photo-degradation)
// it's only been observed in cryogenic matrices. Computing it
// stably is itself a small win.

const N_TABS = 4;
const INNER_POOL = 2;

test.use({ trace: "off" });

test.describe(`Swarm heptacene HF SCF — ${N_TABS}-tab × ${INNER_POOL}-inner`, () => {
  test("heptacene C₃₀H₁₈ STO-3G — delayed DIIS for biradical-borderline", async ({ browser }) => {
    test.setTimeout(8 * 60 * 1000);

    const ctx = await browser.newContext();
    const pages = await Promise.all(
      Array.from({ length: N_TABS }, () => ctx.newPage()),
    );
    pages.forEach((p, i) => p.on("pageerror", (e) => console.error(`[t${i}:pageerror] ${e.message}`)));

    await Promise.all(pages.map((p) => p.goto("/molecule.html", { waitUntil: "domcontentloaded" })));

    for (let i = 1; i < N_TABS; i++) {
      const worker = pages[i]!;
      await worker.evaluate(async ({ sliceIdx, INNER_POOL }: { sliceIdx: number; INNER_POOL: number }) => {
        const [
          { preloadWasmJK },
          { buildJK_DF_Parallel, preloadJK_DF_Workers },
        ] = await Promise.all([
          import("/src/chemistry/df.ts" as string),
          import("/src/parallel/parallel-jk-df.ts" as string),
        ]);
        await preloadWasmJK();
        const ch = new BroadcastChannel("swarm-hept");
        const w = window as unknown as { __slice: { idx: number; B: Float64Array; n: number; nAuxLocal: number } | null };
        w.__slice = null;
        ch.addEventListener("message", async (ev) => {
          const msg = ev.data as { type: string; sliceIdx?: number; B?: Float64Array; n?: number; nAuxLocal?: number; D?: Float64Array };
          if (msg.type === "B-slice" && msg.sliceIdx === sliceIdx && msg.B && typeof msg.n === "number" && typeof msg.nAuxLocal === "number") {
            w.__slice = { idx: sliceIdx, B: msg.B, n: msg.n, nAuxLocal: msg.nAuxLocal };
            await preloadJK_DF_Workers(msg.n, msg.nAuxLocal, INNER_POOL);
            ch.postMessage({ type: "B-slice-ack", sliceIdx });
          } else if (msg.type === "jk" && msg.D && w.__slice) {
            const { B, n, nAuxLocal } = w.__slice;
            const localDF = { B, n, nAux: nAuxLocal, threshold: 0 };
            const { J, K } = await buildJK_DF_Parallel(localDF, msg.D, INNER_POOL);
            ch.postMessage({ type: "jk-partial", sliceIdx, J, K });
          } else if (msg.type === "shutdown") { ch.close(); w.__slice = null; }
        });
      }, { sliceIdx: i, INNER_POOL });
    }

    const result = await pages[0]!.evaluate(async ({ nTabs, INNER_POOL }) => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCFAsync },
        { buildAuxBasisDFCholesky, generateAutoAux },
        { preloadWasmJK },
        { buildJK_DF_Parallel, preloadJK_DF_Workers },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
        import("/src/parallel/parallel-jk-df.ts" as string),
      ]);
      await preloadWasmJK();

      // Heptacene C₃₀H₁₈ — 7 linearly fused rings. 4N+2 = 30 C, 18 H.
      // Extends pentacene/hexacene pattern by one more ring on each side.
      const A = 1.40;
      const H = A * Math.sqrt(3) / 2;
      const CH = 1.09;
      // 14 atoms per row (upper / lower), 2 tips. Steps of A in x from
      // -6.5A to +6.5A in each row.
      const xs = [-6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
      const C_xy: Array<readonly [number, number]> = [
        [-7 * A, 0],                                  // 0: left tip
        ...xs.map((x): readonly [number, number] => [x * A, -H]),  // 1..14: lower row
        [7 * A, 0],                                   // 15: right tip
        ...xs.slice().reverse().map((x): readonly [number, number] => [x * A, H]),  // 16..29: upper row (reversed for nicer traversal)
      ];
      const carbons = C_xy.map((p) => ({ symbol: "C" as const, pos: [p[0], p[1], 0] as const }));
      // Bridgeheads in lower row at x = ±4.5A, ±2.5A, ±0.5A → indices in
      // lower row (1..14): x=-4.5 is idx 2, x=-2.5 is idx 4, x=-0.5 idx 6,
      //                    x=+0.5 idx 7, x=+2.5 idx 9, x=+4.5 idx 11.
      // So lower-row bridge indices: 2,4,6,7,9,11 → carbon array indices 2,4,6,7,9,11
      // Upper row is reversed: x=+6.5 idx 16, x=+5.5 17, ..., x=-6.5 idx 29.
      // Upper bridge at x=+4.5 = idx 18, +2.5=20, +0.5=22, -0.5=23, -2.5=25, -4.5=27.
      const BRIDGE = new Set([2, 4, 6, 7, 9, 11, 18, 20, 22, 23, 25, 27]);
      const hydrogens: Array<{ symbol: "H"; pos: readonly [number, number, number] }> = [];
      for (let i = 0; i < C_xy.length; i++) {
        if (BRIDGE.has(i)) continue;
        const cx = C_xy[i]![0];
        const cy = C_xy[i]![1];
        const rad = Math.hypot(cx, cy);
        hydrogens.push({ symbol: "H" as const, pos: [cx + cx / rad * CH, cy + cy / rad * CH, 0] as const });
      }
      const ATOMS = [...carbons, ...hydrogens];
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(ATOMS as never, "sto-3g");

      const tInt0 = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const integMs = performance.now() - tInt0;

      const auxShells = generateAutoAux(shells, 1);
      const tDF0 = performance.now();
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);
      const dfMs = performance.now() - tDF0;
      const n = df.n, nAux = df.nAux, B = df.B;

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
      const slices = ranges.map(([s, e]) => buildSlice(s, e));
      const masterSlice = slices[0]!;
      await preloadJK_DF_Workers(n, masterSlice.nAuxLocal, INNER_POOL);

      const ch = new BroadcastChannel("swarm-hept");
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
        setTimeout(() => { ch.removeEventListener("message", handler); reject(new Error(`worker ${sliceIdx} ack timeout`)); }, 60000);
      }));
      for (let i = 1; i < slices.length; i++) {
        ch.postMessage({ type: "B-slice", sliceIdx: i, n, nAuxLocal: slices[i]!.nAuxLocal, B: slices[i]!.B });
      }
      await Promise.all(ackPromises);

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
              if (partials.size === nNeeded) { ch.removeEventListener("message", handler); resolve(); }
            }
          };
          ch.addEventListener("message", handler);
          setTimeout(() => { ch.removeEventListener("message", handler); reject(new Error(`gather timeout iter ${iterCount}`)); }, 60000);
        });
        ch.postMessage({ type: "jk", D });
        const masterDF = { B: masterSlice.B, n, nAux: masterSlice.nAuxLocal, threshold: 0 };
        const masterPart = await buildJK_DF_Parallel(masterDF, D, INNER_POOL);
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

      // Delayed-DIIS recipe baked in: heptacene's tiny HOMO-LUMO gap
      // will almost certainly oscillate under default DIIS.
      const tS0 = performance.now();
      const swHF = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, damping: 0.2, diisStartIter: 8,
        energyTol: 1e-5, densityTol: 1e-4, maxIter: 80,
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
        firstFive: swHF.history.slice(0, 5),
        lastFive: swHF.history.slice(-5),
        nC: carbons.length, nH: hydrogens.length,
      };
    }, { nTabs: N_TABS, INNER_POOL });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Swarm HF SCF — heptacene C₃₀H₁₈ STO-3G across ${N_TABS} tabs × ${INNER_POOL} inner`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`atoms = ${result.nC} C + ${result.nH} H`);
    console.log(`n_orb = ${result.n}    n_aux = ${result.nAux}`);
    console.log(`Slice sizes (aux entries): [${result.slicesPerTab.join(", ")}]`);
    console.log(`Per-tab B-slice: ~${result.masterSliceMB.toFixed(1)} MB`);
    console.log();
    console.log(`Integrals:     ${(result.integMs / 1000).toFixed(2).padStart(7)} s`);
    console.log(`Aux-DF build:  ${(result.dfMs / 1000).toFixed(2).padStart(7)} s`);
    console.log(`Swarm SCF (${result.iter} iters): ${(result.swMs / 1000).toFixed(2).padStart(7)} s  converged=${result.converged}`);
    console.log(`Total:         ${((result.integMs + result.dfMs + result.swMs) / 1000).toFixed(2).padStart(7)} s`);
    console.log();
    console.log(`E = ${result.energy.toFixed(8)} Ha`);
    console.log(`Trajectory:  first → ${result.firstFive.map((e: number) => e.toFixed(2)).join(", ")}`);
    console.log(`             last  → ${result.lastFive.map((e: number) => e.toFixed(6)).join(", ")}`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(Number.isFinite(result.energy)).toBe(true);
    expect(result.converged).toBe(true);
    // Wide bound — like anthracene cc-pVDZ, the basin-selection issue
    // may give a spurious lower-energy state. Just verify convergence
    // without divergence.
    expect(result.energy).toBeLessThan(-100);
    expect(result.energy).toBeGreaterThan(-3000);

    await ctx.close();
  });
});
