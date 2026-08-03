import { test, expect } from "@playwright/test";
import { writeSwarmArtifact } from "./lib/swarm-artifact";

// Pentacene C₂₂H₁₄ STO-3G via 4-tab swarm — 5 linearly-fused benzene
// rings. 124 basis functions, n_aux ~ 600-700, B-tensor ~80 MB total.
// Extends the naphthalene/anthracene swarm pattern to a real PAH the
// chemistry community recognizes.

const N_TABS = 4;
const INNER_POOL = 2;

test.describe(`Swarm pentacene HF SCF — ${N_TABS}-tab × ${INNER_POOL}-inner`, () => {
  test("pentacene C₂₂H₁₄ STO-3G — swarm converges", async ({ browser }) => {
    test.setTimeout(3 * 60 * 1000);

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
        const ch = new BroadcastChannel("swarm-pent");
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

      // Pentacene C₂₂H₁₄ — 5 linearly fused hexagonal rings.
      // Geometry follows the same pattern as anthracene (3 rings):
      // upper/lower carbon rows at y=±H, plus 2 tip atoms at the ends.
      // For N=5 rings: 22 C (= 4N+2) and 14 H (one per non-bridgehead C).
      const A = 1.40;
      const H = A * Math.sqrt(3) / 2;
      const CH = 1.09;
      const C_xy: ReadonlyArray<readonly [number, number]> = [
        // Outer-left ring (5 unique atoms)
        [-5 * A,     0   ],   // 0: left tip
        [-4.5 * A,  -H   ],   // 1
        [-3.5 * A,  -H   ],   // 2: bridge 1-2
        [-3.5 * A,   H   ],   // 3: bridge 1-2
        [-4.5 * A,   H   ],   // 4
        // Ring 2 (4 new atoms)
        [-2.5 * A,  -H   ],   // 5
        [-1.5 * A,  -H   ],   // 6: bridge 2-3
        [-1.5 * A,   H   ],   // 7: bridge 2-3
        [-2.5 * A,   H   ],   // 8
        // Ring 3 (4 new atoms)
        [-0.5 * A,  -H   ],   // 9
        [ 0.5 * A,  -H   ],   // 10: bridge 3-4
        [ 0.5 * A,   H   ],   // 11: bridge 3-4
        [-0.5 * A,   H   ],   // 12
        // Ring 4 (4 new atoms)
        [ 1.5 * A,  -H   ],   // 13
        [ 2.5 * A,  -H   ],   // 14: bridge 4-5
        [ 2.5 * A,   H   ],   // 15: bridge 4-5
        [ 1.5 * A,   H   ],   // 16
        // Outer-right ring (5 unique atoms)
        [ 3.5 * A,  -H   ],   // 17
        [ 4.5 * A,  -H   ],   // 18
        [ 5 * A,     0   ],   // 19: right tip
        [ 4.5 * A,   H   ],   // 20
        [ 3.5 * A,   H   ],   // 21
      ];
      const carbons = C_xy.map((p) => ({ symbol: "C" as const, pos: [p[0], p[1], 0] as const }));
      const BRIDGE = new Set([2, 3, 6, 7, 10, 11, 14, 15]);
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

      // Partition by P.
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

      const ch = new BroadcastChannel("swarm-pent");
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

      const tS0 = performance.now();
      const swHF = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 60,
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
        firstFiveE: swHF.history.slice(0, 5),
        lastFiveE: swHF.history.slice(-5),
      };
    }, { nTabs: N_TABS, INNER_POOL });

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Swarm HF SCF — pentacene C₂₂H₁₄ STO-3G across ${N_TABS} tabs × ${INNER_POOL} inner`);
    console.log(`══════════════════════════════════════════════════════════`);
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
    console.log(`Trajectory:  first → ${result.firstFiveE.map((e: number) => e.toFixed(2)).join(", ")}`);
    console.log(`             last  → ${result.lastFiveE.map((e: number) => e.toFixed(6)).join(", ")}`);
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    expect(Number.isFinite(result.energy)).toBe(true);
    expect(result.converged).toBe(true);

    await writeSwarmArtifact(
      pages[0]!,
      { molecule: "pentacene", formula: "C22H14", basis: "STO-3G", nTabs: N_TABS, innerPool: INNER_POOL },
      result,
    );

    await ctx.close();
  });
});
