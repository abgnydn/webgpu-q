import { test, expect } from "@playwright/test";
import { writeSwarmArtifact } from "./lib/swarm-artifact";

// C₆₀ buckminsterfullerene HF SCF via 4-tab swarm. 60 carbons on a
// truncated icosahedron, 300 basis functions at STO-3G. The most
// iconic molecule in chemistry — closed-shell singlet, high I_h
// symmetry, well-defined geometry.
//
// Memory peak at master ~1 GB during V+B build, ~130 MB per tab
// after partitioning. SCF likely needs DIIS to converge through
// the symmetry-degenerate frontier orbitals.

const N_TABS = 4;
const INNER_POOL = 2;

test.use({ trace: "off" });   // Playwright trace fixture caps at 10 min — disable so the test gets its full setTimeout budget

test.describe(`Swarm C₆₀ HF SCF — ${N_TABS}-tab × ${INNER_POOL}-inner`, () => {
  test("C60 STO-3G buckyball — swarm HF", async ({ browser }) => {
    test.setTimeout(20 * 60 * 1000);

    const ctx = await browser.newContext();
    const pages = await Promise.all(
      Array.from({ length: N_TABS }, () => ctx.newPage()),
    );
    pages.forEach((p, i) => p.on("pageerror", (e) => console.error(`[t${i}:pageerror] ${e.message}`)));

    // Surface progress so we see where it is when slow.
    pages.forEach((p, idx) => {
      p.on("console", (msg) => {
        const t = msg.text();
        if (t.startsWith("[m]") || t.startsWith(`[w${idx}]`)) console.log(t);
      });
    });

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
        const ch = new BroadcastChannel("swarm-c60");
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

      // ── C₆₀ geometry: 60 vertices of a truncated icosahedron ──
      // Generated from the three coord-class families involving the
      // golden ratio φ = (1+√5)/2:
      //   group 1: (0, ±1, ±3φ)            cyclic perms  → 12 verts
      //   group 2: (±1, ±(2+φ), ±2φ)       cyclic perms  → 24 verts
      //   group 3: (±2, ±(1+2φ), ±φ)       cyclic perms  → 24 verts
      // All 60 lie on a sphere; nearest-neighbor distance in these
      // dimensionless coords is 2, so scale by 1.42/2 to get
      // bond ≈ 1.42 Å (avg of C60's two bond lengths).
      const phi = (1 + Math.sqrt(5)) / 2;
      const verts: Array<[number, number, number]> = [];
      for (const s1 of [1, -1]) for (const s2 of [1, -1]) {
        verts.push([0, s1, s2 * 3 * phi]);
        verts.push([s1, s2 * 3 * phi, 0]);
        verts.push([s2 * 3 * phi, 0, s1]);
      }
      for (const s1 of [1, -1]) for (const s2 of [1, -1]) for (const s3 of [1, -1]) {
        verts.push([s1, s2 * (2 + phi), s3 * 2 * phi]);
        verts.push([s2 * (2 + phi), s3 * 2 * phi, s1]);
        verts.push([s3 * 2 * phi, s1, s2 * (2 + phi)]);
        verts.push([s1 * 2, s2 * (1 + 2 * phi), s3 * phi]);
        verts.push([s2 * (1 + 2 * phi), s3 * phi, s1 * 2]);
        verts.push([s3 * phi, s1 * 2, s2 * (1 + 2 * phi)]);
      }
      const scale = 1.42 / 2;
      const ATOMS = verts.map((v) => ({
        symbol: "C" as const,
        pos: [v[0] * scale, v[1] * scale, v[2] * scale] as const,
      }));
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(ATOMS as never, "sto-3g");

      // eslint-disable-next-line no-console
      console.log(`[m] C₆₀ n=${shells.length} basis functions, n_electrons=${nElectrons}`);

      const tInt0 = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const integMs = performance.now() - tInt0;
      // eslint-disable-next-line no-console
      console.log(`[m] integrals: ${(integMs / 1000).toFixed(2)} s`);

      const auxShells = generateAutoAux(shells, 1);
      const tDF0 = performance.now();
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);
      const dfMs = performance.now() - tDF0;
      const n = df.n, nAux = df.nAux, B = df.B;
      // eslint-disable-next-line no-console
      console.log(`[m] DF: n=${n}, n_aux=${nAux}, B=${(B.byteLength / 1e6).toFixed(0)} MB, in ${(dfMs / 1000).toFixed(2)} s`);

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

      const ch = new BroadcastChannel("swarm-c60");
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
      }
      const masterSlice = buildSlice(ranges[0]![0], ranges[0]![1]);
      // Free master's full B copy
      (df as { B: Float64Array }).B = new Float64Array(0);
      await Promise.all(ackPromises);
      await preloadJK_DF_Workers(n, masterSlice.nAuxLocal, INNER_POOL);
      // eslint-disable-next-line no-console
      console.log(`[m] B distributed; master holds ${(masterSlice.B.byteLength / 1e6).toFixed(0)} MB slice`);

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
      let lastIter = 0;
      const swHF = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-5, densityTol: 1e-4, maxIter: 40,
        customJKBuilder,
        profileCallback: (iter: number, ms: Record<string, number>) => {
          lastIter = iter;
          if (iter <= 5 || iter % 5 === 0) {
            // eslint-disable-next-line no-console
            console.log(`[m] iter ${iter}: ${(ms.total ?? 0).toFixed(0)} ms (jk=${(ms.jk ?? 0).toFixed(0)})`);
          }
        },
      });
      void lastIter;
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
      };
    }, { nTabs: N_TABS, INNER_POOL });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Swarm HF SCF — C₆₀ buckminsterfullerene STO-3G across ${N_TABS} tabs × ${INNER_POOL} inner`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`n_orb = ${result.n}    n_aux = ${result.nAux}`);
    console.log(`Slice sizes (aux entries): [${result.slicesPerTab.join(", ")}]`);
    console.log(`Per-tab B-slice: ~${result.masterSliceMB.toFixed(0)} MB`);
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

    await writeSwarmArtifact(
      pages[0]!,
      { molecule: "c60", formula: "C60", basis: "STO-3G", nTabs: N_TABS, innerPool: INNER_POOL },
      result,
    );

    await ctx.close();
  });
});
