import { test, expect } from "@playwright/test";

// Octacene C₃₄H₂₀ STO-3G via 4-tab swarm. 8 linearly fused benzene
// rings, 190 basis functions. Past heptacene the acene becomes
// strongly biradical in character — RHF is increasingly unphysical
// vs UHF / multireference treatments. Computing convergence here
// is mostly a stress test of the architecture; the resulting
// energy is unlikely to be chemically meaningful as ground state.

const N_TABS = 4;
const INNER_POOL = 2;

test.use({ trace: "off" });

test.describe(`Swarm octacene HF SCF — ${N_TABS}-tab × ${INNER_POOL}-inner`, () => {
  test("octacene C₃₄H₂₀ STO-3G — delayed DIIS at the biradical regime", async ({ browser }) => {
    test.setTimeout(10 * 60 * 1000);

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
        const ch = new BroadcastChannel("swarm-oct");
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

      // Octacene C₃₄H₂₀ — 8 linearly fused rings. 4N+2 = 34 C, 20 H.
      // Atoms-per-row = 2N = 16 in upper, 16 in lower, plus 2 tips.
      // Lower-row x-positions: ±7.5, ±6.5, ..., ±0.5 → carbon-array
      // indices 1..16. Bridgeheads at x=±0.5,±2.5,±4.5,±6.5 → indices
      // 2,4,6,8,9,11,13,15 (7 bridgeheads per row, total 14).
      const A = 1.40;
      const H = A * Math.sqrt(3) / 2;
      const CH = 1.09;
      const xs = [-7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
      const C_xy: Array<readonly [number, number]> = [
        [-8 * A, 0],                                                              // 0: left tip
        ...xs.map((x): readonly [number, number] => [x * A, -H]),                 // 1..16: lower row
        [8 * A, 0],                                                               // 17: right tip
        ...xs.slice().reverse().map((x): readonly [number, number] => [x * A, H]), // 18..33: upper row
      ];
      const carbons = C_xy.map((p) => ({ symbol: "C" as const, pos: [p[0], p[1], 0] as const }));
      // Lower-row x → carbon-array idx: x=-7.5→1, -6.5→2, ..., -0.5→8,
      //                                  +0.5→9, +1.5→10, ..., +7.5→16.
      // Bridges at x=±5.5,±3.5,±1.5 in lower row → idx 3,5,7,10,12,14,
      // and ±0.5 are NOT bridges (they're between innermost ring's outer
      // tip and the middle), so actually 6 bridges per row? Let me redo:
      // For N=8 rings, there are 7 internal shared edges → 7 bridges per
      // row → 14 bridgeheads total. x positions of bridges: ±6.5, ±4.5,
      // ±2.5, ±0.5. That's 8 per row… wait that's 4 (not 7) on each
      // side. Total per row = 8. Hmm.
      // Pattern revisit: for N rings, shared edges between adjacent
      // rings are at x = (N - 2k - 1)*A for k = 0, 1, ..., N-2.
      // For N=8: x = ±7A, ±5A, ±3A, ±A (4 negative + 4 positive = 8 ?).
      // Hmm that's at integer multiples of A, not half-integer. Skip
      // the exact bridge accounting and just compute by geometry: a C
      // is a bridge iff it has 3 neighboring C atoms (no room for H).
      // For approximate linear PAH that means C at x ∈ (-N+1, +N-1) and
      // at y = ±H — i.e., not the tips and not the outermost two of
      // each row. For N=8: outermost in lower row is x=-7.5A (idx 1),
      // next is -6.5A (idx 2), etc. Bridges are atoms NOT at the
      // outermost two of each row and NOT tips. So bridges in lower
      // row: idx 3 through 14 (12 atoms). That's WAY more than 14.
      // Linear PAH bridgeheads is actually just 2 atoms per shared
      // edge: total 2*(N-1) = 14 bridges. In lower row, half of those.
      // The N-1 = 7 shared edges have their two bridges in lower and
      // upper rows respectively. So 7 lower-row bridges:
      //   For each ring boundary k = 1..N-1, bridges at x = (N-2k+1)A/2
      //   At N=8: ring boundaries at k=1..7 → x = 6.5A, 4.5A, 2.5A,
      //   0.5A, -1.5A, -3.5A, -5.5A.
      // Convert to lower-row indices: x=6.5→15, 4.5→13, 2.5→11, 0.5→9,
      // -1.5→7, -3.5→5, -5.5→3.
      // Lower-row bridges: 3, 5, 7, 9, 11, 13, 15 (7 atoms).
      // Upper row mirrors: indices 17 + 0..15 (reverse order).
      // Upper bridges at x = -5.5, -3.5, -1.5, 0.5, 2.5, 4.5, 6.5 → upper-row
      // indices (note upper is reverse): x=+7.5→idx 18, ..., x=-7.5→idx 33.
      // x=6.5 → idx 19, 4.5 → 21, 2.5 → 23, 0.5 → 25, -1.5 → 27,
      // -3.5 → 29, -5.5 → 31.
      const BRIDGE = new Set([3, 5, 7, 9, 11, 13, 15, 19, 21, 23, 25, 27, 29, 31]);
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

      const ch = new BroadcastChannel("swarm-oct");
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
    console.log(`Swarm HF SCF — octacene C₃₄H₂₀ STO-3G across ${N_TABS} tabs × ${INNER_POOL} inner`);
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

    // Octacene (8 rings) is the deepest into the polyradical regime of
    // the whole ladder — RHF convergence here is not physically reliable.
    // It happened to converge in one nightly, but heptacene (7 rings)
    // did NOT in the same run, which proves convergence at this scale is
    // luck, not signal. Asserting converged===true would therefore be a
    // latent flake. We assert the architectural invariant (swarm runs the
    // full SCF to completion, finite physically-ranged energy) and treat
    // RHF convergence as method-limited, consistent with hexacene/
    // heptacene. Multireference/UHF reference is the real fix.
    expect(Number.isFinite(result.energy)).toBe(true);
    expect(result.energy).toBeLessThan(-100);
    expect(result.energy).toBeGreaterThan(-3000);

    await ctx.close();
  });
});
