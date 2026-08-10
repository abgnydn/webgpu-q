import { test, expect } from "@playwright/test";
import { assertHydrocarbonEnergySane } from "./lib/energy-gate.js";

// Anthracene C₁₄H₁₀ cc-pVDZ — the bigger basis variant that failed
// with default DIIS (E went to +5352 Ha) and with plain damping
// (iter 5 jumped to +801 Ha). With the new `diisStartIter` option
// (b450896) the recipe is:
//   useDIIS: true, damping: 0.2, diisStartIter: 8
// — heavy damping for the first 7 iters keeps the density in the
// right basin while the SCF "warms up", then DIIS takes over for
// fast convergence.
//
// Memory: master peaks ~1.6 GB during V+B build, then keeps 195 MB
// slice. Each of 3 worker tabs holds 195 MB. On Ubuntu CI runners
// (16 GB RAM) this should fit comfortably; C₆₀ STO-3G at 3 GB peak
// hits a Chromium SAB ceiling — anthracene cc-pVDZ at 1.6 GB is
// below that threshold.

const N_TABS = 4;
const INNER_POOL = 2;
const N_CARBON = 14;

// Literature-scale anthracene HF/cc-pVDZ. Cross-checks against this repo's own
// ladder (benzene/cc-pVDZ ≈ -38.45 Ha/C, naphthalene ≈ -38.34 Ha/C ⟹ C₁₄ ≈ -537).
// The tolerance is sanity-scale, not chemical accuracy — the geometry here is
// idealised (regular hexagons, 1.40 Å) so a literature-digit comparison isn't
// available anyway. Even at 5 Ha of slack the run below misses by 343 Ha.
const LIT_HF_CCPVDZ = -537.0;
const LIT_TOL_HA = 5.0;

// Captured by the architecture test, re-asserted by the documented-negative
// physics test below. Serial describe: ONE 20-minute SCF, two separate claims —
// the architecture claim is allowed to be green, the physics claim is not.
// (playwright.config.ts: workers: 1, fullyParallel: false.)
let observedEnergy: number | undefined;

test.use({ trace: "off" });   // 10-min trace fixture cap would clip this — must be top-level, not inside describe

test.describe.serial(`Swarm anthracene cc-pVDZ HF SCF — ${N_TABS}-tab × ${INNER_POOL}-inner`, () => {
  test("architecture — delayed-DIIS recipe reaches SCF convergence without divergence", async ({ browser }) => {
    test.setTimeout(20 * 60 * 1000);

    const ctx = await browser.newContext();
    const pages = await Promise.all(
      Array.from({ length: N_TABS }, () => ctx.newPage()),
    );
    pages.forEach((p, i) => {
      p.on("pageerror", (e) => console.error(`[t${i}:pageerror] ${e.message}`));
      p.on("console", (msg) => {
        const t = msg.text();
        if (t.startsWith("[m]") || t.startsWith(`[w${i}]`)) console.log(t);
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
        const ch = new BroadcastChannel("swarm-anth-ccpvdz");
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

      // Same anthracene geometry as STO-3G test
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

       console.log(`[m] anthracene cc-pVDZ n=${shells.length}, building DF B (memory peak ~1.6 GB)...`);

      const tInt0 = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const integMs = performance.now() - tInt0;

      const auxShells = generateAutoAux(shells, 1);
      const tDF0 = performance.now();
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);
      const dfMs = performance.now() - tDF0;
      const n = df.n, nAux = df.nAux, B = df.B;
       console.log(`[m] DF: n=${n}, n_aux=${nAux}, B=${(B.byteLength / 1e6).toFixed(0)} MB in ${(dfMs / 1000).toFixed(2)} s`);

      // Partition by P
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

      const ch = new BroadcastChannel("swarm-anth-ccpvdz");
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
      // Free master's full B copy — workers now own their slices
      (df as { B: Float64Array }).B = new Float64Array(0);
      await Promise.all(ackPromises);
      await preloadJK_DF_Workers(n, masterSlice.nAuxLocal, INNER_POOL);
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

      // The recipe: heavy damping for first 7 iters, then DIIS takes over.
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
        history: swHF.history,
      };
    }, { nTabs: N_TABS, INNER_POOL });

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Swarm HF SCF — anthracene C₁₄H₁₀ cc-pVDZ across ${N_TABS} tabs × ${INNER_POOL} inner`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`n_orb = ${result.n}    n_aux = ${result.nAux}`);
    console.log(`Slice sizes (aux entries): [${result.slicesPerTab.join(", ")}]`);
    console.log(`Per-tab B-slice: ~${result.masterSliceMB.toFixed(0)} MB`);
    console.log(`Full B-tensor:    ${(result.n * result.n * result.nAux * 8 / 1e9).toFixed(2)} GB total`);
    console.log();
    console.log(`Integrals:     ${(result.integMs / 1000).toFixed(2).padStart(7)} s`);
    console.log(`Aux-DF build:  ${(result.dfMs / 1000).toFixed(2).padStart(7)} s`);
    console.log(`Swarm SCF (${result.iter} iters): ${(result.swMs / 1000).toFixed(2).padStart(7)} s  converged=${result.converged}`);
    console.log(`Total:         ${((result.integMs + result.dfMs + result.swMs) / 1000).toFixed(2).padStart(7)} s`);
    console.log();
    console.log(`E = ${result.energy.toFixed(8)} Ha`);
    console.log(`Trajectory (first 5 / last 5):`);
    const hist = result.history;
    const head = hist.slice(0, Math.min(5, hist.length));
    const tail = hist.length > 10 ? hist.slice(-5) : hist.slice(5);
    for (let i = 0; i < head.length; i++) console.log(`  iter ${(i + 1).toString().padStart(3)}: ${head[i]!.toFixed(6)} Ha`);
    if (tail.length > 0 && tail !== head) { console.log(`  ...`);
      const startIdx = hist.length - tail.length;
      for (let i = 0; i < tail.length; i++) { console.log(`  iter ${(startIdx + i + 1).toString().padStart(3)}: ${tail[i]!.toFixed(6)} Ha`);
      }
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    observedEnergy = result.energy;

    // ── ARCHITECTURE ONLY. This test makes NO claim about the energy. ──
    // What it proves: the 4-tab × 2-inner swarm runs a cc-pVDZ SCF end to end
    // and the delayed-DIIS recipe (damping 0.2 + diisStartIter 8) reaches a
    // stationary point instead of the +5352 Ha divergence that default DIIS
    // produced, or the +801 Ha iter-5 jump that plain damping produced.
    // Whether the stationary point is the RIGHT one is asserted separately,
    // in the documented-negative test below — and it currently is not.
    expect(Number.isFinite(result.energy)).toBe(true);
    expect(result.converged).toBe(true);
    expect(result.energy, "SCF diverged to a positive energy").toBeLessThan(0);

    await ctx.close();
  });

  // ── DOCUMENTED NEGATIVE — this test is REQUIRED to fail. ───────────────────
  // The run above converges to ≈ -880 Ha; anthracene HF/cc-pVDZ is ≈ -537 Ha.
  // That is a 343 Ha error reported with `converged: true` and a finite,
  // plausible-looking number — the most dangerous failure mode in the repo,
  // LIMITATIONS.md §3. Until 2026-08 it was certified GREEN by an assertion
  // window of `-1500 < E < -100`.
  //
  // Cause: the damped warm-up plus the idealised planar geometry steers the SCF
  // into a non-physical orbital occupation that is variationally lower but is
  // not the ground-state singlet. Fixing it needs MOM (maximum overlap method),
  // SOSCF, or a SAD initial guess.
  //
  // test.fail() means Playwright REQUIRES this to fail. If someone fixes the
  // SCF, this test starts passing, Playwright reports "expected to fail but
  // passed", and CI goes red until the annotation is removed. The wrong-basin
  // result can therefore never again be silently sold as a validated energy.
  test("physics — energy vs literature anthracene HF/cc-pVDZ ≈ -537 Ha [EXPECTED FAILURE: wrong-basin SCF, LIMITATIONS.md §3]", () => {
    test.fail();

    expect(observedEnergy, "architecture test must run first (serial describe)").toBeDefined();
    const E = observedEnergy ?? Number.NaN;

    // Fails first, and loudly: ≈ -880 Ha is -62.9 Ha/C, far past the floor.
    assertHydrocarbonEnergySane(E, N_CARBON);
    expect(
      Math.abs(E - LIT_HF_CCPVDZ),
      `E = ${E.toFixed(3)} Ha vs literature ${LIT_HF_CCPVDZ} Ha`,
    ).toBeLessThan(LIT_TOL_HA);
  });
});
