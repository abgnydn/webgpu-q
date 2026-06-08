import { test, expect } from "@playwright/test";
import { writeSwarmArtifact } from "./lib/swarm-artifact";

// Gradual scaling ladder for the streaming swarm — increase system size step by
// step until the full DF B-tensor crosses the 2 GB single-ArrayBuffer wall, at
// which point the calculation is IMPOSSIBLE on any single tab (a Float64Array
// cannot exceed 2^31-1 bytes in any JS engine) yet tractable across N tabs.
//
// Correctness oracle: a lattice of well-separated H2 molecules is non-
// interacting, so its exact RHF energy is M·E(H2). This gives ground truth at
// scales where no single-tab reference can exist by definition. Each tab builds
// ONLY its own mode-slice (streaming, partition:{tab,of:N}) — no full V or full
// B on any tab. The distributed SCF sums partial (J,K) every iteration.
//
// LADDER env (comma-separated H2 counts) and TABS env let CI dial the rungs:
//   LADDER=4,16 TABS=2  (local/PR)      LADDER=36,64,100 TABS=8  (CI headline)

const LADDER = (process.env.LADDER ?? "2,4").split(",").map((s) => parseInt(s, 10));
const N_TABS = parseInt(process.env.TABS ?? "2", 10);
const BASIS = process.env.BASIS ?? "cc-pvdz";
const ARRAYBUFFER_WALL = 2 ** 31 - 1; // 2 GB single-allocation limit

// M H2 molecules on a cubic grid, spaced far apart (non-interacting).
function h2Lattice(M: number): Array<{ symbol: string; pos: [number, number, number] }> {
  const SPACING = 12.0; // Å — far beyond any orbital overlap
  const BOND = 0.74;
  const side = Math.ceil(Math.cbrt(M));
  const atoms: Array<{ symbol: string; pos: [number, number, number] }> = [];
  let placed = 0;
  for (let i = 0; i < side && placed < M; i++)
    for (let j = 0; j < side && placed < M; j++)
      for (let k = 0; k < side && placed < M; k++) {
        const cx = i * SPACING, cy = j * SPACING, cz = k * SPACING;
        atoms.push({ symbol: "H", pos: [cx, cy, cz - BOND / 2] });
        atoms.push({ symbol: "H", pos: [cx, cy, cz + BOND / 2] });
        placed++;
      }
  return atoms;
}

test.describe(`Streaming swarm scaling ladder (${N_TABS} tabs, ${BASIS})`, () => {
  for (const M of LADDER) {
    test(`${M} H2 (${BASIS}) — ${N_TABS}-tab streaming swarm, energy oracle M·E(H2)`, async ({ browser }) => {
      test.setTimeout(100 * 60 * 1000); // generous margin for slow CI runners
      const atoms = h2Lattice(M);

      const ctx = await browser.newContext();
      const pages = await Promise.all(Array.from({ length: N_TABS }, () => ctx.newPage()));
      pages.forEach((p, i) => p.on("pageerror", (e) => console.error(`[t${i}:pageerror] ${e.message}`)));
      pages[0]!.on("console", (msg) => { if (msg.text().startsWith("[m]")) console.log(msg.text()); });
      await Promise.all(pages.map((p) => p.goto("/molecule.html", { waitUntil: "domcontentloaded" })));

      const CH = "swarm-ladder";

      // Worker tabs 1..N-1: each self-builds its slice, then answers jk.
      const workerSetups = pages.slice(1).map((wp, idx) =>
        wp.evaluate(async ({ atoms, basis, tab, of, ch }) => {
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
          const { shells } = moleculeToShellsNuclei(atoms as never, basis);
          const aux = generateAutoAux(shells, 1);
          const slice = await buildAuxBasisDFStreaming(shells, aux, 1e-10, { partition: { tab, of } });
          const w = window as unknown as { __s: { B: Float64Array; n: number; nAux: number } };
          w.__s = { B: slice.B, n: slice.n, nAux: slice.nAux };
          const bc = new BroadcastChannel(ch);
          bc.addEventListener("message", (ev) => {
            const msg = ev.data as { type: string; D?: Float64Array };
            if (msg.type === "ping") bc.postMessage({ type: "ready", tab });
            else if (msg.type === "jk" && msg.D) {
              const { B, n, nAux } = w.__s;
              const { J, K } = buildJK_DF({ B, n, nAux, threshold: 0 }, msg.D);
              bc.postMessage({ type: "jk-partial", tab, J, K });
            } else if (msg.type === "shutdown") bc.close();
          });
        }, { atoms, basis: BASIS, tab: idx + 1, of: N_TABS, ch: CH }));

      // Master tab 0: self-builds slice, coordinates the distributed SCF.
      const result = await pages[0]!.evaluate(async ({ atoms, basis, of, ch, M }) => {
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

        const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, basis);
        const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
        const aux = generateAutoAux(shells, 1);
        const n = shells.length;

        // Oracle: a single H2 in the same basis; total should be M·E(H2).
        const { shells: sh1, nuclei: nu1, nElectrons: ne1 } =
          moleculeToShellsNuclei([{ symbol: "H", pos: [0, 0, -0.37] }, { symbol: "H", pos: [0, 0, 0.37] }] as never, basis);
        const i1 = computeMolecularIntegrals(sh1, nu1, { skipERI: true, skipOAO: true });
        const aux1 = generateAutoAux(sh1, 1);
        const df1 = await buildAuxBasisDFStreaming(sh1, aux1);
        const eH2 = runRHFSCF(i1, ne1, { useDIIS: true, energyTol: 1e-8, densityTol: 1e-7, maxIter: 60, useDF: df1 }).energy;

        // Master's slice (tab 0). The eigendecomposition is deterministic, so
        // nKeptTotal is the same on every tab — full B size is known exactly
        // (n²·nKeptTotal·8) without ever building the full tensor.
        const tB0 = performance.now();
        const mSlice = await buildAuxBasisDFStreaming(shells, aux, 1e-10, { partition: { tab: 0, of } });
        const buildMs = performance.now() - tB0;
        const totalModes = mSlice.nKeptTotal;

        const bc = new BroadcastChannel(ch);
        // Wait for all workers ready (ping until every tab has answered).
        const ready = new Set<number>([0]);
        await new Promise<void>((resolve) => {
          const h = (ev: MessageEvent): void => {
            const msg = ev.data as { type: string; tab?: number };
            if (msg.type === "ready" && typeof msg.tab === "number") ready.add(msg.tab);
            if (ready.size === of) { clearInterval(ping); bc.removeEventListener("message", h); resolve(); }
          };
          bc.addEventListener("message", h);
          const ping = setInterval(() => bc.postMessage({ type: "ping" }), 500);
          bc.postMessage({ type: "ping" });
        });
        const customJKBuilder = async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
          const partials = new Map<number, { J: Float64Array; K: Float64Array }>();
          const wait = new Promise<void>((resolve, reject) => {
            const h = (ev: MessageEvent): void => {
              const msg = ev.data as { type: string; tab?: number; J?: Float64Array; K?: Float64Array };
              if (msg.type === "jk-partial" && typeof msg.tab === "number" && msg.J && msg.K) {
                partials.set(msg.tab, { J: msg.J, K: msg.K });
                if (partials.size === of - 1) { bc.removeEventListener("message", h); resolve(); }
              }
            };
            bc.addEventListener("message", h);
            setTimeout(() => { bc.removeEventListener("message", h); reject(new Error("jk timeout")); }, 300000);
          });
          bc.postMessage({ type: "jk", D });
          const mPart = buildJK_DF({ B: mSlice.B, n, nAux: mSlice.nAux, threshold: 0 }, D);
          await wait;
          const J = new Float64Array(n * n), K = new Float64Array(n * n);
          J.set(mPart.J); K.set(mPart.K);
          for (const { J: pj, K: pk } of partials.values())
            for (let q = 0; q < J.length; q++) { J[q] = J[q]! + pj[q]!; K[q] = K[q]! + pk[q]!; }
          return { J, K };
        };

        const t0 = performance.now();
        const swHF = await runRHFSCFAsync(integrals, nElectrons, {
          useDIIS: true, energyTol: 1e-8, densityTol: 1e-7, maxIter: 100, parallel: 1, customJKBuilder,
        });
        const swMs = performance.now() - t0;
        bc.postMessage({ type: "shutdown" });
        bc.close();

        return {
          n, nElectrons, eH2, M,
          energy: swHF.energy, iter: swHF.iter, converged: swHF.converged,
          masterModes: mSlice.nAux, masterPeakVBytes: mSlice.peakVFloats * 8,
          masterSliceBytes: mSlice.B.byteLength, totalModes, swMs, buildMs,
        };
      }, { atoms, basis: BASIS, of: N_TABS, ch: CH, M });
      await Promise.all(workerSetups);

      // Full-B size is exact: n² · (total kept modes) · 8.
      const fullBBytes = result.n * result.n * result.totalModes * 8;
      const oracle = result.M * result.eH2;
      const dOracle = Math.abs(result.energy - oracle);
      const crossedWall = fullBBytes > ARRAYBUFFER_WALL;

      /* eslint-disable no-console */
      console.log(`\n══════════════════════════════════════════════════════════`);
      console.log(`Scaling rung: ${result.M} H2 (${BASIS}), ${N_TABS} tabs`);
      console.log(`  n=${result.n}, electrons=${result.nElectrons} | per-tab slice build=${(result.buildMs / 1000).toFixed(1)} s, SCF=${(result.swMs / 1000).toFixed(1)} s (${result.iter} iters)`);
      console.log(`  full B ≈ ${(fullBBytes / 1e9).toFixed(2)} GB  ${crossedWall ? "→ EXCEEDS the 2 GB single-ArrayBuffer wall (impossible single-tab)" : "(still fits one tab)"}`);
      console.log(`  master slice = ${(result.masterSliceBytes / 1e6).toFixed(1)} MB, peak V/tab = ${(result.masterPeakVBytes / 1e6).toFixed(1)} MB`);
      console.log(`  swarm E   = ${result.energy.toFixed(8)} Ha`);
      console.log(`  oracle M·E(H2) = ${oracle.toFixed(8)} Ha   |Δ| = ${dOracle.toExponential(2)} (E(H2)=${result.eH2.toFixed(8)})`);
      console.log(`══════════════════════════════════════════════════════════\n`);
      /* eslint-enable no-console */

      // Commit a traceable, env-stamped artifact for this rung (energy is
      // deterministic; failures are emitted too, per research-grade discipline).
      await writeSwarmArtifact(
        pages[0]!,
        { molecule: `h2lattice-${result.M}`, formula: `(H2)${result.M}`, basis: BASIS, nTabs: N_TABS, innerPool: 1 },
        {
          n: result.n, nAux: result.totalModes, swMs: result.swMs, energy: result.energy,
          iter: result.iter, converged: result.converged, refEnergy: oracle, deltaE: dOracle,
          masterSliceMB: result.masterSliceBytes / 1e6,
        },
      );

      expect(result.converged).toBe(true);
      // Energy oracle: non-interacting lattice ⇒ M·E(H2). Residual is accumulated
      // convergence/DF numerical noise, ~linear in M (sub-µHa per molecule). The
      // bar (≈2 µHa/molecule) is far below chemical accuracy yet trips on any
      // real partition/streaming bug, which would be mHa+ or non-convergence.
      expect(dOracle).toBeLessThan(Math.max(5e-6, result.M * 2e-6));

      await ctx.close();
    });
  }
});
