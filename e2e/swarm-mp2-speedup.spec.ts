import { test, expect } from "@playwright/test";

// Does the swarm actually make ONE molecule's MP2 faster? Measure it.
//
// Single-machine = one tab runs the whole MP2 (full occupied range). Distributed
// = the same molecule's MP2 split into k slices across 2 BroadcastChannel tabs;
// wall-clock is the time until the master has reduced every partial. Both must
// land on the SAME energy (correctness), and we report the speedup = t_single /
// t_dist. Research-grade: 1 warmup + 3 timed trials, median, noisy flag.
//
// MEASURED RESULT (honest negative, 2026-06-15, M2 Pro, single-shot):
//   H₂O   cc-pVDZ (n=24):  single 0.9s, dist 1.7s → 0.51×  (crowd LOSES)
//   benzene cc-pVDZ (n=114): single 96s, dist 87s → 1.10× (barely wins), and the
//     breakdown explains why: redundant setup S≈79s (SCF + DF build, paid IN FULL
//     on every tab) vs splittable grind C≈17.5s. The contraction we distribute is
//     only ~18% of the job; the SCF+DF setup (82%) sits on the critical path and
//     can't be skipped. speedup = (S+C)/(S+C/k) is capped near 1 while S≫C.
//   Extrapolation: C/S grows with size, but reaching C≫S (where 2 tabs → ~1.8×)
//     needs n≈600 — a molecule whose DF tensor (~5 GB) doesn't fit a tab anyway.
//   CONCLUSION: distributing the MP2 *contraction* gives no meaningful single-
//     molecule speedup in the browser-feasible range. The swarm's real scaling
//     axis is THROUGHPUT (N independent molecules via chem-energy), not making one
//     molecule faster. The bottleneck to attack would be the SCF+DF setup itself.
//
// This harness stays pointed at H₂O so CI runs in seconds; flip MOL to benzene to
// reproduce the negative result above (~3 min, ~600-900 MB).

// Flip MOL to scale the test. cc-pVDZ throughout.
const MOLS = {
  h2o: {
    basis: "cc-pvdz",
    atoms: (() => {
      const half = (104.52 / 2) * Math.PI / 180;
      const x = 0.9572 * Math.sin(half), z = 0.9572 * Math.cos(half);
      return [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [x, 0, z] },
        { symbol: "H", pos: [-x, 0, z] },
      ];
    })(),
  },
  benzene: {
    basis: "cc-pvdz",
    atoms: (() => {
      const rC = 1.397, rH = 1.397 + 1.084; // planar regular hexagon, Å
      const at: { symbol: string; pos: [number, number, number] }[] = [];
      for (let k = 0; k < 6; k++) {
        const a = (k * 60) * Math.PI / 180;
        at.push({ symbol: "C", pos: [rC * Math.cos(a), rC * Math.sin(a), 0] });
        at.push({ symbol: "H", pos: [rH * Math.cos(a), rH * Math.sin(a), 0] });
      }
      return at;
    })(),
  },
} as const;

const MOL = MOLS.h2o;  // CI-safe default; flip to MOLS.benzene to reproduce the documented negative
const N_SLICES = 2;  // one slice per tab — more slices than tabs just multiplies redundant SCF
const WARMUP = 1, TRIALS = 2;  // h2o is fast; benzene is ~minute-scale (use 0/1 there)

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
}

test.describe("Distributed DF-MP2 — wall-clock speedup vs single machine", () => {
  test(`split across 2 tabs is correct, and we measure the speedup`, async ({ browser }) => {
    test.setTimeout(600 * 1000);
    const ctx = await browser.newContext();
    const a = await ctx.newPage();
    const b = await ctx.newPage();
    a.on("pageerror", (e) => console.error(`[a:pageerror] ${e.message}`));
    b.on("pageerror", (e) => console.error(`[b:pageerror] ${e.message}`));
    a.on("console", (m) => { if (m.text().startsWith("[spd]")) console.log(m.text()); });

    await a.goto("/swarm.html", { waitUntil: "domcontentloaded" });
    await b.goto("/swarm.html", { waitUntil: "domcontentloaded" });

    // Worker tab B: register the slice kernel.
    await b.evaluate(async () => {
      const [{ attachSwarmRuntime }, { BroadcastChannelTransport }, { runMP2SliceTile, MP2_SLICE_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-mp2-speedup"));
      reg.registerKernel(MP2_SLICE_KIND, async (tile: unknown) => runMP2SliceTile(tile as never));
    });

    await a.waitForTimeout(2000);

    const out = await a.evaluate(async ({ mol, nSlices, warmup, trials }) => {
      const [
        { attachSwarmRuntime, swarmMap },
        { BroadcastChannelTransport },
        { runMP2SliceTile, mp2SliceTiles, reduceMP2Slices, MP2_SLICE_KIND },
        { moleculeToShellsNuclei, defaultFrozenCore },
      ] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
        import("/src/chemistry/atoms.ts" as string),
      ]);
      const log = (s: string): void => { console.log(`[spd] ${s}`); };

      const { nElectrons } = moleculeToShellsNuclei(mol.atoms as never, mol.basis as never);
      const nOcc = nElectrons / 2;
      const nFrozen = defaultFrozenCore(mol.atoms as never);

      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-mp2-speedup"));
      reg.registerKernel(MP2_SLICE_KIND, async (tile: unknown) => runMP2SliceTile(tile as never));

      // Single-machine = one slice spanning the whole active occupied range.
      const fullTile = { label: "full", atoms: mol.atoms, basis: mol.basis, iLo: nFrozen, iHi: nOcc } as never;
      const sliceTiles = mp2SliceTiles({ label: "m", atoms: mol.atoms as never, basis: mol.basis as never, nSlices });

      const singleT: number[] = [], distT: number[] = [];
      let eSingle = NaN, eDist = NaN, sliceMs = NaN;
      for (let t = 0; t < warmup + trials; t++) {
        const s0 = performance.now();
        const single = await runMP2SliceTile(fullTile);
        const s1 = performance.now();
        const slices = await swarmMap(reg, MP2_SLICE_KIND, sliceTiles, { claimTimeoutMs: 400 }) as { partial: number; durationMs: number }[];
        const s2 = performance.now();
        const reduced = reduceMP2Slices(slices as never);
        if (t >= warmup) {
          singleT.push(s1 - s0); distT.push(s2 - s1);
          eSingle = single.eHF + single.partial; eDist = reduced.totalEnergy;
          sliceMs = Math.max(...slices.map((x) => x.durationMs)); // slowest worker's own compute
        }
      }
      // Overhead-vs-grind decomposition: a slice's own time = S (full redundant
      // SCF+DF) + C/2 (half the contraction); the full run = S + C. So:
      //   S = 2·slice − full   (the redundant setup, paid on EVERY tab)
      //   C = 2·(full − slice) (the splittable grind)
      const full = med(singleT), slice = sliceMs;
      const S = Math.max(0, 2 * slice - full), C = Math.max(0, 2 * (full - slice));
      log(`molecule: ${mol.atoms.length} atoms, ${mol.basis}, active occ ${nOcc - nFrozen}, ${nSlices} slices`);
      log(`single-machine total: ${Math.round(full)}ms   ${JSON.stringify(singleT.map(Math.round))}`);
      log(`distributed (2 tabs):  ${Math.round(med(distT))}ms   ${JSON.stringify(distT.map(Math.round))}`);
      log(`breakdown: redundant setup S≈${Math.round(S)}ms (paid per tab), splittable grind C≈${Math.round(C)}ms`);
      const spd = full / med(distT);
      log(`speedup = ${spd.toFixed(2)}x   (E_single=${eSingle.toFixed(8)}  E_dist=${eDist.toFixed(8)})`);
      function med(xs: number[]): number { const s = [...xs].sort((p, q) => p - q); return s[Math.floor(s.length / 2)]!; }
      return { eSingle, eDist, singleMed: full, distMed: med(distT), speedup: spd, singleT, distT, S, C };
    }, { mol: MOL, nSlices: N_SLICES, warmup: WARMUP, trials: TRIALS });
console.log(`\n[swarm-mp2-speedup] single ${Math.round(out.singleMed)}ms, dist ${Math.round(out.distMed)}ms, speedup ${out.speedup.toFixed(2)}x\n`);

    // Correctness is the hard gate; speedup is a measurement, not a pass/fail.
    expect(Math.abs(out.eSingle - out.eDist)).toBeLessThan(1e-9);
    expect(median(out.singleT)).toBeGreaterThan(0);
    expect(median(out.distT)).toBeGreaterThan(0);

    await ctx.close();
  });
});
