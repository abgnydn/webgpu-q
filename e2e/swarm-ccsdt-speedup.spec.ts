import { test, expect } from "@playwright/test";

// Does the swarm make ONE molecule's CCSD(T) faster? Measure it — the (T) analogue
// of swarm-mp2-speedup, and the direct answer to the distributed-MP2 honest negative.
//
// MP2 distribution pinned near 1.10× because the redundant SCF+DF setup (S) dwarfs
// MP2's O(N⁵) grind (C): speedup = (S+C)/(S+C/k) ≈ 1 while S ≫ C. The (T) correction
// is O(N⁷), non-iterative, and parallel over the outer occupied spin-orbital i. So
// the HYPOTHESIS is the inverse: distributing (T) should beat MP2. MEASURED (H₂O
// cc-pVDZ, frozen-core, single-threaded): CCSD=53 s, (T)=40 s → S=53 s, C=40 s,
// C/S=0.76, predicted 2-tab speedup 1.28× — better than MP2's 1.10×, but the
// redundant **CCSD** (not SCF/DF) is the bottleneck, so (T) does NOT dominate.
// MEASURED: C/S is FLAT ~0.9 across HF (n=20) and H₂O (n=25) — a bigger basis does
// NOT help (both (T) and CCSD are O(N_v⁴); C/S ∝ N_o, the electron count). So this
// is a robust ~1.3× (beats MP2's 1.10×), flat across size. CCSD itself is not
// distributed here — that's the next frontier. See
// experiments/results/2026-06-23/federated-ccsdt-regime/.
//
// CI default is H₂O STO-3G: (T) is sub-second, so setup dominates and this run mainly
// proves CORRECTNESS (E_single == E_dist < 1e-9) + the harness. Flip MOL to h2o_ccpvdz
// to reproduce the ~1.28× measurement above (~minutes); go larger for the crossover.

const MOLS = {
  h2o_sto3g: {
    basis: "sto-3g",
    atoms: [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, -0.757, 0.587] },
      { symbol: "H", pos: [0, 0.757, 0.587] },
    ],
  },
  h2o_ccpvdz: {
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
} as const;

const MOL = MOLS.h2o_sto3g; // CI-safe; flip to MOLS.h2o_ccpvdz for the (T)-dominated crossover
const N_SLICES = 2; // one slice per tab
const WARMUP = 1, TRIALS = 2; // STO-3G is fast; for cc-pVDZ use 0/1

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
}

test.describe("Federated CCSD(T) — wall-clock speedup vs single machine", () => {
  test("split (T) across 2 tabs is correct, and we measure the speedup", async ({ browser }) => {
    test.setTimeout(900 * 1000);
    const ctx = await browser.newContext();
    const a = await ctx.newPage();
    const b = await ctx.newPage();
    a.on("pageerror", (e) => console.error(`[a:pageerror] ${e.message}`));
    b.on("pageerror", (e) => console.error(`[b:pageerror] ${e.message}`));
    a.on("console", (m) => { if (m.text().startsWith("[spd]")) console.log(m.text()); });

    await a.goto("/swarm.html", { waitUntil: "domcontentloaded" });
    await b.goto("/swarm.html", { waitUntil: "domcontentloaded" });

    // Worker tab B: register the (T) slice kernel.
    await b.evaluate(async () => {
      const [{ attachSwarmRuntime }, { BroadcastChannelTransport }, { runCCSDTSliceTile, CCSDT_SLICE_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-ccsdt-speedup"));
      reg.registerKernel(CCSDT_SLICE_KIND, async (tile: unknown) => runCCSDTSliceTile(tile as never));
    });

    await a.waitForTimeout(2000);

    const out = await a.evaluate(async ({ mol, nSlices, warmup, trials }) => {
      const [
        { attachSwarmRuntime, swarmMap },
        { BroadcastChannelTransport },
        { runCCSDTSliceTile, ccsdtSliceTiles, reduceCCSDTSlices, CCSDT_SLICE_KIND },
        { moleculeToShellsNuclei, defaultFrozenCore },
      ] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
        import("/src/chemistry/atoms.ts" as string),
      ]);
      const log = (s: string): void => { console.log(`[spd] ${s}`); };

      const { nElectrons } = moleculeToShellsNuclei(mol.atoms as never, mol.basis as never);
      const NOCC = nElectrons; // closed-shell: occupied spin-orbitals = nElectrons
      const nFrozen = defaultFrozenCore(mol.atoms as never);
      const loSO = 2 * nFrozen; // RHF interleaved → frozen SOs [0, 2·nFrozen)

      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-ccsdt-speedup"));
      reg.registerKernel(CCSDT_SLICE_KIND, async (tile: unknown) => runCCSDTSliceTile(tile as never));

      // Single-machine = one slice spanning the whole active occupied SO range.
      const fullTile = { label: "full", atoms: mol.atoms, basis: mol.basis, iLo: loSO, iHi: NOCC } as never;
      const sliceTiles = ccsdtSliceTiles({ label: "m", atoms: mol.atoms as never, basis: mol.basis as never, nSlices });

      const singleT: number[] = [], distT: number[] = [];
      let eSingle = NaN, eDist = NaN, sliceMs = NaN;
      for (let t = 0; t < warmup + trials; t++) {
        const s0 = performance.now();
        const single = await runCCSDTSliceTile(fullTile);
        const s1 = performance.now();
        const slices = await swarmMap(reg, CCSDT_SLICE_KIND, sliceTiles, { claimTimeoutMs: 800 }) as { partial: number; durationMs: number }[];
        const s2 = performance.now();
        const reduced = reduceCCSDTSlices(slices as never);
        if (t >= warmup) {
          singleT.push(s1 - s0); distT.push(s2 - s1);
          eSingle = single.eCCSD + single.partial; eDist = reduced.totalEnergy;
          sliceMs = Math.max(...slices.map((x) => x.durationMs)); // slowest worker's own compute
        }
      }
      function med(xs: number[]): number { const s = [...xs].sort((p, q) => p - q); return s[Math.floor(s.length / 2)]!; }
      // slice = S + C/2 (S = redundant SCF+CCSD setup, C = (T) grind); full = S + C:
      //   S = 2·slice − full   (redundant, paid on EVERY tab)
      //   C = 2·(full − slice) (the splittable (T) grind)
      const full = med(singleT), slice = sliceMs;
      const S = Math.max(0, 2 * slice - full), C = Math.max(0, 2 * (full - slice));
      log(`molecule: ${mol.atoms.length} atoms, ${mol.basis}, active occ-SO ${NOCC - loSO}, ${nSlices} slices`);
      log(`single-machine CCSD(T): ${Math.round(full)}ms   ${JSON.stringify(singleT.map(Math.round))}`);
      log(`distributed (2 tabs):   ${Math.round(med(distT))}ms   ${JSON.stringify(distT.map(Math.round))}`);
      log(`breakdown: redundant SCF+CCSD setup S≈${Math.round(S)}ms (per tab), splittable (T) grind C≈${Math.round(C)}ms`);
      const spd = full / med(distT);
      log(`speedup = ${spd.toFixed(2)}x   (E_single=${eSingle.toFixed(8)}  E_dist=${eDist.toFixed(8)})`);
      return { eSingle, eDist, singleMed: full, distMed: med(distT), speedup: spd, singleT, distT, S, C };
    }, { mol: MOL, nSlices: N_SLICES, warmup: WARMUP, trials: TRIALS });

    console.log(`\n[swarm-ccsdt-speedup] single ${Math.round(out.singleMed)}ms, dist ${Math.round(out.distMed)}ms, ` +
      `speedup ${out.speedup.toFixed(2)}x (S≈${Math.round(out.S)}ms C≈${Math.round(out.C)}ms)\n`);

    // Correctness is the hard gate; speedup is a measurement, not a pass/fail.
    expect(Math.abs(out.eSingle - out.eDist)).toBeLessThan(1e-9);
    expect(median(out.singleT)).toBeGreaterThan(0);
    expect(median(out.distT)).toBeGreaterThan(0);

    await ctx.close();
  });
});
