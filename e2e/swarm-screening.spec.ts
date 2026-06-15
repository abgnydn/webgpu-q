import { test, expect } from "@playwright/test";

// SCREENING DEMO — the swarm's good axis: many independent molecules, one per
// tab, zero waste. We rank a small library of candidate molecules by their
// HOMO–LUMO gap (a real descriptor: small gap → absorbs visible light / more
// reactive; large gap → transparent / stable), first on ONE tab (baseline +
// ranking), then split across TWO tabs. The ranking must be identical and the
// wall-clock must drop — the linear-throughput win that distributing a single
// molecule could never give (see swarm-mp2-speedup, the honest negative).
//
// HF/cc-pVDZ gaps are qualitative (no correlation, crude virtuals) — the point
// is the distributed screening workflow + speedup, not publication gaps.

type Atom = { symbol: string; pos: [number, number, number] };
type Cand = { label: string; atoms: Atom[] };

// A small candidate library — all closed-shell, small, equilibrium geometries (Å).
const LIBRARY: Cand[] = [
  { label: "H2",   atoms: [{ symbol: "H", pos: [0, 0, 0] }, { symbol: "H", pos: [0, 0, 0.741] }] },
  { label: "LiH",  atoms: [{ symbol: "Li", pos: [0, 0, 0] }, { symbol: "H", pos: [0, 0, 1.595] }] },
  { label: "N2",   atoms: [{ symbol: "N", pos: [0, 0, 0] }, { symbol: "N", pos: [0, 0, 1.098] }] },
  { label: "CO",   atoms: [{ symbol: "C", pos: [0, 0, 0] }, { symbol: "O", pos: [0, 0, 1.128] }] },
  { label: "HF",   atoms: [{ symbol: "H", pos: [0, 0, 0] }, { symbol: "F", pos: [0, 0, 0.917] }] },
  { label: "H2O",  atoms: (() => { const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
    return [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }] as Atom[]; })() },
  { label: "CH4",  atoms: (() => { const s = 1.087 / Math.sqrt(3);
    return [{ symbol: "C", pos: [0, 0, 0] }, { symbol: "H", pos: [s, s, s] }, { symbol: "H", pos: [s, -s, -s] },
      { symbol: "H", pos: [-s, s, -s] }, { symbol: "H", pos: [-s, -s, s] }] as Atom[]; })() },
  { label: "C2H2", atoms: [{ symbol: "C", pos: [0, 0, 0.6015] }, { symbol: "C", pos: [0, 0, -0.6015] },
    { symbol: "H", pos: [0, 0, 1.6625] }, { symbol: "H", pos: [0, 0, -1.6625] }] },
  { label: "C2H4", atoms: [{ symbol: "C", pos: [0, 0, 0.6695] }, { symbol: "C", pos: [0, 0, -0.6695] },
    { symbol: "H", pos: [0, 0.9289, 1.2321] }, { symbol: "H", pos: [0, -0.9289, 1.2321] },
    { symbol: "H", pos: [0, 0.9289, -1.2321] }, { symbol: "H", pos: [0, -0.9289, -1.2321] }] },
  { label: "CH2O", atoms: [{ symbol: "C", pos: [0, 0, -0.5293] }, { symbol: "O", pos: [0, 0, 0.6722] },
    { symbol: "H", pos: [0, 0.9367, -1.1172] }, { symbol: "H", pos: [0, -0.9367, -1.1172] }] },
];

const tilesFor = (lib: Cand[]) => lib.map((c) => ({ label: c.label, atoms: c.atoms, method: "hf" as const, basis: "cc-pvdz" as const }));

test.describe("Swarm screening — rank a molecule library by HOMO–LUMO gap", () => {
  test("two tabs rank the library faster, identical ranking to one tab", async ({ browser }) => {
    test.setTimeout(240 * 1000);
    const ctx = await browser.newContext();
    const a = await ctx.newPage();
    const b = await ctx.newPage();
    a.on("pageerror", (e) => console.error(`[a:pageerror] ${e.message}`));
    b.on("pageerror", (e) => console.error(`[b:pageerror] ${e.message}`));
    a.on("console", (m) => { if (m.text().startsWith("[scr]")) console.log(m.text()); });

    await a.goto("/swarm.html", { waitUntil: "domcontentloaded" });
    await b.goto("/swarm.html", { waitUntil: "domcontentloaded" });

    // Worker tab B: register the chem-energy kernel + count tiles it runs.
    await b.evaluate(async () => {
      const [{ attachSwarmRuntime }, { BroadcastChannelTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const w = window as unknown as { __ran: number };
      w.__ran = 0;
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-screening"));
      reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => { w.__ran++; return runChemEnergyTile(tile as never); });
    });

    await a.waitForTimeout(2000); // let B join the roster

    const out = await a.evaluate(async (tiles) => {
      const [{ attachSwarmRuntime, swarmMap }, { BroadcastChannelTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const log = (s: string): void => { console.log(`[scr] ${s}`); };
      const w = window as unknown as { __ran: number };
      w.__ran = 0;
      type R = { label: string; homoLumoGapEv: number; converged: boolean };

      // ── Baseline: one tab, sequential. ──
      const s0 = performance.now();
      const single: R[] = [];
      for (const t of tiles) single.push(await runChemEnergyTile(t as never) as R);
      const singleMs = performance.now() - s0;

      // ── Distributed: split the library across both tabs. ──
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-screening"));
      reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => { w.__ran++; return runChemEnergyTile(tile as never); });
      const d0 = performance.now();
      const dist = await swarmMap(reg, CHEM_ENERGY_KIND, tiles as never[], { claimTimeoutMs: 300 }) as R[];
      const distMs = performance.now() - d0;

      // ── Rank both by gap (ascending: smallest gap = most reactive/colored first). ──
      const rank = (rs: R[]): string[] => [...rs].sort((p, q) => p.homoLumoGapEv - q.homoLumoGapEv).map((r) => r.label);
      const rankSingle = rank(single), rankDist = rank(dist);
      const gapOf = new Map(single.map((r) => [r.label, r.homoLumoGapEv]));

      log(`library: ${tiles.length} molecules, HF/cc-pVDZ`);
      log(`ranked by HOMO–LUMO gap (smallest first = most reactive/colored):`);
      rankSingle.forEach((lbl, i) => log(`  ${String(i + 1).padStart(2)}. ${lbl.padEnd(5)} ${gapOf.get(lbl)!.toFixed(2)} eV`));
      log(`single-tab: ${Math.round(singleMs)}ms   |   two-tab: ${Math.round(distMs)}ms   →  ${(singleMs / distMs).toFixed(2)}x`);
      log(`split: master ran ${tiles.length - w.__ran}, worker ran ${w.__ran}`);

      // gaps must match between the two paths (same deterministic compute).
      let maxGapDiff = 0;
      const dg = new Map(dist.map((r) => [r.label, r.homoLumoGapEv]));
      for (const [lbl, g] of gapOf) maxGapDiff = Math.max(maxGapDiff, Math.abs(g - (dg.get(lbl) ?? NaN)));

      return {
        rankSingle, rankDist, singleMs, distMs, speedup: singleMs / distMs,
        workerRan: w.__ran, allConverged: single.every((r) => r.converged) && dist.every((r) => r.converged),
        maxGapDiff, n: tiles.length,
      };
    }, tilesFor(LIBRARY));

    console.log(`\n[swarm-screening] ${out.n} molecules, single ${Math.round(out.singleMs)}ms → two-tab ${Math.round(out.distMs)}ms (${out.speedup.toFixed(2)}x), worker ran ${out.workerRan}\n`);

    expect(out.allConverged).toBe(true);
    expect(out.maxGapDiff).toBeLessThan(1e-6);          // identical gaps both ways
    expect(out.rankDist).toEqual(out.rankSingle);       // identical ranking
    expect(out.workerRan).toBeGreaterThan(0);           // work really split across tabs
    expect(out.speedup).toBeGreaterThan(1.2);           // the no-waste axis actually speeds up

    await ctx.close();
  });
});
