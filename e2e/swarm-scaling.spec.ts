import { test, expect, type Page } from "@playwright/test";

// HONEST multi-tab scaling — does adding tabs actually speed up a screen?
//
// Lessons baked in from the flawed first screening pass:
//   • WARMUP every tab first (JIT is cold on the first run; each page is its own
//     JS context with its own JIT, so each must be warmed independently).
//   • EVEN split (round-robin), measured directly rather than through swarmMap's
//     scheduler — to isolate the parallel-execution CEILING from any scheduler
//     variance (swarmMap now balances via greedy pull, but a fixed even split is
//     the cleanest scaling baseline). Shares run truly concurrently (Promise.all
//     across pages); wall-clock = the slowest tab.
//   • Same library each time; speedup(N) = wall(1 tab) / wall(N tabs).
//
// This isolates the throughput axis: N tabs each doing a whole molecule, no
// redundant setup, no cross-talk — the win the single-molecule split couldn't give.

type Atom = { symbol: string; pos: [number, number, number] };
const LIBRARY: { label: string; atoms: Atom[] }[] = [
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

const tile = (c: { label: string; atoms: Atom[] }) => ({ label: c.label, atoms: c.atoms, method: "hf" as const, basis: "cc-pvdz" as const });
const MAX_TABS = 4;

test.describe("Swarm scaling — speedup vs number of tabs", () => {
  test("screen a library across 1..4 tabs; measure the scaling curve", async ({ browser }) => {
    test.setTimeout(360 * 1000);
    const ctx = await browser.newContext();
    const pages: Page[] = [];
    for (let i = 0; i < MAX_TABS; i++) {
      const p = await ctx.newPage();
      p.on("pageerror", (e) => console.error(`[p${i}:pageerror] ${e.message}`));
      await p.goto("/swarm.html", { waitUntil: "domcontentloaded" });
      pages.push(p);
    }

    // Run a list of tiles sequentially in one page, return wall-time + gaps.
    const runShare = (page: Page, tiles: unknown[]) => page.evaluate(async (ts: unknown[]) => {
      const { runChemEnergyTile } = await import("/src/swarm/chemistry-kernel.ts" as string);
      const t0 = performance.now();
      const res: { label: string; gap: number; conv: boolean }[] = [];
      for (const t of ts) {
        const r = await runChemEnergyTile(t as never) as { label: string; homoLumoGapEv: number; converged: boolean };
        res.push({ label: r.label, gap: r.homoLumoGapEv, conv: r.converged });
      }
      return { ms: performance.now() - t0, res };
    }, tiles);

    // ── Warm every page (each JS context warms its own JIT) on 3 sizes. ──
    const warm = [tile(LIBRARY[0]!), tile(LIBRARY[5]!), tile(LIBRARY[8]!)];
    await Promise.all(pages.map((p) => runShare(p, warm)));

    const tiles = LIBRARY.map(tile);
    const rankOf = (rs: { label: string; gap: number }[]) => [...rs].sort((a, b) => a.gap - b.gap).map((r) => r.label);

    // ── Measure wall-clock at N = 1..MAX_TABS, even round-robin shares. ──
    const rows: { tabs: number; wallMs: number; speedup: number; shares: number[] }[] = [];
    let baseRank: string[] | null = null, wall1 = 0;
    for (let N = 1; N <= MAX_TABS; N++) {
      const shares: unknown[][] = Array.from({ length: N }, () => []);
      tiles.forEach((t, i) => shares[i % N]!.push(t));
      const out = await Promise.all(pages.slice(0, N).map((p, i) => runShare(p, shares[i]!)));
      const wallMs = Math.max(...out.map((o) => o.ms));      // slowest tab = wall-clock
      const allRes = out.flatMap((o) => o.res);
      expect(allRes.every((r) => r.conv)).toBe(true);
      const rank = rankOf(allRes);
      if (baseRank === null) { baseRank = rank; wall1 = wallMs; }
      else expect(rank).toEqual(baseRank);                  // ranking stable regardless of #tabs
      rows.push({ tabs: N, wallMs, speedup: wall1 / wallMs, shares: shares.map((s) => s.length) });
    }
console.log(`\n[swarm-scaling] ${tiles.length} molecules, HF/cc-pVDZ, warmed:`);
    for (const r of rows) { console.log(`  ${r.tabs} tab${r.tabs > 1 ? "s" : " "}: wall ${String(Math.round(r.wallMs)).padStart(6)}ms  speedup ${r.speedup.toFixed(2)}x  (efficiency ${(100 * r.speedup / r.tabs).toFixed(0)}%)  shares=[${r.shares.join(",")}]`);
    }
    console.log("");

    // Honest expectations: more tabs help, but sub-linear (uneven molecule costs
    // + 10 not divisible by 3/4 → the slowest tab caps the win).
    const last = rows[rows.length - 1]!;
    expect(rows[1]!.speedup).toBeGreaterThan(1.3);   // 2 tabs clearly help
    expect(last.speedup).toBeGreaterThan(rows[1]!.speedup); // 4 tabs beat 2 tabs
    expect(last.speedup).toBeLessThanOrEqual(MAX_TABS + 0.1); // can't beat ideal linear

    await ctx.close();
  });
});
