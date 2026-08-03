import { test, expect, type Page } from "@playwright/test";

// Does cost-aware (LPT) scheduling beat FIFO for a swarm screen? Measure it.
//
// The throughput axis (N independent molecules) is where the swarm genuinely wins
// — but swarm-scaling found it sub-linear (~59% at 4 tabs) because molecule costs
// are uneven (H₂ n=10 ≪ C₂H₄ n=48) and FIFO can hand a heavy molecule out late, so
// one tab tails past everyone. LPT (`costFn: chemTileCost`, sort heaviest-first)
// starts the heavy molecule at t=0. This runs the SAME library across tabs twice —
// FIFO then LPT — asserts identical energies, and reports the makespan delta.
//
// Honest framing: LPT is a worst-case/average win (Graham's 4/3 bound), not a
// pointwise guarantee — the gain depends on how uneven + how mis-ordered the
// library is. Speedup is a measurement, not a pass/fail; correctness is the gate.

type Atom = { symbol: string; pos: [number, number, number] };
const LIBRARY: { label: string; atoms: Atom[] }[] = [
  { label: "H2", atoms: [{ symbol: "H", pos: [0, 0, 0] }, { symbol: "H", pos: [0, 0, 0.741] }] },
  { label: "LiH", atoms: [{ symbol: "Li", pos: [0, 0, 0] }, { symbol: "H", pos: [0, 0, 1.595] }] },
  { label: "N2", atoms: [{ symbol: "N", pos: [0, 0, 0] }, { symbol: "N", pos: [0, 0, 1.098] }] },
  { label: "CO", atoms: [{ symbol: "C", pos: [0, 0, 0] }, { symbol: "O", pos: [0, 0, 1.128] }] },
  { label: "HF", atoms: [{ symbol: "H", pos: [0, 0, 0] }, { symbol: "F", pos: [0, 0, 0.917] }] },
  { label: "H2O", atoms: [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [0, -0.757, 0.587] }, { symbol: "H", pos: [0, 0.757, 0.587] }] },
  { label: "CH4", atoms: (() => { const s = 1.087 / Math.sqrt(3); return [{ symbol: "C", pos: [0, 0, 0] }, { symbol: "H", pos: [s, s, s] }, { symbol: "H", pos: [s, -s, -s] }, { symbol: "H", pos: [-s, s, -s] }, { symbol: "H", pos: [-s, -s, s] }] as Atom[]; })() },
  { label: "C2H2", atoms: [{ symbol: "C", pos: [0, 0, 0.6015] }, { symbol: "C", pos: [0, 0, -0.6015] }, { symbol: "H", pos: [0, 0, 1.6625] }, { symbol: "H", pos: [0, 0, -1.6625] }] },
  { label: "C2H4", atoms: [{ symbol: "C", pos: [0, 0, 0.6695] }, { symbol: "C", pos: [0, 0, -0.6695] }, { symbol: "H", pos: [0, 0.9289, 1.2321] }, { symbol: "H", pos: [0, -0.9289, 1.2321] }, { symbol: "H", pos: [0, 0.9289, -1.2321] }, { symbol: "H", pos: [0, -0.9289, -1.2321] }] },
  { label: "CH2O", atoms: [{ symbol: "C", pos: [0, 0, -0.5293] }, { symbol: "O", pos: [0, 0, 0.6722] }, { symbol: "H", pos: [0, 0.9367, -1.1172] }, { symbol: "H", pos: [0, -0.9367, -1.1172] }] },
];
const TABS = Number(process.env.SWARM_TABS ?? 4); // set to the box's core count (e.g. SWARM_TABS=2 on a 2-core cloud runner)
const CH = "test-sched";

function median(xs: number[]): number { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]!; }

test.describe("Swarm scheduling — LPT vs FIFO on an uneven library", () => {
  test("cost-aware (LPT) scheduling: identical results, measure the makespan delta", async ({ browser }) => {
    test.setTimeout(600 * 1000);
    const ctx = await browser.newContext();
    const pages: Page[] = [];
    for (let i = 0; i < TABS; i++) {
      const p = await ctx.newPage();
      p.on("pageerror", (e) => console.error(`[p${i}:pageerror] ${e.message}`));
      p.on("console", (m) => { if (m.text().startsWith("[sched]")) console.log(m.text()); });
      await p.goto("/swarm.html", { waitUntil: "domcontentloaded" });
      pages.push(p);
    }

    // Worker tabs register the chem-energy kernel.
    for (let i = 1; i < TABS; i++) {
      await pages[i]!.evaluate(async (ch) => {
        const [{ attachSwarmRuntime }, { BroadcastChannelTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
          import("/src/parallel/swarm/swarm-map.ts" as string),
          import("/src/parallel/swarm/broadcast-transport.ts" as string),
          import("/src/swarm/chemistry-kernel.ts" as string),
        ]);
        const reg = attachSwarmRuntime(new BroadcastChannelTransport(ch));
        reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => runChemEnergyTile(tile as never));
      }, CH);
    }
    await pages[0]!.waitForTimeout(2000);

    const out = await pages[0]!.evaluate(async ({ lib, ch }) => {
      const [
        { attachSwarmRuntime, swarmMap },
        { BroadcastChannelTransport },
        { runChemEnergyTile, chemTileCost, CHEM_ENERGY_KIND },
      ] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const log = (s: string): void => { console.log(`[sched] ${s}`); };
      const reg = attachSwarmRuntime(new BroadcastChannelTransport(ch));
      reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => runChemEnergyTile(tile as never));

      const tiles = lib.map((m: { label: string; atoms: unknown }) => ({ label: m.label, atoms: m.atoms, method: "hf", basis: "cc-pvdz" }));
      const energies = (rs: { energy: number }[]): number[] => rs.map((r) => r.energy).sort((a, b) => a - b);

      // Warm every tab once (cold JIT), then time FIFO and LPT.
      await swarmMap(reg, CHEM_ENERGY_KIND, tiles, { claimTimeoutMs: 300 });
      const fifoT: number[] = [], lptT: number[] = [];
      let eFifo: number[] = [], eLpt: number[] = [];
      for (let t = 0; t < 2; t++) {
        let s = performance.now();
        const rf = await swarmMap(reg, CHEM_ENERGY_KIND, tiles, { claimTimeoutMs: 300 }) as { energy: number }[];
        fifoT.push(performance.now() - s);
        s = performance.now();
        const rl = await swarmMap(reg, CHEM_ENERGY_KIND, tiles, { claimTimeoutMs: 300, costFn: (tile: unknown) => chemTileCost(tile as never) }) as { energy: number }[];
        lptT.push(performance.now() - s);
        eFifo = energies(rf); eLpt = energies(rl);
      }
      const fifo = median(fifoT), lpt = median(lptT);
      function median(xs: number[]): number { const a = [...xs].sort((p, q) => p - q); return a[Math.floor(a.length / 2)]!; }
      log(`${lib.length} molecules, cc-pVDZ HF`);
      log(`FIFO ${Math.round(fifo)}ms  LPT ${Math.round(lpt)}ms  speedup ${(fifo / lpt).toFixed(2)}x`);
      return { fifo, lpt, eFifo, eLpt };
    }, { lib: LIBRARY, ch: CH });
console.log(`\n[swarm-scheduling] FIFO ${Math.round(out.fifo)}ms → LPT ${Math.round(out.lpt)}ms  (${(out.fifo / out.lpt).toFixed(2)}× )\n`);

    // Correctness is the hard gate: LPT must return the SAME set of energies.
    expect(out.eLpt.length).toBe(out.eFifo.length);
    for (let i = 0; i < out.eFifo.length; i++) {
      expect(Math.abs(out.eLpt[i]! - out.eFifo[i]!)).toBeLessThan(1e-9);
    }
    expect(median([out.fifo])).toBeGreaterThan(0);

    await ctx.close();
  });
});
