import { test, expect } from "@playwright/test";

// Swarm × GPU: the two threads of this project meeting. Two browser tabs form a
// BroadcastChannel swarm; a batch of N2 cc-pVDZ single-points (force DF + fast →
// the hybrid GPU/WASM build) is distributed across them, each tab running
// runRHFAuto on the GPU for its tiles. "The crowd is the cluster" AND "the GPU is
// in the tab" — a heterogeneous batch where every tile is GPU-accelerated and the
// work is split across machines (here: tabs). Same-origin multi-tab, no infra.

// N2 closed-shell (14 e⁻), cc-pVDZ (n=28, N has d) — force DF + fast so each tile
// takes the hybrid GPU path regardless of size. A few bond lengths.
const N2_TILES = [1.05, 1.0976, 1.15, 1.20, 1.25, 1.30].map((r) => ({
  label: `N2 r=${r.toFixed(3)}`,
  atoms: [{ symbol: "N", pos: [0, 0, 0] }, { symbol: "N", pos: [0, 0, r] }],
  method: "hf", basis: "cc-pvdz", force: "df", fast: true,
}));

test.describe("Swarm × GPU — distributed GPU-accelerated single-points", () => {
  test("two tabs split an N2 cc-pVDZ batch, every tile runs hybrid GPU DF", async ({ browser }) => {
    test.setTimeout(180 * 1000);
    const ctx = await browser.newContext();
    const a = await ctx.newPage();
    const b = await ctx.newPage();
    a.on("pageerror", (e) => console.error(`[a:pageerror] ${e.message}`));
    b.on("pageerror", (e) => console.error(`[b:pageerror] ${e.message}`));
    a.on("console", (m) => { if (m.text().startsWith("[sg]")) console.log(m.text()); });

    await a.goto("/swarm.html", { waitUntil: "domcontentloaded" });
    await b.goto("/swarm.html", { waitUntil: "domcontentloaded" });

    // Worker tab B: attach a runtime on a private channel + register the kernel,
    // counting how many tiles it computes so we can prove distribution.
    await b.evaluate(async () => {
      const [{ attachSwarmRuntime }, { BroadcastChannelTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const w = window as unknown as { __ran: number };
      w.__ran = 0;
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-gpu-swarm"));
      reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => { w.__ran++; return runChemEnergyTile(tile as never); });
    });

    // Give hellos time to propagate so B is in A's roster before dispatch.
    await a.waitForTimeout(2000);

    // Master tab A: attach + register (so A can also compute), then distribute.
    const results = await a.evaluate(async (tiles) => {
      const [{ attachSwarmRuntime, swarmMap }, { BroadcastChannelTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const w = window as unknown as { __ran: number };
      w.__ran = 0;
      const log = (s: string): void => { console.log(`[sg] ${s}`); };
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-gpu-swarm"));
      reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => { w.__ran++; return runChemEnergyTile(tile as never); });
      const out = await swarmMap(reg, CHEM_ENERGY_KIND, tiles as never[], { claimTimeoutMs: 400 }) as { label: string; energy: number; engine: string; dfMethod: string; converged: boolean }[];
      for (const r of out) log(`${r.label}: ${r.energy.toFixed(5)} [${r.dfMethod}/${r.engine}] conv=${r.converged}`);
      return { results: out, masterRan: w.__ran };
    }, N2_TILES);

    const workerRan = await b.evaluate(() => (window as unknown as { __ran: number }).__ran);
    console.log(`\n[swarm-gpu] master ran ${results.masterRan}, worker ran ${workerRan}, ${results.results.length} tiles\n`);

    const out = results.results;
    expect(out.length).toBe(6);
    // Every tile is correct and GPU-accelerated, wherever it ran.
    for (const r of out) {
      expect(r.converged).toBe(true);
      expect(r.dfMethod).toBe("density-fitting");
      expect(r.engine).toBe("gpu+wasm");          // hybrid GPU path on every tile
      expect(r.energy).toBeLessThan(-108);        // N2 cc-pVDZ HF ≈ -108.95 Ha
      expect(r.energy).toBeGreaterThan(-110);
    }
    // The batch was actually split across the two tabs (not all run on master).
    expect(workerRan).toBeGreaterThan(0);
    expect(results.masterRan + workerRan).toBeGreaterThanOrEqual(6);

    await ctx.close();
  });

  test("two tabs split an OH• radical (UHF) bond scan, every tile GPU-DF + open-shell", async ({ browser }) => {
    test.setTimeout(180 * 1000);
    const ctx = await browser.newContext();
    const a = await ctx.newPage();
    const b = await ctx.newPage();
    a.on("pageerror", (e) => console.error(`[a:pageerror] ${e.message}`));
    b.on("pageerror", (e) => console.error(`[b:pageerror] ${e.message}`));
    a.on("console", (m) => { if (m.text().startsWith("[sgr]")) console.log(m.text()); });
    await a.goto("/swarm.html", { waitUntil: "domcontentloaded" });
    await b.goto("/swarm.html", { waitUntil: "domcontentloaded" });

    await b.evaluate(async () => {
      const [{ attachSwarmRuntime }, { BroadcastChannelTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const w = window as unknown as { __ran2: number };
      w.__ran2 = 0;
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-gpu-radical"));
      reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => { w.__ran2++; return runChemEnergyTile(tile as never); });
    });
    await a.waitForTimeout(2000);

    // OH radical: 9 e⁻ → doublet (5α/4β). Open-shell, DF, GPU.
    const ohTiles = [0.95, 0.97, 1.00, 1.05, 1.10, 1.15].map((r) => ({
      label: `OH• r=${r.toFixed(2)}`,
      atoms: [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [0, 0, r] }],
      method: "hf", basis: "cc-pvdz", open: { nAlpha: 5, nBeta: 4 }, force: "df", fast: true,
    }));

    const results = await a.evaluate(async (tiles) => {
      const [{ attachSwarmRuntime, swarmMap }, { BroadcastChannelTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const w = window as unknown as { __ran2: number };
      w.__ran2 = 0;
      const log = (s: string): void => { console.log(`[sgr] ${s}`); };
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-gpu-radical"));
      reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => { w.__ran2++; return runChemEnergyTile(tile as never); });
      const out = await swarmMap(reg, CHEM_ENERGY_KIND, tiles as never[], { claimTimeoutMs: 400 }) as { label: string; energy: number; engine: string; dfMethod: string; converged: boolean; scf: string }[];
      for (const r of out) log(`${r.label}: ${r.energy.toFixed(5)} [${r.scf}/${r.dfMethod}/${r.engine}] conv=${r.converged}`);
      return { results: out, masterRan: w.__ran2 };
    }, ohTiles);

    const workerRan = await b.evaluate(() => (window as unknown as { __ran2: number }).__ran2);
    console.log(`\n[swarm-gpu-radical] master ran ${results.masterRan}, worker ran ${workerRan}\n`);

    const out = results.results;
    expect(out.length).toBe(6);
    for (const r of out) {
      expect(r.converged).toBe(true);
      expect(r.scf).toBe("uhf");                   // open-shell path
      expect(r.dfMethod).toBe("density-fitting");
      expect(r.engine).toBe("gpu+wasm");           // hybrid GPU on every radical tile
      expect(r.energy).toBeLessThan(-75);          // OH UHF/cc-pVDZ ≈ -75.4 Ha
      expect(r.energy).toBeGreaterThan(-76);
    }
    expect(workerRan).toBeGreaterThan(0);          // distributed across both tabs

    await ctx.close();
  });
});
