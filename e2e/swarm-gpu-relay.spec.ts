import { test, expect } from "@playwright/test";

// Swarm × GPU over the CROSS-MACHINE transport. swarm-gpu.spec proved GPU tiles
// distribute over same-origin BroadcastChannel; this proves the same over the
// RelayTransport — two independent peers coordinating through a free public MQTT
// broker (the path proven to cross cloud NAT). Run here as two browser contexts
// on one box, but the coordination is genuinely through the external broker, not
// in-process — so it exercises the real cross-machine path. Skips if the broker
// is unreachable (CI/offline), like the WebRTC de-risk.
//
// This is the project's two theses fused: GPU-accelerated chemistry-grade
// single-points (GPU in the tab) distributed across independent peers through a
// free hub (the crowd is the cluster), no infra of our own.

const ROOM = `wgq-gpu-${Date.now().toString(36)}`;

// 3 N2 cc-pVDZ tiles (relay is higher-latency than BroadcastChannel → fewer).
const N2_TILES = [1.0976, 1.15, 1.25].map((r) => ({
  label: `N2 r=${r.toFixed(3)}`,
  atoms: [{ symbol: "N", pos: [0, 0, 0] }, { symbol: "N", pos: [0, 0, r] }],
  method: "hf", basis: "cc-pvdz", force: "df", fast: true,
}));

const SETUP = async ({ room, id }: { room: string; id: string }): Promise<boolean> => {
  const [{ attachSwarmRuntime }, { RelayTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
    import("/src/parallel/swarm/swarm-map.ts" as string),
    import("/src/parallel/swarm/relay-transport.ts" as string),
    import("/src/swarm/chemistry-kernel.ts" as string),
  ]);
  const w = window as unknown as { __reg: unknown; __ran: number; __ready: boolean };
  w.__ran = 0;
  const connected = await new Promise<boolean>((resolve) => {
    // Only resolve on onReady (subscribe complete). mqtt emits transient errors
    // during connect because it has reconnectPeriod — treating those as failure
    // would skip spuriously; the timeout is the real failure path.
    const t = new RelayTransport({ room, id, onReady: () => resolve(true) });
    setTimeout(() => resolve(false), 25000);
    const reg = attachSwarmRuntime(t);
    reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => { w.__ran++; return runChemEnergyTile(tile as never); });
    w.__reg = reg;
  });
  return connected;
};

test.describe("Swarm × GPU over a cross-machine relay", () => {
  test("two relay peers split an N2 cc-pVDZ batch, every tile runs hybrid GPU DF", async ({ browser }) => {
    test.setTimeout(5 * 60 * 1000);
    const ctx = await browser.newContext();
    const a = await ctx.newPage();
    const b = await ctx.newPage();
    a.on("pageerror", (e) => console.error(`[a:pageerror] ${e.message}`));
    b.on("pageerror", (e) => console.error(`[b:pageerror] ${e.message}`));
    a.on("console", (m) => { if (m.text().startsWith("[sgr]")) console.log(m.text()); });
    await a.goto("/molecule.html", { waitUntil: "domcontentloaded" });
    await b.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    // Worker first, then master — both connect outbound to the broker.
    const bConn = await b.evaluate(SETUP, { room: ROOM, id: "worker" }).catch(() => false);
    const aConn = await a.evaluate(SETUP, { room: ROOM, id: "master" }).catch(() => false);
    test.skip(!aConn || !bConn, "MQTT broker unreachable (offline/CI) — cross-machine relay path not exercised");

    // Let hellos propagate through the broker so master sees the worker.
    await a.waitForTimeout(5000);

    const results = await a.evaluate(async (tiles) => {
      const { swarmMap } = await import("/src/parallel/swarm/swarm-map.ts" as string);
      const w = window as unknown as { __reg: unknown; __ran: number };
      const { CHEM_ENERGY_KIND } = await import("/src/swarm/chemistry-kernel.ts" as string);
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[sgr] ${s}`); };
      const out = await swarmMap(w.__reg as never, CHEM_ENERGY_KIND, tiles as never[], { claimTimeoutMs: 3000 }) as { label: string; energy: number; engine: string; dfMethod: string; converged: boolean }[];
      for (const r of out) log(`${r.label}: ${r.energy.toFixed(5)} [${r.dfMethod}/${r.engine}] conv=${r.converged}`);
      return { results: out, masterRan: w.__ran };
    }, N2_TILES);

    const workerRan = await b.evaluate(() => (window as unknown as { __ran: number }).__ran);
    console.log(`\n[swarm-gpu-relay] master ran ${results.masterRan}, worker ran ${workerRan}, ${results.results.length} tiles\n`);

    const out = results.results;
    expect(out.length).toBe(3);
    for (const r of out) {
      expect(r.converged).toBe(true);
      expect(r.dfMethod).toBe("density-fitting");
      expect(r.engine).toBe("gpu+wasm");      // hybrid GPU on every tile, wherever it ran
      expect(r.energy).toBeLessThan(-108);
      expect(r.energy).toBeGreaterThan(-110);
    }
    // The batch was coordinated through the broker (both peers connected); the
    // worker carrying ≥1 tile proves cross-peer distribution over the relay.
    expect(workerRan).toBeGreaterThan(0);

    await ctx.close();
  });
});
