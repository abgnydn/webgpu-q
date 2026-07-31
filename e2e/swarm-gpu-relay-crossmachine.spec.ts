import { test, expect } from "@playwright/test";

// Cross-MACHINE GPU-DF batch: N separate Actions VMs, each a real machine on the
// public internet, coordinate through a free MQTT broker and split a batch of
// chemistry single-points via swarmMap. Each tile runs runRHFAuto (force DF +
// fast → the hybrid GPU/WASM build where a GPU is present, else f64 WASM-DF) —
// so the crowd computes density-fitted single-points in parallel, no infra.
//
// This is the swarmMap-of-tiles path over RelayTransport, distinct from the
// hand-rolled distributed SCF in swarm-hf-relay-crossmachine: independent tiles,
// not one tightly-coupled SCF. (It also exercises the send-queue fix that first
// made swarmMap work over RelayTransport at all.)
//
// CI runners may lack a real GPU (SwiftShader → no navigator.gpu), so the engine
// degrades to WASM-DF there; we assert DF + correct energy + actual cross-VM
// distribution (≥1 tile computed by a remote peer), and LOG the engine. The GPU
// engine itself is proven locally (swarm-gpu, swarm-gpu-relay).
//
// idx 0 = master (distributes + asserts); idx 1..N-1 = workers (serve tiles).

const IDX = parseInt(process.env.IDX ?? "0", 10);
const NTOTAL = parseInt(process.env.NTOTAL ?? "2", 10);
const ROOM = process.env.ROOM ?? "local-dev";
const ID = (i: number): string => `wgq-${ROOM}-${i}`;

// N2 cc-pVDZ batch (closed-shell, N has d → DF/GPU regime). A few bond lengths.
const N2_TILES = [1.0976, 1.10, 1.15, 1.20].map((r) => ({
  label: `N2 r=${r.toFixed(3)}`,
  atoms: [{ symbol: "N", pos: [0, 0, 0] }, { symbol: "N", pos: [0, 0, r] }],
  method: "hf", basis: "cc-pvdz", force: "df", fast: true,
}));

test.describe(`Cross-machine GPU-DF batch over a relay (N=${NTOTAL})`, () => {
  test(`node ${IDX}/${NTOTAL} — swarmMap distributes the batch across VMs`, async ({ page }) => {
    test.setTimeout(8 * 60 * 1000);
    page.on("pageerror", (e) => console.error(`[${IDX}:pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[gx]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const result = await page.evaluate(async ({ idx, ntot, room, masterId, tiles }) => {
      const log = (s: string): void => { console.log(`[gx] ${s}`); };
      const [{ attachSwarmRuntime, swarmMap }, { RelayTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/relay-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const self = `wgq-${room}-${idx}`;
      const t = new RelayTransport({ room, id: self });
      try { await t.openAsync(); } catch (e) { return { ok: false, stage: "broker", error: String(e) }; }

      // Register the kernel, stamping each result with the peer that computed it.
      let ran = 0;
      const reg = attachSwarmRuntime(t);
      reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => {
        ran++;
        const r = await runChemEnergyTile(tile as never);
        return { ...r, computedBy: self };
      });

      if (idx !== 0) {
        // Worker: announce readiness until the master signals the batch is done,
        // serving claimed tiles via the swarm runtime in the meantime.
        let done = false;
        t.onMessage((msg: { from: string; type: string }) => { if (msg.type === "batch-done") done = true; });
        await new Promise<void>((resolve) => {
          const hi = setInterval(() => { if (!done) t.send({ to: masterId, type: "ready" }); }, 1500);
          t.send({ to: masterId, type: "ready" });
          const poll = setInterval(() => { if (done) { clearInterval(hi); clearInterval(poll); resolve(); } }, 500);
          setTimeout(() => { clearInterval(hi); clearInterval(poll); resolve(); }, 300000);
        });
        t.close();
        return { ok: ran > 0, stage: "worker", ran };
      }

      // Master: wait for all N-1 workers to be ready, then distribute the batch.
      const readied = new Set<string>();
      const allReady = await new Promise<boolean>((resolve) => {
        const off = t.onMessage((msg: { from: string; type: string }) => {
          if (msg.type === "ready") { readied.add(msg.from); if (readied.size >= ntot - 1) { off(); resolve(true); } }
        });
        setTimeout(() => resolve(readied.size >= ntot - 1), 120000);
      });
      if (!allReady) { t.close(); return { ok: false, stage: "handshake", error: `only ${readied.size}/${ntot - 1} workers` }; }
      log(`${readied.size} workers ready — distributing ${tiles.length} tiles`);

      const out = await swarmMap(reg, CHEM_ENERGY_KIND, tiles as never[], { claimTimeoutMs: 4000 }) as
        { label: string; energy: number; engine: string; dfMethod: string; converged: boolean; computedBy: string }[];
      t.send({ type: "batch-done" });
      await new Promise((r) => setTimeout(r, 2000));
      t.close();

      const remote = out.filter((r) => r.computedBy !== self).length;
      for (const r of out) log(`${r.label}: ${r.energy.toFixed(5)} [${r.dfMethod}/${r.engine}] by=${r.computedBy === self ? "master" : "worker"} conv=${r.converged}`);
      const allConv = out.every((r) => r.converged);
      const allDF = out.every((r) => r.dfMethod === "density-fitting");
      const sane = out.every((r) => r.energy < -108 && r.energy > -110);
      return {
        ok: allConv && allDF && sane && remote > 0, stage: "master",
        nTiles: out.length, remote, masterRan: ran,
        engines: Array.from(new Set(out.map((r) => r.engine))),
      };
    }, { idx: IDX, ntot: NTOTAL, room: ROOM, masterId: ID(0), tiles: N2_TILES });
console.log(`\n[gpu-relay-x node ${IDX}/${NTOTAL}] ${JSON.stringify(result)}\n`);
    expect(result.ok, `stage=${result.stage} ${result.error ?? ""}`).toBe(true);
  });
});
