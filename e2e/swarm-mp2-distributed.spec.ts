import { test, expect } from "@playwright/test";

// Distributed DF-MP2 — the swarm's FIRST collaborative single-molecule reduction.
// Two BroadcastChannel tabs split one H₂O cc-pVDZ molecule's MP2 correlation
// energy by outer occupied index, each returning a scalar partial; the master
// sums them (reduceMP2Slices) → E_corr. The reduced total must match the
// single-machine DF-MP2 to ~1e-9 (float reassociation), and the work must
// actually have been split across the two tabs. This is "the crowd computes one
// molecule together," network-optimal: spec in, scalar out, one round.

const H2O = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half), z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
})();

test.describe("Distributed DF-MP2 — collaborative single-molecule reduction", () => {
  test("two tabs split H₂O cc-pVDZ MP2; reduced total == single-machine (≤1e-9)", async ({ browser }) => {
    test.setTimeout(180 * 1000);
    const ctx = await browser.newContext();
    const a = await ctx.newPage();
    const b = await ctx.newPage();
    a.on("pageerror", (e) => console.error(`[a:pageerror] ${e.message}`));
    b.on("pageerror", (e) => console.error(`[b:pageerror] ${e.message}`));
    a.on("console", (m) => { if (m.text().startsWith("[mp2]")) console.log(m.text()); });

    await a.goto("/swarm.html", { waitUntil: "domcontentloaded" });
    await b.goto("/swarm.html", { waitUntil: "domcontentloaded" });

    // Worker tab B: register the mp2-slice kernel, count tiles it runs.
    await b.evaluate(async () => {
      const [{ attachSwarmRuntime }, { BroadcastChannelTransport }, { runMP2SliceTile, MP2_SLICE_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const w = window as unknown as { __ran: number };
      w.__ran = 0;
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-mp2-swarm"));
      reg.registerKernel(MP2_SLICE_KIND, async (tile: unknown) => { w.__ran++; return runMP2SliceTile(tile as never); });
    });

    await a.waitForTimeout(2000); // let B join A's roster

    const out = await a.evaluate(async (atoms) => {
      const [
        { attachSwarmRuntime, swarmMap },
        { BroadcastChannelTransport },
        { runMP2SliceTile, mp2SliceTiles, reduceMP2Slices, MP2_SLICE_KIND },
        { moleculeToShellsNuclei, defaultFrozenCore },
        { runRHFAuto },
        { mp2EnergyDF },
        { generateAutoAux, buildAuxBasisDFStreaming },
      ] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/rhf-auto.ts" as string),
        import("/src/chemistry/mp2.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
      ]);
      const log = (s: string): void => { console.log(`[mp2] ${s}`); };
      const w = window as unknown as { __ran: number };
      w.__ran = 0;

      // Single-machine reference DF-MP2 (same path the kernel uses).
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const n = shells.length, nOcc = nElectrons / 2;
      const ref = await runRHFAuto(shells, nuclei, nElectrons, {});
      const df = await buildAuxBasisDFStreaming(shells, generateAutoAux(shells, 1));
      const nFrozen = defaultFrozenCore(atoms as never);
      const refCorr = mp2EnergyDF(ref.hf.C_MO, ref.hf.orbitalEnergies, nOcc, n, nFrozen, df);
      const refTotal = ref.hf.energy + refCorr;
      log(`single-machine: E_HF=${ref.hf.energy.toFixed(8)} E_corr=${refCorr.toFixed(8)} E_tot=${refTotal.toFixed(8)} (active occ=${nOcc - nFrozen})`);

      // Distribute: split the active occupied range into 4 slices across the swarm.
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("test-mp2-swarm"));
      reg.registerKernel(MP2_SLICE_KIND, async (tile: unknown) => { w.__ran++; return runMP2SliceTile(tile as never); });
      const tiles = mp2SliceTiles({ label: "H2O", atoms: atoms as never, basis: "cc-pvdz", nSlices: 4 });
      const slices = await swarmMap(reg, MP2_SLICE_KIND, tiles, { claimTimeoutMs: 400 }) as never[];
      const reduced = reduceMP2Slices(slices as never);
      for (const s of slices as { label: string; partial: number }[]) log(`  ${s.label}: partial=${s.partial.toFixed(8)}`);
      log(`distributed: E_corr=${reduced.correlationEnergy.toFixed(8)} E_tot=${reduced.totalEnergy.toFixed(8)}`);

      return { refCorr, refTotal, nTiles: tiles.length, masterRan: w.__ran,
        distCorr: reduced.correlationEnergy, distTotal: reduced.totalEnergy };
    }, H2O);

    const workerRan = await b.evaluate(() => (window as unknown as { __ran: number }).__ran);
    console.log(`\n[swarm-mp2] master ran ${out.masterRan}, worker ran ${workerRan} of ${out.nTiles} slices\n`);

    // Reduction is correct: distributed == single-machine to reassociation noise.
    expect(Math.abs(out.distCorr - out.refCorr)).toBeLessThan(1e-9);
    expect(Math.abs(out.distTotal - out.refTotal)).toBeLessThan(1e-9);
    expect(out.refCorr).toBeLessThan(0);
    expect(out.refTotal).toBeLessThan(-76);   // H₂O cc-pVDZ MP2 ≈ -76.23 Ha
    expect(out.refTotal).toBeGreaterThan(-77);
    // The molecule's MP2 was genuinely split across the two tabs.
    expect(workerRan).toBeGreaterThan(0);
    expect(out.masterRan + workerRan).toBe(out.nTiles);

    await ctx.close();
  });
});
