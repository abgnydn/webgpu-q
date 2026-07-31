import { test, expect } from "@playwright/test";

// N-MACHINE cross-machine distributed Hartree–Fock over a free public relay.
//
// N separate GitHub Actions VMs each build only their own mode-slice of the
// density-fitting tensor (no full tensor on any), then run a distributed DF-HF
// SCF: every iteration the master broadcasts the density D through a free MQTT
// broker, each VM computes its partial (J,K) on its slice, workers return their
// partials, the master sums all N and drives the SCF. The converged energy must
// match the single-machine reference.
//
// The scaling axis is MEMORY, not wall-time: each machine holds full_B / N, so N
// machines pool N× the capacity. (Wall-time doesn't speed up here — cross-machine
// builds are independent/redundant and small-n SCF is broker-latency-bound; we do
// NOT report a fake speedup curve.) Run at N=2 and N=4 to show full/N scaling.
//
// idx 0 = master; idx 1..N-1 = workers. Rendezvous via deterministic ids from a
// shared room. Transport = RelayTransport (hub-relayed; the path proven to cross
// cloud NAT for free).

const IDX = parseInt(process.env.IDX ?? "0", 10);     // 0 = master
const NTOTAL = parseInt(process.env.NTOTAL ?? "2", 10);
const ROOM = process.env.ROOM ?? "local-dev";
const ID = (i: number): string => `wgq-${ROOM}-${i}`;

// Benzene cc-pVDZ — big enough that the per-machine slice (full/N) is visible.
const BENZENE = (() => {
  const rCC = 1.395, rCH = 1.087;
  const a: Array<{ symbol: string; pos: [number, number, number] }> = [];
  for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; a.push({ symbol: "C", pos: [rCC * Math.cos(t), rCC * Math.sin(t), 0] }); }
  for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3; const r2 = rCC + rCH; a.push({ symbol: "H", pos: [r2 * Math.cos(t), r2 * Math.sin(t), 0] }); }
  return a;
})();

test.describe(`N=${NTOTAL} cross-machine distributed HF over a free relay`, () => {
  test(`node ${IDX}/${NTOTAL} — distributed DF-HF SCF energy matches single-machine`, async ({ page }) => {
    test.setTimeout(8 * 60 * 1000);
    page.on("pageerror", (e) => console.error(`[${IDX}:pageerror] ${e.message}`));
    page.on("console", (m) => { const t = m.text(); if (t.startsWith("[hfN]")) console.log(t); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const result = await page.evaluate(async ({ idx, ntot, room, atoms, masterId }) => {
      const log = (s: string): void => { console.log(`[hfN] ${s}`); };
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF, runRHFSCFAsync },
        { buildAuxBasisDFStreaming, generateAutoAux },
        { buildJK_DF, preloadWasmJK },
        { RelayTransport },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
        import("/src/parallel/swarm/relay-transport.ts" as string),
      ]);
      await preloadWasmJK();

      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const aux = generateAutoAux(shells, 1);
      const n = shells.length;
      const mySlice = await buildAuxBasisDFStreaming(shells, aux, 1e-10, { partition: { tab: idx, of: ntot } });
      const myDF = { B: mySlice.B, n, nAux: mySlice.nAux, threshold: 0 };
      const sliceMB = mySlice.B.byteLength / 1e6;
      const fullMB = (n * n * mySlice.nKeptTotal * 8) / 1e6;
      log(`node ${idx}: slice modes ${mySlice.nAux}/${mySlice.nKeptTotal}, ${sliceMB.toFixed(2)} MB (full ${fullMB.toFixed(1)} MB → /${ntot})`);

      const self = `wgq-${room}-${idx}`;
      const t = new RelayTransport({ room, id: self });
      try { await t.openAsync(); } catch (e) { return { ok: false, stage: "broker", error: String(e) }; }

      if (idx !== 0) {
        // Worker: announce readiness until engaged, then serve JK to the master.
        let served = 0; let done = false;
        t.onMessage((msg: { from: string; type: string; payload?: unknown }) => {
          if (msg.type === "D") {
            const D = Float64Array.from((msg.payload as { D: number[] }).D);
            const { J, K } = buildJK_DF(myDF, D);
            t.send({ to: masterId, type: "jk-partial", payload: { J: Array.from(J), K: Array.from(K) } });
            served++;
          } else if (msg.type === "done") { done = true; }
        });
        await new Promise<void>((resolve) => {
          const hi = setInterval(() => { if (!done && served === 0) t.send({ to: masterId, type: "ready" }); }, 1500);
          t.send({ to: masterId, type: "ready" });
          const poll = setInterval(() => { if (done) { clearInterval(hi); clearInterval(poll); resolve(); } }, 500);
          setTimeout(() => { clearInterval(hi); clearInterval(poll); resolve(); }, 360000);
        });
        t.close();
        return { ok: served > 0 && done, stage: "worker", served, sliceMB: Number(sliceMB.toFixed(2)) };
      }

      // Master: single-machine reference, then the distributed SCF.
      const fullB = await buildAuxBasisDFStreaming(shells, aux, 1e-10);
      const ref = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-8, densityTol: 1e-7, maxIter: 100, useDF: fullB }).energy;
      log(`single-machine reference E = ${ref.toFixed(8)}`);

      // Wait for all N-1 workers to be ready.
      const readied = new Set<string>();
      const allReady = await new Promise<boolean>((resolve) => {
        const off = t.onMessage((msg: { from: string; type: string }) => {
          if (msg.type === "ready") { readied.add(msg.from); if (readied.size >= ntot - 1) { off(); resolve(true); } }
        });
        setTimeout(() => resolve(readied.size >= ntot - 1), 180000);
      });
      if (!allReady) { t.close(); return { ok: false, stage: "handshake", error: `only ${readied.size}/${ntot - 1} workers ready` }; }
      log(`${readied.size} workers ready — starting distributed SCF`);

      let iters = 0;
      const t0 = Date.now();
      const customJKBuilder = async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
        iters++;
        const need = ntot - 1;
        const got = new Map<string, { J: number[]; K: number[] }>();
        const gather = new Promise<void>((resolve, reject) => {
          const off = t.onMessage((msg: { from: string; type: string; payload?: unknown }) => {
            if (msg.type === "jk-partial") {
              got.set(msg.from, msg.payload as { J: number[]; K: number[] });
              if (got.size >= need) { off(); resolve(); }
            }
          });
          setTimeout(() => { off(); reject(new Error(`gather timeout iter ${iters}: ${got.size}/${need}`)); }, 90000);
        });
        t.send({ type: "D", payload: { D: Array.from(D) } }); // broadcast to all workers
        const J = new Float64Array(n * n), K = new Float64Array(n * n);
        const mPart = buildJK_DF(myDF, D);
        J.set(mPart.J); K.set(mPart.K);
        await gather;
        for (const p of got.values()) for (let i = 0; i < J.length; i++) { J[i] = J[i]! + p.J[i]!; K[i] = K[i]! + p.K[i]!; }
        return { J, K };
      };

      let swEnergy = NaN;
      try {
        const sw = await runRHFSCFAsync(integrals, nElectrons, { useDIIS: true, energyTol: 1e-8, densityTol: 1e-7, maxIter: 100, parallel: 1, customJKBuilder });
        swEnergy = sw.energy;
      } catch (e) { t.send({ type: "done" }); t.close(); return { ok: false, stage: "scf", error: String(e) }; }
      const scfMs = Date.now() - t0;
      t.send({ type: "done" });
      await new Promise((r) => setTimeout(r, 2000));
      t.close();

      const dE = Math.abs(swEnergy - ref);
      log(`N=${ntot}: distributed E = ${swEnergy.toFixed(8)}, |ΔE| = ${dE.toExponential(2)}, ${iters} iters, ${(scfMs / 1000).toFixed(1)}s, per-machine ${sliceMB.toFixed(2)} MB`);
      return { ok: dE < 1e-7, stage: "scf", N: ntot, refEnergy: ref, swEnergy, deltaE: dE, iters, scfMs, sliceMB: Number(sliceMB.toFixed(2)), fullMB: Number(fullMB.toFixed(1)) };
    }, { idx: IDX, ntot: NTOTAL, room: ROOM, atoms: BENZENE, masterId: ID(0) });
console.log(`\n[hfN-relay node ${IDX}/${NTOTAL}] result:`, JSON.stringify(result), "\n");
    expect(result.ok, `stage=${result.stage} ${result.error ?? ""}`).toBe(true);
  });
});
