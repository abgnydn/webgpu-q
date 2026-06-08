import { test, expect } from "@playwright/test";

// N=2 CROSS-MACHINE distributed Hartree–Fock over a free public relay.
//
// Two SEPARATE GitHub Actions VMs each build only their own mode-slice of the
// density-fitting tensor (no full tensor on either), then run a distributed
// DF-HF SCF: every iteration the master broadcasts the density D through a free
// MQTT broker, each VM computes its partial (J,K) on its slice, the worker
// returns its partial through the broker, the master sums and drives the SCF.
// The converged energy must match the single-machine reference.
//
// This is the step that turns "we can coordinate across machines" into "the
// cluster computes a molecule." Rendezvous via deterministic ids from the shared
// run id; transport = RelayTransport (hub-relayed, the path proven to cross
// cloud NAT for free — WebRTC P2P needs a TURN key).

const ROLE = process.env.ROLE ?? "a";          // a=master, b=worker
const ROOM = process.env.ROOM ?? "local-dev";
const ID_A = `wgq-${ROOM}-a`;
const ID_B = `wgq-${ROOM}-b`;

const H2O_ATOMS = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half), z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
})();

test.describe("N=2 cross-machine distributed HF over a free relay", () => {
  test(`role ${ROLE} — distributed DF-HF SCF energy matches single-machine`, async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    page.on("pageerror", (e) => console.error(`[${ROLE}:pageerror] ${e.message}`));
    page.on("console", (m) => { const t = m.text(); if (t.startsWith("[hf2]")) console.log(t); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const result = await page.evaluate(async ({ role, room, idA, idB, atoms }) => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[hf2] ${s}`); };
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
      const tab = role === "a" ? 0 : 1;
      const mySlice = await buildAuxBasisDFStreaming(shells, aux, 1e-10, { partition: { tab, of: 2 } });
      const myDF = { B: mySlice.B, n, nAux: mySlice.nAux, threshold: 0 };
      log(`built slice: modes ${mySlice.nAux}/${mySlice.nKeptTotal}, ${(mySlice.B.byteLength / 1e6).toFixed(2)} MB`);

      const self = role === "a" ? idA : idB;
      const t = new RelayTransport({ room, id: self });
      try {
        await t.openAsync();
      } catch (e) { return { ok: false, stage: "broker", error: String(e) }; }
      log(`relay connected as ${self}`);

      if (role === "b") {
        // Worker: announce readiness until the master engages, then serve JK.
        let served = 0; let done = false;
        t.onMessage((msg: { from: string; type: string; payload?: unknown }) => {
          if (msg.type === "D") {
            const D = Float64Array.from((msg.payload as { D: number[] }).D);
            const { J, K } = buildJK_DF(myDF, D);
            t.send({ to: idA, type: "jk-partial", payload: { J: Array.from(J), K: Array.from(K) } });
            served++;
          } else if (msg.type === "done") { done = true; }
        });
        await new Promise<void>((resolve) => {
          const hi = setInterval(() => { if (!done && served === 0) t.send({ to: idA, type: "ready" }); }, 1500);
          t.send({ to: idA, type: "ready" });
          const poll = setInterval(() => { if (done) { clearInterval(hi); clearInterval(poll); resolve(); } }, 500);
          setTimeout(() => { clearInterval(hi); clearInterval(poll); resolve(); }, 240000);
        });
        t.close();
        return { ok: served > 0 && done, stage: "worker", served };
      }

      // Master: single-machine reference, then the distributed SCF.
      const fullB = await buildAuxBasisDFStreaming(shells, aux, 1e-10);
      const ref = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-8, densityTol: 1e-7, maxIter: 100, useDF: fullB }).energy;
      log(`single-machine reference E = ${ref.toFixed(8)}`);

      // Wait for the worker to be ready.
      const ready = await new Promise<boolean>((resolve) => {
        const off = t.onMessage((msg: { from: string; type: string }) => { if (msg.type === "ready") { off(); resolve(true); } });
        setTimeout(() => resolve(false), 120000);
      });
      if (!ready) { t.close(); return { ok: false, stage: "handshake", error: "worker never readied" }; }
      log("worker ready — starting distributed SCF");

      let iters = 0;
      const customJKBuilder = async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
        iters++;
        const workerPartial = new Promise<{ J: Float64Array; K: Float64Array }>((resolve, reject) => {
          const off = t.onMessage((msg: { from: string; type: string; payload?: unknown }) => {
            if (msg.type === "jk-partial") {
              off();
              const p = msg.payload as { J: number[]; K: number[] };
              resolve({ J: Float64Array.from(p.J), K: Float64Array.from(p.K) });
            }
          });
          setTimeout(() => { off(); reject(new Error(`worker jk timeout (iter ${iters})`)); }, 60000);
        });
        t.send({ to: idB, type: "D", payload: { D: Array.from(D) } });
        const mPart = buildJK_DF(myDF, D);
        const wPart = await workerPartial;
        const J = new Float64Array(n * n), K = new Float64Array(n * n);
        for (let i = 0; i < J.length; i++) { J[i] = mPart.J[i]! + wPart.J[i]!; K[i] = mPart.K[i]! + wPart.K[i]!; }
        return { J, K };
      };

      let swEnergy = NaN;
      try {
        const sw = await runRHFSCFAsync(integrals, nElectrons, {
          useDIIS: true, energyTol: 1e-8, densityTol: 1e-7, maxIter: 100, parallel: 1, customJKBuilder,
        });
        swEnergy = sw.energy;
      } catch (e) { t.send({ to: idB, type: "done" }); t.close(); return { ok: false, stage: "scf", error: String(e) }; }
      t.send({ to: idB, type: "done" });
      // give the broker a beat to deliver "done" before tearing down
      await new Promise((r) => setTimeout(r, 1500));
      t.close();

      const dE = Math.abs(swEnergy - ref);
      log(`distributed E = ${swEnergy.toFixed(8)}, |ΔE| vs single-machine = ${dE.toExponential(2)} (${iters} iters)`);
      return { ok: dE < 1e-7, stage: "scf", refEnergy: ref, swEnergy, deltaE: dE, iters };
    }, { role: ROLE, room: ROOM, idA: ID_A, idB: ID_B, atoms: H2O_ATOMS });

    console.log(`\n[hf2-relay ${ROLE}] result:`, JSON.stringify(result), "\n");
    expect(result.ok, `stage=${result.stage} ${result.error ?? ""}`).toBe(true);
  });
});
