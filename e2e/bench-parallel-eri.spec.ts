import { test, expect } from "@playwright/test";

// Bench parallel ERI build vs sequential — closes the bottleneck the
// pair-cache exposed. The ERI build is now the wall-time dominant cost
// at n=60+ (sequential: 24 s ethane, 740 s benzene). Workers partition
// over outer μ; canonical encoding guarantees no race condition.

test.describe("Parallel ERI build", () => {
  test("benzene cc-pVDZ — sequential vs parallel=8 (single trial)", async ({ page }) => {
    test.setTimeout(30 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { buildERIParallel },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/parallel-eri.ts" as string),
      ]);
      const rCC = 1.395, rCH = 1.087;
      const atoms: Array<{ symbol: string; pos: readonly [number, number, number] }> = [];
      for (let i = 0; i < 6; i++) {
        const θ = i * Math.PI / 3;
        atoms.push({ symbol: "C", pos: [rCC * Math.cos(θ), rCC * Math.sin(θ), 0] });
      }
      for (let i = 0; i < 6; i++) {
        const θ = i * Math.PI / 3;
        const r2 = rCC + rCH;
        atoms.push({ symbol: "H", pos: [r2 * Math.cos(θ), r2 * Math.sin(θ), 0] });
      }
      const { shells, nuclei } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      const tSeq = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const seqMs = performance.now() - tSeq;
      const eriSeq = integrals.eri_AO;

      const tPar = performance.now();
      const eriPar = await buildERIParallel(shells, integrals.n, 1e-10, 8);
      const parMs = performance.now() - tPar;
      let maxDelta = 0;
      for (let i = 0; i < eriPar.length; i++) {
        const d = Math.abs(eriPar[i]! - eriSeq[i]!);
        if (d > maxDelta) maxDelta = d;
      }
      return { n: integrals.n, seqMs, parMs, maxDelta };
    });

     console.log(`\n── Parallel ERI build — benzene cc-pVDZ (n=${r.n}) ──`);
    console.log(`sequential:       ${(r.seqMs / 1000).toFixed(1).padStart(6)} s`);
    console.log(`parallel=8:       ${(r.parMs / 1000).toFixed(1).padStart(6)} s  → ${(r.seqMs / r.parMs).toFixed(2)}× vs sequential`);
    console.log(`max |Δ|: ${r.maxDelta.toExponential(2)} Ha`);
     
    expect(r.maxDelta).toBeLessThan(1e-12);
  });

  test("ethane cc-pVDZ — sequential vs parallel (N=2,4,8)", async ({ page }) => {
    test.setTimeout(10 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { buildERIParallel },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/parallel-eri.ts" as string),
      ]);
      const cc = 1.535, ch = 1.094;
      const hch = 107.8 * Math.PI / 180;
      const hcc = (Math.PI - hch) / 2 + Math.PI / 6;
      const sH = ch * Math.sin(hcc), cH = ch * Math.cos(hcc);
      const atoms = [
        { symbol: "C", pos: [0, 0, -cc / 2] },
        { symbol: "C", pos: [0, 0,  cc / 2] },
        { symbol: "H", pos: [ sH, 0, -cc / 2 - cH] },
        { symbol: "H", pos: [-sH / 2,  sH * Math.sqrt(3) / 2, -cc / 2 - cH] },
        { symbol: "H", pos: [-sH / 2, -sH * Math.sqrt(3) / 2, -cc / 2 - cH] },
        { symbol: "H", pos: [-sH, 0,  cc / 2 + cH] },
        { symbol: "H", pos: [ sH / 2,  sH * Math.sqrt(3) / 2,  cc / 2 + cH] },
        { symbol: "H", pos: [ sH / 2, -sH * Math.sqrt(3) / 2,  cc / 2 + cH] },
      ];
      const { shells, nuclei } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      // Sequential reference.
      const tSeq = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const seqMs = performance.now() - tSeq;
      const eriSeq = integrals.eri_AO;

      // Parallel at N = 2, 4, 8.
      const parResults: Record<string, { ms: number; maxDelta: number }> = {};
      for (const N of [2, 4, 8] as const) {
        const t0 = performance.now();
        const eriPar = await buildERIParallel(shells, integrals.n, 1e-10, N);
        const ms = performance.now() - t0;
        // Verify element-wise match to sequential.
        let maxDelta = 0;
        for (let i = 0; i < eriPar.length; i++) {
          const d = Math.abs(eriPar[i]! - eriSeq[i]!);
          if (d > maxDelta) maxDelta = d;
        }
        parResults[`parallel=${N}`] = { ms, maxDelta };
      }

      return { n: integrals.n, seqMs, parResults };
    });

     console.log(`\n── Parallel ERI build — ethane cc-pVDZ (n=${r.n}) ──`);
    console.log(`sequential:       ${r.seqMs.toFixed(0).padStart(7)} ms`);
    for (const [name, p] of Object.entries(r.parResults)) {
      const speedup = (r.seqMs / p.ms).toFixed(2);
      console.log(`${name.padEnd(15)} ${p.ms.toFixed(0).padStart(7)} ms  → ${speedup.padStart(5)}× vs sequential   max |Δ| = ${p.maxDelta.toExponential(2)} Ha`);
    }
     

    // Correctness: parallel must bit-match sequential.
    for (const p of Object.values(r.parResults)) {
      expect(p.maxDelta).toBeLessThan(1e-12);
    }
  });
});
