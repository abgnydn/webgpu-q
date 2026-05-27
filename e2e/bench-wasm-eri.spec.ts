import { test, expect } from "@playwright/test";

// Bench the Rust+WASM ERI kernel vs the sequential TypeScript ERI build.
// Goal: measure the native-compile-vs-JIT delta on the n⁴ ERI inner
// loop. Same algorithm, same Schwarz screening, same 8-fold symmetry —
// the only difference is the language the inner loop runs in.

test.describe("WASM ERI kernel", () => {
  test("ethane cc-pVDZ — TypeScript sequential vs Rust+WASM", async ({ page }) => {
    test.setTimeout(10 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { buildERIWasm },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/wasm-eri.ts" as string),
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

      // TS sequential reference.
      const tTs = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const tsMs = performance.now() - tTs;
      const eriTs = integrals.eri_AO;

      // WASM warmup (first call loads + initializes wasm).
      await buildERIWasm(shells, 1e-10);

      // WASM measurement.
      const tWasm = performance.now();
      const eriWasm = await buildERIWasm(shells, 1e-10);
      const wasmMs = performance.now() - tWasm;

      // Verify bit-equality (or as close as the two paths get).
      let maxDelta = 0;
      let maxRel = 0;
      for (let i = 0; i < eriTs.length; i++) {
        const d = Math.abs(eriTs[i]! - eriWasm[i]!);
        if (d > maxDelta) maxDelta = d;
        const refMag = Math.max(Math.abs(eriTs[i]!), 1e-300);
        const rel = d / refMag;
        if (rel > maxRel && refMag > 1e-10) maxRel = rel;
      }
      return { n: integrals.n, tsMs, wasmMs, maxDelta, maxRel };
    });

    /* eslint-disable no-console */
    console.log(`\n── WASM ERI vs TypeScript ERI — ethane cc-pVDZ (n=${r.n}) ──`);
    console.log(`TypeScript sequential:  ${r.tsMs.toFixed(0).padStart(7)} ms`);
    console.log(`Rust+WASM sequential:   ${r.wasmMs.toFixed(0).padStart(7)} ms  → ${(r.tsMs / r.wasmMs).toFixed(2)}× vs TS`);
    console.log(`max |Δ|: ${r.maxDelta.toExponential(2)} Ha    max relative diff: ${r.maxRel.toExponential(2)}`);
    /* eslint-enable no-console */

    // Bit-level match: WASM and TS implement the same algorithm; expect ≤ 1e-10.
    expect(r.maxDelta).toBeLessThan(1e-10);
  });
});
