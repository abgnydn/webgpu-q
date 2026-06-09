import { test, expect } from "@playwright/test";

// CAPSTONE: a molecule whose exact 4-index ERI cannot fit in a browser tab, run
// to a converged Hartree-Fock energy anyway — in the tab — via runRHFAuto's
// automatic density-fitting path with the hybrid GPU/WASM integral build.
//
// Naphthalene C10H8 / cc-pVDZ: n ≈ 190 → exact ERI = 190⁴·8 B ≈ 9.7 GB. That
// allocation simply fails in a browser. DF never builds it (V + B are ~n²·n_aux,
// hundreds of MB), so the calculation runs. This is the payoff of the whole
// GPU-DF arc: chemistry-grade HF on a real PAH, from a URL.
//
// NOTE: this is a CI-tier capstone (~8 min). The engine here is the all-WASM
// streaming DF, by design: at n=190 buildDFForRegime gates the GPU hybrid off
// (its per-block JS merge, O(n²·nAux), runs >2× slower than WASM-SIMD streaming
// at PAH scale — the GPU hybrid is a MEDIUM-molecule optimization, benzene
// 1.31× V-build). DF is what makes the molecule feasible regardless of engine —
// the test asserts that (method = density-fitting, ERI never built), not speed.

// Planar D2h naphthalene (Å), reasonable geometry (feasibility demo, not a
// precision benchmark — we assert a sane physical energy, not a literature digit).
const NAPHTHALENE: { symbol: string; pos: number[] }[] = [
  { symbol: "C", pos: [0.0, 0.716, 0.0] },
  { symbol: "C", pos: [0.0, -0.716, 0.0] },
  { symbol: "C", pos: [1.241, 1.403, 0.0] },
  { symbol: "C", pos: [1.241, -1.403, 0.0] },
  { symbol: "C", pos: [-1.241, 1.403, 0.0] },
  { symbol: "C", pos: [-1.241, -1.403, 0.0] },
  { symbol: "C", pos: [2.433, 0.710, 0.0] },
  { symbol: "C", pos: [2.433, -0.710, 0.0] },
  { symbol: "C", pos: [-2.433, 0.710, 0.0] },
  { symbol: "C", pos: [-2.433, -0.710, 0.0] },
  { symbol: "H", pos: [1.235, 2.489, 0.0] },
  { symbol: "H", pos: [1.235, -2.489, 0.0] },
  { symbol: "H", pos: [-1.235, 2.489, 0.0] },
  { symbol: "H", pos: [-1.235, -2.489, 0.0] },
  { symbol: "H", pos: [3.376, 1.247, 0.0] },
  { symbol: "H", pos: [3.376, -1.247, 0.0] },
  { symbol: "H", pos: [-3.376, 1.247, 0.0] },
  { symbol: "H", pos: [-3.376, -1.247, 0.0] },
];

test.describe("Capstone — naphthalene cc-pVDZ HF in a tab (exact ERI infeasible)", () => {
  test("runRHFAuto auto-picks DF, skips the 8 GB ERI, converges to a sane energy", async ({ page }) => {
    test.setTimeout(15 * 60 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[cap]")) console.log(m.text()); });
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async (atoms: { symbol: string; pos: number[] }[]) => {
      const log = (s: string): void => { /* eslint-disable-next-line no-console */ console.log(`[cap] ${s}`); };
      const [{ moleculeToShellsNuclei }, { runRHFAuto }] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/rhf-auto.ts" as string),
      ]);
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const n = shells.length;
      const eriBytes = Math.pow(n, 4) * 8;          // the allocation DF avoids
      const eriGB = eriBytes / 1024 / 1024 / 1024;

      const t0 = performance.now();
      const res = await runRHFAuto(shells as never, nuclei as never, nElectrons, { fast: true });
      const sec = (performance.now() - t0) / 1000;

      log(`naphthalene n=${n}, ${nElectrons} e⁻: would-be exact ERI = ${eriGB.toFixed(1)} GB (skipped)`);
      log(`  HF = ${res.hf.energy.toFixed(6)} Ha [${res.provenance.method}/${res.provenance.engine}/${res.provenance.precision}, nAux=${res.provenance.nAux}], ${res.hf.iter} iters, ${sec.toFixed(1)}s, converged=${res.hf.converged}`);
      log(`  provenance.expectedError: ${res.provenance.expectedError}`);
      return {
        n, eriGB, energy: res.hf.energy, prov: res.provenance,
        converged: res.hf.converged, iters: res.hf.iter, eriLen: res.integrals.eri_AO.length, sec,
      };
    }, NAPHTHALENE);

    console.log(`\n[naphthalene-capstone] ${JSON.stringify(r)}\n`);
    expect(r.n).toBeGreaterThan(150);             // genuinely large
    expect(r.eriGB).toBeGreaterThan(4);           // exact ERI is multi-GB → tab-infeasible
    expect(r.prov.method).toBe("density-fitting"); // auto-gate chose DF
    expect(r.eriLen).toBe(0);                      // the 8 GB ERI was never built
    expect(r.converged).toBe(true);               // and it actually converged
    // Sane physical RHF/cc-pVDZ energy for naphthalene (~ -383 Ha at this geom).
    expect(r.energy).toBeLessThan(-380);
    expect(r.energy).toBeGreaterThan(-386);
  });
});
