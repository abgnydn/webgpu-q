import { test, expect } from "@playwright/test";

// Naphthalene cc-pVDZ (n≈190): the first molecule where direct n⁴ ERI
// (~10 GB) exceeds the browser tab heap. aux-DF Cholesky should fit
// (B tensor ~200 MB) and converge.
//
// Proof of concept that aux-DF unlocks new territory beyond what
// direct can reach.

test.describe("Naphthalene cc-pVDZ — direct OOMs, aux-DF unlocks", () => {
  test("aux-DF Cholesky HF on naphthalene", async ({ page }) => {
    test.setTimeout(20 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCFAsync },
        { buildAuxBasisDFCholesky, generateAutoAux },
        { preloadWasmJK },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
      ]);
      await preloadWasmJK();

      // Naphthalene (C10H8) at equilibrium, planar in z=0, Å.
      // Bridgehead carbons C8a, C4a at center-of-ring axis (0, ±0.7176).
      // Approximate experimental geometry.
      const ATOMS: Array<{ symbol: "C" | "H"; pos: readonly [number, number, number] }> = [
        // 10 carbons — two fused hexagons sharing the (C8a-C4a) edge.
        { symbol: "C", pos: [-2.4225, -0.7176, 0] }, // C1
        { symbol: "C", pos: [-1.2451, -1.4032, 0] }, // C2
        { symbol: "C", pos: [ 0.0,    -0.7176, 0] }, // C8a
        { symbol: "C", pos: [ 0.0,     0.7176, 0] }, // C4a
        { symbol: "C", pos: [-1.2451,  1.4032, 0] }, // C3 (← but it's on the LEFT ring upper edge)
        { symbol: "C", pos: [-2.4225,  0.7176, 0] }, // C4
        { symbol: "C", pos: [ 1.2451, -1.4032, 0] }, // C5
        { symbol: "C", pos: [ 2.4225, -0.7176, 0] }, // C6
        { symbol: "C", pos: [ 2.4225,  0.7176, 0] }, // C7
        { symbol: "C", pos: [ 1.2451,  1.4032, 0] }, // C8
        // 8 hydrogens on non-bridgehead carbons.
        // H displaced outward (away from ring centers) by 1.09 Å.
        // Ring center 1 at (-1.245, 0), ring center 2 at (+1.245, 0).
        // For each non-bridgehead C, dir = (C - ring_center)/|C - ring_center| × 1.09.
        ...([[-2.4225, -0.7176], [-1.2451, -1.4032], [-1.2451, 1.4032], [-2.4225, 0.7176],
            [1.2451, -1.4032], [2.4225, -0.7176], [2.4225, 0.7176], [1.2451, 1.4032]] as const)
          .map((c) => {
            const cx: number = c[0]; const cy: number = c[1];
            const rcX = cx < 0 ? -1.245 : 1.245;
            const dx = cx - rcX;
            const len = Math.sqrt(dx * dx + cy * cy);
            return { symbol: "H" as const, pos: [cx + 1.09 * dx / len, cy + 1.09 * cy / len, 0] as const };
          }),
      ];

      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(ATOMS as never, "cc-pvdz");

      // Skip direct ERI build entirely. Direct path can't fit anyway.
      const tInteg = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const integMs = performance.now() - tInteg;

      const auxShells = generateAutoAux(shells, 1);
      const tDF = performance.now();
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);
      const dfMs = performance.now() - tDF;

      const tHF = performance.now();
      const hf = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 100,
        useDF: df, parallel: 8,
      });
      const hfMs = performance.now() - tHF;

      return {
        n: integrals.n, nAuxIn: auxShells.length, nAuxKept: df.nAux,
        nElectrons, integMs, dfMs, hfMs,
        energy: hf.energy, iter: hf.iter, converged: hf.converged,
      };
    });

    const totalMs = r.integMs + r.dfMs + r.hfMs;
     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Naphthalene C₁₀H₈ cc-pVDZ aux-DF HF SCF`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`n_orb = ${r.n}    n_electrons = ${r.nElectrons}`);
    console.log(`n_aux_in = ${r.nAuxIn}   Cholesky rank = ${r.nAuxKept}`);
    console.log();
    console.log(`Direct ERI would be: ${(r.n ** 4 * 8 / 1e9).toFixed(1)} GB (does not fit)`);
    console.log(`B-tensor footprint:  ${(r.n * r.n * r.nAuxKept * 8 / 1e6).toFixed(0)} MB (fits)`);
    console.log();
    console.log(`Integrals (skipERI+skipOAO):  ${(r.integMs / 1000).toFixed(2).padStart(7)} s`);
    console.log(`Aux-DF B-tensor (Cholesky):   ${(r.dfMs / 1000).toFixed(2).padStart(7)} s`);
    console.log(`HF SCF (${r.iter} iters):        ${(r.hfMs / 1000).toFixed(2).padStart(7)} s   converged=${r.converged}`);
    console.log(`Total end-to-end:             ${(totalMs / 1000).toFixed(2).padStart(7)} s`);
    console.log();
    console.log(`E = ${r.energy.toFixed(8)} Ha`);
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    expect(Number.isFinite(r.energy)).toBe(true);
    expect(r.converged).toBe(true);
  });
});
