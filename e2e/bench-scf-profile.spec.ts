import { test, expect } from "@playwright/test";

// Per-iter SCF wall-time breakdown on naphthalene cc-pVDZ.
// Tells us which phase to attack next: JK_DF build, F assembly,
// DIIS, F' transform, eigendecomp, or density update.

test.describe("SCF per-iter profiling", () => {
  test("naphthalene cc-pVDZ — phase breakdown", async ({ page }) => {
    test.setTimeout(10 * 60 * 1000);

    const profiles: Array<{ iter: number; ms: Record<string, number> }> = [];
    page.on("console", (msg) => {
      const t = msg.text();
      if (t.startsWith("[scf-profile]")) {
        const json = t.slice("[scf-profile]".length).trim();
        try { profiles.push(JSON.parse(json) as { iter: number; ms: Record<string, number> }); } catch { /* ignore */ }
      }
    });

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

      const ATOMS: Array<{ symbol: "C" | "H"; pos: readonly [number, number, number] }> = [
        { symbol: "C", pos: [-2.4225, -0.7176, 0] },
        { symbol: "C", pos: [-1.2451, -1.4032, 0] },
        { symbol: "C", pos: [ 0.0,    -0.7176, 0] },
        { symbol: "C", pos: [ 0.0,     0.7176, 0] },
        { symbol: "C", pos: [-1.2451,  1.4032, 0] },
        { symbol: "C", pos: [-2.4225,  0.7176, 0] },
        { symbol: "C", pos: [ 1.2451, -1.4032, 0] },
        { symbol: "C", pos: [ 2.4225, -0.7176, 0] },
        { symbol: "C", pos: [ 2.4225,  0.7176, 0] },
        { symbol: "C", pos: [ 1.2451,  1.4032, 0] },
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
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const auxShells = generateAutoAux(shells, 1);
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);

      const hf = await runRHFSCFAsync(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-6, densityTol: 1e-5, maxIter: 60,
        useDF: df, parallel: 8,
        profileCallback: (iter: number, ms: Record<string, number>) => {
          // eslint-disable-next-line no-console
          console.log("[scf-profile]", JSON.stringify({ iter, ms }));
        },
      });
      return { n: integrals.n, energy: hf.energy, iter: hf.iter, converged: hf.converged };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`SCF per-iter breakdown — naphthalene cc-pVDZ (n=${r.n})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`iter |   jk   | f_asm  | trans  |   eig   | dens  | total`);
    console.log(`-----+--------+--------+--------+---------+-------+-------`);
    for (const p of profiles) {
      const ms = p.ms;
      console.log(
        `${String(p.iter).padStart(4)} | ` +
        `${ms.jk!.toFixed(0).padStart(6)} | ` +
        `${ms.f_assemble!.toFixed(0).padStart(6)} | ` +
        `${ms.transform!.toFixed(0).padStart(6)} | ` +
        `${ms.eig!.toFixed(0).padStart(7)} | ` +
        `${ms.density!.toFixed(0).padStart(5)} | ` +
        `${ms.total!.toFixed(0).padStart(5)}`,
      );
    }
    // Aggregate columns
    const sum = (k: string): number => profiles.reduce((s, p) => s + p.ms[k]!, 0);
    const tot = sum("total");
    console.log(`-----+--------+--------+--------+---------+-------+-------`);
    console.log(
      ` SUM | ` +
      `${sum("jk").toFixed(0).padStart(6)} | ` +
      `${sum("f_assemble").toFixed(0).padStart(6)} | ` +
      `${sum("transform").toFixed(0).padStart(6)} | ` +
      `${sum("eig").toFixed(0).padStart(7)} | ` +
      `${sum("density").toFixed(0).padStart(5)} | ` +
      `${tot.toFixed(0).padStart(5)}`,
    );
    console.log(`     (% of total)`);
    console.log(
      `     | ` +
      `${(100*sum("jk")/tot).toFixed(0).padStart(5)}% | ` +
      `${(100*sum("f_assemble")/tot).toFixed(0).padStart(5)}% | ` +
      `${(100*sum("transform")/tot).toFixed(0).padStart(5)}% | ` +
      `${(100*sum("eig")/tot).toFixed(0).padStart(6)}% | ` +
      `${(100*sum("density")/tot).toFixed(0).padStart(4)}% |   100%`,
    );
    console.log();
    console.log(`E = ${r.energy.toFixed(8)} Ha   (iter=${r.iter}, conv=${r.converged})`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.converged).toBe(true);
    expect(profiles.length).toBeGreaterThan(0);
  });
});
