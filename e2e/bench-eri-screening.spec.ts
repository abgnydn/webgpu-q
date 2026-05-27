import { test, expect } from "@playwright/test";

// Measure ERI screening hit-rate and the wall-time impact of tightening
// / loosening the Schwarz threshold. Goal: find out whether our current
// 1e-10 threshold is leaving free speedup on the table.

test.describe("ERI screening — hit-rate + threshold sweep", () => {
  test("ethane cc-pVDZ — Schwarz threshold sweep, 1e-12 → 1e-6", async ({ page }) => {
    test.setTimeout(15 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [{ moleculeToShellsNuclei }, { computeMolecularIntegrals }, { runRHFSCF }] =
        await Promise.all([
          import("/src/chemistry/atoms.ts" as string),
          import("/src/chemistry/cg-molecular.ts" as string),
          import("/src/chemistry/hf-scf.ts" as string),
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
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      const rows: Array<{
        tol: number; eriMs: number; hfMs: number;
        computed: number; skipped: number; totalUnique: number;
        energy: number;
      }> = [];
      const tols = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4];
      for (const tol of tols) {
        const stats = { computed: 0, skipped: 0, totalUnique: 0 };
        const tEri = performance.now();
        const integrals = computeMolecularIntegrals(shells, nuclei,
          { screenStats: stats, schwarzTol: tol });
        const eriMs = performance.now() - tEri;
        const tHf = performance.now();
        const hf = runRHFSCF(integrals, nElectrons, {
          useDIIS: true, energyTol: 1e-9, densityTol: 1e-7, maxIter: 200,
        });
        const hfMs = performance.now() - tHf;
        rows.push({
          tol, eriMs, hfMs,
          computed: stats.computed, skipped: stats.skipped,
          totalUnique: stats.totalUnique,
          energy: hf.energy,
        });
      }
      return { n: rows[0]!.totalUnique > 0 ? 60 : 0, rows };
    });

    /* eslint-disable no-console */
    console.log(`\n── Schwarz screening sweep — ethane cc-pVDZ ──`);
    console.log("threshold      | unique-pairs       computed      skipped    skip%    ERI build    HF run     total      energy");
    console.log("─────────────────────────────────────────────────────────────────────────────────────────────────────────────────");
    const e0 = r.rows[1]!.energy; // 1e-10 reference
    for (const row of r.rows) {
      const skipPct = (100 * row.skipped / row.totalUnique).toFixed(1);
      const dE = Math.abs(row.energy - e0).toExponential(2);
      console.log(
        `${row.tol.toExponential(0).padStart(8)}     | ${
          row.totalUnique.toString().padStart(11)
        }   ${row.computed.toString().padStart(10)}   ${
          row.skipped.toString().padStart(10)
        }   ${skipPct.padStart(5)}%   ${
          row.eriMs.toFixed(0).padStart(6)
        } ms   ${row.hfMs.toFixed(0).padStart(5)} ms   ${
          (row.eriMs + row.hfMs).toFixed(0).padStart(6)
        } ms   |ΔE|=${dE} Ha`,
      );
    }
    /* eslint-enable no-console */

    // Energy at the 1e-6 threshold should still be within chemical accuracy.
    const e_loose = r.rows[r.rows.length - 1]!.energy;
    expect(Math.abs(e_loose - e0)).toBeLessThan(2e-3);
  });
});
