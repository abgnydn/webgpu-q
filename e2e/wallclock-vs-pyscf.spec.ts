// E2E test for E34 — wall-clock vs PySCF reference.
//
// Runs E34 (4 molecules × 2 basis × {HF, MP2, CCSD, CCSD(T) CPU/GPU})
// in a headless WebGPU Chromium and asserts:
//   - every row either converges or is explicitly skipped (no throws)
//   - all energies are finite where expected
//   - artifact JSON is well-formed for downstream merging with the
//     PySCF reference script output
//
// Performance numbers are recorded but NOT pass/fail-gated — wall-clock
// is informational data, not a regression target. We don't want flaky
// CI when an Apple Silicon runner is busier than usual.
import * as fs from "node:fs";
import * as path from "node:path";
import { expect, test } from "@playwright/test";
import { bootExperimentsPage } from "./lib/run-level.js";

test.describe("E34 wall-clock vs PySCF", () => {
  test("runs HF → MP2 → CCSD → CCSD(T) on 4 molecules × 2 basis sets", async ({ page }) => {
    test.setTimeout(30 * 60 * 1000);
    await bootExperimentsPage(page);
    const artifact = await page.evaluate(async () => {
      const hook = window.__webgpuq;
      if (!hook) throw new Error("window.__webgpuq missing");
      return await hook.runE34();
    });
console.log(`[e2e:E34] status=${artifact.status} — ${artifact.diagnosis}`);
    for (const r of artifact.rows) {
      const flag = r.success ? "✓" : "✗";
      const energy = isNaN(r.energy_Ha) ? "       (skip)" : r.energy_Ha.toFixed(8);
      const notes = r.notes ? `   ${r.notes}` : "";
       console.log(
        `  ${flag} ${r.molecule.padEnd(5)} ${r.basis.padEnd(8)} ${r.method.padEnd(14)} ` +
        `${r.seconds.toFixed(4).padStart(10)} s   E = ${energy} Ha${notes}`,
      );
    }

    expect(artifact.status).toBe("pass");
    expect(artifact.rows.length).toBeGreaterThan(0);

    // Save the artifact as evidence under experiments/results/<UTC date>/level-6/.
    const dateUTC = new Date().toISOString().slice(0, 10);
    const outDir = path.resolve("experiments/results", dateUTC, "level-6");
    fs.mkdirSync(outDir, { recursive: true });
    const outPath = path.join(outDir, "E34-wallclock-vs-pyscf.json");
    fs.writeFileSync(outPath, JSON.stringify(artifact, null, 2));
     console.log(`[e2e:E34] wrote ${outPath}`);

    // Every row either succeeded or carries an explicit notes field
    // explaining the skip — never a silent failure.
    for (const r of artifact.rows) {
      if (!r.success) {
        throw new Error(`E34 row failed without an explicit skip note: ${JSON.stringify(r)}`);
      }
      // Energies must be finite for rows that actually ran (non-skipped).
      if (r.notes?.startsWith("skipped")) continue;
      if (r.notes?.startsWith("not run")) continue;
      if (r.notes?.startsWith("cc-pVDZ")) continue;
      // Trivial-case CCSD(T) on H₂ (NOCC < 3 spin-orbitals → no triples)
      // returns seconds=0 by short-circuit. Energy is still finite.
      expect(Number.isFinite(r.energy_Ha)).toBe(true);
      // seconds is informational, not gated — wall-clock varies and
      // can be 0 for short-circuited algorithmic edge cases.
      expect(r.seconds).toBeGreaterThanOrEqual(0);
    }
  });
});
