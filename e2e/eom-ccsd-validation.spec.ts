// E2E test for E35 — EOM-CCSD cross-validation vs PySCF.
//
// Runs E35 (6 small organics × HF → CCSD → EE-EOM-CCSD) in
// headless WebGPU Chromium and saves the artifact JSON for offline
// comparison against scripts/run-pyscf-eom-reference.py output.
//
// Pass bar: every molecule converges through the pipeline. Excitation
// energies are recorded as data — the merge-and-diff step is offline.
import * as fs from "node:fs";
import * as path from "node:path";
import { expect, test } from "@playwright/test";
import { bootExperimentsPage } from "./lib/run-level.js";

test.describe("E35 EOM-CCSD validation", () => {
  test("runs HF → CCSD → EOM-CCSD on 6 small organics at STO-3G", async ({ page }) => {
    test.setTimeout(30 * 60 * 1000);
    await bootExperimentsPage(page);
    const artifact = await page.evaluate(async () => {
      const hook = window.__webgpuq;
      if (!hook) throw new Error("window.__webgpuq missing");
      return await hook.runE35();
    });
console.log(`[e2e:E35] status=${artifact.status} — ${artifact.diagnosis}`);
    for (const r of artifact.rows) {
      if (!r.success) { console.log(`  ✗ ${r.molecule}: ${r.notes ?? "failed"}`);
        continue;
      }
      const exc = r.excitations_eV.map((e: number) => e.toFixed(3)).join("  ");
      console.log(
        `  ✓ ${r.molecule.padEnd(5)} HF=${r.E_HF_Ha.toFixed(6)}  ` +
        `CCSD=${r.E_CCSD_Ha.toFixed(6)}  excitations (eV): ${exc}`,
      );
    }

    expect(artifact.status).toBe("pass");
    expect(artifact.rows.length).toBeGreaterThan(0);

    for (const r of artifact.rows) {
      expect(r.success).toBe(true);
      expect(Number.isFinite(r.E_HF_Ha)).toBe(true);
      expect(Number.isFinite(r.E_CCSD_Ha)).toBe(true);
      expect(r.excitations_eV.length).toBeGreaterThan(0);
      // All returned excitations should be real, positive, and finite.
      for (const e of r.excitations_eV) {
        expect(Number.isFinite(e)).toBe(true);
        expect(e).toBeGreaterThan(0);
      }
    }

    const dateUTC = new Date().toISOString().slice(0, 10);
    const outDir = path.resolve("experiments/results", dateUTC, "level-6");
    fs.mkdirSync(outDir, { recursive: true });
    const outPath = path.join(outDir, "E35-eom-ccsd-validation.json");
    fs.writeFileSync(outPath, JSON.stringify(artifact, null, 2));
    console.log(`[e2e:E35] wrote ${outPath}`);
  });
});
