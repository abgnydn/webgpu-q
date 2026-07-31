// E2E test for the WebGPU CCSD(T) port (Tier 2 stage 27).
//
// Runs E32 (CCSD(T) GPU vs CPU cross-check on LiH / BeH₂ / H₂O
// STO-3G) in a headless WebGPU Chromium and asserts every row
// passes the |E_(T)_GPU − E_(T)_CPU| ≤ 5e-6 Ha bar.
import { expect, test } from "@playwright/test";
import { bootExperimentsPage } from "./lib/run-level.js";

test.describe("CCSD(T) GPU port", () => {
  test("E32 cross-checks GPU against CPU within f32 noise", async ({ page }) => {
    test.setTimeout(15 * 60 * 1000);
    await bootExperimentsPage(page);
    const artifact = await page.evaluate(async () => {
      const hook = window.__webgpuq;
      if (!hook) throw new Error("window.__webgpuq missing");
      return await hook.runE32();
    });
     console.log(`[e2e:E32] status=${artifact.status} — ${artifact.diagnosis}`);
    for (const r of artifact.rows) { console.log(`  ${r.molecule}: CPU ${r.cpuET.toExponential(4)}, GPU ${r.gpuET.toExponential(4)}, |Δ| ${r.deltaHartree.toExponential(2)} Ha, speedup ${r.speedup.toFixed(1)}×`);
    }
    expect(artifact.status).toBe("pass");
    for (const r of artifact.rows) {
      expect(r.pass).toBe(true);
      expect(r.deltaHartree).toBeLessThan(5e-6);
    }
  });
});
