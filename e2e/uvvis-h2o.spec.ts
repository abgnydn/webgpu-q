// E2E test for E33 — H₂O EOM-CCSD UV-vis spectrum (Tier 2 stage 36).
import { expect, test } from "@playwright/test";
import { bootExperimentsPage } from "./lib/run-level.js";

test.describe("H₂O EOM-CCSD UV-vis (E33)", () => {
  test("STO-3G excitations: real, dipole-allowed singlet present, ordering correct", async ({ page }) => {
    test.setTimeout(2 * 60 * 1000);
    await bootExperimentsPage(page);
    const artifact = await page.evaluate(async () => {
      const hook = window.__webgpuq;
      if (!hook) throw new Error("window.__webgpuq missing");
      return await hook.runE33();
    });
     
    console.log(`[e2e:E33] status=${artifact.status} — ${artifact.diagnosis}`);
    for (const r of artifact.rows.slice(0, 10)) {
       
      console.log(
        `  root ${r.root}: ${r.energyEV.toFixed(3)} eV  f=${r.oscillator.toExponential(2)}  ${r.assignment}`,
      );
    }
    expect(artifact.status).toBe("pass");
    expect(artifact.rows.length).toBeGreaterThan(0);
    // At least one row classified as singlet with f > 0.05.
    const dipoleAllowed = artifact.rows.filter(
      (r) => r.assignment === "singlet" && r.oscillator > 1e-4,
    );
    expect(dipoleAllowed.length).toBeGreaterThan(0);
    // Every imaginary part at roundoff.
    for (const r of artifact.rows) expect(Math.abs(r.imag)).toBeLessThan(1e-6);
  });
});
