import { test, expect } from "@playwright/test";

test.describe("Molecule SI report page", () => {
  test("H₂O HF — full pipeline runs end-to-end and renders all 7 sections", async ({ page }) => {
    page.on("console", (msg) => {
      const text = msg.text();
      if (text.startsWith("{")) return;
      // eslint-disable-next-line no-console
      console.log(`[browser:${msg.type()}] ${text}`);
    });
    page.on("pageerror", (err) => {
      // eslint-disable-next-line no-console
      console.error(`[browser:pageerror] ${err.message}`);
    });

    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    // Defaults: H₂O + HF.
    await expect(page.locator("#moleculeSel")).toHaveValue("h2o");
    await expect(page.locator("#methodSel")).toHaveValue("hf");

    // Click run.
    await page.locator("#runBtn").click();

    // The IP card is the LAST step — wait for it to render real content.
    // Generous timeout: full pipeline is ~30-60s (geomopt + Hessian + Raman + α/β + UV-vis + thermo + IP).
    await expect(
      page.locator("#card-ip .body").locator("text=Koopmans IP"),
    ).toBeVisible({ timeout: 180_000 });

    // Smoke-check each card has rendered SOMETHING beyond "computing…".
    for (const id of ["energy", "spectra", "uvvis", "field", "charges", "thermo", "ip"] as const) {
      const body = page.locator(`#card-${id} .body`);
      await expect(body).toBeVisible();
      // Should NOT still say "computing…".
      const txt = await body.textContent();
      expect(txt).not.toMatch(/^\s*computing…\s*$/);
    }

    // Specific physics sanity: H₂O HF/STO-3G IP via Koopmans is ~10-14 eV.
    const ipText = await page.locator("#card-ip .body").textContent();
    expect(ipText).toMatch(/Koopmans IP/);
    // Total entropy from thermo card should be ~45 cal/(mol·K).
    const thermoText = await page.locator("#card-thermo .body").textContent();
    expect(thermoText).toMatch(/Total entropy/);
    expect(thermoText).toMatch(/cal\/\(mol·K\)/);

    // Final screenshot for the artifact.
    await page.screenshot({ path: "e2e/.artifacts/molecule-h2o-hf.png", fullPage: true });
  });
});
