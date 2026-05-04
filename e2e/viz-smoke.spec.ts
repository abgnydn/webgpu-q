import { test, expect } from "@playwright/test";

test.describe("Viz page", () => {
  test("all panels render; TFIM phase-transition mode works", async ({ page }) => {
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

    await page.goto("/viz.html", { waitUntil: "domcontentloaded" });

    const statusL = page.locator("#statusL");
    const statusR = page.locator("#statusR");
    const statusNet = page.locator("#statusNet");
    await expect(statusL).not.toHaveText("init…", { timeout: 15_000 });
    await expect(statusR).not.toHaveText("init…", { timeout: 5_000 });
    await expect(statusNet).not.toHaveText("init…", { timeout: 5_000 });

    async function setRange(id: string, v: string): Promise<void> {
      await page.locator(`#${id}`).evaluate((el, val) => {
        const input = el as HTMLInputElement;
        input.value = val;
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }, v);
    }

    await page.screenshot({ path: "e2e/.artifacts/viz-phase5-brickwall.png", fullPage: false });

    // Switch to TFIM ground-state mode.
    await page.locator("#model").selectOption("tfim");
    await page.waitForTimeout(2500);   // imag-time evolution takes ~1s for N=12, h=1
    const tfimStatus = await statusNet.textContent();
    // eslint-disable-next-line no-console
    console.log(`[viz-smoke] TFIM h=1: ${tfimStatus}`);
    await page.screenshot({ path: "e2e/.artifacts/viz-phase5-tfim-critical.png", fullPage: false });

    // Slide h down to ferromagnetic phase.
    await setRange("hField", "0.1");
    await page.waitForTimeout(2500);
    const tfimFerro = await statusNet.textContent();
    // eslint-disable-next-line no-console
    console.log(`[viz-smoke] TFIM h=0.1: ${tfimFerro}`);
    await page.screenshot({ path: "e2e/.artifacts/viz-phase5-tfim-ferro.png", fullPage: false });

    // Slide h up to paramagnetic phase.
    await setRange("hField", "2.5");
    await page.waitForTimeout(2500);
    const tfimPara = await statusNet.textContent();
    // eslint-disable-next-line no-console
    console.log(`[viz-smoke] TFIM h=2.5: ${tfimPara}`);
    await page.screenshot({ path: "e2e/.artifacts/viz-phase5-tfim-para.png", fullPage: false });

    // Scroll the controls column down and capture the order-parameter sweep.
    await page.locator("#orderRow").scrollIntoViewIfNeeded();
    await page.waitForTimeout(400);
    await page.screenshot({ path: "e2e/.artifacts/viz-phase5-sweep-panel.png", fullPage: false });

    // Final: scrub time to end of the imag-time evolution.
    const maxT = await page.locator("#time").evaluate((el) => (el as HTMLInputElement).max);
    await setRange("time", maxT);
    await page.waitForTimeout(300);
    await page.screenshot({ path: "e2e/.artifacts/viz-phase5-tfim-end.png", fullPage: false });

    // Quench dynamics — light cone.
    await page.locator("#model").selectOption("quench");
    await setRange("hField", "1.0");
    await page.waitForTimeout(4000);
    const quenchStat = await statusNet.textContent();
    // eslint-disable-next-line no-console
    console.log(`[viz-smoke] quench: ${quenchStat}`);
    const maxT2 = await page.locator("#time").evaluate((el) => (el as HTMLInputElement).max);
    await setRange("time", String(Math.floor(parseInt(maxT2, 10) / 2)));
    await page.waitForTimeout(200);
    await page.screenshot({ path: "e2e/.artifacts/viz-phase5-quench-mid.png", fullPage: false });
    await setRange("time", maxT2);
    await page.waitForTimeout(200);
    await page.screenshot({ path: "e2e/.artifacts/viz-phase5-quench-end.png", fullPage: false });

    // Monitored trajectory — γ = 0 (pure unitary), γ_c, γ ≫ 1 (frozen).
    await page.locator("#model").selectOption("trajectory");
    await page.waitForTimeout(2000);
    await setRange("gamma", "0");
    await page.waitForTimeout(2500);
    const trajUnitary = await statusNet.textContent();
    // eslint-disable-next-line no-console
    console.log(`[viz-smoke] trajectory γ=0: ${trajUnitary}`);
    const maxT3 = await page.locator("#time").evaluate((el) => (el as HTMLInputElement).max);
    await setRange("time", maxT3);
    await page.waitForTimeout(200);
    await page.screenshot({ path: "e2e/.artifacts/viz-phase6-traj-unitary.png", fullPage: false });

    await setRange("gamma", "0.20");
    await page.waitForTimeout(2500);
    const trajCrit = await statusNet.textContent();
    // eslint-disable-next-line no-console
    console.log(`[viz-smoke] trajectory γ=0.20: ${trajCrit}`);
    const maxT4 = await page.locator("#time").evaluate((el) => (el as HTMLInputElement).max);
    await setRange("time", maxT4);
    await page.waitForTimeout(200);
    await page.screenshot({ path: "e2e/.artifacts/viz-phase6-traj-crit.png", fullPage: false });

    await setRange("gamma", "1.40");
    await page.waitForTimeout(2500);
    const trajFrozen = await statusNet.textContent();
    // eslint-disable-next-line no-console
    console.log(`[viz-smoke] trajectory γ=1.40: ${trajFrozen}`);
    const maxT5 = await page.locator("#time").evaluate((el) => (el as HTMLInputElement).max);
    await setRange("time", maxT5);
    await page.waitForTimeout(200);
    await page.screenshot({ path: "e2e/.artifacts/viz-phase6-traj-frozen.png", fullPage: false });
  });
});
