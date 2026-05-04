import { test, expect } from "@playwright/test";

test.describe("Landing page", () => {
  test("renders, links work, screenshot saved", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/webgpu-q/);
    await expect(page.locator("h1")).toContainText("Quantum simulation");
    // Screenshot full page so we can eyeball it.
    await page.screenshot({ path: "e2e/.artifacts/landing.png", fullPage: true });

    // Verify the three demo links resolve to the right targets.
    const hyperscope = page.locator('a[href="/viz.html"]').first();
    await expect(hyperscope).toBeVisible();
    const experiments = page.locator('a[href="/experiments/"]').first();
    await expect(experiments).toBeVisible();
    const demo = page.locator('a[href="/demo.html"]').first();
    await expect(demo).toBeVisible();

    // Click the hyperscope CTA → page navigates.
    await hyperscope.click();
    await page.waitForURL("**/viz.html");
    await expect(page).toHaveTitle(/Hyperscope/);
  });
});
