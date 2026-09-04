import { test, expect } from "@playwright/test";

test.describe("Landing page", () => {
  test("renders, links work, screenshot saved", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/webgpu-q/);
    await expect(page.locator("h1")).toContainText(/quantum chemistry/i);
    // Screenshot full page so we can eyeball it.
    await page.screenshot({ path: "e2e/.artifacts/landing.png", fullPage: true });

    // Verify the demo links resolve to the right targets.
    const hyperscope = page.locator('a[href="/viz.html"]').first();
    await expect(hyperscope).toBeVisible();
    const molecule = page.locator('a[href="/molecule.html"]').first();
    await expect(molecule).toBeVisible();
    const experiments = page.locator('a[href="/experiments/index.html"]').first();
    await expect(experiments).toBeVisible();
    const demo = page.locator('a[href="/demo.html"]').first();
    await expect(demo).toBeVisible();

    // Click the hyperscope CTA → page navigates.
    await hyperscope.click();
    await page.waitForURL("**/viz.html");
    await expect(page).toHaveTitle(/Hyperscope/);
  });
});
