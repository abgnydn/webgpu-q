import { test, expect } from "@playwright/test";

// iPhone 14-ish viewport. We don't need device emulation for layout checks —
// a 390 × 844 window is enough to exercise every @media (max-width: 720px) rule.
const MOBILE = { width: 390, height: 844 };

test.use({ viewport: MOBILE });

test.describe("Mobile-first layout (390 × 844)", () => {
  test("landing page: no horizontal scroll, hero CTAs stack full-width", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/webgpu-q/);

    // Body shouldn't be wider than the viewport — the canonical "did the page blow out the phone" check.
    const overflow = await page.evaluate(() => ({
      doc: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
      win: window.innerWidth,
    }));
    expect(overflow.doc).toBeLessThanOrEqual(overflow.win + 1);
    expect(overflow.body).toBeLessThanOrEqual(overflow.win + 1);

    // Primary CTA should be near full-width on phone (column layout).
    const cta = page.locator(".hero .ctas .btn.primary");
    await expect(cta).toBeVisible();
    const ctaBox = await cta.boundingBox();
    expect(ctaBox).toBeTruthy();
    // Button width should be most of the viewport (allow for hero side-padding).
    expect(ctaBox!.width).toBeGreaterThan(MOBILE.width * 0.7);

    await page.screenshot({ path: "e2e/.artifacts/mobile-landing.png", fullPage: true });
  });

  test("experiments dashboard: tables scroll horizontally, no body overflow", async ({ page }) => {
    await page.goto("/experiments/");
    await expect(page.locator("h1")).toContainText("experiments");

    const overflow = await page.evaluate(() => ({
      doc: document.documentElement.scrollWidth,
      win: window.innerWidth,
    }));
    expect(overflow.doc).toBeLessThanOrEqual(overflow.win + 1);

    // Each table should be wrapped in a .table-wrap that scrolls horizontally.
    const wrappers = await page.locator(".table-wrap").count();
    expect(wrappers).toBeGreaterThanOrEqual(4);

    await page.screenshot({ path: "e2e/.artifacts/mobile-experiments.png", fullPage: true });
  });

  test("viz page: panes stack vertically, controls reachable below", async ({ page }) => {
    await page.goto("/viz.html");
    await expect(page).toHaveTitle(/Hyperscope/);

    const overflow = await page.evaluate(() => ({
      doc: document.documentElement.scrollWidth,
      win: window.innerWidth,
    }));
    expect(overflow.doc).toBeLessThanOrEqual(overflow.win + 1);

    // The three panes should stack — measure the y-positions and confirm they're not overlapping.
    const ys = await page.evaluate(() => {
      const panes = Array.from(document.querySelectorAll(".pane")) as HTMLElement[];
      return panes.map((p) => p.getBoundingClientRect().top + window.scrollY);
    });
    expect(ys.length).toBeGreaterThanOrEqual(3);
    for (let i = 1; i < ys.length; i++) {
      expect(ys[i]).toBeGreaterThan(ys[i - 1]!);
    }

    await page.screenshot({ path: "e2e/.artifacts/mobile-viz.png", fullPage: true });
  });

  test("gpu-mps bench page: tables wrapped, no overflow", async ({ page }) => {
    await page.goto("/experiments/gpu-mps/");
    await expect(page.locator("h1")).toContainText("GPU MPS");

    const overflow = await page.evaluate(() => ({
      doc: document.documentElement.scrollWidth,
      win: window.innerWidth,
    }));
    expect(overflow.doc).toBeLessThanOrEqual(overflow.win + 1);

    const wrappers = await page.locator(".table-wrap").count();
    expect(wrappers).toBeGreaterThanOrEqual(5);

    await page.screenshot({ path: "e2e/.artifacts/mobile-gpu-mps.png", fullPage: true });
  });
});
