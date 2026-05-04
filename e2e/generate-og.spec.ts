import { test } from "@playwright/test";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

// Generate OG card (1200×630) and favicon variants by screenshotting
// the static HTML templates in tools/og/. Output goes to public/.

test("generate og-image.png (1200×630)", async ({ browser }) => {
  const ctx = await browser.newContext({
    viewport: { width: 1200, height: 630 },
    deviceScaleFactor: 2,
  });
  const page = await ctx.newPage();
  const url = pathToFileURL(resolve("tools/og/og-card.html")).href;
  await page.goto(url);
  await page.waitForTimeout(150);
  await page.screenshot({
    path: "public/og-image.png",
    fullPage: false,
    omitBackground: false,
    type: "png",
    clip: { x: 0, y: 0, width: 1200, height: 630 },
  });
  await ctx.close();
});

const ICON_SIZES = [
  { size: 32,  name: "favicon-32.png" },
  { size: 180, name: "apple-touch-icon.png" },
  { size: 192, name: "icon-192.png" },
  { size: 512, name: "icon-512.png" },
];

for (const { size, name } of ICON_SIZES) {
  test(`generate ${name} (${size}×${size})`, async ({ browser }) => {
    const ctx = await browser.newContext({
      viewport: { width: size, height: size },
      deviceScaleFactor: 1,
    });
    const page = await ctx.newPage();
    const url = pathToFileURL(resolve("tools/og/icon.html")).href;
    await page.goto(url);
    // The icon HTML is hard-coded at 512×512; let CSS scale via viewport.
    await page.evaluate((s) => {
      document.body.style.zoom = String(s / 512);
      document.body.style.width = `${s}px`;
      document.body.style.height = `${s}px`;
    }, size);
    await page.waitForTimeout(100);
    await page.screenshot({
      path: `public/${name}`,
      type: "png",
      clip: { x: 0, y: 0, width: size, height: size },
    });
    await ctx.close();
  });
}
