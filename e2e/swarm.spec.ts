import { test, expect } from "@playwright/test";

// Drive two tabs of /swarm.html in the same browser context (same origin)
// and verify the BroadcastChannel-based swarm distributes a primes job
// across them.

test.describe("Swarm — BroadcastChannel multi-tab", () => {
  test("two tabs distribute a primes job and report a consistent total", async ({ browser }) => {
    const ctx = await browser.newContext();
    const a = await ctx.newPage();
    const b = await ctx.newPage();

    a.on("pageerror", (err) => console.error(`[a:pageerror] ${err.message}`));
    b.on("pageerror", (err) => console.error(`[b:pageerror] ${err.message}`));

    await a.goto("/swarm.html", { waitUntil: "domcontentloaded" });
    await b.goto("/swarm.html", { waitUntil: "domcontentloaded" });

    // Wait for both peers to see each other in the roster.
    await a.waitForFunction(() =>
      document.querySelectorAll("#peers .peer").length >= 2,
      { timeout: 5000 },
    );
    await b.waitForFunction(() =>
      document.querySelectorAll("#peers .peer").length >= 2,
      { timeout: 5000 },
    );

    // Set a smaller N for the e2e — we want the test to finish quickly.
    await a.locator("#N").fill("100000");
    await a.locator("#tiles").fill("4");

    // Dispatch from tab A.
    await a.locator("#runBtn").click();

    // Wait for the success log line on A.
    await expect(a.locator("#log")).toContainText("Total primes in [0, 100,000)", { timeout: 30000 });

    const text = await a.locator("#log").textContent();
    // π(10^5) = 9592 — quick sanity bound.
    expect(text).toContain("9,592");

    await ctx.close();
  });
});
