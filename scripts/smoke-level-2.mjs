// Node-side smoke driver — opens the experiments page in a real Chrome
// instance (with WebGPU enabled) and invokes runE5/E6/E7 via the module
// runtime with tiny opts. Reports status + diagnosis per artifact.
//
// Requires vite dev server already running at http://localhost:5175.

import { chromium } from "playwright";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const URL = "http://localhost:5175/experiments/";

const args = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan",
  "--no-first-run",
  "--no-default-browser-check",
];

const browser = await chromium.launch({
  executablePath: CHROME,
  args,
  headless: false, // WebGPU is more reliable with a real display context
});
const ctx = await browser.newContext();
const page = await ctx.newPage();
page.on("console", (msg) => {
  const t = msg.text();
  // Skip the pretty-JSON dump noise emitEmit spews — keep the one-line banner only.
  if (t.startsWith("{") || t.startsWith("  ")) return;
  console.log("[page]", t);
});
page.on("pageerror", (e) => console.error("[pageerror]", e.message));

await page.goto(URL);
await page.waitForFunction(() => !!navigator.gpu, { timeout: 5000 });
console.log("WebGPU present:", await page.evaluate(() => !!navigator.gpu));

// Call runE5/E6/E7 inside the page. Use tiny opts so this finishes in
// seconds, not minutes. We import the ESM modules dynamically at the
// Vite dev URL so they run in the exact same pipeline as the UI.
const result = await page.evaluate(async () => {
  // @ts-ignore — dynamic import at runtime, resolved by Vite.
  const { runE5 } = await import("/experiments/level-2-mps/E5-mps-correctness.ts");
  const { runE6 } = await import("/experiments/level-2-mps/E6-qubit-ceiling.ts");
  const { runE7 } = await import("/experiments/level-2-mps/E7-chi-scaling.ts");

  const summary = [];

  console.log("[smoke] running E5 (tiny)…");
  const e5 = await runE5({ trials: 1, Ns: [4, 6, 8], chi: 32, cheapestChi: false });
  summary.push({ id: "E5", status: e5.status, diagnosis: e5.diagnosis, rows: e5.rows.length });

  console.log("[smoke] running E6 (tiny)…");
  const e6 = await runE6({ trials: 3, warmup: 1, Ns: [16, 32], chi: 16, depth: 2 });
  summary.push({ id: "E6", status: e6.status, diagnosis: e6.diagnosis, rows: e6.rows.length });

  console.log("[smoke] running E7 (tiny)…");
  const e7 = await runE7({ nQubits: 8, seedsPerDepth: 2, depths: [1, 2, 3], chiLadder: [2, 4, 8, 16] });
  summary.push({ id: "E7", status: e7.status, diagnosis: e7.diagnosis, rows: e7.rows.length, fit: e7.fit });

  return summary;
});

console.log("\n===== SMOKE RESULTS =====");
for (const r of result) {
  console.log(`${r.id}: ${r.status}  (${r.rows} rows)`);
  console.log(`     ${r.diagnosis}`);
  if (r.fit) {
    console.log(`     fit: slope=${r.fit.slope.toFixed(3)}, R²=${r.fit.r2.toFixed(3)}, depths=[${r.fit.depths.join(",")}]`);
  }
}

await browser.close();
