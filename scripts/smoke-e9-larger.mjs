// E9 at larger batch size to unmask fence/timer bias in the smoke.

import { chromium } from "playwright";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const browser = await chromium.launch({
  executablePath: CHROME,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan",
         "--no-first-run", "--no-default-browser-check"],
  headless: false,
});
const ctx = await browser.newContext();
const page = await ctx.newPage();
page.on("console", (msg) => {
  const t = msg.text();
  if (t.startsWith("{") || t.startsWith("  ")) return;
  console.log("[page]", t);
});
page.on("pageerror", (e) => console.error("[pageerror]", e.message));

await page.goto("http://localhost:5175/experiments/");
await page.waitForFunction(() => !!navigator.gpu, { timeout: 5000 });

const result = await page.evaluate(async () => {
  // @ts-ignore
  const { runE9 } = await import("/experiments/level-3-fusion/E9-dispatch-collapse.ts");
  // Default-like opts: gatesPerBatch=256 matches E4.
  const e9 = await runE9({
    trials: 12, warmup: 3,
    Ks: [1, 2, 4, 8, 16, 32],
    Ns: [8, 12, 16],
    gatesPerBatch: 256,
  });
  return {
    status: e9.status,
    diagnosis: e9.diagnosis,
    perKFits: e9.summary.perKFits.map((f) => ({ k: f.k, aMicros: f.alphaEffMicros, r2: f.r2 })),
    kScaling: e9.summary.kScalingFit,
    collapseFactor: e9.summary.collapseFactor,
  };
});

console.log("\n===== E9 LARGER-BATCH =====");
console.log(`status: ${result.status}`);
console.log(`diagnosis: ${result.diagnosis}`);
console.log(`α_raw = ${result.kScaling.alphaRawMicros.toFixed(2)} μs, C = ${result.kScaling.cResidualMicros.toFixed(2)} μs, R² = ${result.kScaling.r2.toFixed(3)}`);
console.log(`collapse factor (α_eff(1)/α_eff(k_max)) = ${result.collapseFactor.toFixed(2)}×`);
for (const f of result.perKFits) {
  console.log(`  k=${f.k.toString().padStart(2)}  α_eff=${f.aMicros.toFixed(2).padStart(7)} μs  R²=${f.r2.toFixed(3)}`);
}
await browser.close();
