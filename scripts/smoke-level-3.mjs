// Browser smoke for Level 3 — fusion correctness + dispatch collapse +
// throughput. Tiny opts to finish in seconds. Requires vite dev server
// running at http://localhost:5175.

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

await page.goto(URL);
await page.waitForFunction(() => !!navigator.gpu, { timeout: 5000 });
console.log("WebGPU present:", await page.evaluate(() => !!navigator.gpu));

const result = await page.evaluate(async () => {
  // @ts-ignore — dynamic import resolved by Vite.
  const { runE8 } = await import("/experiments/level-3-fusion/E8-fusion-correctness.ts");
  const { runE9 } = await import("/experiments/level-3-fusion/E9-dispatch-collapse.ts");
  const { runE10 } = await import("/experiments/level-3-fusion/E10-throughput.ts");

  const summary = [];

  console.log("[smoke] running E8 (tiny)…");
  const e8 = await runE8({ trials: 2, Ks: [1, 4, 16, 32], Ns: [8, 12] });
  summary.push({ id: "E8", status: e8.status, diagnosis: e8.diagnosis, rows: e8.rows.length });

  console.log("[smoke] running E9 (tiny)…");
  // gatesPerBatch stays at the E9 default (256) — the map-fence timing
  // bias is T_fence / gatesPerBatch per gate, and the α_eff(k_max) ≤ 20μs
  // bar needs this to be ≤ ~10μs. Smaller batches push us above the bar.
  const e9 = await runE9({
    trials: 4, warmup: 2,
    Ks: [1, 4, 16, 32],
    Ns: [8, 12, 16],
  });
  summary.push({
    id: "E9", status: e9.status, diagnosis: e9.diagnosis, rows: e9.rows.length,
    perKFits: e9.summary.perKFits.map((f) => ({ k: f.k, alphaMicros: f.alphaEffMicros, r2: f.r2 })),
    kScaling: e9.summary.kScalingFit,
  });

  console.log("[smoke] running E10 (tiny)…");
  const e10 = await runE10({
    trials: 4, warmup: 2,
    Ns: [10, 14, 18, 20],
    depths: [32, 128],
    chainK: 32,
  });
  summary.push({
    id: "E10", status: e10.status, diagnosis: e10.diagnosis, rows: e10.rows.length,
    best: e10.summary.bestSpeedup,
    atN20: e10.summary.speedupAtN20,
  });

  return summary;
});

console.log("\n===== SMOKE RESULTS =====");
for (const r of result) {
  console.log(`${r.id}: ${r.status}  (${r.rows} rows)`);
  console.log(`     ${r.diagnosis}`);
  if (r.perKFits) {
    for (const f of r.perKFits) {
      console.log(`     k=${f.k}  α_eff=${f.alphaMicros.toFixed(2)}μs  R²=${f.r2.toFixed(3)}`);
    }
    if (r.kScaling) {
      console.log(`     1/k fit: α_raw=${r.kScaling.alphaRawMicros.toFixed(2)}μs  C=${r.kScaling.cResidualMicros.toFixed(2)}μs  R²=${r.kScaling.r2.toFixed(3)}`);
    }
  }
  if (r.best !== undefined) {
    console.log(`     best speedup ${r.best.toFixed(2)}×, N=20 speedup ${r.atN20?.toFixed(2) ?? "n/a"}×`);
  }
}

await browser.close();
