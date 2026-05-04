// E9 re-smoke against the JIT-unrolled fused shader. Pre-warms the
// pipeline cache so cold-compile doesn't pollute the trial timing,
// then runs E9 with enough batch to beat fence bias (gatesPerBatch=256).

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
  const { initGPU, QuantumCircuit } = await import("/src/quantum.ts");
  // @ts-ignore
  const { runE9 } = await import("/experiments/level-3-fusion/E9-dispatch-collapse.ts");

  // Pre-warm the JIT cache for each k E9 will touch, so cold compile
  // doesn't skew the trial timing. Cold compile times are reported
  // separately on the QuantumCircuit.fusedColdCompileMs() map.
  const device = await initGPU();
  const scratch = new QuantumCircuit({ device, nQubits: 8 });
  const Ks = [1, 2, 4, 8, 16, 32, 64];
  scratch.warmFusedCache(Ks);
  const coldMs = Object.fromEntries(scratch.fusedColdCompileMs());
  scratch.dispose();

  const e9 = await runE9({
    trials: 12, warmup: 3,
    Ks,
    Ns: [8, 12, 16],
    gatesPerBatch: 256,
  });

  return {
    status: e9.status,
    diagnosis: e9.diagnosis,
    perKFits: e9.summary.perKFits.map((f) => ({ k: f.k, aMicros: f.alphaEffMicros, r2: f.r2 })),
    kScaling: e9.summary.kScalingFit,
    collapse: e9.summary.collapseFactor,
    coldCompileMs: coldMs,
  };
});

console.log("\n===== E9 JIT =====");
console.log(`status: ${result.status}`);
console.log(`diagnosis: ${result.diagnosis}`);
console.log(`α_raw = ${result.kScaling.alphaRawMicros.toFixed(2)} μs, C = ${result.kScaling.cResidualMicros.toFixed(2)} μs, R² = ${result.kScaling.r2.toFixed(3)}`);
console.log(`collapse factor = ${result.collapse.toFixed(2)}×`);
console.log(`\nper-k α_eff:`);
for (const f of result.perKFits) {
  console.log(`  k=${f.k.toString().padStart(2)}  α_eff=${f.aMicros.toFixed(2).padStart(7)} μs  R²=${f.r2.toFixed(3)}`);
}
console.log(`\ncold compile (ms) per k:`);
for (const [k, ms] of Object.entries(result.coldCompileMs)) {
  console.log(`  k=${k.padStart(2)}  ${Number(ms).toFixed(2)} ms`);
}
await browser.close();
