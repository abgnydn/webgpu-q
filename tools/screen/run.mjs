#!/usr/bin/env node
// run.mjs — headless molecular screening runner.
//
// Drives a running webgpu-q instance and calls window.__webgpuq.screen(...)
// on a molecule library, printing the ranked artifact JSON to stdout. This is
// the executable "tool" a driver (Claude Science, a Modal job, a cron) invokes
// to turn a library → ranked property table without a UI.
//
// screen() runs on the CPU/WASM path (quickReport), so NO GPU is required —
// any Chromium works. (The GPU kernels — statevector / fusion / CCSD(T) — are a
// separate path; see tools/modal/webgpu_t4_probe.py for the GPU-gated runner.)
//
// Usage:
//   # 1. serve the app locally (dev or a built preview):
//   npm run dev            # http://localhost:5175/experiments/
//   # 2. run the screen:
//   node tools/screen/run.mjs molecules.json > artifact.json
//   node tools/screen/run.mjs molecules.json --rankBy dipole --basis cc-pvdz
//   node tools/screen/run.mjs molecules.json --url https://webgpu-q.vercel.app/experiments/
//
// molecules.json shape:
//   {
//     "molecules": [
//       { "name": "water", "xyz": "3\nwater\nO 0 0 0\nH 0 0 0.958\nH 0.926 0 -0.24\n" },
//       { "name": "methane", "atoms": [ { "symbol": "C", "pos": [0,0,0] }, ... ] }
//     ],
//     "opts": { "basis": "sto-3g", "rankBy": "gap", "descending": true }
//   }
// opts may also be overridden by CLI flags (--basis, --rankBy, --descending).

import { readFileSync } from "node:fs";
import { chromium } from "playwright";

function parseArgs(argv) {
  const args = { file: undefined, url: "http://localhost:5175/experiments/", opts: {} };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--url") args.url = argv[++i];
    else if (a === "--basis") args.opts.basis = argv[++i];
    else if (a === "--rankBy") args.opts.rankBy = argv[++i];
    else if (a === "--descending") args.opts.descending = argv[++i] !== "false";
    else if (a === "--addD2") args.opts.addD2 = true;
    else if (!a.startsWith("--")) args.file = a;
  }
  if (!args.file) {
    console.error("usage: node tools/screen/run.mjs <molecules.json> [--url U] [--basis B] [--rankBy gap|dipole|energy|ip] [--descending false] [--addD2]");
    process.exit(2);
  }
  return args;
}

async function main() {
  const { file, url, opts: cliOpts } = parseArgs(process.argv);
  const spec = JSON.parse(readFileSync(file, "utf8"));
  const molecules = spec.molecules ?? spec; // allow a bare array too
  const opts = { ...(spec.opts ?? {}), ...cliOpts };

  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(120_000);
    await page.goto(url, { waitUntil: "domcontentloaded" });
    await page.waitForFunction("window.__webgpuq && window.__webgpuq.ready === true");
    const hasScreen = await page.evaluate("typeof window.__webgpuq.screen === 'function'");
    if (!hasScreen) {
      throw new Error(
        `window.__webgpuq.screen is not present at ${url} — the served build predates the screen() entry point. ` +
        "Serve this branch (npm run dev / build+preview), or deploy it, then retry.",
      );
    }
    const artifact = await page.evaluate(
      ([mols, o]) => window.__webgpuq.screen(mols, o),
      [molecules, opts],
    );
    process.stdout.write(JSON.stringify(artifact, null, 2) + "\n");
  } finally {
    await browser.close();
  }
}

main().catch((e) => {
  console.error(e instanceof Error ? e.stack : String(e));
  process.exit(1);
});
