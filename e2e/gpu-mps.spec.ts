import { test, expect } from "@playwright/test";

test.describe("GPU MPS Phase 1A — complex matmul", () => {
  test("GPU matmul matches CPU across shapes; bench page screenshots saved", async ({ page }) => {
    page.on("console", (msg) => {
      const text = msg.text();
      if (text.startsWith("{")) return;
      // eslint-disable-next-line no-console
      console.log(`[browser:${msg.type()}] ${text}`);
    });
    page.on("pageerror", (err) => {
      // eslint-disable-next-line no-console
      console.error(`[browser:pageerror] ${err.message}`);
    });

    await page.goto("/experiments/gpu-mps/", { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.__gpuMpsBench?.ready === true, undefined, { timeout: 15_000 });

    // Drive the benchmark headlessly.
    const rows = await page.evaluate(async () => {
      return await window.__gpuMpsBench!.run();
    });

    // eslint-disable-next-line no-console
    console.log(`[gpu-mps] ${rows.length} shapes tested`);
    for (const r of rows) {
      // eslint-disable-next-line no-console
      console.log(`  ${r.shape}: rel=${r.rel.toExponential(2)} match=${r.match} cpu=${r.cpuMs.toFixed(2)}ms gpu=${r.gpuMs.toFixed(2)}ms speedup=${r.speedup.toFixed(2)}×`);
      // FP32 matmul should agree with FP32 reference to ~1e-4 relative.
      expect(r.rel, `shape ${r.shape}`).toBeLessThan(1e-3);
      expect(r.match).toBe(true);
    }

    await page.screenshot({ path: "e2e/.artifacts/gpu-mps-bench.png", fullPage: true });
  });

  test("GPU Jacobi SVD reconstructs the input within FP32 tolerance", async ({ page }) => {
    page.on("console", (msg) => {
      const text = msg.text();
      if (text.startsWith("{")) return;
      // eslint-disable-next-line no-console
      console.log(`[browser:${msg.type()}] ${text}`);
    });
    page.on("pageerror", (err) => {
      // eslint-disable-next-line no-console
      console.error(`[browser:pageerror] ${err.message}`);
    });

    await page.goto("/experiments/gpu-mps/", { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.__gpuMpsBench?.ready === true, undefined, { timeout: 15_000 });

    const svdRows = await page.evaluate(async () => {
      return await window.__gpuMpsBench!.runSvd();
    });

    // eslint-disable-next-line no-console
    console.log(`[gpu-svd] ${svdRows.length} sizes tested`);
    for (const r of svdRows) {
      // eslint-disable-next-line no-console
      console.log(`  n=${r.n}: rel=${r.reconstructRel.toExponential(2)} match=${r.match} σ_max=${r.sigmaMax.toFixed(3)} gpu=${r.gpuMs.toFixed(2)}ms`);
      // f32 Jacobi reconstruction floor: ~1e-5 for n ≤ 32 random Gaussians.
      expect(r.reconstructRel, `n=${r.n}`).toBeLessThan(1e-3);
      expect(r.match).toBe(true);
    }

    await page.screenshot({ path: "e2e/.artifacts/gpu-mps-svd.png", fullPage: true });
  });

  test("Phase 2: full MPS evolution via GPU SVD matches CPU MPS", async ({ page }) => {
    page.on("console", (msg) => {
      const text = msg.text();
      if (text.startsWith("{")) return;
      // eslint-disable-next-line no-console
      console.log(`[browser:${msg.type()}] ${text}`);
    });
    page.on("pageerror", (err) => {
      // eslint-disable-next-line no-console
      console.error(`[browser:pageerror] ${err.message}`);
    });

    await page.goto("/experiments/gpu-mps/", { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.__gpuMpsBench?.ready === true, undefined, { timeout: 15_000 });

    const rows = await page.evaluate(async () => {
      return await window.__gpuMpsBench!.runMps();
    });

    // eslint-disable-next-line no-console
    console.log(`[gpu-mps-phase2] ${rows.length} configs tested`);
    for (const r of rows) {
      // eslint-disable-next-line no-console
      console.log(`  N=${r.N} χ=${r.chi}: F=${r.fidelity.toFixed(7)} cpu=${r.cpuMs.toFixed(2)}ms gpu=${r.gpuMs.toFixed(2)}ms speedup=${r.speedup.toFixed(2)}×`);
      expect(r.match, `N=${r.N} χ=${r.chi} fidelity=${r.fidelity}`).toBe(true);
    }
    await page.screenshot({ path: "e2e/.artifacts/gpu-mps-phase2.png", fullPage: true });
  });

  test("Phase 4a: GPU-resident MPS single-qubit chain matches CPU MPS", async ({ page }) => {
    page.on("console", (msg) => {
      const text = msg.text();
      if (text.startsWith("{")) return;
      // eslint-disable-next-line no-console
      console.log(`[browser:${msg.type()}] ${text}`);
    });
    page.on("pageerror", (err) => {
      // eslint-disable-next-line no-console
      console.error(`[browser:pageerror] ${err.message}`);
    });

    await page.goto("/experiments/gpu-mps/", { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.__gpuMpsBench?.ready === true, undefined, { timeout: 15_000 });

    const rows = await page.evaluate(async () => {
      return await window.__gpuMpsBench!.runResident();
    });

    // eslint-disable-next-line no-console
    console.log(`[gpu-mps-phase4a] ${rows.length} configs tested`);
    for (const r of rows) {
      // eslint-disable-next-line no-console
      console.log(`  N=${r.N} gates=${r.gateCount}: F=${r.fidelity.toFixed(7)} cpu=${r.cpuMs.toFixed(2)}ms gpu=${r.gpuMs.toFixed(2)}ms speedup=${r.speedup.toFixed(2)}×`);
      expect(r.match, `N=${r.N} fidelity=${r.fidelity}`).toBe(true);
    }
    await page.screenshot({ path: "e2e/.artifacts/gpu-mps-phase4a.png", fullPage: true });
  });

  test("Phase 5 v1: GPU canonicalize() round-trip preserves the physical state", async ({ page }) => {
    page.on("console", (msg) => {
      const text = msg.text();
      if (text.startsWith("{")) return;
      // eslint-disable-next-line no-console
      console.log(`[browser:${msg.type()}] ${text}`);
    });
    page.on("pageerror", (err) => {
      // eslint-disable-next-line no-console
      console.error(`[browser:pageerror] ${err.message}`);
    });

    await page.goto("/experiments/gpu-mps/", { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.__gpuMpsBench?.ready === true, undefined, { timeout: 15_000 });

    const rows = await page.evaluate(async () => {
      return await window.__gpuMpsBench!.runPhase5v1();
    });

    // eslint-disable-next-line no-console
    console.log(`[gpu-mps-phase5v1] ${rows.length} configs tested`);
    for (const r of rows) {
      // eslint-disable-next-line no-console
      console.log(`  N=${r.N} gates=${r.gateCount} + canonicalize: F=${r.fidelity.toFixed(7)} gpu=${r.gpuMs.toFixed(2)}ms`);
      expect(r.match, `N=${r.N} fidelity=${r.fidelity}`).toBe(true);
    }
    await page.screenshot({ path: "e2e/.artifacts/gpu-mps-phase5v1.png", fullPage: true });
  });

  test("Phase 4b/5: full GPU two-site pipeline matches CPU MPS on rectangular SVD circuits", async ({ page }) => {
    page.on("console", (msg) => {
      const text = msg.text();
      if (text.startsWith("{")) return;
      // eslint-disable-next-line no-console
      console.log(`[browser:${msg.type()}] ${text}`);
    });
    page.on("pageerror", (err) => {
      // eslint-disable-next-line no-console
      console.error(`[browser:pageerror] ${err.message}`);
    });

    await page.goto("/experiments/gpu-mps/", { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.__gpuMpsBench?.ready === true, undefined, { timeout: 15_000 });

    const rows = await page.evaluate(async () => {
      return await window.__gpuMpsBench!.runPhase4b();
    });

    // eslint-disable-next-line no-console
    console.log(`[gpu-mps-phase4b] ${rows.length} configs tested`);
    for (const r of rows) {
      // eslint-disable-next-line no-console
      console.log(`  N=${r.N} gates=${r.gateCount}: F=${r.fidelity.toFixed(7)} cpu=${r.cpuMs.toFixed(2)}ms gpu=${r.gpuMs.toFixed(2)}ms`);
      expect(r.match, `N=${r.N} fidelity=${r.fidelity}`).toBe(true);
    }
    await page.screenshot({ path: "e2e/.artifacts/gpu-mps-phase4b.png", fullPage: true });
  });
});
