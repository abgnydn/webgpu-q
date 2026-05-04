// ─────────────────────────────────────────────────────────────
// run-level.ts — Playwright helper that drives a single level
// in a headless WebGPU Chromium and saves each per-experiment
// artifact JSON under experiments/results/YYYY-MM-DD/level-N/.
//
// This is the "everything reproducible end-to-end" harness:
// the same code path the user clicks in the browser is invoked
// from CI / scripts, the JSONs land on disk, and the test
// asserts each artifact's status field.
// ─────────────────────────────────────────────────────────────

import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { expect, type Page } from "@playwright/test";

export type LevelKey = "runLevel1" | "runLevel2" | "runLevel3" | "runLevel6";

export interface ArtifactLike {
  meta?: { protocol?: string };
  status?: "pass" | "fail" | "noisy";
  diagnosis?: string;
}

export interface LevelResult {
  status: "pass" | "fail" | "noisy";
  artifacts: ArtifactLike[];
}

// Playwright runs from the project root, so cwd is stable here.
const REPO_ROOT = process.cwd();

function todayUTC(): string {
  return new Date().toISOString().slice(0, 10);
}

/**
 * Navigate to the experiments dashboard and wait for the window
 * hook to be installed. Surfaces page console + page errors so a
 * GPU validation error in the runner becomes a Playwright failure
 * instead of a silent hang.
 */
export async function bootExperimentsPage(page: Page): Promise<void> {
  page.on("console", (msg) => {
    const text = msg.text();
    // Skip the firehose of artifact JSON dumps from emitArtifact —
    // we keep the [artifact:…] header line for grep visibility.
    if (text.startsWith("{")) return;
    // eslint-disable-next-line no-console
    console.log(`[browser:${msg.type()}] ${text}`);
  });
  page.on("pageerror", (err) => {
    // eslint-disable-next-line no-console
    console.error(`[browser:pageerror] ${err.message}`);
  });

  await page.goto("/experiments/", { waitUntil: "domcontentloaded" });

  // Wait for module graph to load and for the window hook to install.
  await page.waitForFunction(() => window.__webgpuq?.ready === true, undefined, {
    timeout: 60_000,
  });

  // Sanity: WebGPU must actually exist in this Chromium. If the launch
  // flags didn't take, fail loudly here rather than chasing a confusing
  // "no adapter" error 30 s into a sweep.
  const gpuOk = await page.evaluate(async () => {
    if (!("gpu" in navigator)) return { ok: false, reason: "navigator.gpu missing" };
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) return { ok: false, reason: "no adapter" };
      const info = adapter.info ?? {};
      return { ok: true, vendor: info.vendor, arch: info.architecture, desc: info.description };
    } catch (err) {
      return { ok: false, reason: String(err) };
    }
  });
  if (!gpuOk.ok) {
    throw new Error(`WebGPU unavailable in test browser: ${gpuOk.reason}`);
  }
  // eslint-disable-next-line no-console
  console.log(`[e2e] adapter: ${JSON.stringify(gpuOk)}`);
}

/**
 * Drive `window.__webgpuq[levelKey]()` headlessly, save every
 * artifact JSON to disk, return the array for assertions.
 */
export async function runLevelE2E(
  page: Page,
  levelKey: LevelKey,
  levelN: number,
): Promise<LevelResult> {
  await bootExperimentsPage(page);

  const result = await page.evaluate(async (key) => {
    const hook = window.__webgpuq;
    if (!hook) throw new Error("window.__webgpuq missing");
    const fn = hook[key as keyof typeof hook];
    if (typeof fn !== "function") throw new Error(`runner ${key} not exposed`);
    return await (fn as () => Promise<{ status: string; artifacts: unknown[] }>)();
  }, levelKey);

  const artifacts = (result.artifacts ?? []) as ArtifactLike[];

  const outDir = resolve(REPO_ROOT, "experiments", "results", todayUTC(), `level-${levelN}`);
  await mkdir(outDir, { recursive: true });

  for (const art of artifacts) {
    const protocol = art.meta?.protocol ?? `unknown-${Math.random().toString(36).slice(2, 8)}`;
    const filename = `${protocol}.json`;
    const fullPath = resolve(outDir, filename);
    await writeFile(fullPath, JSON.stringify(art, null, 2), "utf-8");
    // eslint-disable-next-line no-console
    console.log(
      `[e2e] level-${levelN} ${protocol}: ${art.status} — ${art.diagnosis ?? ""} → ${fullPath}`,
    );
  }

  return { status: result.status as LevelResult["status"], artifacts };
}

/**
 * E2E pipeline assertion (intentionally NOT a per-hypothesis pass-bar
 * gate). The project's research discipline says failures are evidence
 * — RESEARCH.md §"Honest negative results" rejects silent reruns and
 * requires committing failed JSONs with `status:"fail"` and a diagnosis.
 *
 * So the e2e suite checks the *pipeline* invariants, not the science:
 *   • Every artifact has the locked schema (meta.protocol, status,
 *     diagnosis), proving the experiment ran to completion and the
 *     run-all wiring didn't silently swallow an exception.
 *   • Each artifact's status is one of the documented values.
 *
 * Whether an experiment hit its pass bar is a separate question
 * answered by the JSON itself; CI gating on "all pass" would make us
 * either suppress real findings or stop publishing the negatives.
 */
export function assertLevelStatus(result: LevelResult, _allowNoisy = true): void {
  for (const art of result.artifacts) {
    expect(art.meta?.protocol, "artifact missing meta.protocol").toBeTruthy();
    expect(["pass", "fail", "noisy"]).toContain(art.status);
    if (art.status !== "pass") {
      // eslint-disable-next-line no-console
      console.log(`[e2e][note] ${art.meta?.protocol} ${art.status}: ${art.diagnosis}`);
    }
  }
  expect(result.artifacts.length, "no artifacts produced").toBeGreaterThan(0);
}
