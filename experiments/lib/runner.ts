// ─────────────────────────────────────────────────────────────
// runner.ts — the shared timing harness for every experiment.
//
// Rules (RESEARCH.md §Timing):
//   • W warmup samples discarded (default 5)
//   • T trials retained (default 20)
//   • performance.now() with forced GPU sync before AND after
//     (a mapped readback on a tiny buffer; queue.submit alone is
//     non-blocking so pure performance.now() timing is fiction)
//   • pushErrorScope("validation") + popErrorScope around each
//     trial; any error aborts the run with a diagnosis string
//
// Every run returns a JSON artifact that can be fed straight to
// the results README without any post-processing.
// ─────────────────────────────────────────────────────────────

import { stats, type Stats } from "./stats.js";
import type { EnvBlock } from "./env.js";

export interface RunConfig {
  readonly warmup?: number;   // default 5
  readonly trials?: number;   // default 20
}

export interface TimedResult {
  /** Seconds per trial, one entry per retained trial. */
  readonly samples: readonly number[];
  readonly stats: Stats;
  /** Any validation error that fired during a trial. null = clean run. */
  readonly gpuError: string | null;
}

/**
 * Run `fn` once per trial with the standard warmup + sync pattern.
 * `fn` should submit GPU work; the harness injects the sync barrier.
 */
export async function timedRun(
  device: GPUDevice,
  fn: () => void | Promise<void>,
  cfg: RunConfig = {},
): Promise<TimedResult> {
  const warmup = cfg.warmup ?? 5;
  const trials = cfg.trials ?? 20;

  // Tiny "fence" buffer we map-read to force a full submission drain.
  // 4 bytes is enough — the wait cost dominates at this size.
  const fence = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: "runner-fence",
  });

  // 4-byte aligned payload — WebGPU rejects sub-word writes.
  const fenceWord = new Uint32Array([0]);
  async function sync(): Promise<void> {
    device.queue.writeBuffer(fence, 0, fenceWord);
    await fence.mapAsync(GPUMapMode.READ);
    fence.unmap();
  }

  // Warmup runs: include shader compile, pipeline warm, TLB warm.
  for (let i = 0; i < warmup; i++) {
    await fn();
    await sync();
  }

  const samples: number[] = [];
  let firstError: string | null = null;

  for (let i = 0; i < trials; i++) {
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");

    await sync(); // clean starting line

    const t0 = performance.now();
    await fn();
    await sync();
    const t1 = performance.now();

    const oomErr = await device.popErrorScope();
    const valErr = await device.popErrorScope();
    if (!firstError && (oomErr || valErr)) {
      firstError = (valErr?.message ?? oomErr?.message ?? "unknown").toString();
    }

    samples.push((t1 - t0) / 1000); // seconds
  }

  fence.destroy();

  return { samples, stats: stats(samples), gpuError: firstError };
}

// ── JSON artifact schema ─────────────────────────────────────
//
// Every experiment emits a JSON blob with this shape. Keep the
// keys stable — external analysis scripts depend on them.

export interface ArtifactMeta {
  readonly protocol: string;        // short identifier, e.g. "E1-gate-fidelity"
  readonly hypothesis: string;      // falsifiable claim being tested
  readonly passBar: string;         // e.g. "F ≥ 1 − 1e-5"
  readonly seed: string;            // seed name from seeds.ts
  readonly warmup: number;
  readonly trials: number;
}

export interface Artifact<Row = Record<string, unknown>> {
  readonly meta: ArtifactMeta;
  readonly env: EnvBlock;
  readonly rows: readonly Row[];
  /** Overall status: "pass" / "fail" / "noisy". Failures are evidence. */
  readonly status: "pass" | "fail" | "noisy";
  /** Short human-readable diagnosis, required if status ≠ "pass". */
  readonly diagnosis: string;
}

/** Print artifact to stdout + console in pretty JSON. */
export function emitArtifact<R>(a: Artifact<R>): void {
  const json = JSON.stringify(a, null, 2);
   
  console.log(`[artifact:${a.meta.protocol}] ${a.status} — ${a.diagnosis}`);
   
  console.log(json);
}

/** Trigger a browser download of the artifact as JSON. Call from a click handler. */
export function downloadArtifact<R>(a: Artifact<R>, filename?: string): void {
  if (typeof window === "undefined") return;
  const blob = new Blob([JSON.stringify(a, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename ?? `${a.meta.protocol}-${a.env.timestamp.replace(/[:.]/g, "-")}.json`;
  link.click();
  URL.revokeObjectURL(url);
}
