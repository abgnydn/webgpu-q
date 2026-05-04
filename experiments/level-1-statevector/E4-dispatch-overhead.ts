// ─────────────────────────────────────────────────────────────
// E4 — dispatch overhead.
//
// Hypothesis: per-gate cost decomposes as T(N) = α + β · 2^N, where α
// is the fixed submit + sync cost in microseconds and β is the
// per-amplitude time in nanoseconds. α in [50, 500] μs on consumer
// WebGPU stacks, dominating at N < 12.
//
// This establishes the baseline that Level 3 (kernel fusion) attacks.
// The fusion speedup story isn't real until α is measured and cited.
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit } from "../../src/quantum.js";
import { H } from "../../src/gates.js";
import { captureEnv } from "../lib/env.js";
import { timedRun } from "../lib/runner.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

export interface E4Row {
  readonly nQubits: number;
  readonly nStates: number;
  readonly gatesPerBatch: number;
  readonly medianSeconds: number;
  readonly secondsPerGate: number;
  readonly microsPerGate: number;
  readonly noisy: boolean;
}

export interface E4Fit {
  /** Fixed per-gate cost in seconds. */
  readonly alpha: number;
  /** Per-amplitude cost in seconds per amplitude. */
  readonly beta: number;
  readonly r2: number;
  readonly alphaMicros: number;
  readonly betaNsPerAmp: number;
}

// OLS: y = α + β · x, return {alpha, beta, R²}.
function olsFit(xs: readonly number[], ys: readonly number[]): { alpha: number; beta: number; r2: number } {
  const n = xs.length;
  if (n < 2) return { alpha: NaN, beta: NaN, r2: NaN };
  const sx = xs.reduce((a, b) => a + b, 0);
  const sy = ys.reduce((a, b) => a + b, 0);
  const mx = sx / n, my = sy / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i]! - mx) * (ys[i]! - my);
    sxx += (xs[i]! - mx) ** 2;
    syy += (ys[i]! - my) ** 2;
  }
  const beta = sxx === 0 ? 0 : sxy / sxx;
  const alpha = my - beta * mx;
  const r2 = syy === 0 ? 1 : (sxy * sxy) / (sxx * syy);
  return { alpha, beta, r2 };
}

export async function runE4(
  opts: {
    trials?: number;
    warmup?: number;
    Ns?: readonly number[];
    gatesPerBatch?: number;
  } = {},
): Promise<Artifact<E4Row> & { fit: E4Fit }> {
  const trials = opts.trials ?? 20;
  const warmup = opts.warmup ?? 5;
  // Wide N sweep: below N≈14 α dominates (flat regime), above N≈20 β·2^N
  // dominates (linear regime). OLS over both gives a real α/β split.
  // (Empirically on M-series: α ≈ 40 μs; at N=18 β·2^N ≈ 2 μs, still in
  // noise. Need N ≥ 22 to see β unambiguously.)
  const Ns = opts.Ns ?? [4, 8, 12, 14, 16, 18, 20, 22];
  const gatesPerBatch = opts.gatesPerBatch ?? 256;

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E4Row[] = [];
  for (const n of Ns) {
     
    console.log(`[E4] N=${n} gates=${gatesPerBatch}`);
    const gpu = new QuantumCircuit({ device, nQubits: n });
    try {
      const { stats: s } = await timedRun(device, () => {
        for (let k = 0; k < gatesPerBatch; k++) gpu.apply(H, 0);
      }, { trials, warmup });

      const nStates = 1 << n;
      const secondsPerGate = s.median / gatesPerBatch;
      rows.push({
        nQubits: n,
        nStates,
        gatesPerBatch,
        medianSeconds: s.median,
        secondsPerGate,
        microsPerGate: secondsPerGate * 1e6,
        noisy: s.noisy,
      });
       
      console.log(`      → ${(secondsPerGate*1e6).toFixed(2)} μs/gate ${s.noisy ? "[NOISY]" : ""}`);
    } finally {
      gpu.dispose();
    }
  }

  // Linear fit of sec/gate vs 2^N.
  const xs = rows.map((r) => r.nStates);
  const ys = rows.map((r) => r.secondsPerGate);
  const { alpha, beta, r2 } = olsFit(xs, ys);

  const fit: E4Fit = {
    alpha, beta, r2,
    alphaMicros: alpha * 1e6,
    betaNsPerAmp: beta * 1e9,
  };

  const alphaOk = Number.isFinite(alpha) && alpha >= 5e-6 && alpha <= 500e-6;
  const betaOk = Number.isFinite(beta) && beta > 0 && beta < 100e-9;
  const r2Ok = Number.isFinite(r2) && r2 >= 0.85;
  // A cell is "noisy" individually; only fail the whole run if > half are.
  const noisyFrac = rows.filter((r) => r.noisy).length / rows.length;
  const noisyBlocker = noisyFrac > 0.5;
  const status: Artifact<E4Row>["status"] =
    noisyBlocker ? "noisy"
    : (alphaOk && betaOk && r2Ok) ? "pass"
    : "fail";

  const meta: ArtifactMeta = {
    protocol: "E4-dispatch-overhead",
    hypothesis: "Per-gate time T(N) = α + β · 2^N; α ∈ [5, 500] μs, β > 0, R² ≥ 0.85 over N ∈ [4, 18].",
    passBar: "α ∈ [5, 500] μs AND β > 0 AND R² ≥ 0.85",
    seed: "E4_OVERHEAD",
    warmup,
    trials,
  };

  const noisyTail = rows.filter((r) => r.noisy).map((r) => r.nQubits);
  const noisyNote = noisyTail.length === 0 ? "" : ` (noisy cells: N=${noisyTail.join(",")})`;
  const diagnosis =
    noisyBlocker ? `>50% of cells NOISY — investigate before publishing${noisyNote}.` :
    (alphaOk && betaOk && r2Ok) ? `α=${fit.alphaMicros.toFixed(1)} μs, β=${fit.betaNsPerAmp.toFixed(3)} ns/amp, R²=${fit.r2.toFixed(4)}${noisyNote}.` :
    `α=${fit.alphaMicros.toFixed(1)} μs (bar [5, 500]), β=${fit.betaNsPerAmp.toFixed(3)} ns/amp (>0?), R²=${fit.r2.toFixed(4)} (≥0.85?). Fit violated${noisyNote}.`;

  return { meta, env, rows, status, diagnosis, fit };
}
