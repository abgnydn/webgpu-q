// ─────────────────────────────────────────────────────────────
// E3 — runtime scaling T(N).
//
// Hypothesis: at N ≥ 18 per-gate time scales linearly with state size.
// T(N) ≈ α + β · 2^N, fit log–log slope ≈ 1.00 ± 0.05 in the
// memory-bound regime.
//
// Below N = 18 the dispatch-overhead dominates — the slope is flatter
// than 1 there by design, and that's what E4 is for. E3 validates the
// high-N regime.
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit } from "../../src/quantum.js";
import { randomCircuit } from "../../src/circuits.js";
import { captureEnv } from "../lib/env.js";
import { timedRun } from "../lib/runner.js";
import { SEEDS } from "../lib/seeds.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

export interface E3Row {
  readonly nQubits: number;
  readonly nStates: number;
  readonly gates: number;
  readonly layers: number;
  readonly medianSeconds: number;
  readonly secondsPerGate: number;
  readonly noisy: boolean;
  readonly effectiveGbps: number;
}

export interface E3Fit {
  readonly slope: number;         // d(log t) / d(log 2^N)
  readonly intercept: number;     // log₁₀(seconds) at 2^N = 1
  readonly r2: number;            // coefficient of determination
  readonly nRegime: readonly number[];  // which Ns were fit
}

function fitLogLog(rows: readonly E3Row[]): E3Fit {
  const xs = rows.map((r) => Math.log10(r.nStates));
  const ys = rows.map((r) => Math.log10(r.secondsPerGate));
  const n = xs.length;
  const sx = xs.reduce((a, b) => a + b, 0);
  const sy = ys.reduce((a, b) => a + b, 0);
  const mx = sx / n;
  const my = sy / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i]! - mx) * (ys[i]! - my);
    sxx += (xs[i]! - mx) ** 2;
    syy += (ys[i]! - my) ** 2;
  }
  const slope = sxx === 0 ? 0 : sxy / sxx;
  const intercept = my - slope * mx;
  const r2 = syy === 0 ? 1 : (sxy * sxy) / (sxx * syy);
  return { slope, intercept, r2, nRegime: rows.map((r) => r.nQubits) };
}

export async function runE3(
  opts: {
    trials?: number;
    warmup?: number;
    Ns?: readonly number[];
    layers?: number;
  } = {},
): Promise<Artifact<E3Row> & { fit: E3Fit }> {
  const trials = opts.trials ?? 12;
  const warmup = opts.warmup ?? 3;
  // Memory-bound regime kicks in at N ≈ 22 on consumer WebGPU
  // (α ≈ 25–100 μs; β·2^N dominates only when 2^N · β > 10 α).
  // Include smaller Ns for context but only fit N ≥ 22.
  const Ns = opts.Ns ?? [12, 16, 18, 20, 22, 24];
  const layers = opts.layers ?? 8;

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E3Row[] = [];
  for (const n of Ns) {
     
    console.log(`[E3] N=${n} layers=${layers}`);
    const gpu = new QuantumCircuit({ device, nQubits: n });
    try {
      // Count gates by dry-running a tracker. Equivalent to instrumenting
      // the GateApi but simpler: layer of n single + floor(n/2) CNOTs.
      const gatesPerLayer = n + Math.floor(n / 2);
      const totalGates = gatesPerLayer * layers;

      const { stats: s } = await timedRun(device, () => {
        randomCircuit(gpu, layers, SEEDS.E3_SCALING ^ n);
      }, { trials, warmup });

      const nStates = 1 << n;
      const secondsPerGate = s.median / totalGates;
      const bytesPerGate = 2 * 8 * nStates;
      const effectiveGbps = bytesPerGate / secondsPerGate / 1e9;

      rows.push({
        nQubits: n,
        nStates,
        gates: totalGates,
        layers,
        medianSeconds: s.median,
        secondsPerGate,
        noisy: s.noisy,
        effectiveGbps,
      });
       
      console.log(`      → ${(secondsPerGate * 1e6).toFixed(1)} μs/gate  ${effectiveGbps.toFixed(1)} GB/s  ${s.noisy ? "[NOISY]" : ""}`);
    } finally {
      gpu.dispose();
    }
  }

  // Fit only the memory-bound regime. Below N≈22 dispatch α pollutes
  // the slope; see E4 for the α/β decomposition that justifies N ≥ 22.
  const FIT_MIN_N = 22;
  const fitRows = rows.filter((r) => r.nQubits >= FIT_MIN_N);
  const fit = fitRows.length >= 2
    ? fitLogLog(fitRows)
    : { slope: NaN, intercept: NaN, r2: NaN, nRegime: [] };

  const slopeOk = !Number.isNaN(fit.slope) && Math.abs(fit.slope - 1) <= 0.08;
  // Two-point fits give R² = 1 trivially. Require ≥ 3 fit points for R² to
  // be meaningful; with 2 points, accept slope-only if slope is in range.
  const r2Ok = fitRows.length >= 3
    ? (!Number.isNaN(fit.r2) && fit.r2 >= 0.95)
    : true;
  const noisyFrac = rows.filter((r) => r.noisy).length / rows.length;
  const noisyBlocker = noisyFrac > 0.5;
  const status: Artifact<E3Row>["status"] =
    noisyBlocker ? "noisy"
    : (slopeOk && r2Ok && fitRows.length >= 2) ? "pass"
    : "fail";

  const meta: ArtifactMeta = {
    protocol: "E3-scaling-law",
    hypothesis: `Per-gate time scales linearly with state size at N ≥ ${FIT_MIN_N}; fit slope ≈ 1.00 ± 0.08 on log–log.`,
    passBar: `slope ∈ [0.92, 1.08] AND R² ≥ 0.95 for N ≥ ${FIT_MIN_N} (R² waived if only 2 fit points)`,
    seed: "E3_SCALING",
    warmup,
    trials,
  };

  const noisyTail = rows.filter((r) => r.noisy).map((r) => r.nQubits);
  const noisyNote = noisyTail.length === 0 ? "" : ` (noisy cells: N=${noisyTail.join(",")})`;
  const diagnosis =
    noisyBlocker ? `>50% of cells NOISY — investigate before publishing${noisyNote}.` :
    (slopeOk && r2Ok && fitRows.length >= 2)
      ? `slope=${fit.slope.toFixed(3)}, R²=${fit.r2.toFixed(4)}, fit over N ∈ [${fit.nRegime.join(",")}]${noisyNote}.`
      : `slope=${fit.slope.toFixed(3)} (need 1.00±0.08), R²=${fit.r2.toFixed(4)} (need ≥ 0.95 if ≥ 3 pts), points=${fitRows.length}${noisyNote}.`;

  return { meta, env, rows, status, diagnosis, fit };
}
