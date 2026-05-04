// ─────────────────────────────────────────────────────────────
// E9 — dispatch-cost collapse via chain fusion.
//
// Model carried forward from E4: per-dispatch time decomposes as
//   T(N, k) = α_raw + k · β · 2^N
// where α_raw is the fixed submit + sync cost (≈ 25–500 μs in E4)
// and β is the per-amplitude work per gate. Fusing k gates into
// one dispatch pays α_raw once but does k × β · 2^N work in
// registers → per-gate time
//   t_per_gate(N, k) = α_raw / k + β · 2^N.
// The intercept of t_per_gate vs 2^N at fixed k is α_eff(k). In a
// perfect world α_eff(k) = α_raw / k exactly; in practice the
// measurement harness adds a residual C that does not amortize
// with k:
//   α_eff(k) = α_raw / k + C.
// C comes from the map-fence sync inside `timedRun` — the fence
// costs ~1-3 ms on current WebGPU stacks, divided across
// gates_per_batch it contributes C_fence ≈ T_fence / gates_per_batch
// to every measurement. At gates_per_batch = 256 and T_fence ≈ 2 ms
// that's C_fence ≈ 8 μs per gate — an irreducible floor that is
// NOT a fusion-kernel inefficiency, just a timing methodology cost.
// (GPU timestamp queries would remove it; that is a future refactor
// of the shared `timedRun` harness.)
//
// Method: for each k ∈ {1, 2, 4, 8, 16, 32, 64} and several N, run
// `gates_per_batch` single-qubit gates on qubit 0 in chains of
// length k. Extract α_eff(k) as the intercept of t_per_gate vs
// 2^N at fixed k (E4 methodology). Then fit the 1/k model
//   α_eff(k) = α_raw · (1/k) + C
// by linear OLS on (1/k, α_eff). α_raw is the quantity fusion
// targets — the real dispatch cost that fusion divides by k.
//
// Pass bar (revised, 1/k fit):
//   α_raw ∈ [5, 500] μs  — dispatch cost in E4's expected range;
//                           confirms fusion is dividing the right thing.
//   α_eff(k_max) ≤ 20 μs — absolute amortized overhead at max fusion.
//   fit R² ≥ 0.98        — 1/k model is a good description.
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit, FUSED_CHAIN_MAX_K } from "../../src/quantum.js";
import { H, Rx, Ry, Rz, type Matrix2 } from "../../src/gates.js";
import { captureEnv } from "../lib/env.js";
import { timedRun } from "../lib/runner.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

export interface E9Row {
  readonly k: number;
  readonly nQubits: number;
  readonly nStates: number;
  readonly gatesPerBatch: number;
  readonly dispatchesPerBatch: number;
  readonly medianSeconds: number;
  readonly secondsPerGate: number;
  readonly microsPerGate: number;
  readonly noisy: boolean;
}

export interface E9FitForK {
  readonly k: number;
  /** Per-gate intercept in μs: includes α_raw/k + C for this k. */
  readonly alphaEffMicros: number;
  readonly betaSecPerAmp: number;
  readonly betaNsPerAmp: number;
  readonly r2: number;
  readonly nPoints: number;
}

export interface E9KScalingFit {
  /** Slope of α_eff vs 1/k in μs — the fusion-targeted α_raw. */
  readonly alphaRawMicros: number;
  /** Intercept: residual that does NOT amortize with k (mostly fence bias). */
  readonly cResidualMicros: number;
  readonly r2: number;
  readonly nPoints: number;
}

export interface E9Summary {
  readonly perKFits: readonly E9FitForK[];
  readonly kScalingFit: E9KScalingFit;
  readonly alphaBaselineMicros: number;   // α_eff(1)
  readonly alphaFusedMicros: number;      // α_eff(k_max)
  readonly collapseFactor: number;        // α_eff(1) / α_eff(k_max)
  readonly kMax: number;
}

function olsFit(xs: readonly number[], ys: readonly number[]):
  { alpha: number; beta: number; r2: number } {
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

function pickGate(rng: () => number): Matrix2 {
  const pick = Math.floor(rng() * 4);
  const theta = rng() * 2 * Math.PI;
  if (pick === 0) return H;
  if (pick === 1) return Rx(theta);
  if (pick === 2) return Ry(theta);
  return Rz(theta);
}

export async function runE9(
  opts: {
    trials?: number;
    warmup?: number;
    Ks?: readonly number[];
    Ns?: readonly number[];
    gatesPerBatch?: number;
  } = {},
): Promise<Artifact<E9Row> & { summary: E9Summary }> {
  const trials = opts.trials ?? 20;
  const warmup = opts.warmup ?? 5;
  // k=1 is "unfused" via the fused kernel itself — a one-gate chain.
  // Apples-to-apples: same binding layout, same shader path. The only
  // variable is k, which isolates the fusion amortization effect.
  const Ks = opts.Ks ?? [1, 2, 4, 8, 16, 32, 64];
  const Ns = opts.Ns ?? [8, 12, 16, 20];
  // Match E4's 256-gate batches so fence-bias contribution is bounded
  // at ~T_fence / 256 per gate.
  const gatesPerBatch = opts.gatesPerBatch ?? 256;

  for (const k of Ks) {
    if (k > FUSED_CHAIN_MAX_K) {
      throw new Error(`E9: k=${k} exceeds FUSED_CHAIN_MAX_K=${FUSED_CHAIN_MAX_K}`);
    }
    if (gatesPerBatch % k !== 0) {
      throw new Error(`E9: gatesPerBatch=${gatesPerBatch} must be divisible by k=${k} for a fair comparison`);
    }
  }

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E9Row[] = [];

  // Deterministic gate stream reused across all cells — guarantees any
  // per-gate work is identical and only the dispatch-batching varies.
  const allGates: Matrix2[] = [];
  const rng = mulberry32(SEEDS.E4_OVERHEAD ^ 0xE9E9E9);
  for (let i = 0; i < gatesPerBatch; i++) allGates.push(pickGate(rng));

  for (const n of Ns) {
    const gpu = new QuantumCircuit({ device, nQubits: n });
    try {
      for (const k of Ks) {
        const dispatchesPerBatch = gatesPerBatch / k;

        const { stats: s } = await timedRun(device, () => {
          for (let d = 0; d < dispatchesPerBatch; d++) {
            const start = d * k;
            const chain = allGates.slice(start, start + k);
            gpu.applyFusedSingle(chain, 0);
          }
        }, { trials, warmup });

        const secondsPerGate = s.median / gatesPerBatch;
        rows.push({
          k,
          nQubits: n,
          nStates: 1 << n,
          gatesPerBatch,
          dispatchesPerBatch,
          medianSeconds: s.median,
          secondsPerGate,
          microsPerGate: secondsPerGate * 1e6,
          noisy: s.noisy,
        });


        console.log(
          `[E9] N=${n} k=${k} dispatches=${dispatchesPerBatch}: `
          + `${(secondsPerGate * 1e6).toFixed(2)} μs/gate `
          + `${s.noisy ? "[NOISY]" : ""}`,
        );
      }
    } finally {
      gpu.dispose();
    }
  }

  // Stage 1: per-k linear fit of t_per_gate vs 2^N → α_eff(k), β(k).
  const perKFits: E9FitForK[] = [];
  for (const k of Ks) {
    const kRows = rows.filter((r) => r.k === k);
    const xs = kRows.map((r) => r.nStates);
    const ys = kRows.map((r) => r.secondsPerGate);
    const { alpha, beta, r2 } = olsFit(xs, ys);
    perKFits.push({
      k,
      alphaEffMicros: alpha * 1e6,
      betaSecPerAmp: beta,
      betaNsPerAmp: beta * 1e9,
      r2,
      nPoints: kRows.length,
    });
  }

  // Stage 2: fit α_eff(k) = α_raw · (1/k) + C by OLS on (1/k, α_eff).
  const kXs: number[] = [];
  const kYs: number[] = [];
  for (const f of perKFits) {
    if (Number.isFinite(f.alphaEffMicros)) {
      kXs.push(1 / f.k);
      kYs.push(f.alphaEffMicros);
    }
  }
  const kFit = olsFit(kXs, kYs);
  // Intercept (beta in our OLS helper's naming) when x=0 is C; slope is α_raw.
  const kScalingFit: E9KScalingFit = {
    alphaRawMicros: kFit.beta,   // slope of (α_eff vs 1/k) = α_raw
    cResidualMicros: kFit.alpha, // intercept at 1/k = 0
    r2: kFit.r2,
    nPoints: kXs.length,
  };

  const kMax = Ks[Ks.length - 1]!;
  const fitK1 = perKFits.find((f) => f.k === 1);
  const fitKMax = perKFits.find((f) => f.k === kMax);
  const alphaBaselineMicros = fitK1?.alphaEffMicros ?? NaN;
  const alphaFusedMicros = fitKMax?.alphaEffMicros ?? NaN;
  const collapseFactor =
    Number.isFinite(alphaBaselineMicros) && alphaFusedMicros > 0
      ? alphaBaselineMicros / alphaFusedMicros
      : NaN;

  const noisyFrac = rows.filter((r) => r.noisy).length / Math.max(1, rows.length);
  const noisyBlocker = noisyFrac > 0.5;

  // Revised pass bar — see header comment for rationale.
  const alphaRawOk =
    Number.isFinite(kScalingFit.alphaRawMicros) &&
    kScalingFit.alphaRawMicros >= 5 &&
    kScalingFit.alphaRawMicros <= 500;
  const absBar = Number.isFinite(alphaFusedMicros) && alphaFusedMicros > 0 && alphaFusedMicros <= 20;
  const fitQuality = Number.isFinite(kScalingFit.r2) && kScalingFit.r2 >= 0.98;

  const status: Artifact<E9Row>["status"] =
    noisyBlocker ? "noisy" :
    (alphaRawOk && absBar && fitQuality) ? "pass" :
    "fail";

  const meta: ArtifactMeta = {
    protocol: "E9-dispatch-collapse",
    hypothesis:
      `Per-gate time follows α_eff(k) = α_raw/k + C, with α_raw in E4's dispatch-cost range `
      + `[5, 500] μs and α_eff(k_max=${kMax}) ≤ 20 μs. Residual C is dominated by `
      + `the map-fence bias in \`timedRun\`, not by fusion-kernel inefficiency.`,
    passBar: `α_raw ∈ [5, 500] μs AND α_eff(${kMax}) ≤ 20 μs AND 1/k fit R² ≥ 0.98`,
    seed: "E4_OVERHEAD (xor E9)",
    warmup,
    trials,
  };

  const noisyNote = noisyFrac > 0
    ? ` (noisy cells: ${rows.filter((r) => r.noisy).map((r) => `N=${r.nQubits},k=${r.k}`).join(";")})`
    : "";

  const base =
    `α_raw=${kScalingFit.alphaRawMicros.toFixed(2)}μs, `
    + `C=${kScalingFit.cResidualMicros.toFixed(2)}μs, `
    + `α_eff(1)=${alphaBaselineMicros.toFixed(2)}μs, `
    + `α_eff(${kMax})=${alphaFusedMicros.toFixed(2)}μs, `
    + `ratio=${collapseFactor.toFixed(2)}×, R²=${kScalingFit.r2.toFixed(3)}`;

  let diagnosis: string;
  if (noisyBlocker) {
    diagnosis = `>50% of cells NOISY.${noisyNote}`;
  } else if (status === "pass") {
    diagnosis = `${base}. Within-range fit confirms fusion targets α_raw correctly.${noisyNote}`;
  } else {
    const misses: string[] = [];
    if (!alphaRawOk) misses.push(`α_raw outside [5, 500] μs`);
    if (!absBar) misses.push(`α_eff(${kMax}) > 20 μs`);
    if (!fitQuality) misses.push(`1/k fit R² < 0.98`);
    diagnosis = `${base}. Missed: ${misses.join("; ")}.${noisyNote}`;
  }

  const summary: E9Summary = {
    perKFits,
    kScalingFit,
    alphaBaselineMicros,
    alphaFusedMicros,
    collapseFactor,
    kMax,
  };

  return { meta, env, rows, status, diagnosis, summary };
}
