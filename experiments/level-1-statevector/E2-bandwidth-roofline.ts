// ─────────────────────────────────────────────────────────────
// E2 — bandwidth vs roofline.
//
// Hypothesis: at N ≥ 20, per-gate throughput is bandwidth-bound.
// Effective BW ≥ 40% of the adapter's peak memory BW.
//
// The "peak BW" number is not in the WebGPU API. We read the adapter
// description string and use a lookup table of common GPU families.
// If we can't identify the adapter, we fall back to reporting raw BW
// and flag the run with a warning in the diagnosis field — never fake
// a pass bar against a made-up peak.
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit } from "../../src/quantum.js";
import { H } from "../../src/gates.js";
import { captureEnv, type EnvBlock } from "../lib/env.js";
import { timedRun } from "../lib/runner.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

// Hand-curated peak BW by adapter description. Values are published LPDDR5 /
// GDDR6X / HBM2e bandwidths — the nominal figure from vendor specs.
// Extend as new devices are characterized.
const PEAK_BW_TABLE: Array<[RegExp, number, string]> = [
  [/apple m1\b/i,       68, "Apple M1 (LPDDR4X, 68 GB/s)"],
  [/apple m1 pro\b/i,  200, "Apple M1 Pro (LPDDR5, 200 GB/s)"],
  [/apple m1 max\b/i,  400, "Apple M1 Max (LPDDR5, 400 GB/s)"],
  [/apple m2\b/i,      100, "Apple M2 (LPDDR5, 100 GB/s)"],
  [/apple m2 pro\b/i,  200, "Apple M2 Pro (LPDDR5, 200 GB/s)"],
  [/apple m2 max\b/i,  400, "Apple M2 Max (LPDDR5, 400 GB/s)"],
  [/apple m3\b/i,      100, "Apple M3 (LPDDR5, 100 GB/s)"],
  [/apple m3 pro\b/i,  150, "Apple M3 Pro (LPDDR5, 150 GB/s)"],
  [/apple m3 max\b/i,  400, "Apple M3 Max (LPDDR5, 400 GB/s)"],
  [/apple m4\b/i,      120, "Apple M4 (LPDDR5X, 120 GB/s)"],
  [/apple m4 pro\b/i,  273, "Apple M4 Pro (LPDDR5X, 273 GB/s)"],
  [/apple m4 max\b/i,  546, "Apple M4 Max (LPDDR5X, 546 GB/s)"],
  [/rtx 4090/i,       1008, "RTX 4090 (GDDR6X, 1008 GB/s)"],
  [/rtx 4080/i,        716, "RTX 4080 (GDDR6X, 716 GB/s)"],
  [/rtx 3090/i,        936, "RTX 3090 (GDDR6X, 936 GB/s)"],
  [/rtx 3080/i,        760, "RTX 3080 (GDDR6X, 760 GB/s)"],
  [/rtx 3070/i,        448, "RTX 3070 (GDDR6, 448 GB/s)"],
  [/radeon rx 7900 xtx/i, 960, "Radeon RX 7900 XTX (GDDR6, 960 GB/s)"],

  // NVIDIA datacenter parts — the cloud-GPU CI targets (see
  // docs/webgpu-ci-providers.md). NB: headless Chrome on the web masks
  // device/description to just `vendor=nvidia, architecture=turing|ampere|ada`
  // (same fingerprinting reason as Apple below), so these device-name rows only
  // match where the description is actually populated (self-hosted runners, some
  // browsers). On a masked adapter use `?peak=NNN` — e.g. the Tesla T4 is 320.
  [/a100[^0-9]*80|a100.*80\s*gb/i, 2039, "NVIDIA A100 80GB (HBM2e, 2039 GB/s)"],
  [/\ba100\b/i,                    1555, "NVIDIA A100 40GB (HBM2e, 1555 GB/s)"],
  [/tesla v100|\bv100\b/i,          900, "NVIDIA V100 (HBM2, 900 GB/s)"],
  [/a10g\b|\ba10\b/i,               600, "NVIDIA A10/A10G (GDDR6, 600 GB/s)"],
  [/tesla t4|nvidia t4|\bt4\b/i,    320, "NVIDIA T4 (GDDR6, 320 GB/s)"],
  [/\bl4\b|nvidia l4/i,             300, "NVIDIA L4 (GDDR6, 300 GB/s)"],
  // NVIDIA consumer Turing/Ampere (exposed by some non-Chrome WebGPU stacks)
  [/rtx 2080 ti/i,                  616, "RTX 2080 Ti (GDDR6, 616 GB/s)"],
  [/rtx 2070/i,                     448, "RTX 2070 (GDDR6, 448 GB/s)"],
  [/rtx 2060/i,                     336, "RTX 2060 (GDDR6, 336 GB/s)"],

  // Chromium intentionally returns only `vendor=apple`, `architecture=metal-3`
  // (no device or description) for fingerprinting reasons, so we can't pin
  // down which Apple Silicon variant we're on. Map "apple metal-3" to the
  // mid-range 200 GB/s of M{2,3} Pro — accurate for the most common dev
  // machine, conservative for Max chips, and over-generous for base chips.
  // Override via `?peak=NNN` in the URL or `runE2({ peakGbps: NNN })` when
  // the exact part is known. Order matters: place AFTER the specific M-chip
  // matches so a fully-described "Apple M2 Max" still wins.
  [/apple metal-3/i,    200, "Apple Silicon (Metal-3, est. 200 GB/s — override via ?peak=N for exact)"],
  [/apple metal-2/i,    100, "Apple Silicon (Metal-2, est. 100 GB/s — override via ?peak=N for exact)"],
];

interface PeakBW {
  readonly gbps: number;
  readonly source: string;        // which table entry or fallback
  readonly guessed: boolean;      // true = fell back to a guess
}

function lookupPeakBW(env: EnvBlock, override?: number): PeakBW {
  if (override && override > 0) {
    return { gbps: override, source: `user-supplied peak BW (override=${override} GB/s)`, guessed: false };
  }
  const haystack = `${env.adapter.vendor} ${env.adapter.architecture} ${env.adapter.device} ${env.adapter.description}`;
  for (const [re, gbps, source] of PEAK_BW_TABLE) {
    if (re.test(haystack)) return { gbps, source, guessed: false };
  }
  // Fallback: assume 200 GB/s and flag it. Better than silently passing.
  return { gbps: 200, source: `UNRECOGNIZED adapter '${haystack.trim() || "<empty>"}' — using 200 GB/s placeholder`, guessed: true };
}

/** Query-param override — `?peak=400` in the URL makes the run self-service. */
function readPeakFromUrl(): number | undefined {
  if (typeof window === "undefined") return undefined;
  const p = new URLSearchParams(window.location.search).get("peak");
  if (!p) return undefined;
  const n = Number(p);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

export interface E2Row {
  readonly nQubits: number;
  readonly nStates: number;
  readonly gatesPerBatch: number;
  readonly medianSeconds: number;
  readonly p90Seconds: number;
  readonly noisy: boolean;
  readonly bytesPerGate: number;
  readonly effectiveGbps: number;
  readonly peakGbps: number;
  readonly fractionOfPeak: number;
}

export async function runE2(
  opts: { trials?: number; warmup?: number; Ns?: readonly number[]; peakGbps?: number } = {},
): Promise<Artifact<E2Row>> {
  const trials = opts.trials ?? 20;
  const warmup = opts.warmup ?? 5;
  const Ns = opts.Ns ?? [16, 18, 20, 22, 24];

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no adapter for env block");
  const env = await captureEnv(device, adapter);
  const peak = lookupPeakBW(env, opts.peakGbps ?? readPeakFromUrl());

  const rows: E2Row[] = [];
  for (const n of Ns) {
     
    console.log(`[E2] N=${n} peak=${peak.gbps} GB/s (${peak.source})`);
    const gpu = new QuantumCircuit({ device, nQubits: n });
    const gatesPerBatch = 64;
    try {
      const { stats: s } = await timedRun(device, () => {
        for (let k = 0; k < gatesPerBatch; k++) gpu.apply(H, 0);
      }, { trials, warmup });

      const nStates = 1 << n;
      const bytesPerGate = 2 * 8 * nStates; // read + write vec2<f32>
      const effectiveGbps = (bytesPerGate * gatesPerBatch) / s.median / 1e9;

      rows.push({
        nQubits: n,
        nStates,
        gatesPerBatch,
        medianSeconds: s.median,
        p90Seconds: s.p90,
        noisy: s.noisy,
        bytesPerGate,
        effectiveGbps,
        peakGbps: peak.gbps,
        fractionOfPeak: effectiveGbps / peak.gbps,
      });
       
      console.log(`      → ${effectiveGbps.toFixed(1)} GB/s  (${(effectiveGbps/peak.gbps*100).toFixed(1)}% of peak)  ${s.noisy ? "[NOISY]" : ""}`);
    } finally {
      gpu.dispose();
    }
  }

  // Pass bar: N=22 and N=24 both ≥ 40% of peak.
  const highNRows = rows.filter((r) => r.nQubits >= 22);
  const allHighPassed = highNRows.length > 0 && highNRows.every((r) => r.fractionOfPeak >= 0.4);
  const anyNoisy = rows.some((r) => r.noisy);
  const status: Artifact<E2Row>["status"] =
    peak.guessed ? "noisy"
    : anyNoisy ? "noisy"
    : allHighPassed ? "pass"
    : "fail";

  const meta: ArtifactMeta = {
    protocol: "E2-bandwidth-roofline",
    hypothesis: "At N ≥ 20 the single-qubit gate kernel is memory-bandwidth-bound, reaching ≥ 40% of peak BW at N ≥ 22.",
    passBar: "effectiveGbps ≥ 0.40 × peakGbps for N ∈ {22, 24}",
    seed: "E2_BANDWIDTH",
    warmup,
    trials,
  };

  const worst = rows.length === 0 ? null : rows.reduce((w, r) => r.fractionOfPeak < w.fractionOfPeak ? r : w);
  const diagnosis =
    peak.guessed ? `peak BW guessed — ${peak.source}. Run again once this adapter is in the table.` :
    anyNoisy ? `NOISY samples at N=${rows.filter((r) => r.noisy).map((r) => r.nQubits).join(",")}. Investigate before publishing.` :
    allHighPassed ? `All N ≥ 22 cells ≥ 40% of peak. Best: N=${bestPeakRow(rows).nQubits} at ${(bestPeakRow(rows).fractionOfPeak*100).toFixed(1)}%.` :
    `Below 40% of peak: N=${worst!.nQubits} at ${(worst!.fractionOfPeak*100).toFixed(1)}%. Suggests dispatch overhead — see E4.`;

  return { meta, env, rows, status, diagnosis };
}

function bestPeakRow(rows: readonly E2Row[]): E2Row {
  return rows.reduce((b, r) => r.fractionOfPeak > b.fractionOfPeak ? r : b);
}
