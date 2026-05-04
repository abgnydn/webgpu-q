// ─────────────────────────────────────────────────────────────
// E6 — qubit-count ceiling for MPS in a single browser tab.
//
// Hypothesis (RESEARCH.md / protocol.md): at χ = 32 and depth-4
// brick-wall circuits, the MPS simulator reaches N ≥ 72 qubits
// with median sweep time ≤ 1.0 s on commodity hardware.
//
// Method: for each N ∈ {32, 48, 64, 72, 96, 128}, run 20 trials
// (tunable). Each trial builds a fresh MPS at χ = 32 and applies
// a deterministic depth-4 brick-wall circuit (same builder as E5).
// Record: sweep wall-seconds, peak bond dim reached, approximate
// resident RAM footprint, and whether any trial threw (OOM or a
// downstream exception).
//
// Pass bar: N = 72 completes without OOM AND median sweep ≤ 1.0 s.
// Negative results are evidence — we commit the JSON either way.
//
// Level 2 is CPU-only (pure TS), so this file is a timing harness
// rather than a GPU `timedRun` caller. The shared `stats()` helper
// gives us median/p10/p90/IQR and the noisy flag.
// ─────────────────────────────────────────────────────────────

import { MPS } from "../../src/mps.js";
import { H, X, Rz } from "../../src/gates.js";
import type { Matrix2 } from "../../src/gates.js";
import { captureEnv } from "../lib/env.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import { initGPU } from "../../src/quantum.js";
import { stats } from "../lib/stats.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

export interface E6Row {
  readonly nQubits: number;
  readonly chi: number;
  readonly depth: number;
  readonly trials: number;
  readonly completedTrials: number;
  readonly medianSeconds: number;
  readonly p10Seconds: number;
  readonly p90Seconds: number;
  readonly p99Seconds: number;
  readonly iqrSeconds: number;
  readonly stdSeconds: number;
  readonly noisy: boolean;
  readonly peakMaxBond: number;     // max across completed trials
  readonly medianMaxBond: number;
  /** Rough upper bound on tensor storage: N · 2 · χ² · 16 B. */
  readonly ramBytesBound: number;
  readonly oom: boolean;
  readonly oomReason: string | null;
}

interface GateApi {
  readonly nQubits: number;
  apply: (m: Matrix2, q: number) => void;
  cnot: (c: number, t: number) => void;
}

function applyBrickwall(c: GateApi, layers: number, rng: () => number): void {
  const N = c.nQubits;
  for (let l = 0; l < layers; l++) {
    for (let q = 0; q < N; q++) {
      const pick = Math.floor(rng() * 3);
      if (pick === 0) c.apply(H, q);
      else if (pick === 1) c.apply(Rz(rng() * Math.PI * 2), q);
      else c.apply(X, q);
    }
    const offset = l & 1;
    for (let q = offset; q + 1 < N; q += 2) c.cnot(q, q + 1);
  }
}

function seedFor(N: number, trial: number): number {
  return (SEEDS.E6_MPS_CEILING ^ (N * 2654435761) ^ (trial * 0x9e3779b1)) >>> 0;
}

export async function runE6(
  opts: {
    trials?: number;
    warmup?: number;
    Ns?: readonly number[];
    chi?: number;
    depth?: number;
    /** Abort a single trial after this many seconds (default: none). Useful in UI runs. */
    trialBudgetSec?: number;
  } = {},
): Promise<Artifact<E6Row>> {
  const trials = opts.trials ?? 20;
  const warmup = opts.warmup ?? 2;
  const Ns = opts.Ns ?? ([32, 48, 64, 72, 96, 128] as const);
  const chi = opts.chi ?? 32;
  const depth = opts.depth ?? 4;
  const trialBudgetSec = opts.trialBudgetSec ?? Infinity;

  // Level 2 is CPU-only, but we still capture GPU env for provenance
  // consistency with the rest of the research ladder.
  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E6Row[] = [];

  for (const N of Ns) {
    // ─── warmup ───
    let warmOom: string | null = null;
    for (let w = 0; w < warmup; w++) {
      try {
        const mps = new MPS({ nQubits: N, chiMax: chi });
        const rng = mulberry32(seedFor(N, -1 - w));
        applyBrickwall(mps, depth, rng);
      } catch (err) {
        warmOom = err instanceof Error ? err.message : String(err);
        break;
      }
    }

    // ─── trials ───
    const samples: number[] = [];
    const maxBondPerTrial: number[] = [];
    let oom = warmOom !== null;
    let oomReason: string | null = warmOom === null ? null : `warmup: ${warmOom}`;

    if (!oom) {
      for (let t = 0; t < trials; t++) {
        let mps: MPS;
        try {
          mps = new MPS({ nQubits: N, chiMax: chi });
        } catch (err) {
          oom = true;
          oomReason = `trial ${t} init: ${err instanceof Error ? err.message : String(err)}`;
          break;
        }
        const rng = mulberry32(seedFor(N, t));
        const t0 = performance.now();
        try {
          applyBrickwall(mps, depth, rng);
        } catch (err) {
          oom = true;
          oomReason = `trial ${t} apply: ${err instanceof Error ? err.message : String(err)}`;
          break;
        }
        const wallSeconds = (performance.now() - t0) / 1000;
        samples.push(wallSeconds);
        const bonds = mps.bondDimensions();
        maxBondPerTrial.push(bonds.length === 0 ? 1 : Math.max(...bonds, 1));

        // Early-stop once the median has stabilized (tolerant bar).
        // Honest rule: once ≥ 5 trials AND IQR ≤ 0.1·median AND each trial
        // has cost > 0.2 · remaining budget, stop early.
        if (samples.length >= 5) {
          const s = stats(samples);
          if (s.iqr <= 0.1 * s.median && wallSeconds > trialBudgetSec / 4) break;
        }
        if (wallSeconds > trialBudgetSec) break;
      }
    }

    const s = samples.length >= 1 ? stats(samples) : null;
    const peakMaxBond = maxBondPerTrial.length === 0 ? 0 : Math.max(...maxBondPerTrial);
    const sorted = maxBondPerTrial.slice().sort((a, b) => a - b);
    const medianMaxBond = sorted.length === 0 ? 0 : (sorted[Math.floor(sorted.length / 2)] ?? 0);
    const ramBytesBound = N * 2 * chi * chi * 16;

    rows.push({
      nQubits: N,
      chi,
      depth,
      trials,
      completedTrials: samples.length,
      medianSeconds: s?.median ?? NaN,
      p10Seconds: s?.p10 ?? NaN,
      p90Seconds: s?.p90 ?? NaN,
      p99Seconds: s?.p99 ?? NaN,
      iqrSeconds: s?.iqr ?? NaN,
      stdSeconds: s?.std ?? NaN,
      noisy: s?.noisy ?? false,
      peakMaxBond,
      medianMaxBond,
      ramBytesBound,
      oom,
      oomReason,
    });


    console.log(
      `[E6] N=${N} χ=${chi} depth=${depth} n=${samples.length}: `
      + `median=${(s?.median ?? 0).toFixed(3)}s p90=${(s?.p90 ?? 0).toFixed(3)}s `
      + `peakχ=${peakMaxBond} RAM≲${(ramBytesBound / 1e6).toFixed(1)}MB `
      + (oom ? `FAIL: ${oomReason}` : (s?.noisy ? "[NOISY]" : "ok")),
    );
  }

  // Pass bar: N = 72 completes AND median ≤ 1.0 s.
  const target = rows.find((r) => r.nQubits === 72);
  const noisyFrac = rows.filter((r) => r.noisy).length / Math.max(1, rows.length);
  const noisyBlocker = noisyFrac > 0.5;

  const status: Artifact<E6Row>["status"] =
    noisyBlocker ? "noisy" :
    (target && !target.oom && Number.isFinite(target.medianSeconds) && target.medianSeconds <= 1.0) ? "pass" :
    "fail";

  const meta: ArtifactMeta = {
    protocol: "E6-qubit-ceiling",
    hypothesis: `MPS at χ=${chi}, depth=${depth} brick-wall reaches N ≥ 72 in one browser tab with median sweep ≤ 1.0 s.`,
    passBar: "at N=72: no OOM AND median sweep seconds ≤ 1.0",
    seed: "E6_MPS_CEILING",
    warmup,
    trials,
  };

  const firstOom = rows.find((r) => r.oom);
  const noisyNote = noisyFrac > 0 ? ` (noisy cells: N=${rows.filter((r) => r.noisy).map((r) => r.nQubits).join(",")})` : "";

  let diagnosis: string;
  if (noisyBlocker) {
    diagnosis = `>50% of cells NOISY (std/median > 0.1)${noisyNote} — investigate before publishing.`;
  } else if (status === "pass" && target) {
    const biggest = rows.filter((r) => !r.oom).slice(-1)[0] ?? target;
    diagnosis =
      `N=72 median=${target.medianSeconds.toFixed(3)}s peakχ=${target.peakMaxBond}. `
      + `Largest completed: N=${biggest.nQubits} at ${biggest.medianSeconds.toFixed(3)}s peakχ=${biggest.peakMaxBond}.`
      + noisyNote;
  } else if (firstOom) {
    const tgtDesc = target
      ? (target.oom ? `OOM: ${target.oomReason}` : `${target.medianSeconds.toFixed(3)}s (bar 1.0s)`)
      : "not run";
    diagnosis = `First failure at N=${firstOom.nQubits}: ${firstOom.oomReason}. N=72 → ${tgtDesc}.` + noisyNote;
  } else if (target) {
    diagnosis =
      `N=72 median=${target.medianSeconds.toFixed(3)}s > 1.0s bar. `
      + `p90=${target.p90Seconds.toFixed(3)}s peakχ=${target.peakMaxBond}.`
      + noisyNote;
  } else {
    diagnosis = `N=72 not in sweep (${rows.map((r) => r.nQubits).join(",")}).` + noisyNote;
  }

  return { meta, env, rows, status, diagnosis };
}
