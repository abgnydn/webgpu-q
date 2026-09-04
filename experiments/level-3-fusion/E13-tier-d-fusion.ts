// ─────────────────────────────────────────────────────────────
// E13 — Tier D 4-qubit tile fusion.
//
// E11 (Tier B): 3 ops per pair → 1 dispatch (2.69× headline)
// E12 (Tier C): 5 ops per triple → 1 dispatch (4.22× headline)
// E13 (Tier D): 7 ops per quadruple → 1 dispatch — same playbook,
// wider window. The cascade pattern is
//
//   (single ⊗ single ⊗ single ⊗ single)
//     · CNOT(qLo, qMid1) · CNOT(qMid1, qMid2) · CNOT(qMid2, qHi)
//
// Layer pattern: at each depth step, every contiguous 4-tuple
// (q, q+1, q+2, q+3) for q ∈ {0, 4, 8, …} runs the cascade.
//
// Hypothesis: at N ≥ 12 with depth ≥ 20, fused throughput beats
// unfused by ≥ 5× (7 dispatches → 1 per tile).
//
// Pass bar: best (median over warmup+trials) speedup ≥ 5.0× AND
// worst F ≥ 1 − 1e-5.
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit } from "../../src/quantum.js";
import { Rx, Ry, Rz, type Matrix2 } from "../../src/gates.js";
import { fuseQuadCascade } from "../../src/four-qubit-dense.js";
import { captureEnv } from "../lib/env.js";
import { timedRun } from "../lib/runner.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";
import { stateMetrics, passed, FIDELITY_PASS_BAR } from "../lib/fidelity.js";

export interface E13Row {
  readonly nQubits: number;
  readonly depth: number;
  readonly tilesPerLayer: number;
  readonly unfusedSec: number;
  readonly fusedSec: number;
  readonly speedup: number;
  readonly fidelity: number;
  readonly unfusedNoisy: boolean;
  readonly fusedNoisy: boolean;
}

export interface E13Summary {
  readonly bestSpeedup: number;
  readonly bestSpeedupAt: { nQubits: number; depth: number };
  readonly worstFidelity: number;
}

interface QuadLayer {
  readonly tiles: readonly {
    qLo: number; qMid1: number; qMid2: number; qHi: number;
    Ulo: Matrix2; Umid1: Matrix2; Umid2: Matrix2; Uhi: Matrix2;
  }[];
}

function pickGate(rng: () => number): Matrix2 {
  const pick = Math.floor(rng() * 3);
  const theta = rng() * 2 * Math.PI;
  if (pick === 0) return Rx(theta);
  if (pick === 1) return Ry(theta);
  return Rz(theta);
}

function buildLayers(n: number, depth: number, rng: () => number): QuadLayer[] {
  const layers: QuadLayer[] = [];
  for (let l = 0; l < depth; l++) {
    const tiles: QuadLayer["tiles"][number][] = [];
    for (let q = 0; q + 3 < n; q += 4) {
      tiles.push({
        qLo: q, qMid1: q + 1, qMid2: q + 2, qHi: q + 3,
        Ulo: pickGate(rng),
        Umid1: pickGate(rng),
        Umid2: pickGate(rng),
        Uhi: pickGate(rng),
      });
    }
    layers.push({ tiles });
  }
  return layers;
}

function applyUnfusedLayer(c: QuantumCircuit, layer: QuadLayer): void {
  for (const t of layer.tiles) {
    c.apply(t.Ulo, t.qLo);
    c.apply(t.Umid1, t.qMid1);
    c.apply(t.Umid2, t.qMid2);
    c.apply(t.Uhi, t.qHi);
    c.cnot(t.qLo, t.qMid1);
    c.cnot(t.qMid1, t.qMid2);
    c.cnot(t.qMid2, t.qHi);
  }
}

function applyFusedLayer(c: QuantumCircuit, layer: QuadLayer): void {
  for (const t of layer.tiles) {
    c.applyDense16x16(
      fuseQuadCascade(t.Ulo, t.Umid1, t.Umid2, t.Uhi),
      t.qLo, t.qMid1, t.qMid2, t.qHi,
    );
  }
}

async function readState(c: QuantumCircuit): Promise<Float32Array> {
  return c.amplitudes();
}

export async function runE13(
  opts: {
    trials?: number;
    warmup?: number;
    Ns?: readonly number[];
    depths?: readonly number[];
  } = {},
): Promise<Artifact<E13Row> & { summary: E13Summary }> {
  const trials = opts.trials ?? 5;
  const warmup = opts.warmup ?? 3;
  const Ns = opts.Ns ?? [12, 16, 20];
  const depths = opts.depths ?? [20, 80];

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E13: no adapter");
  const env = await captureEnv(device, adapter);

  const rows: E13Row[] = [];
  let allPassed = true;

  for (const n of Ns) {
    if (n < 4) continue;
    const cUnfused = new QuantumCircuit({ device, nQubits: n });
    const cFused = new QuantumCircuit({ device, nQubits: n });

    try {
      for (const D of depths) {
        const seed = ((SEEDS.E13_TIERD_FUSION ^ (n * 2654435761) ^ (D * 0x9e3779b1)) >>> 0);
        const layers = buildLayers(n, D, mulberry32(seed));
        const tilesPerLayer = layers[0]?.tiles.length ?? 0;
        if (tilesPerLayer === 0) continue;

        cUnfused.reset();
        for (const l of layers) applyUnfusedLayer(cUnfused, l);
        const refState = await readState(cUnfused);

        cFused.reset();
        for (const l of layers) applyFusedLayer(cFused, l);
        const fusedState = await readState(cFused);

        const m = stateMetrics(refState, fusedState);
        const F = m.fidelity;
        allPassed = allPassed && passed(m);

        const { stats: unS } = await timedRun(device, () => {
          cUnfused.reset();
          for (const l of layers) applyUnfusedLayer(cUnfused, l);
        }, { trials, warmup });

        const { stats: fuS } = await timedRun(device, () => {
          cFused.reset();
          for (const l of layers) applyFusedLayer(cFused, l);
        }, { trials, warmup });

        const speedup = unS.median / fuS.median;
        rows.push({
          nQubits: n,
          depth: D,
          tilesPerLayer,
          unfusedSec: unS.median,
          fusedSec: fuS.median,
          speedup,
          fidelity: F,
          unfusedNoisy: unS.noisy,
          fusedNoisy: fuS.noisy,
        });

        console.log(
          `[E13] N=${n} D=${D} tiles=${tilesPerLayer}: ` +
          `unfused=${(unS.median * 1000).toFixed(2)} ms, ` +
          `fused=${(fuS.median * 1000).toFixed(2)} ms, ` +
          `speedup=${speedup.toFixed(2)}× F=${F.toFixed(7)} |ψ|²=${m.normTest.toFixed(7)} ` +
          `${unS.noisy || fuS.noisy ? "[NOISY]" : ""}`,
        );
      }
    } finally {
      cUnfused.dispose();
      cFused.dispose();
    }
  }

  const SPEEDUP_PASS = 5.0;

  const worstF = rows.length > 0 ? rows.reduce((a, r) => Math.min(a, r.fidelity), 1) : 0;
  const best = rows.length > 0
    ? rows.reduce((a, r) => (r.speedup > a.speedup ? r : a), rows[0]!)
    : null;

  const noisyFrac = rows.filter((r) => r.unfusedNoisy || r.fusedNoisy).length / Math.max(1, rows.length);
  const correctnessOk = rows.length > 0 && worstF >= FIDELITY_PASS_BAR && allPassed;
  const speedupOk = !!best && best.speedup >= SPEEDUP_PASS;
  const noisyBlocker = noisyFrac > 0.5;

  const status: Artifact<E13Row>["status"] =
    rows.length === 0 ? "fail" :
    !correctnessOk ? "fail" :
    noisyBlocker ? "noisy" :
    speedupOk ? "pass" : "fail";

  const summary: E13Summary = {
    bestSpeedup: best?.speedup ?? 0,
    bestSpeedupAt: best ? { nQubits: best.nQubits, depth: best.depth } : { nQubits: 0, depth: 0 },
    worstFidelity: worstF,
  };

  const meta: ArtifactMeta = {
    protocol: "E13-tier-d-fusion",
    hypothesis:
      "Four-qubit tile fusion (7 ops per tile per layer → 1 dense 16×16 dispatch) "
      + "delivers ≥ 5× speedup vs the unfused 7-dispatch path at N ≥ 12, while "
      + "staying within F ≥ 1 − 1e-5 of the unfused statevector.",
    passBar: "best (median over warmup+trials) speedup ≥ 5.0× AND worst F ≥ 1 − 1e-5",
    seed: "E13_TIERD_FUSION",
    warmup,
    trials,
  };

  // Honest negative case worth a written-out explanation: the pass bar
  // assumed dispatch overhead dominates, but per-block compute scales
  // 4× per tier (4×4 → 8×8 → 16×16) while memory traffic only 2×, so
  // Tier D's per-block arithmetic intensity (≈1024 fmuls per 16-amp
  // block) crosses the bandwidth → compute boundary on Apple Metal-3.
  // The 3× speedup IS real and useful — Tier D just doesn't deliver
  // the linear "ops collapsed = speedup" the protocol hypothesized.
  const diagnosis = rows.length === 0
    ? "No cells ran (every N < 4?)."
    : !correctnessOk
      ? `Correctness FAIL: worst F=${worstF.toFixed(7)} < ${FIDELITY_PASS_BAR}.`
      : status === "pass" && best
        ? `Best speedup ${best.speedup.toFixed(2)}× at N=${best.nQubits} D=${best.depth}. ` +
          `Worst F=${worstF.toFixed(7)} (≥ ${FIDELITY_PASS_BAR}).`
        : status === "noisy"
          ? `Correctness OK (worst F=${worstF.toFixed(7)}) but timing too noisy.`
          : `Correctness OK (worst F=${worstF.toFixed(7)}) but best speedup ${best?.speedup.toFixed(2) ?? "n/a"}× < ${SPEEDUP_PASS}. ` +
            `Honest negative: Tier D crosses into compute-bound territory — per-block ` +
            `cmul count (256) grows 4× over Tier C while memory traffic only 2×, so ` +
            `dispatch collapse no longer translates 1:1 into wall-time speedup. ` +
            // Deliberately unnumbered: hardcoding Tier C's speedup here made every
            // E13 artifact contradict the E12 artifact generated beside it once
            // Tier C moved 4.18× → 4.22×. Read the number off E12's own artifact.
            `Tier C (8×8) remains the bandwidth-bound sweet spot.`;

  return { meta, env, rows, status, diagnosis, summary };
}
