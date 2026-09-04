// ─────────────────────────────────────────────────────────────
// E12 — Tier C 3-qubit tile fusion.
//
// E11 (Tier B) collapsed (single ⊗ single) · CNOT into one 4×4
// dispatch — 3 ops → 1 dispatch per pair. E12 widens the tile:
// (single ⊗ single ⊗ single) · CNOT(qLo, qMid) · CNOT(qMid, qHi)
// is a 5-op cascade folded into one 8×8 dispatch.
//
// Layer pattern: at each depth step, every contiguous 3-tuple
// (q, q+1, q+2) for q ∈ {0, 3, 6, …} runs the cascade.
//
// Hypothesis: at N ≥ 12 with depth ≥ 20, fused throughput beats
// unfused by ≥ 3× (5 dispatches → 1 per tile).
//
// Pass bar: best (median over warmup+trials) speedup ≥ 3.0× AND
// worst F ≥ 1 − 1e-5. Speedup is the headline number, fidelity
// is the hard gate.
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit } from "../../src/quantum.js";
import { Rx, Ry, Rz, type Matrix2 } from "../../src/gates.js";
import { fuseTriCascade } from "../../src/three-qubit-dense.js";
import { captureEnv } from "../lib/env.js";
import { timedRun } from "../lib/runner.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";
import { stateMetrics, passed, FIDELITY_PASS_BAR } from "../lib/fidelity.js";

export interface E12Row {
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

export interface E12Summary {
  readonly bestSpeedup: number;
  readonly bestSpeedupAt: { nQubits: number; depth: number };
  readonly worstFidelity: number;
}

interface CascadeLayer {
  readonly tiles: readonly { qLo: number; qMid: number; qHi: number; Ulo: Matrix2; Umid: Matrix2; Uhi: Matrix2 }[];
}

function pickGate(rng: () => number): Matrix2 {
  const pick = Math.floor(rng() * 3);
  const theta = rng() * 2 * Math.PI;
  if (pick === 0) return Rx(theta);
  if (pick === 1) return Ry(theta);
  return Rz(theta);
}

function buildLayers(n: number, depth: number, rng: () => number): CascadeLayer[] {
  const layers: CascadeLayer[] = [];
  for (let l = 0; l < depth; l++) {
    const tiles: CascadeLayer["tiles"][number][] = [];
    for (let q = 0; q + 2 < n; q += 3) {
      tiles.push({
        qLo: q,
        qMid: q + 1,
        qHi: q + 2,
        Ulo: pickGate(rng),
        Umid: pickGate(rng),
        Uhi: pickGate(rng),
      });
    }
    layers.push({ tiles });
  }
  return layers;
}

function applyUnfusedLayer(c: QuantumCircuit, layer: CascadeLayer): void {
  for (const t of layer.tiles) {
    c.apply(t.Ulo, t.qLo);
    c.apply(t.Umid, t.qMid);
    c.apply(t.Uhi, t.qHi);
    c.cnot(t.qLo, t.qMid);
    c.cnot(t.qMid, t.qHi);
  }
}

function applyFusedLayer(c: QuantumCircuit, layer: CascadeLayer): void {
  for (const t of layer.tiles) {
    c.applyDense8x8(fuseTriCascade(t.Ulo, t.Umid, t.Uhi), t.qLo, t.qMid, t.qHi);
  }
}

async function readState(c: QuantumCircuit): Promise<Float32Array> {
  return c.amplitudes();
}

export async function runE12(
  opts: {
    trials?: number;
    warmup?: number;
    Ns?: readonly number[];
    depths?: readonly number[];
  } = {},
): Promise<Artifact<E12Row> & { summary: E12Summary }> {
  const trials = opts.trials ?? 5;
  const warmup = opts.warmup ?? 3;
  const Ns = opts.Ns ?? [12, 15, 18];
  const depths = opts.depths ?? [20, 80];

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E12: no adapter");
  const env = await captureEnv(device, adapter);

  const rows: E12Row[] = [];
  let allPassed = true;

  for (const n of Ns) {
    if (n < 3) continue;
    const cUnfused = new QuantumCircuit({ device, nQubits: n });
    const cFused = new QuantumCircuit({ device, nQubits: n });

    try {
      for (const D of depths) {
        const seed = ((SEEDS.E12_TIERC_FUSION ^ (n * 2654435761) ^ (D * 0x9e3779b1)) >>> 0);
        const layers = buildLayers(n, D, mulberry32(seed));
        const tilesPerLayer = layers[0]?.tiles.length ?? 0;
        if (tilesPerLayer === 0) continue;

        // Correctness: run both paths once, compare statevectors.
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
          `[E12] N=${n} D=${D} tiles=${tilesPerLayer}: ` +
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

  const SPEEDUP_PASS = 3.0;

  const worstF = rows.length > 0 ? rows.reduce((a, r) => Math.min(a, r.fidelity), 1) : 0;
  const best = rows.length > 0
    ? rows.reduce((a, r) => (r.speedup > a.speedup ? r : a), rows[0]!)
    : null;

  const noisyFrac = rows.filter((r) => r.unfusedNoisy || r.fusedNoisy).length / Math.max(1, rows.length);
  const correctnessOk = rows.length > 0 && worstF >= FIDELITY_PASS_BAR && allPassed;
  const speedupOk = !!best && best.speedup >= SPEEDUP_PASS;
  const noisyBlocker = noisyFrac > 0.5;

  const status: Artifact<E12Row>["status"] =
    rows.length === 0 ? "fail" :
    !correctnessOk ? "fail" :
    noisyBlocker ? "noisy" :
    speedupOk ? "pass" : "fail";

  const summary: E12Summary = {
    bestSpeedup: best?.speedup ?? 0,
    bestSpeedupAt: best ? { nQubits: best.nQubits, depth: best.depth } : { nQubits: 0, depth: 0 },
    worstFidelity: worstF,
  };

  const meta: ArtifactMeta = {
    protocol: "E12-tier-c-fusion",
    hypothesis:
      "Three-qubit tile fusion (5 ops per tile per layer → 1 dense 8×8 dispatch) "
      + "delivers ≥ 3× speedup vs the unfused 5-dispatch path at N ≥ 12, while "
      + "staying within F ≥ 1 − 1e-5 of the unfused statevector.",
    passBar: "best (median over warmup+trials) speedup ≥ 3.0× AND worst F ≥ 1 − 1e-5",
    seed: "E12_TIERC_FUSION",
    warmup,
    trials,
  };

  const diagnosis = rows.length === 0
    ? "No cells ran (every N < 3?)."
    : !correctnessOk
      ? `Correctness FAIL: worst F=${worstF.toFixed(7)} < ${FIDELITY_PASS_BAR}.`
      : status === "pass" && best
        ? `Best speedup ${best.speedup.toFixed(2)}× at N=${best.nQubits} D=${best.depth}. ` +
          `Worst F=${worstF.toFixed(7)} (≥ ${FIDELITY_PASS_BAR}).`
        : status === "noisy"
          ? `Correctness OK (worst F=${worstF.toFixed(7)}) but timing too noisy.`
          : `Correctness OK (worst F=${worstF.toFixed(7)}) but best speedup ${best?.speedup.toFixed(2) ?? "n/a"}× < ${SPEEDUP_PASS}.`;

  return { meta, env, rows, status, diagnosis, summary };
}
