// ─────────────────────────────────────────────────────────────
// E11 — Brick-wall layer fusion (the real "kernel fusion" thesis).
//
// E8–E10 fuse k single-qubit gates on the SAME qubit into one
// dispatch. E11 fuses each brick-wall pair (single_q ⊗ single_{q+1}
// followed by CNOT(q, q+1)) into a single 4×4 dense unitary —
// 3 sequential dispatches collapse to 1 across each adjacent pair
// at every depth. That's a 3× drop in dispatch count AND a 3× drop
// in memory traffic per layer (one round-trip of the statevector
// per pair instead of three).
//
// Hypothesis: at N ≥ 14 with depth D ≥ 20, the layer-fused path
// runs at ≥ 2× the unfused throughput AND matches the unfused
// statevector at F ≥ 1 − 1e-5.
//
// Pass bar:  median speedup ≥ 2.0× on at least one (N, D) cell
// AND  fidelity ≥ 1 − 1e-5 on every cell  (correctness is a hard
// gate, throughput is the headline number).
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit } from "../../src/quantum.js";
import { Rx, Ry, Rz, type Matrix2 } from "../../src/gates.js";
import { fuseBrickwallPair } from "../../src/two-qubit-dense.js";
import { captureEnv } from "../lib/env.js";
import { timedRun } from "../lib/runner.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";
import { stateMetrics, passed, FIDELITY_PASS_BAR } from "../lib/fidelity.js";

export interface E11Row {
  readonly nQubits: number;
  readonly depth: number;
  readonly unfusedSec: number;
  readonly fusedSec: number;
  readonly speedup: number;
  readonly fidelity: number;
  readonly unfusedNoisy: boolean;
  readonly fusedNoisy: boolean;
}

export interface E11Summary {
  readonly bestSpeedup: number;
  readonly bestSpeedupAt: { nQubits: number; depth: number };
  readonly worstFidelity: number;
}

function pickGate(rng: () => number): Matrix2 {
  const pick = Math.floor(rng() * 3);
  const theta = rng() * 2 * Math.PI;
  if (pick === 0) return Rx(theta);
  if (pick === 1) return Ry(theta);
  return Rz(theta);
}

interface BrickwallLayer {
  readonly singles: readonly Matrix2[];     // length n; one gate per qubit
  readonly cnotOffset: 0 | 1;               // pairs at (offset, offset+1), (offset+2, offset+3), …
}

function buildLayers(n: number, depth: number, rng: () => number): BrickwallLayer[] {
  const layers: BrickwallLayer[] = [];
  for (let l = 0; l < depth; l++) {
    const singles: Matrix2[] = [];
    for (let q = 0; q < n; q++) singles.push(pickGate(rng));
    layers.push({ singles, cnotOffset: (l & 1) as 0 | 1 });
  }
  return layers;
}

function applyUnfusedLayer(c: QuantumCircuit, layer: BrickwallLayer): void {
  const n = c.nQubits;
  for (let q = 0; q < n; q++) c.apply(layer.singles[q]!, q);
  for (let q = layer.cnotOffset; q + 1 < n; q += 2) c.cnot(q, q + 1);
}

function applyFusedLayer(c: QuantumCircuit, layer: BrickwallLayer): void {
  const n = c.nQubits;
  // Fuse each (q, q+1) pair where the CNOT lives. For qubits not in any
  // pair (boundary qubit of an offset=1 layer), apply the single gate
  // alone — there's no CNOT partner to fuse with.
  const usedInPair = new Uint8Array(n);
  for (let q = layer.cnotOffset; q + 1 < n; q += 2) {
    const Ulo = layer.singles[q]!;
    const Uhi = layer.singles[q + 1]!;
    c.applyDense4x4(fuseBrickwallPair(Ulo, Uhi, "lo"), q, q + 1);
    usedInPair[q] = 1;
    usedInPair[q + 1] = 1;
  }
  for (let q = 0; q < n; q++) {
    if (!usedInPair[q]) c.apply(layer.singles[q]!, q);
  }
}

async function readState(c: QuantumCircuit): Promise<Float32Array> {
  return c.amplitudes();
}

export async function runE11(
  opts: {
    trials?: number;
    warmup?: number;
    Ns?: readonly number[];
    depths?: readonly number[];
  } = {},
): Promise<Artifact<E11Row> & { summary: E11Summary }> {
  const trials = opts.trials ?? 5;
  const warmup = opts.warmup ?? 3;
  const Ns = opts.Ns ?? [12, 16, 20];
  const depths = opts.depths ?? [20, 80];

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E11: no adapter");
  const env = await captureEnv(device, adapter);

  const rows: E11Row[] = [];
  let allPassed = true;

  for (const n of Ns) {
    const cUnfused = new QuantumCircuit({ device, nQubits: n });
    const cFused = new QuantumCircuit({ device, nQubits: n });

    try {
      for (const D of depths) {
        // Reproducible layer sequence per (N, D).
        const seed = ((SEEDS.E4_OVERHEAD ^ 0xE11E11 ^ (n * 2654435761) ^ (D * 0x9e3779b1)) >>> 0);
        const layers = buildLayers(n, D, mulberry32(seed));

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

        // Throughput: D layers per trial.
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
          unfusedSec: unS.median,
          fusedSec: fuS.median,
          speedup,
          fidelity: F,
          unfusedNoisy: unS.noisy,
          fusedNoisy: fuS.noisy,
        });


        console.log(
          `[E11] N=${n} D=${D}: unfused=${(unS.median * 1000).toFixed(2)} ms, ` +
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

  const SPEEDUP_PASS = 2.0;

  const worstF = rows.reduce((a, r) => Math.min(a, r.fidelity), 1);
  const best = rows.reduce((a, r) => (r.speedup > a.speedup ? r : a), rows[0]!);

  const noisyFrac = rows.filter((r) => r.unfusedNoisy || r.fusedNoisy).length / Math.max(1, rows.length);
  const correctnessOk = worstF >= FIDELITY_PASS_BAR && allPassed;
  const speedupOk = best.speedup >= SPEEDUP_PASS;
  const noisyBlocker = noisyFrac > 0.5;

  const status: Artifact<E11Row>["status"] =
    !correctnessOk ? "fail" :
    noisyBlocker ? "noisy" :
    speedupOk ? "pass" : "fail";

  const summary: E11Summary = {
    bestSpeedup: best.speedup,
    bestSpeedupAt: { nQubits: best.nQubits, depth: best.depth },
    worstFidelity: worstF,
  };

  const meta: ArtifactMeta = {
    protocol: "E11-brickwall-fusion",
    hypothesis:
      "Brick-wall layer fusion (3 dispatches per pair per layer → 1 dense 4×4 dispatch) "
      + "delivers ≥ 2× speedup vs unfused brick-wall at N ≥ 14, while staying within "
      + "F ≥ 1 − 1e-5 of the unfused statevector.",
    passBar: "best (median over warmup+trials) speedup ≥ 2.0× AND worst F ≥ 1 − 1e-5",
    seed: "E4_OVERHEAD (xor E11)",
    warmup,
    trials,
  };

  const diagnosis = !correctnessOk
    ? `Correctness FAIL: worst F=${worstF.toFixed(7)} < ${FIDELITY_PASS_BAR}.`
    : status === "pass"
      ? `Best speedup ${best.speedup.toFixed(2)}× at N=${best.nQubits} D=${best.depth}. ` +
        `Worst F=${worstF.toFixed(7)} (≥ ${FIDELITY_PASS_BAR}).`
      : status === "noisy"
        ? `Correctness OK (worst F=${worstF.toFixed(7)}) but timing too noisy.`
        : `Correctness OK (worst F=${worstF.toFixed(7)}) but best speedup ${best.speedup.toFixed(2)}× < ${SPEEDUP_PASS}.`;

  return { meta, env, rows, status, diagnosis, summary };
}
