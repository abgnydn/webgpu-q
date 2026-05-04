// ─────────────────────────────────────────────────────────────
// E8 — fusion correctness vs unfused reference.
//
// Hypothesis (protocol.md): for every chain of length
// k ∈ {1, 2, 4, 8, 16, 32} of single-qubit gates on the same qubit,
// the fused kernel produces a statevector matching the unfused path
// at fidelity F ≥ 1 − 1e-5 and |norm − 1| < 1e-4.
//
// Method: seeded random single-qubit gate chains. Two GPU circuits:
// one runs k unfused `apply` calls; the other runs one
// `applyFusedSingle(chain)`. Both start from a non-trivial state
// (H on every qubit) so we exercise real entanglement-preserving
// paths, not just |0…0⟩ cancellation. Compare `amplitudes()`
// output with `stateMetrics`.
//
// Why start from the Hadamard-all state: with |0…⟩ as the input,
// a single-qubit chain on qubit 0 only excites a 2-dimensional
// subspace — any f32 cancellation bug that cancels on the
// |0⟩-only path is invisible. Spreading amplitude to every
// basis state first stresses the same kernel across 2^N lanes.
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit, FUSED_CHAIN_MAX_K } from "../../src/quantum.js";
import { H, Rx, Ry, Rz, S, T, type Matrix2 } from "../../src/gates.js";
import { captureEnv } from "../lib/env.js";
import { FIDELITY_PASS_BAR, stateMetrics } from "../lib/fidelity.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

export interface E8Row {
  readonly k: number;
  readonly nQubits: number;
  readonly qubit: number;
  readonly trial: number;
  readonly fidelity: number;
  readonly infidelity: number;
  readonly maxDp: number;
  readonly normFused: number;
  readonly normUnfused: number;
  readonly passed: boolean;
}

const DEFAULT_KS = [1, 2, 4, 8, 16, 32] as const;
const DEFAULT_NS = [8, 16, 24] as const;
// Reserved for E8: xor into base seed so different depths use independent chains.
const SEED_BASE = SEEDS.E1_FIDELITY ^ 0xF0C0FF;

function randomGate(rng: () => number): Matrix2 {
  // Uniform mix over {H, S, T, Rx(θ), Ry(θ), Rz(θ)} with θ ∈ [0, 2π).
  const pick = Math.floor(rng() * 6);
  const theta = rng() * 2 * Math.PI;
  if (pick === 0) return H;
  if (pick === 1) return S;
  if (pick === 2) return T;
  if (pick === 3) return Rx(theta);
  if (pick === 4) return Ry(theta);
  return Rz(theta);
}

function randomChain(k: number, rng: () => number): Matrix2[] {
  const out: Matrix2[] = [];
  for (let i = 0; i < k; i++) out.push(randomGate(rng));
  return out;
}

export async function runE8(
  opts: {
    trials?: number;
    Ks?: readonly number[];
    Ns?: readonly number[];
  } = {},
): Promise<Artifact<E8Row>> {
  const trials = opts.trials ?? 20;
  const Ks = opts.Ks ?? DEFAULT_KS;
  const Ns = opts.Ns ?? DEFAULT_NS;

  for (const k of Ks) {
    if (k > FUSED_CHAIN_MAX_K) {
      throw new Error(`E8: k=${k} exceeds FUSED_CHAIN_MAX_K=${FUSED_CHAIN_MAX_K}`);
    }
  }

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E8Row[] = [];

  for (const n of Ns) {
    const qubit = Math.floor(n / 2); // non-trivial index so pair indexing exercises both low+high masks
    for (const k of Ks) {
      for (let t = 0; t < trials; t++) {
        const seed = (SEED_BASE ^ (n * 2654435761) ^ (k * 0x9e3779b1) ^ (t * 0x85ebca77)) >>> 0;
        const chain = randomChain(k, mulberry32(seed));

        // Unfused: k sequential `apply` calls.
        const unfused = new QuantumCircuit({ device, nQubits: n });
        // Non-trivial initial state: H on every qubit → uniform superposition.
        for (let q = 0; q < n; q++) unfused.apply(H, q);
        for (const M of chain) unfused.apply(M, qubit);
        const psiUn = await unfused.amplitudes();
        unfused.dispose();

        // Fused: one `applyFusedSingle`.
        const fused = new QuantumCircuit({ device, nQubits: n });
        for (let q = 0; q < n; q++) fused.apply(H, q);
        fused.applyFusedSingle(chain, qubit);
        const psiF = await fused.amplitudes();
        fused.dispose();

        const m = stateMetrics(psiUn, psiF);
        const passed = m.fidelity >= FIDELITY_PASS_BAR && Math.abs(m.normTest - 1) < 1e-4;

        rows.push({
          k,
          nQubits: n,
          qubit,
          trial: t,
          fidelity: m.fidelity,
          infidelity: m.infidelity,
          maxDp: m.maxDp,
          normFused: m.normTest,
          normUnfused: m.normRef,
          passed,
        });


        console.log(
          `[E8] N=${n} q=${qubit} k=${k} trial=${t}: `
          + `F=${m.fidelity.toFixed(9)} 1−F=${m.infidelity.toExponential(2)} `
          + `‖ψ_f‖=${m.normTest.toFixed(6)} ${passed ? "✓" : "✗"}`,
        );
      }
    }
  }

  const fails = rows.filter((r) => !r.passed);
  const status: Artifact<E8Row>["status"] = fails.length === 0 ? "pass" : "fail";

  const minF = rows.reduce((a, b) => (a.fidelity < b.fidelity ? a : b), rows[0]!);
  const worstNorm = rows.reduce(
    (a, b) => (Math.abs(a.normFused - 1) > Math.abs(b.normFused - 1) ? a : b),
    rows[0]!,
  );

  const meta: ArtifactMeta = {
    protocol: "E8-fusion-correctness",
    hypothesis: `For every chain of length k ∈ {${Ks.join(",")}} on the same qubit, the fused kernel matches the unfused reference at F ≥ ${FIDELITY_PASS_BAR}.`,
    passBar: `F ≥ ${FIDELITY_PASS_BAR} AND |‖ψ_fused‖ − 1| < 1e-4 on ALL cells.`,
    seed: "E1_FIDELITY (xor E8)",
    warmup: 0,
    trials,
  };

  const diagnosis = status === "pass"
    ? `All ${rows.length} cells passed. Worst-F: N=${minF.nQubits} k=${minF.k} trial=${minF.trial} F=${minF.fidelity.toFixed(9)} (1−F=${minF.infidelity.toExponential(2)}). Worst-norm: ${worstNorm.normFused.toFixed(6)} at N=${worstNorm.nQubits} k=${worstNorm.k}.`
    : `${fails.length} / ${rows.length} cells failed. First miss: N=${fails[0]!.nQubits} k=${fails[0]!.k} trial=${fails[0]!.trial} F=${fails[0]!.fidelity.toFixed(9)} ‖ψ_f‖=${fails[0]!.normFused.toFixed(6)}.`;

  return { meta, env, rows, status, diagnosis };
}
