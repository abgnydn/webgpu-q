// ─────────────────────────────────────────────────────────────
// E1 — gate fidelity vs CPU reference.
//
// Hypothesis: for every circuit in the suite and every N ∈ [4, 16],
// the GPU statevector agrees with a Float64 CPU reference at fidelity
// F ≥ 1 − 1e-5.
//
// This is the *correctness* gate for the entire stack. If E1 fails we
// stop — no amount of benchmark win makes a wrong answer valuable.
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit } from "../../src/quantum.js";
import { CpuCircuit } from "../../src/cpu-reference.js";
import {
  bell, ghz, qft, deutschJozsaConstant, randomCircuit, type GateApi,
} from "../../src/circuits.js";
import { captureEnv } from "../lib/env.js";
import { stateMetrics, FIDELITY_PASS_BAR, fmtMetrics } from "../lib/fidelity.js";
import { SEEDS } from "../lib/seeds.js";
import { stats } from "../lib/stats.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

// A registered circuit in the suite. `build` takes a GateApi so the
// same builder drives both CPU and GPU without duplication.
interface Suite {
  readonly name: string;
  readonly minN: number;
  readonly maxN: number;
  readonly build: (c: GateApi, n: number, trialSeed: number) => void;
}

const SUITE: readonly Suite[] = [
  { name: "bell",   minN: 2, maxN: 16, build: (c) => bell(c) },
  { name: "ghz",    minN: 2, maxN: 16, build: (c) => ghz(c) },
  { name: "qft",    minN: 2, maxN: 12, build: (c) => qft(c) },
  { name: "dj",     minN: 2, maxN: 14, build: (c) => deutschJozsaConstant(c) },
  // Random brick-wall stresses the two-qubit path + variable rotations.
  { name: "random-6L", minN: 4, maxN: 16, build: (c, _n, seed) => randomCircuit(c, 6, seed) },
];

export interface E1Row {
  readonly circuit: string;
  readonly nQubits: number;
  readonly trials: number;
  readonly fidelityMedian: number;
  readonly fidelityMin: number;
  readonly infidelityMedian: number;
  readonly maxDpMedian: number;
  readonly tvdMedian: number;
  readonly normGpuMedian: number;
  readonly passed: boolean;
  readonly notes: string;
}

/** Run a single (circuit, N) cell for `trials` trials. */
async function runCell(
  device: GPUDevice,
  s: Suite,
  n: number,
  trials: number,
): Promise<E1Row> {
  const fids: number[] = [];
  const maxDps: number[] = [];
  const tvds: number[] = [];
  const norms: number[] = [];

  let minF = 1;
  let notes = "";

  for (let t = 0; t < trials; t++) {
    // Per-trial seed derived deterministically from the experiment seed
    // and the circuit name. Guarantees full reproducibility without
    // storing every intermediate value.
    const trialSeed = (SEEDS.E1_FIDELITY ^ hashString(s.name) ^ (t * 0x9E3779B1)) >>> 0;

    const cpu = new CpuCircuit(n);
    const gpu = new QuantumCircuit({ device, nQubits: n });
    try {
      s.build(cpu, n, trialSeed);
      s.build(gpu, n, trialSeed);

      const gpuAmps = await gpu.amplitudes();
      const m = stateMetrics(cpu.psi, gpuAmps);
      fids.push(m.fidelity);
      maxDps.push(m.maxDp);
      tvds.push(m.tvd);
      norms.push(m.normTest);
      if (m.fidelity < minF) minF = m.fidelity;
      if (t === 0) notes = fmtMetrics(m); // keep first trial for sanity
    } finally {
      gpu.dispose();
    }
  }

  const medF = stats(fids).median;
  const medDp = stats(maxDps).median;
  const medTvd = stats(tvds).median;
  const medNorm = stats(norms).median;

  const pass = minF >= FIDELITY_PASS_BAR && Math.abs(medNorm - 1) < 1e-4;

  return {
    circuit: s.name,
    nQubits: n,
    trials,
    fidelityMedian: medF,
    fidelityMin: minF,
    infidelityMedian: 1 - medF,
    maxDpMedian: medDp,
    tvdMedian: medTvd,
    normGpuMedian: medNorm,
    passed: pass,
    notes,
  };
}

export async function runE1(
  opts: { trials?: number; maxN?: number } = {},
): Promise<Artifact<E1Row>> {
  const trials = opts.trials ?? 5; // fidelity is deterministic within f32 — 5 is plenty
  const capN = opts.maxN ?? 14;    // CPU reference caps at 24, but 14 keeps runtime < 5 s

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E1Row[] = [];
  for (const s of SUITE) {
    const hi = Math.min(s.maxN, capN);
    // sample a reasonable set of Ns, not every integer (keeps E1 under a minute)
    const Ns = dedupe([s.minN, Math.max(s.minN, 4), 8, 12, hi].filter((n) => n >= s.minN && n <= hi));
    for (const n of Ns) {
       
      console.log(`[E1] ${s.name} N=${n} trials=${trials}`);
      const row = await runCell(device, s, n, trials);
       
      console.log(`      → F=${row.fidelityMedian.toFixed(9)}  min=${row.fidelityMin.toFixed(9)}  ${row.passed ? "PASS" : "FAIL"}`);
      rows.push(row);
    }
  }

  const allPass = rows.every((r) => r.passed);
  const worst = rows.reduce((w, r) => r.fidelityMin < w.fidelityMin ? r : w);

  const meta: ArtifactMeta = {
    protocol: "E1-gate-fidelity",
    hypothesis: "GPU statevector agrees with Float64 CPU reference at F ≥ 1 − 1e-5 across the circuit suite.",
    passBar: `F ≥ ${FIDELITY_PASS_BAR} AND |norm − 1| < 1e-4`,
    seed: "E1_FIDELITY",
    warmup: 0,
    trials,
  };

  const artifact: Artifact<E1Row> = {
    meta,
    env,
    rows,
    status: allPass ? "pass" : "fail",
    diagnosis: allPass
      ? `${rows.length} cells all within pass bar (worst min F=${worst.fidelityMin.toFixed(9)} on ${worst.circuit} N=${worst.nQubits})`
      : `FAIL on ${worst.circuit} N=${worst.nQubits}: min F=${worst.fidelityMin.toFixed(9)} (bar ${FIDELITY_PASS_BAR})`,
  };

  return artifact;
}

// ── helpers ──────────────────────────────────────────────────
function hashString(s: string): number {
  let h = 0x811C9DC5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
function dedupe<T>(xs: readonly T[]): T[] {
  return Array.from(new Set(xs));
}
