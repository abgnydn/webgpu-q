// ─────────────────────────────────────────────────────────────
// E5 — MPS correctness vs statevector ground truth.
//
// Hypothesis: for every circuit in the suite and every N ∈ [4, 20],
// the MPS state at χ = 64 agrees with the CPU statevector at F ≥ 0.999.
//
// Circuits (all adjacent-gate, since MPS v1 requires |c − t| = 1):
//   • ghz         — H(0) + CNOT ladder. Schmidt rank 2 everywhere.
//   • brick-depth-2
//   • brick-depth-4
//   • heisenberg-1-trotter — 1 Trotter step of the 1D XXZ model.
//
// Pass bar: F ≥ 0.999 on every (circuit, N) cell at χ = 64.
// Secondary: report smallest χ ∈ {2, 4, 8, 16, 32, 64} achieving
// F ≥ 0.999 per cell ("cheapest χ" curve).
// ─────────────────────────────────────────────────────────────

import { CpuCircuit } from "../../src/cpu-reference.js";
import { MPS } from "../../src/mps.js";
import { H, X, Rx, Ry, Rz } from "../../src/gates.js";
import { stateMetrics } from "../lib/fidelity.js";
import { captureEnv } from "../lib/env.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import { initGPU } from "../../src/quantum.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

export interface E5Row {
  readonly circuit: string;
  readonly nQubits: number;
  readonly chi: number;              // χ used (final, at pass bar)
  readonly trial: number;
  readonly fidelity: number;         // |⟨ψ_cpu|ψ_mps⟩|²
  readonly tvd: number;
  readonly maxDp: number;
  readonly normMps: number;          // ‖ψ_mps‖² (should ≈ 1)
  readonly maxBond: number;          // largest bond dim after circuit
  readonly cheapestChi: number | null; // smallest χ hitting F ≥ 0.999, null if 64 also fails
  readonly truncLoss: number;        // cumulative truncation loss
  readonly wallSeconds: number;      // MPS wall time (f64 CPU)
}

type CircuitName = "ghz" | "brick-depth-2" | "brick-depth-4" | "heisenberg-1-trotter";

const SUITE: readonly CircuitName[] = [
  "ghz",
  "brick-depth-2",
  "brick-depth-4",
  "heisenberg-1-trotter",
];

const DEFAULT_NS = [4, 6, 8, 10, 12, 14, 16, 18, 20] as const;
const CHEAPEST_CHI_LADDER = [2, 4, 8, 16, 32, 64] as const;

interface GateApi {
  readonly nQubits: number;
  apply: (m: import("../../src/gates.js").Matrix2, q: number) => void;
  cnot: (c: number, t: number) => void;
  applyControlled: (m: import("../../src/gates.js").Matrix2, c: number, t: number) => void;
}

// ── builders ─────────────────────────────────────────────────
function build(circuit: CircuitName, c: GateApi, rng: () => number): void {
  const N = c.nQubits;
  switch (circuit) {
    case "ghz":
      c.apply(H, 0);
      for (let k = 0; k < N - 1; k++) c.cnot(k, k + 1);
      return;

    case "brick-depth-2":
      applyBrickwall(c, 2, rng);
      return;

    case "brick-depth-4":
      applyBrickwall(c, 4, rng);
      return;

    case "heisenberg-1-trotter":
      // One Trotter step of H = Σ_j (Rx+Ry+Rz) · adjacent XX: approximate via
      // single-qubit rotations + adjacent CNOT bricks at random small angles.
      // Full Heisenberg XX would need non-adjacent or richer two-site gates;
      // this preserves the ADJACENT TEBD flavor while exercising real
      // off-diagonal 4×4 dynamics.
      for (let q = 0; q < N; q++) c.apply(Rx(0.3 + 0.01 * q), q);
      for (let q = 0; q + 1 < N; q += 2) {
        c.cnot(q, q + 1);
        c.apply(Rz(0.22 + 0.003 * q), q + 1);
        c.cnot(q, q + 1);
      }
      for (let q = 1; q + 1 < N; q += 2) {
        c.cnot(q, q + 1);
        c.apply(Rz(0.17 + 0.004 * q), q + 1);
        c.cnot(q, q + 1);
      }
      for (let q = 0; q < N; q++) c.apply(Ry(0.2 + 0.011 * q), q);
      return;
  }
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

function ampsFromCpu(cpu: CpuCircuit): Float64Array {
  // CpuCircuit.psi is already [re, im, re, im, ...] Float64.
  return cpu.psi;
}

export async function runE5(
  opts: {
    trials?: number;
    Ns?: readonly number[];
    chi?: number;
    cheapestChi?: boolean;
  } = {},
): Promise<Artifact<E5Row>> {
  const trials = opts.trials ?? 5;
  const Ns = opts.Ns ?? DEFAULT_NS;
  const chi = opts.chi ?? 64;
  const computeCheapest = opts.cheapestChi ?? true;

  // Env block — we need an adapter for captureEnv. Level 2 is CPU-only so we
  // still record GPU info for provenance with the rest of the level ladder.
  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E5Row[] = [];

  for (const circuit of SUITE) {
    for (const N of Ns) {
      if (N > 20) continue; // CpuCircuit caps at 24, but 2^20 is our published envelope
      for (let trial = 0; trial < trials; trial++) {
        const subseed = SEEDS.E5_MPS_CORRECTNESS ^ (N * 2654435761) ^ (trial * 0x9E3779B1);
        const seed = subseed >>> 0;


        const cpu = new CpuCircuit(N);
        const mps = new MPS({ nQubits: N, chiMax: chi });

        const rngCpu = mulberry32(seed);
        const rngMps = mulberry32(seed);

        const t0 = performance.now();
        build(circuit, cpu, rngCpu);
        build(circuit, mps, rngMps);
        const wallSeconds = (performance.now() - t0) / 1000;

        const cpuPsi = ampsFromCpu(cpu);
        const mpsPsi = mps.statevector();
        const m = stateMetrics(cpuPsi, mpsPsi);

        let cheapestChi: number | null = null;
        if (computeCheapest) {
          for (const cChi of CHEAPEST_CHI_LADDER) {
            if (cChi > chi) break;
            const mpsTrial = new MPS({ nQubits: N, chiMax: cChi });
            const rngT = mulberry32(seed);
            build(circuit, mpsTrial, rngT);
            const mT = stateMetrics(cpuPsi, mpsTrial.statevector());
            if (mT.fidelity >= 0.999) {
              cheapestChi = cChi;
              break;
            }
          }
        }

        rows.push({
          circuit,
          nQubits: N,
          chi,
          trial,
          fidelity: m.fidelity,
          tvd: m.tvd,
          maxDp: m.maxDp,
          normMps: m.normTest,
          maxBond: Math.max(...mps.bondDimensions(), 1),
          cheapestChi,
          truncLoss: mps.truncationLoss(),
          wallSeconds,
        });


        console.log(
          `[E5] ${circuit} N=${N} trial=${trial} χ=${chi} ` +
          `F=${m.fidelity.toFixed(6)} ‖ψ‖=${m.normTest.toFixed(6)} ` +
          `maxBond=${Math.max(...mps.bondDimensions(), 1)} ` +
          `cheapestχ=${cheapestChi ?? "fail"} ${wallSeconds.toFixed(3)}s`,
        );
      }
    }
  }

  // Pass bar: F ≥ 0.999 on every cell.
  const fails = rows.filter((r) => r.fidelity < 0.999 || Math.abs(r.normMps - 1) > 1e-3);
  const status: Artifact<E5Row>["status"] = fails.length === 0 ? "pass" : "fail";

  const minF = rows.reduce((a, b) => a.fidelity < b.fidelity ? a : b, rows[0]!);
  const worstNorm = rows.reduce((a, b) => Math.abs(a.normMps - 1) > Math.abs(b.normMps - 1) ? a : b, rows[0]!);

  const meta: ArtifactMeta = {
    protocol: "E5-mps-correctness",
    hypothesis: "At χ = 64, MPS state agrees with CPU statevector at F ≥ 0.999 for every (circuit, N) in the suite and every N ∈ [4, 20].",
    passBar: "F ≥ 0.999 AND |‖ψ_mps‖ − 1| < 1e-3 on ALL cells.",
    seed: "E5_MPS_CORRECTNESS",
    warmup: 0,
    trials,
  };

  const diagnosis = status === "pass"
    ? `All ${rows.length} cells passed. Worst-F cell: ${minF.circuit} N=${minF.nQubits} trial=${minF.trial} F=${minF.fidelity.toFixed(6)}.`
    : `${fails.length} / ${rows.length} cells failed. First miss: ${fails[0]!.circuit} N=${fails[0]!.nQubits} trial=${fails[0]!.trial} F=${fails[0]!.fidelity.toFixed(6)}, ‖ψ‖=${fails[0]!.normMps.toFixed(6)}. Worst-norm: ${worstNorm.normMps.toFixed(6)} at ${worstNorm.circuit} N=${worstNorm.nQubits}.`;

  return { meta, env, rows, status, diagnosis };
}
