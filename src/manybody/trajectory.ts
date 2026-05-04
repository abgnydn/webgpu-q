// ─────────────────────────────────────────────────────────────
// trajectory.ts — single quantum trajectory under stochastic
// continuous Z-monitoring.
//
// Model (the standard "monitored circuit" / measurement-induced
// phase transition setup):
//
//   1. Real-time TEBD step:  |ψ⟩ ← exp(-i τ H) |ψ⟩
//   2. For each site q independently:
//      with probability γ τ, perform a projective Z-measurement.
//      Outcome ± with probabilities ⟨P_±⟩, project + renormalize.
//
// As γ varies:
//   • γ = 0   : pure unitary, entropy grows ballistically (volume law).
//   • γ → ∞   : state stays product, entropy stays ~0 (area law).
//   • γ = γ_c : entropy grows logarithmically — the measurement-
//               induced phase transition (an active research topic).
//
// We track the per-site ⟨σ_z⟩, the bond entropies, and a list of
// (step, site, outcome) measurement events. The viz renders them
// as a sites×time heatmap with measurement marks overlaid.
// ─────────────────────────────────────────────────────────────

import { MPS } from "../mps.js";
import type { Hamiltonian1D } from "./hamiltonian.js";
import { expmComplex } from "./real-time.js";
import { siteExpectations } from "./observables.js";
import type { Matrix2 } from "../gates.js";

const P_PLUS: Matrix2 = [[1, 0], [0, 0], [0, 0], [0, 0]];
const P_MINUS: Matrix2 = [[0, 0], [0, 0], [0, 0], [1, 0]];

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function entropyFromSigma(sigma: Float64Array): number {
  let norm2 = 0;
  for (let i = 0; i < sigma.length; i++) norm2 += sigma[i]! * sigma[i]!;
  if (norm2 <= 0) return 0;
  let S = 0;
  for (let i = 0; i < sigma.length; i++) {
    const p = (sigma[i]! * sigma[i]!) / norm2;
    if (p > 1e-18) S -= p * Math.log2(p);
  }
  return S;
}

export interface TrajectoryStep {
  /** Per-site ⟨σ_z⟩ after this step, length N. */
  sz: readonly number[];
  /** Bond entropies after this step, length N − 1. */
  entropies: readonly number[];
  /** Mid-cut entropy (proxy for "average" entanglement). */
  midcutEntropy: number;
  /** Sites that were measured this step, with outcomes (+1 = |0⟩, −1 = |1⟩). */
  measurements: ReadonlyArray<{ site: number; outcome: 1 | -1 }>;
}

export interface TrajectoryRecording {
  readonly nQubits: number;
  /** γ used (per-step measurement probability is γ·τ). */
  readonly gamma: number;
  readonly tau: number;
  readonly steps: number;
  readonly history: readonly TrajectoryStep[];
  /** Max entropy across the trajectory (for color scaling). */
  readonly maxEntropy: number;
}

export interface TrajectoryOptions {
  hamiltonian: Hamiltonian1D;
  /** Measurement rate γ (per site per unit time). */
  gamma: number;
  steps?: number;
  tau?: number;
  chiMax?: number;
  seed?: number;
  /** Initial state. "neel" = |↑↓↑↓…⟩, "all-up" = |↑↑↑…⟩, "x-pol" = |+⟩^⊗N. */
  initialState?: "neel" | "all-up" | "x-pol";
  /** If true, also record the bond-entropy spectrum (cost: ~50%). Default true. */
  recordEntropies?: boolean;
}

/**
 * Apply the σ_z eigenstate projector P_± at site q to an MPS, then
 * renormalize. Uses the existing single-qubit-gate path which is
 * non-unitary-tolerant; renormalize() restores ‖ψ‖ = 1.
 */
function projectZ(mps: MPS, q: number, outcome: 1 | -1): void {
  mps.apply(outcome === 1 ? P_PLUS : P_MINUS, q);
  mps.renormalize();
}

/** Measure σ_z at site q stochastically. Returns the outcome and projects the state. */
function measureZ(mps: MPS, q: number, rng: () => number): 1 | -1 {
  // Compute ⟨P_+⟩ via expectation: (1 + ⟨σ_z⟩) / 2.
  const psi = mps.statevector();
  let pPlus = 0;
  const dim = 1 << mps.nQubits;
  for (let b = 0; b < dim; b++) {
    if (((b >> q) & 1) === 0) {
      const re = psi[b * 2]!;
      const im = psi[b * 2 + 1]!;
      pPlus += re * re + im * im;
    }
  }
  const u = rng();
  const outcome: 1 | -1 = u < pPlus ? 1 : -1;
  projectZ(mps, q, outcome);
  return outcome;
}

export function recordTrajectory(opts: TrajectoryOptions): TrajectoryRecording {
  const N = opts.hamiltonian.nQubits;
  const gamma = opts.gamma;
  const tau = opts.tau ?? 0.1;
  const steps = opts.steps ?? 60;
  const chiMax = opts.chiMax ?? 32;
  const seed = opts.seed ?? 0xCAFE5E0;
  const init = opts.initialState ?? "x-pol";
  const recordE = opts.recordEntropies ?? true;

  const mps = new MPS({ nQubits: N, chiMax });

  if (init === "neel") {
    const Xgate: Matrix2 = [[0, 0], [1, 0], [1, 0], [0, 0]];
    for (let q = 1; q < N; q += 2) mps.apply(Xgate, q);
  } else if (init === "x-pol") {
    // |+⟩ = (|0⟩ + |1⟩)/√2. Hadamard on every site.
    const sq = 1 / Math.SQRT2;
    const Hgate: Matrix2 = [[sq, 0], [sq, 0], [sq, 0], [-sq, 0]];
    for (let q = 0; q < N; q++) mps.apply(Hgate, q);
  }
  // "all-up" leaves the default |0…0⟩ state.

  const rng = mulberry32(seed);
  const history: TrajectoryStep[] = [];

  // Pre-build the real-time gates exp(-i τ h_q).
  const gates = opts.hamiltonian.bonds.map((b) => expmComplex(b.h, 4, tau));

  // Initial snapshot.
  {
    const exps = siteExpectations(mps.statevector(), N);
    const sz = Array.from(exps.sz);
    const entropies = recordE ? perBondEntropy(mps) : new Array(N - 1).fill(0);
    const midcut = entropies[Math.floor((N - 2) / 2)] ?? 0;
    history.push({ sz, entropies, midcutEntropy: midcut, measurements: [] });
  }

  for (let step = 0; step < steps; step++) {
    // 1. Real-time evolution τ.
    for (let q = 0; q < N - 1; q += 2) mps.applyTwoSite(gates[q]!, q);
    for (let q = 1; q < N - 1; q += 2) mps.applyTwoSite(gates[q]!, q);

    // 2. Per-site stochastic measurement at rate γ.
    const measurements: { site: number; outcome: 1 | -1 }[] = [];
    const probMeasure = gamma * tau;
    if (probMeasure > 0) {
      for (let q = 0; q < N; q++) {
        if (rng() < probMeasure) {
          const outcome = measureZ(mps, q, rng);
          measurements.push({ site: q, outcome });
        }
      }
    }

    const exps = siteExpectations(mps.statevector(), N);
    const sz = Array.from(exps.sz);
    const entropies = recordE ? perBondEntropy(mps) : new Array(N - 1).fill(0);
    const midcut = entropies[Math.floor((N - 2) / 2)] ?? 0;
    history.push({ sz, entropies, midcutEntropy: midcut, measurements });
  }

  let maxS = 0;
  for (const s of history) for (const e of s.entropies) if (e > maxS) maxS = e;

  return { nQubits: N, gamma, tau, steps, history, maxEntropy: maxS };
}

function perBondEntropy(mps: MPS): number[] {
  const out: number[] = [];
  for (let cut = 0; cut < mps.nQubits - 1; cut++) {
    out.push(entropyFromSigma(mps.schmidtSpectrum(cut)));
  }
  return out;
}
