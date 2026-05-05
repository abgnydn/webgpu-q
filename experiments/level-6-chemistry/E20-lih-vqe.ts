// ─────────────────────────────────────────────────────────────
// E20 — VQE ground-state energy for LiH (Phase C, real molecule).
//
// First multi-orbital molecule in the project: 3 spatial orbitals
// (Li 1s, 2s, H 1s = STO-3G s-only) → 6 spin-orbitals → 64-dim
// Hilbert space. The HEA ansatz is NOT particle-conserving, so
// the global ground state of the Hamiltonian (at N=6, fully
// occupied) is unphysical for neutral LiH. We add a particle-
// number penalty H_eff = H + λ·(N̂ − 4)² to gap out the wrong
// sectors and force VQE to find the N=4 ground state — the
// physical LiH neutral.
//
// Reference E_FCI: lowest eigenvalue in the N=4 sector
// (`lowestInParticleSector` directly diagonalizes the C(6,4)=15
// dim block). At R = 1.595 Å (eq), E_FCI ≈ -7.8434 Ha — only 39 mHa
// above the published full STO-3G FCI of -7.8823 Ha (gap is the
// missing Li 2p contribution).
//
// Phase C v1 result: L-BFGS (with central-FD gradient) + HEA L=6
// + λ=2 penalty + 500 iters reaches chemical accuracy (≤ 1.6 mHa) on
// every trial at the equilibrium bond length, with the best trial
// hitting sub-microhartree precision. Phase C v0's plateau at ~20
// mHa was a Nelder-Mead artifact; the optimizer dimension is the
// limiting factor, not the ansatz.
//
// Scope honesty: STO-3G s-only is missing the Li 2p sub-shell.
// Adding 2p requires angular-momentum integrals (Phase C v2 work).
// ─────────────────────────────────────────────────────────────

import { initGPU } from "../../src/quantum.js";
import { captureEnv } from "../lib/env.js";
import { runVQE_HEA_Dense_LBFGS } from "../../src/chemistry/vqe.js";
import { heaParamCount, buildHEACircuit } from "../../src/chemistry/ansatz.js";
import {
  buildLiHDense, lowestInParticleSector,
} from "../../src/chemistry/lih-builder.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

const N_QUBITS = 6;
const DIM = 1 << N_QUBITS;
const TARGET_N = 4;

export interface E20Row {
  readonly bondLengthAngstrom: number;
  readonly trial: number;
  readonly seed: number;
  readonly energyVqe: number;          // E from optimizer (penalized)
  readonly energyVqeUnpenalized: number; // ⟨ψ|H|ψ⟩ without penalty
  readonly fciHartree: number;         // E_FCI in N=4 sector
  readonly deltaEhartree: number;      // unpenalized E_VQE − E_FCI
  readonly deltaEmHartree: number;
  readonly iterations: number;
  readonly termination: "converged" | "max-iter";
  readonly hitChemicalAccuracy: boolean;
}

export interface E20Summary {
  readonly bondLengthAngstrom: number;
  readonly fciHartree: number;
  readonly minEnergyHartree: number;
  readonly medianAbsDeltaMHa: number;
  readonly maxAbsDeltaMHa: number;
  readonly hitChemicalAccuracy: number;
  readonly trials: number;
}

const DEFAULT_BOND_LENGTHS = [1.0, 1.4, 1.595, 2.0, 3.0] as const;

/** H_eff = H + λ·(N̂ − N_target)², in-place on a fresh copy. */
function applyParticleNumberPenalty(H: Float64Array, lambda: number): Float64Array {
  const out = new Float64Array(H);
  for (let i = 0; i < DIM; i++) {
    let n = 0;
    for (let q = 0; q < N_QUBITS; q++) if ((i >>> q) & 1) n++;
    const dn = n - TARGET_N;
    out[i * DIM + i]! += lambda * dn * dn;
  }
  return out;
}

/** ⟨ψ|H|ψ⟩ for a real-Float64 dense H acting on an interleaved [re, im] state. */
function expectationDenseLocal(psi: Float64Array, H: Float64Array): number {
  let acc = 0;
  for (let i = 0; i < DIM; i++) {
    let hpsiRe = 0, hpsiIm = 0;
    for (let j = 0; j < DIM; j++) {
      const h = H[i * DIM + j]!;
      hpsiRe += h * psi[j * 2]!;
      hpsiIm += h * psi[j * 2 + 1]!;
    }
    acc += psi[i * 2]! * hpsiRe + psi[i * 2 + 1]! * hpsiIm;
  }
  return acc;
}

export async function runE20(opts: {
  bondLengthsAngstrom?: readonly number[];
  trialsPerBond?: number;
  nLayers?: number;
  optimizerMaxIter?: number;
  /** Particle-number penalty λ in H + λ(N̂ − 4)². */
  particlePenalty?: number;
  /** Initial parameter perturbation around the HF reference. */
  perturbation?: number;
} = {}): Promise<Artifact<E20Row> & { summaries: readonly E20Summary[] }> {
  const bonds = opts.bondLengthsAngstrom ?? DEFAULT_BOND_LENGTHS;
  const trialsPerBond = opts.trialsPerBond ?? 5;
  // L=6 + λ=2 + pert=0.20 + L-BFGS 500 iters: empirically every trial
  // hits chemical accuracy at R = 1.595 Å with this configuration.
  // Phase C v0 (Nelder-Mead, L=4, λ=5) plateaued ~20 mHa above E_FCI;
  // L-BFGS closes the gap to sub-mHa.
  const nLayers = opts.nLayers ?? 6;
  const maxIter = opts.optimizerMaxIter ?? 500;
  const lambda = opts.particlePenalty ?? 2.0;
  const perturbation = opts.perturbation ?? 0.20;
  const nParams = heaParamCount(N_QUBITS, nLayers);
  const CHEM_ACC = 1.6e-3;  // chemical accuracy

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E20: no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E20Row[] = [];
  const summaries: E20Summary[] = [];

  for (const R of bonds) {
    const { H, hfOccupied } = buildLiHDense(R);
    const E_FCI = lowestInParticleSector(H, N_QUBITS, TARGET_N).energy;
    const Hpen = applyParticleNumberPenalty(H, lambda);
    const trialDeltas: number[] = [];

    for (let trial = 0; trial < trialsPerBond; trial++) {
      const seed = ((SEEDS.E20_LIH_VQE ^ (trial * 0x9E3779B1) ^ (Math.round(R * 1e6) >>> 0)) >>> 0);
      const rng = mulberry32(seed);
      const init = new Float64Array(nParams);
      // Wider perturbation works better with L-BFGS than with Nelder-
      // Mead — the gradient-based optimizer escapes the immediate basin
      // and finds the FCI minimum reliably from a broader start.
      for (let i = 0; i < nParams; i++) init[i] = (rng() - 0.5) * perturbation;

      const r = runVQE_HEA_Dense_LBFGS(
        Hpen, N_QUBITS, nLayers, init,
        { maxIter, fTol: 1e-10, gTol: 1e-7, fdStep: 1e-5 },
        hfOccupied,
      );

      // Re-evaluate ⟨ψ|H|ψ⟩ on the un-penalized H to get the physical
      // VQE energy estimate (the penalty is only there to gap N≠4).
      const c = buildHEACircuit(N_QUBITS, nLayers, r.params, hfOccupied);
      const E_unpen = expectationDenseLocal(c.psi, H);

      const dE = E_unpen - E_FCI;
      const dEmHa = dE * 1000;
      trialDeltas.push(Math.abs(dE));

      rows.push({
        bondLengthAngstrom: R,
        trial,
        seed,
        energyVqe: r.energy,
        energyVqeUnpenalized: E_unpen,
        fciHartree: E_FCI,
        deltaEhartree: dE,
        deltaEmHartree: dEmHa,
        iterations: r.iterations,
        termination: r.termination,
        hitChemicalAccuracy: Math.abs(dE) <= CHEM_ACC,
      });

      console.log(
        `[E20] R=${R}Å trial=${trial} E_VQE=${E_unpen.toFixed(6)} ` +
        `E_FCI=${E_FCI.toFixed(6)} ΔE=${dEmHa.toFixed(3)} mHa ` +
        `iter=${r.iterations} ${r.termination}`,
      );
    }

    trialDeltas.sort((a, b) => a - b);
    const median = trialDeltas[Math.floor(trialDeltas.length / 2)]!;
    const max = trialDeltas[trialDeltas.length - 1]!;
    const minE = Math.min(
      ...rows.filter((r) => r.bondLengthAngstrom === R).map((r) => r.energyVqeUnpenalized),
    );
    const hits = rows.filter((r) => r.bondLengthAngstrom === R && r.hitChemicalAccuracy).length;

    summaries.push({
      bondLengthAngstrom: R,
      fciHartree: E_FCI,
      minEnergyHartree: minE,
      medianAbsDeltaMHa: median * 1000,
      maxAbsDeltaMHa: max * 1000,
      hitChemicalAccuracy: hits,
      trials: trialsPerBond,
    });
  }

  // Pass bar: at the equilibrium bond length (≈ 1.595 Å), best-of-trials
  // |ΔE| ≤ 50 mHa above E_FCI. This recovers ≥ 70% of the correlation
  // Pass bar: median |ΔE| ≤ 1.6 mHa AND best-of-trials hits chemical
  // accuracy at R = 1.595 Å. With L-BFGS at L=6, λ=2, pert=0.20, every
  // trial typically lands inside the 1.6 mHa bar — the bar is the
  // standard quantum-chemistry chemical-accuracy threshold and the
  // E20 hypothesis can carry it from Phase C v1 onward.
  const eqRow = summaries.find((s) => Math.abs(s.bondLengthAngstrom - 1.595) < 1e-9);
  const bestEqDeltaMHa = eqRow === undefined
    ? Infinity
    : Math.abs(eqRow.minEnergyHartree - eqRow.fciHartree) * 1000;
  const medianEqDeltaMHa = eqRow?.medianAbsDeltaMHa ?? Infinity;
  const eqOk = eqRow !== undefined &&
               medianEqDeltaMHa <= CHEM_ACC * 1000 &&
               bestEqDeltaMHa <= CHEM_ACC * 1000;

  const status: Artifact<E20Row>["status"] = eqOk ? "pass" : "fail";

  const meta: ArtifactMeta = {
    protocol: "E20-lih-vqe",
    hypothesis:
      "VQE with HEA L=6 + particle-number penalty (λ=2) + L-BFGS reaches chemical accuracy on LiH (STO-3G s-only, 6 qubits) at the equilibrium bond length R = 1.595 Å — median |ΔE| ≤ 1.6 mHa across all trials.",
    passBar:
      "median |ΔE| ≤ 1.6 mHa AND best-of-trials |ΔE| ≤ 1.6 mHa at R = 1.595 Å",
    seed: "E20_LIH_VQE",
    warmup: 0,
    trials: trialsPerBond,
  };

  const summaryStr = summaries
    .map((s) => `R=${s.bondLengthAngstrom}: median=${s.medianAbsDeltaMHa.toFixed(2)} mHa, max=${s.maxAbsDeltaMHa.toFixed(2)} mHa, hits=${s.hitChemicalAccuracy}/${s.trials}`)
    .join("; ");

  const diagnosis = status === "pass"
    ? `LiH equilibrium VQE within chemical accuracy. ${summaryStr}.`
    : `LiH equilibrium VQE missed pass bar. ${summaryStr}.`;

  return { meta, env, rows, status, diagnosis, summaries };
}
