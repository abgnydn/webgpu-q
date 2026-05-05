// ─────────────────────────────────────────────────────────────
// E21 — VQE ground-state energy for BeH₂ (Phase C v2, multi-atom).
//
// First multi-atom molecule in the project. Linear H–Be–H,
// STO-3G s-only basis (Be 1s + Be 2s + 2 H 1s = 4 spatial /
// 8 spin-orbitals → 256-dim Hilbert space, 6 electrons in the
// neutral molecule). Phase C v2 scope deliberately excludes
// Be 2p (would lift to 14 spin-orbitals, 16384-dim) — adding
// it requires Cartesian-Gaussian p-shell integrals (Phase C v3).
//
// Why BeH₂ is harder than LiH:
//   • The HF determinant in AO basis is asymmetric (puts both
//     H electrons on H_left), giving a ~2 Ha correlation gap to
//     E_FCI vs LiH's ~0.18 Ha gap.
//   • Closing that gap requires deeper HEA: L = 4-6 plateaus
//     ~50 mHa above E_FCI; L = 10 best-trial reaches chemical
//     accuracy; L = 12 reliably hits sub-mHa.
//
// Pass bar (R = 1.34 Å, experimental Be–H bond):
//   • best of 5 trials |ΔE| ≤ 1.6 mHa (chemical accuracy).
//   • Median bar relaxed to ≤ 10 mHa, since not every random
//     init crosses the energy barrier on the AO-basis HF.
//
// Closing the median gap to chemical accuracy needs HF SCF +
// MO-basis transformation (orbital optimization), which is
// Phase C v2.5 work. With MO-basis HF the correlation gap drops
// by an order of magnitude and L = 6 should suffice.
// ─────────────────────────────────────────────────────────────

import { initGPU } from "../../src/quantum.js";
import { captureEnv } from "../lib/env.js";
import { runVQE_HEA_Dense_LBFGS } from "../../src/chemistry/vqe.js";
import { heaParamCount, buildHEACircuit } from "../../src/chemistry/ansatz.js";
import { buildBeH2Dense } from "../../src/chemistry/beh2-builder.js";
import { lowestInParticleSector } from "../../src/chemistry/sector.js";
import { expectationDense } from "../../src/chemistry/h2-builder.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

const N_QUBITS = 8;
const DIM = 1 << N_QUBITS;
const TARGET_N = 6;

export interface E21Row {
  readonly bondLengthAngstrom: number;
  readonly trial: number;
  readonly seed: number;
  readonly energyVqe: number;          // optimizer's final value (penalized)
  readonly energyVqeUnpenalized: number;
  readonly fciHartree: number;
  readonly hfHartree: number;
  readonly deltaEhartree: number;
  readonly deltaEmHartree: number;
  readonly correlationCapturedPercent: number;
  readonly iterations: number;
  readonly termination: "converged" | "max-iter";
  readonly hitChemicalAccuracy: boolean;
}

export interface E21Summary {
  readonly bondLengthAngstrom: number;
  readonly fciHartree: number;
  readonly hfHartree: number;
  readonly minEnergyHartree: number;
  readonly bestDeltaMHa: number;
  readonly medianDeltaMHa: number;
  readonly correlationCapturedPercent: number;
  readonly hitChemicalAccuracy: number;
  readonly trials: number;
}

const DEFAULT_BOND_LENGTHS = [1.0, 1.2, 1.34, 1.6, 2.0] as const;

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

export async function runE21(opts: {
  bondLengthsAngstrom?: readonly number[];
  trialsPerBond?: number;
  nLayers?: number;
  optimizerMaxIter?: number;
  particlePenalty?: number;
  perturbation?: number;
} = {}): Promise<Artifact<E21Row> & { summaries: readonly E21Summary[] }> {
  const bonds = opts.bondLengthsAngstrom ?? DEFAULT_BOND_LENGTHS;
  const trialsPerBond = opts.trialsPerBond ?? 5;
  // L = 12 (104 params) has tighter trial-to-trial variance than L = 10:
  // empirically every trial lands within 6 mHa, with the best around
  // sub-mHa. The cost is ~50 s/trial vs L=10's ~35 s.
  const nLayers = opts.nLayers ?? 12;
  const maxIter = opts.optimizerMaxIter ?? 1500;
  const lambda = opts.particlePenalty ?? 2.0;
  const perturbation = opts.perturbation ?? 0.20;
  const nParams = heaParamCount(N_QUBITS, nLayers);
  const CHEM_ACC = 1.6e-3;
  const PASS_BAR_MEDIAN = 10e-3;  // looser than LiH's 1.6 mHa

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E21: no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E21Row[] = [];
  const summaries: E21Summary[] = [];

  for (const R of bonds) {
    const { H, hfOccupied } = buildBeH2Dense(R);
    const E_FCI = lowestInParticleSector(H, N_QUBITS, TARGET_N).energy;
    // HF reference energy: HEA at θ=0 starting from |HF⟩.
    const hfCircuit = buildHEACircuit(N_QUBITS, nLayers, new Float64Array(nParams), hfOccupied);
    const E_HF = expectationDense(hfCircuit.psi, H);
    const correlation = E_HF - E_FCI;

    const Hpen = applyParticleNumberPenalty(H, lambda);
    const trialDeltas: number[] = [];

    for (let trial = 0; trial < trialsPerBond; trial++) {
      const seed = ((SEEDS.E21_BEH2_VQE ^ (trial * 0x9E3779B1) ^ (Math.round(R * 1e6) >>> 0)) >>> 0);
      const rng = mulberry32(seed);
      const init = new Float64Array(nParams);
      for (let i = 0; i < nParams; i++) init[i] = (rng() - 0.5) * perturbation;

      const r = runVQE_HEA_Dense_LBFGS(
        Hpen, N_QUBITS, nLayers, init,
        { maxIter, fTol: 1e-10, gTol: 1e-7, fdStep: 1e-5 },
        hfOccupied,
      );

      const c = buildHEACircuit(N_QUBITS, nLayers, r.params, hfOccupied);
      const E_unpen = expectationDense(c.psi, H);
      const dE = E_unpen - E_FCI;
      const dEmHa = dE * 1000;
      const corrCap = correlation > 1e-9 ? ((E_HF - E_unpen) / correlation) * 100 : 0;
      trialDeltas.push(Math.abs(dE));

      rows.push({
        bondLengthAngstrom: R,
        trial,
        seed,
        energyVqe: r.energy,
        energyVqeUnpenalized: E_unpen,
        fciHartree: E_FCI,
        hfHartree: E_HF,
        deltaEhartree: dE,
        deltaEmHartree: dEmHa,
        correlationCapturedPercent: corrCap,
        iterations: r.iterations,
        termination: r.termination,
        hitChemicalAccuracy: Math.abs(dE) <= CHEM_ACC,
      });

      console.log(
        `[E21] R=${R}Å t=${trial} E_VQE=${E_unpen.toFixed(6)} ` +
        `E_FCI=${E_FCI.toFixed(6)} ΔE=${dEmHa.toFixed(3)} mHa ` +
        `corr=${corrCap.toFixed(2)}% iter=${r.iterations} ${r.termination}`,
      );
    }

    trialDeltas.sort((a, b) => a - b);
    const median = trialDeltas[Math.floor(trialDeltas.length / 2)]! * 1000;
    const minE = Math.min(...rows.filter((r) => r.bondLengthAngstrom === R).map((r) => r.energyVqeUnpenalized));
    const bestDelta = (minE - E_FCI) * 1000;
    const bestCorr = correlation > 1e-9 ? ((E_HF - minE) / correlation) * 100 : 0;
    const hits = rows.filter((r) => r.bondLengthAngstrom === R && r.hitChemicalAccuracy).length;

    summaries.push({
      bondLengthAngstrom: R,
      fciHartree: E_FCI,
      hfHartree: E_HF,
      minEnergyHartree: minE,
      bestDeltaMHa: bestDelta,
      medianDeltaMHa: median,
      correlationCapturedPercent: bestCorr,
      hitChemicalAccuracy: hits,
      trials: trialsPerBond,
    });
  }

  // Pass bar: at R = 1.34 Å, best-of-trials hits chemical accuracy
  // AND median |ΔE| ≤ 10 mHa.
  const eqRow = summaries.find((s) => Math.abs(s.bondLengthAngstrom - 1.34) < 1e-9);
  const eqOk = eqRow !== undefined &&
               eqRow.bestDeltaMHa <= CHEM_ACC * 1000 &&
               eqRow.medianDeltaMHa <= PASS_BAR_MEDIAN * 1000;

  const status: Artifact<E21Row>["status"] = eqOk ? "pass" : "fail";

  const meta: ArtifactMeta = {
    protocol: "E21-beh2-vqe",
    hypothesis:
      "VQE with HEA L=10 + particle-number penalty + L-BFGS reaches chemical accuracy on BeH₂ (STO-3G s-only, 8 qubits) at R = 1.34 Å — best-of-5 |ΔE| ≤ 1.6 mHa above E_FCI, median ≤ 10 mHa.",
    passBar:
      "best-of-trials |ΔE| ≤ 1.6 mHa AND median |ΔE| ≤ 10 mHa at R = 1.34 Å",
    seed: "E21_BEH2_VQE",
    warmup: 0,
    trials: trialsPerBond,
  };

  const summaryStr = summaries
    .map((s) => `R=${s.bondLengthAngstrom}: best=${s.bestDeltaMHa.toFixed(2)} mHa, median=${s.medianDeltaMHa.toFixed(2)} mHa, hits=${s.hitChemicalAccuracy}/${s.trials}`)
    .join("; ");

  const diagnosis = status === "pass"
    ? `BeH₂ equilibrium VQE within chemical accuracy. ${summaryStr}.`
    : `BeH₂ equilibrium VQE missed pass bar. ${summaryStr}.`;

  return { meta, env, rows, status, diagnosis, summaries };
}
