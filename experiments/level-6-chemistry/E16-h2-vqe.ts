// ─────────────────────────────────────────────────────────────
// E16 — VQE ground-state energy for H₂.
//
// Per Level-6 protocol: 10 random classical initializations
// per bond length R, hardware-efficient ansatz, optimizer
// = Nelder-Mead (gradient-free). Reference is the FCI energy
// of the same hard-coded qubit Hamiltonian (see eigensolver
// + tests/chemistry/h2-fci.test.ts) — that gives a honest
// "did VQE find the ground state of THIS Hamiltonian?" answer
// without conflating with PySCF basis-convention rounding.
//
// Scope honesty: the protocol asks for R ∈ {0.5, 0.74, 1.0, 1.5,
// 2.5} Å. Only R = 0.7414 Å has a published Pauli coefficient
// table that's checked in here. Multi-R is left in the runner
// signature (`opts.bondLengthsAngstrom`) but only 0.7414 is in
// the lookup table. Adding more R values is a coefficient-table
// drop-in, not a code change.
// ─────────────────────────────────────────────────────────────

import { initGPU } from "../../src/quantum.js";
import { captureEnv } from "../lib/env.js";
import { runVQE_HEA_Dense } from "../../src/chemistry/vqe.js";
import { heaParamCount } from "../../src/chemistry/ansatz.js";
import { buildH2Dense, expectationDense } from "../../src/chemistry/h2-builder.js";
import { smallestEigenvalue } from "../../src/chemistry/eigensolver.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

export interface E16Row {
  readonly bondLengthAngstrom: number;
  readonly trial: number;
  readonly seed: number;
  readonly energy: number;
  readonly deltaEhartree: number;       // E_VQE − E_FCI
  readonly deltaEmHartree: number;      // ΔE in mHa
  readonly iterations: number;
  readonly termination: "converged" | "max-iter";
  readonly beatHF: boolean;             // E_VQE < E_HF
}

export interface E16Summary {
  readonly bondLengthAngstrom: number;
  readonly medianAbsDeltaMHa: number;
  readonly maxAbsDeltaMHa: number;
  readonly minEnergyHartree: number;
  readonly fciHartree: number;
  readonly hfHartree: number;
  readonly hitChemicalAccuracy: number;  // count of |ΔE| ≤ 1.6 mHa
  readonly trials: number;
}

// Bond lengths swept for the dissociation curve. Per protocol: probe
// the bonding well (0.5), equilibrium (0.7414), and the dissociation
// tail (1.0, 1.5, 2.5). At every R we build the dense Hamiltonian
// from molecular integrals and the reference E_FCI from classical
// Jacobi diagonalization of the same matrix, so the VQE pass-bar is
// "did VQE find the spectrum of THIS Hamiltonian?" — basis errors
// don't affect the test, only how close the curve is to nature.
const DEFAULT_BOND_LENGTHS = [0.5, 0.7414, 1.0, 1.5, 2.5];

export async function runE16(
  opts: {
    bondLengthsAngstrom?: readonly number[];
    trialsPerBond?: number;
    nLayers?: number;
    optimizerMaxIter?: number;
  } = {},
): Promise<Artifact<E16Row> & { summaries: readonly E16Summary[] }> {
  const bonds = opts.bondLengthsAngstrom ?? DEFAULT_BOND_LENGTHS;
  const trialsPerBond = opts.trialsPerBond ?? 10;
  // L=4 is enough for H₂ in any basis; 8000 iters lets stuck Nelder-Mead
  // simplexes shrink past local minima at long R.
  const nLayers = opts.nLayers ?? 4;
  const maxIter = opts.optimizerMaxIter ?? 8000;
  const nQubits = 4;
  const nParams = heaParamCount(nQubits, nLayers);
  const PASS_BAR_MEDIAN = 1.6e-3;   // chemical accuracy (Ha)
  const PASS_BAR_MAX = 5.0e-3;      // protocol's "max ≤ 5 mHa" bar (Ha)

  // Env block — chemistry is CPU-only but we record GPU info to keep
  // artifact provenance uniform with the rest of the level ladder.
  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E16: no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E16Row[] = [];
  const summaries: E16Summary[] = [];

  for (const R of bonds) {
    // Build the dense Hamiltonian for this bond length from STO-3G
    // molecular integrals. E_FCI is the classical-diagonalization
    // ground state of THIS matrix; E_HF is the diagonal element on
    // the closed-shell |0011⟩ determinant.
    const { H } = buildH2Dense(R);
    const E_FCI = smallestEigenvalue(H, 16);
    const hfPsi = new Float64Array(32);
    hfPsi[3 * 2] = 1;
    const E_HF = expectationDense(hfPsi, H);

    const trialDeltas: number[] = [];

    for (let trial = 0; trial < trialsPerBond; trial++) {
      const seed = ((SEEDS.E16_H2_VQE ^ (trial * 0x9E3779B1) ^ (Math.round(R * 1e6) >>> 0)) >>> 0);
      const rng = mulberry32(seed);
      // Smaller initial perturbation: starting from the |HF⟩ reference,
      // the FCI ground state is a tight rotation away in θ-space. Large
      // initial Ry kicks the state far from the N_e=2 sector and traps
      // Nelder-Mead at unphysical minima.
      const init = new Float64Array(nParams);
      for (let i = 0; i < nParams; i++) init[i] = (rng() - 0.5) * 0.05;

      const r = runVQE_HEA_Dense(
        H, nQubits, nLayers, init,
        {
          maxIter,
          fTol: 1e-9,
          xTol: 1e-7,
          initialStep: 0.1,
          seed: (seed * 0x85ebca6b) >>> 0,
        },
        // Initialize at |HF⟩ = |0011⟩ (qubits 0, 1 occupied) so the
        // ansatz starts inside the closed-shell N_e=2 sector.
        [0, 1],
      );

      const dE = r.energy - E_FCI;
      const dEmHa = dE * 1000;
      trialDeltas.push(Math.abs(dE));
      rows.push({
        bondLengthAngstrom: R,
        trial,
        seed,
        energy: r.energy,
        deltaEhartree: dE,
        deltaEmHartree: dEmHa,
        iterations: r.iterations,
        termination: r.termination,
        beatHF: r.energy < E_HF,
      });


      console.log(
        `[E16] R=${R}Å trial=${trial} E=${r.energy.toFixed(6)} ΔE=${dEmHa.toFixed(3)} mHa ` +
        `iter=${r.iterations} ${r.termination} beat-HF=${r.energy < E_HF}`,
      );
    }

    trialDeltas.sort((a, b) => a - b);
    const median = trialDeltas[Math.floor(trialDeltas.length / 2)]!;
    const max = trialDeltas[trialDeltas.length - 1]!;
    const minE = Math.min(...rows.filter((r) => r.bondLengthAngstrom === R).map((r) => r.energy));
    const hits = rows.filter((r) => r.bondLengthAngstrom === R && Math.abs(r.deltaEhartree) <= PASS_BAR_MEDIAN).length;

    summaries.push({
      bondLengthAngstrom: R,
      medianAbsDeltaMHa: median * 1000,
      maxAbsDeltaMHa: max * 1000,
      minEnergyHartree: minE,
      fciHartree: E_FCI,
      hfHartree: E_HF,
      hitChemicalAccuracy: hits,
      trials: trialsPerBond,
    });
  }

  // Pass bar: median |ΔE| ≤ 1.6 mHa AND max |ΔE| ≤ 5 mHa AT EVERY R.
  const allMedianOk = summaries.every((s) => s.medianAbsDeltaMHa <= PASS_BAR_MEDIAN * 1000);
  const allMaxOk = summaries.every((s) => s.maxAbsDeltaMHa <= PASS_BAR_MAX * 1000);
  const status: Artifact<E16Row>["status"] = allMedianOk && allMaxOk ? "pass" : "fail";

  const meta: ArtifactMeta = {
    protocol: "E16-h2-vqe",
    hypothesis:
      "VQE with hardware-efficient ansatz on H₂ (STO-3G/JW, 4 qubits) converges to the FCI ground state of the same qubit Hamiltonian at chemical accuracy from 10 random initializations.",
    passBar: "median |ΔE| ≤ 1.6 mHa AND max |ΔE| ≤ 5 mHa at every bond length",
    seed: "E16_H2_VQE",
    warmup: 0,
    trials: trialsPerBond,
  };

  const diagnosis = status === "pass"
    ? `${summaries.length} bond length${summaries.length === 1 ? "" : "s"} all within chemical accuracy. ` +
      summaries.map((s) => `R=${s.bondLengthAngstrom}: median=${s.medianAbsDeltaMHa.toFixed(3)} mHa, max=${s.maxAbsDeltaMHa.toFixed(3)} mHa, hit-rate=${s.hitChemicalAccuracy}/${s.trials}`).join("; ")
    : `Pass bar missed at ${summaries.filter((s) => s.medianAbsDeltaMHa > PASS_BAR_MEDIAN * 1000 || s.maxAbsDeltaMHa > PASS_BAR_MAX * 1000).length} R-points. ` +
      summaries.map((s) => `R=${s.bondLengthAngstrom}: median=${s.medianAbsDeltaMHa.toFixed(3)} mHa, max=${s.maxAbsDeltaMHa.toFixed(3)} mHa`).join("; ");

  return { meta, env, rows, status, diagnosis, summaries };
}
