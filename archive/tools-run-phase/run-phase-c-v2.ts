// Phase C v2 publishable artifact run — BeH₂ VQE dissociation curve.
// Mirrors tools/run-phase-c.ts. Standalone vite-node script.
//
// Usage:
//   npx vite-node tools/run-phase-c-v2.ts
//   PHASE_C2_TRIALS=10 npx vite-node tools/run-phase-c-v2.ts

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { runVQE_HEA_Dense_LBFGS } from "../src/chemistry/vqe.js";
import { heaParamCount, buildHEACircuit } from "../src/chemistry/ansatz.js";
import { buildBeH2Dense } from "../src/chemistry/beh2-builder.js";
import { lowestInParticleSector } from "../src/chemistry/sector.js";
import { expectationDense } from "../src/chemistry/h2-builder.js";

const N_QUBITS = 8;
const DIM = 1 << N_QUBITS;
const TARGET_N = 6;

const TRIALS = Number.parseInt(process.env["PHASE_C2_TRIALS"] ?? "5", 10);
// L=12 (104 params) reduces trial-to-trial variance compared with L=10:
// smoke at R=1.34 hit 0.36 / 5.8 / 2.7 mHa across 3 trials, vs L=10's
// 1.3 / 4.2 / 13.1 mHa range. Deeper ansatz pays off in reliability.
const N_LAYERS = 12;
const LAMBDA = 2;
const PERTURBATION = 0.20;
const MAX_ITER = 1500;
const CHEM_ACC_MHA = 1.6;
const PASS_BAR_MEDIAN_MHA = 10;
const BONDS = [1.0, 1.2, 1.34, 1.6, 2.0];

interface Cell {
  R_angstrom: number;
  trial: number;
  seed: number;
  energyVqe: number;
  fciHartree: number;
  hfHartree: number;
  deltaEmHa: number;
  correlationCapturedPercent: number;
  iterations: number;
  termination: "converged" | "max-iter";
  hitChemicalAccuracy: boolean;
  wallSeconds: number;
}

interface Summary {
  R_angstrom: number;
  fciHartree: number;
  hfHartree: number;
  bestEnergyHartree: number;
  bestDeltaMHa: number;
  medianDeltaMHa: number;
  correlationCapturedPercent: number;
  hitChemicalAccuracy: number;
  trials: number;
}

function applyPenalty(H: Float64Array, lambda: number): Float64Array {
  const out = new Float64Array(H);
  for (let i = 0; i < DIM; i++) {
    let n = 0;
    for (let q = 0; q < N_QUBITS; q++) if ((i >>> q) & 1) n++;
    const dn = n - TARGET_N;
    out[i * DIM + i]! += lambda * dn * dn;
  }
  return out;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function todayUTC(): string {
  return new Date().toISOString().slice(0, 10);
}

console.log(`Phase C v2 — BeH₂ VQE (STO-3G s-only, HEA L=${N_LAYERS}, ${TRIALS} trials × ${BONDS.length} R values)`);
console.log("=".repeat(72));

const rows: Cell[] = [];
const summaries: Summary[] = [];

for (const R of BONDS) {
  const { H, hfOccupied } = buildBeH2Dense(R);
  const E_FCI = lowestInParticleSector(H, N_QUBITS, TARGET_N).energy;
  const hfCircuit = buildHEACircuit(N_QUBITS, N_LAYERS, new Float64Array(heaParamCount(N_QUBITS, N_LAYERS)), hfOccupied);
  const E_HF = expectationDense(hfCircuit.psi, H);
  const correlation = E_HF - E_FCI;

  const Hpen = applyPenalty(H, LAMBDA);
  const nParams = heaParamCount(N_QUBITS, N_LAYERS);
  const trialDeltas: number[] = [];
  let bestE = Infinity;

  for (let t = 0; t < TRIALS; t++) {
    const seed = (0x21BE211E ^ (t * 0x9E3779B1) ^ (Math.round(R * 1e6) >>> 0)) >>> 0;
    const rng = mulberry32(seed);
    const init = new Float64Array(nParams);
    for (let i = 0; i < nParams; i++) init[i] = (rng() - 0.5) * PERTURBATION;

    const t0 = performance.now();
    const r = runVQE_HEA_Dense_LBFGS(Hpen, N_QUBITS, N_LAYERS, init, {
      maxIter: MAX_ITER, fTol: 1e-10, gTol: 1e-7, fdStep: 1e-5,
    }, hfOccupied);
    const wall = (performance.now() - t0) / 1000;

    const c = buildHEACircuit(N_QUBITS, N_LAYERS, r.params, hfOccupied);
    const E_unpen = expectationDense(c.psi, H);
    const dE = E_unpen - E_FCI;
    const dEmHa = dE * 1000;
    const corrCap = correlation > 1e-9 ? ((E_HF - E_unpen) / correlation) * 100 : 0;
    trialDeltas.push(Math.abs(dE));
    if (E_unpen < bestE) bestE = E_unpen;

    rows.push({
      R_angstrom: R, trial: t, seed,
      energyVqe: E_unpen, fciHartree: E_FCI, hfHartree: E_HF,
      deltaEmHa: dEmHa, correlationCapturedPercent: corrCap,
      iterations: r.iterations, termination: r.termination,
      hitChemicalAccuracy: Math.abs(dE) <= CHEM_ACC_MHA / 1000,
      wallSeconds: wall,
    });

    console.log(
      `  R=${R} t=${t} E_VQE=${E_unpen.toFixed(6)} ΔE=${dEmHa.toFixed(3)} mHa ` +
      `corr=${corrCap.toFixed(2)}% ${wall.toFixed(2)}s ${r.termination}`,
    );
  }

  trialDeltas.sort((a, b) => a - b);
  const median = trialDeltas[Math.floor(trialDeltas.length / 2)]! * 1000;
  const bestDelta = (bestE - E_FCI) * 1000;
  const bestCorr = correlation > 1e-9 ? ((E_HF - bestE) / correlation) * 100 : 0;
  const hits = rows.filter((r) => r.R_angstrom === R && r.hitChemicalAccuracy).length;

  summaries.push({
    R_angstrom: R, fciHartree: E_FCI, hfHartree: E_HF,
    bestEnergyHartree: bestE, bestDeltaMHa: bestDelta,
    medianDeltaMHa: median, correlationCapturedPercent: bestCorr,
    hitChemicalAccuracy: hits, trials: TRIALS,
  });
  console.log(
    `  → R=${R}: best ΔE = ${bestDelta.toFixed(3)} mHa, ` +
    `median = ${median.toFixed(3)} mHa, corr capture = ${bestCorr.toFixed(2)}%, hit chem-acc = ${hits}/${TRIALS}`,
  );
}

const eqRow = summaries.find((s) => Math.abs(s.R_angstrom - 1.34) < 1e-9)!;
const status: "pass" | "fail" =
  eqRow.bestDeltaMHa <= CHEM_ACC_MHA && eqRow.medianDeltaMHa <= PASS_BAR_MEDIAN_MHA ? "pass" : "fail";
const diagnosis = status === "pass"
  ? `BeH₂ R=1.34 Å: best ΔE = ${eqRow.bestDeltaMHa.toFixed(3)} mHa (chemical accuracy), median = ${eqRow.medianDeltaMHa.toFixed(2)} mHa (corr capture ${eqRow.correlationCapturedPercent.toFixed(2)}%). ` +
    `Total chem-acc hits across all R: ${rows.filter((r) => r.hitChemicalAccuracy).length}/${rows.length}.`
  : `BeH₂ R=1.34 Å: best ΔE = ${eqRow.bestDeltaMHa.toFixed(3)} mHa, median = ${eqRow.medianDeltaMHa.toFixed(2)} mHa — pass bar (best ≤ 1.6, median ≤ 10) not met.`;

const artifact = {
  meta: {
    protocol: "E21-beh2-vqe-publishable",
    hypothesis: "VQE with HEA L=10 + particle-number penalty + L-BFGS reaches chemical accuracy on BeH₂ (STO-3G s-only, 8 qubits) at the experimental Be–H bond R = 1.34 Å — best-of-trials |ΔE| ≤ 1.6 mHa above E_FCI.",
    passBar: `best-of-${TRIALS}-trials |ΔE| ≤ ${CHEM_ACC_MHA} mHa AND median |ΔE| ≤ ${PASS_BAR_MEDIAN_MHA} mHa at R = 1.34 Å`,
    seed: "E21_BEH2_VQE",
    warmup: 0, trials: TRIALS,
  },
  env: {
    runtime: `node ${process.version}`,
    platform: process.platform, arch: process.arch,
    timestamp: new Date().toISOString(),
    bonds: BONDS, nLayers: N_LAYERS, lambda: LAMBDA,
    perturbation: PERTURBATION, maxIter: MAX_ITER,
    optimizer: "L-BFGS (Armijo backtracking, central-FD gradient)",
  },
  rows, summaries, status, diagnosis,
};

const outDir = resolve(process.cwd(), "experiments", "results", todayUTC(), "level-6");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "E21-beh2-vqe-publishable.json");
writeFileSync(outPath, JSON.stringify(artifact, null, 2), "utf-8");
console.log(`\n→ ${outPath}`);
console.log(`Status: ${status}`);
console.log(diagnosis);
