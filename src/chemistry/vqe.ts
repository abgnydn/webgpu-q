// ─────────────────────────────────────────────────────────────
// vqe.ts — Variational quantum eigensolver orchestrator.
//
// Inputs: a Hamiltonian, an ansatz (params → CpuCircuit or
// directly a statevector), an initial parameter vector, and an
// optimizer. Returns the converged energy and parameters.
//
// Energy oracle:  E(θ) = ⟨ψ(θ)| H |ψ(θ)⟩ summed over Pauli terms.
// Stop criterion delegated to the optimizer.
// ─────────────────────────────────────────────────────────────

import type { Hamiltonian } from "./pauli.js";
import { expectationHamiltonian } from "./pauli.js";
import { buildHEACircuit } from "./ansatz.js";
import { expectationDense } from "./h2-builder.js";
import { nelderMead, type NelderMeadOptions, type NelderMeadResult } from "./optimizer.js";

export interface VQEResult {
  energy: number;
  params: Float64Array;
  iterations: number;
  termination: "converged" | "max-iter";
  history: readonly number[];
}

/**
 * Run VQE with a hardware-efficient ansatz on the given Hamiltonian.
 * Generic over qubit count but trial circuits run on the CPU
 * reference simulator (Float64). For ≤ 20 qubits this is fast.
 */
export function runVQE_HEA(
  H: Hamiltonian,
  nQubits: number,
  nLayers: number,
  initParams: Float64Array,
  optOpts: NelderMeadOptions = {},
): VQEResult {
  const energyOf = (theta: Float64Array): number => {
    const c = buildHEACircuit(nQubits, nLayers, theta);
    return expectationHamiltonian(c.psi, H);
  };
  const r: NelderMeadResult = nelderMead(energyOf, initParams, optOpts);
  return {
    energy: r.bestF,
    params: r.bestX,
    iterations: r.iter,
    termination: r.termination,
    history: r.history,
  };
}

/**
 * Same as runVQE_HEA but the Hamiltonian is supplied as a dense N×N
 * Hermitian matrix. Avoids the Pauli decomposition step — useful when
 * the Hamiltonian comes from molecular integrals (h2-builder) and the
 * cost of evaluating ψ†Hψ directly is cheaper than summing per-Pauli
 * expectations for a long Pauli sum.
 */
export function runVQE_HEA_Dense(
  Hdense: Float64Array,
  nQubits: number,
  nLayers: number,
  initParams: Float64Array,
  optOpts: NelderMeadOptions = {},
  initialOccupied: readonly number[] = [],
): VQEResult {
  const energyOf = (theta: Float64Array): number => {
    const c = buildHEACircuit(nQubits, nLayers, theta, initialOccupied);
    return expectationDense(c.psi, Hdense);
  };
  const r: NelderMeadResult = nelderMead(energyOf, initParams, optOpts);
  return {
    energy: r.bestF,
    params: r.bestX,
    iterations: r.iter,
    termination: r.termination,
    history: r.history,
  };
}

/**
 * Drop-in oracle: any callable θ → ψ(θ) (length 2·2^n Float64Array)
 * is acceptable, so callers can supply Givens / UCC / custom ansatzes.
 */
export function runVQE_Custom(
  H: Hamiltonian,
  ansatzToPsi: (theta: Float64Array) => Float64Array,
  initParams: Float64Array,
  optOpts: NelderMeadOptions = {},
): VQEResult {
  const energyOf = (theta: Float64Array): number => {
    const psi = ansatzToPsi(theta);
    return expectationHamiltonian(psi, H);
  };
  const r = nelderMead(energyOf, initParams, optOpts);
  return {
    energy: r.bestF,
    params: r.bestX,
    iterations: r.iter,
    termination: r.termination,
    history: r.history,
  };
}
