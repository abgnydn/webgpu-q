// ─────────────────────────────────────────────────────────────
// ansatz.ts — Parametric circuits that prepare a trial wave-
// function from a parameter vector. Drives every VQE inner loop.
//
// Two ansatzes are exposed:
//
//   • Hardware-efficient (HEA) — alternating layers of single-qubit
//     Ry rotations and a linear ladder of CNOTs. Hits chemical
//     accuracy on H₂ at modest depth; not particle-conserving so it
//     can wander out of the physical sector during optimization
//     (recovers if the ground state is in N_e = 2).
//
//   • Particle-conserving "double-excitation" — for H₂ the entire
//     correlation energy lives in the |0011⟩ ↔ |1100⟩ subspace, so
//     a 1-parameter Givens rotation between those determinants is
//     literally enough to reach FCI. Use this in tests to sanity-
//     check that VQE *can* converge before pointing it at HEA.
//
// All ansatzes drive the existing CpuCircuit so the statevector
// path is shared with every other CPU-reference test.
// ─────────────────────────────────────────────────────────────

import { CpuCircuit } from "../cpu-reference.js";
import { Ry, X } from "../gates.js";

/**
 * Hardware-efficient ansatz.
 *
 *   |ψ(θ)⟩ = (∏_{l=0..L-1} Entangler · Ry⊗N (θ_l)) · Ry⊗N (θ_L) · |φ_0⟩
 *
 * `initialOccupied` controls |φ_0⟩: each qubit listed gets an X gate
 * applied before the first Ry layer. Default is empty → |0…0⟩.
 *
 * For chemistry the right reference is the Hartree-Fock determinant
 * (N_e qubits flipped on) so that small θ stays near the physical
 * sector — without this, optimization on |0…0⟩ has to traverse states
 * with the wrong electron number, which traps Nelder-Mead at poor
 * local minima especially at long bond lengths.
 *
 * Layer count = nLayers; total parameter count = (nLayers + 1) · nQubits.
 */
export function buildHEACircuit(
  nQubits: number,
  nLayers: number,
  params: Float64Array,
  initialOccupied: readonly number[] = [],
): CpuCircuit {
  const expected = (nLayers + 1) * nQubits;
  if (params.length !== expected) {
    throw new Error(`HEA: param length ${params.length} ≠ ${expected} for nQubits=${nQubits} nLayers=${nLayers}`);
  }
  const c = new CpuCircuit(nQubits);
  for (const q of initialOccupied) c.apply(X, q);

  let p = 0;
  for (let q = 0; q < nQubits; q++) c.apply(Ry(params[p++]!), q);

  for (let l = 0; l < nLayers; l++) {
    for (let q = 0; q < nQubits - 1; q++) c.cnot(q, q + 1);
    for (let q = 0; q < nQubits; q++) c.apply(Ry(params[p++]!), q);
  }
  return c;
}

export function heaParamCount(nQubits: number, nLayers: number): number {
  return (nLayers + 1) * nQubits;
}

/**
 * For H₂: a 1-parameter Givens-style ansatz that mixes |0011⟩ ↔ |1100⟩.
 *
 *   |ψ(θ)⟩ = cos(θ/2)·|0011⟩ + sin(θ/2)·|1100⟩
 *
 * Built directly into the statevector (no quantum circuit decomposition
 * — this is a "trust me" ansatz used as the FCI ceiling for VQE tests).
 * Particle number is preserved by construction.
 */
export function buildH2GivensState(theta: number): Float64Array {
  const N = 16;
  const psi = new Float64Array(2 * N);
  // |0011⟩ = basis index 3 (q0=q1=1).
  // |1100⟩ = basis index 12 (q2=q3=1).
  psi[3 * 2] = Math.cos(theta / 2);
  psi[12 * 2] = Math.sin(theta / 2);
  return psi;
}

