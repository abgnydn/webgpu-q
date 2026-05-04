// ─────────────────────────────────────────────────────────────
// Circuit builders — the textbook circuits used by demo + bench.
//
// Signatures take a generic "Gate API" object so the same builder
// works against QuantumCircuit (GPU) and CpuCircuit (CPU reference).
// ─────────────────────────────────────────────────────────────

import { H, Rz, X } from "./gates.js";

/** Minimal interface every builder uses. */
export interface GateApi {
  readonly nQubits: number;
  apply(matrix: import("./gates.js").Matrix2, qubit: number): void;
  applyControlled(matrix: import("./gates.js").Matrix2, control: number, target: number): void;
  cnot(control: number, target: number): void;
}

// ── Bell pair: (|00⟩ + |11⟩)/√2 on qubits (0, 1) ────────────
export function bell(c: GateApi): void {
  c.apply(H, 0);
  c.cnot(0, 1);
}

// ── GHZ_n: (|0…0⟩ + |1…1⟩)/√2 ───────────────────────────────
export function ghz(c: GateApi): void {
  c.apply(H, 0);
  for (let k = 0; k < c.nQubits - 1; k++) c.cnot(k, k + 1);
}

// ── Quantum Fourier Transform on qubits [0, n) ───────────────
// Standard QFT with swap at the end. Uses CRz approximation of
// the controlled phase gate: controlled-P(θ) = controlled-Rz(θ)
// up to a global phase factor of e^{iθ/2}, which we don't observe
// at the probability level.
export function qft(c: GateApi): void {
  const n = c.nQubits;
  for (let i = 0; i < n; i++) {
    c.apply(H, i);
    for (let j = i + 1; j < n; j++) {
      const theta = Math.PI / (1 << (j - i));
      // controlled-Rz(θ) from j onto i
      c.applyControlled(Rz(theta), j, i);
    }
  }
  // bit-reversal via swap = 3 CNOTs
  for (let i = 0; i < Math.floor(n / 2); i++) {
    const a = i, b = n - 1 - i;
    c.cnot(a, b);
    c.cnot(b, a);
    c.cnot(a, b);
  }
}

// ── Deutsch–Jozsa for a constant-0 oracle ────────────────────
// Prepares |+…+⟩|−⟩, applies identity oracle, H on input → |0…0⟩.
// Good sanity: DJ on constant oracle must yield all-zero probability
// mass concentrated at |0…0⟩ in the input register.
export function deutschJozsaConstant(c: GateApi): void {
  const n = c.nQubits - 1; // last qubit is the ancilla
  c.apply(X, n);
  for (let i = 0; i < c.nQubits; i++) c.apply(H, i);
  // oracle = identity (constant 0)
  for (let i = 0; i < n; i++) c.apply(H, i);
}

// ── Random single+two-qubit layer circuit, seeded ────────────
// Used by the benchmark + by tests that want stressful traffic.
export function randomCircuit(c: GateApi, layers: number, seed = 0xC0FFEE): void {
  const n = c.nQubits;
  const rng = mulberry32(seed >>> 0);
  for (let l = 0; l < layers; l++) {
    // single-qubit sweep
    for (let q = 0; q < n; q++) {
      const pick = Math.floor(rng() * 3);
      if (pick === 0) c.apply(H, q);
      else if (pick === 1) c.apply(Rz(rng() * Math.PI * 2), q);
      else c.apply(X, q);
    }
    // brick-wall CNOTs
    const offset = l & 1;
    for (let q = offset; q + 1 < n; q += 2) c.cnot(q, q + 1);
  }
}

// Deterministic RNG so bench + tests reproduce.
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
