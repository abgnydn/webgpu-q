import { describe, expect, test } from "vitest";
import { CpuCircuit } from "../src/cpu-reference";
import { H, X, Y, Z, S, T, Rx, Ry, Rz } from "../src/gates";

const EPS = 1e-12;

function approxEq(a: number, b: number, eps = EPS): boolean {
  return Math.abs(a - b) < eps;
}

describe("single-qubit gates", () => {
  test("|0⟩ stays |0⟩ under identity of products (XX = I)", () => {
    const c = new CpuCircuit(1);
    c.apply(X, 0);
    c.apply(X, 0);
    expect(approxEq(c.probabilities()[0]!, 1)).toBe(true);
  });

  test("H|0⟩ = (|0⟩ + |1⟩)/√2 → 50/50", () => {
    const c = new CpuCircuit(1);
    c.apply(H, 0);
    const p = c.probabilities();
    expect(approxEq(p[0]!, 0.5)).toBe(true);
    expect(approxEq(p[1]!, 0.5)).toBe(true);
  });

  test("HH = I", () => {
    const c = new CpuCircuit(1);
    c.apply(H, 0);
    c.apply(H, 0);
    expect(approxEq(c.probabilities()[0]!, 1)).toBe(true);
    expect(approxEq(c.probabilities()[1]!, 0)).toBe(true);
  });

  test("X|0⟩ = |1⟩", () => {
    const c = new CpuCircuit(1);
    c.apply(X, 0);
    expect(approxEq(c.probabilities()[1]!, 1)).toBe(true);
  });

  test("Z|+⟩ = |−⟩ (still 50/50 in Z basis)", () => {
    const c = new CpuCircuit(1);
    c.apply(H, 0);
    c.apply(Z, 0);
    const p = c.probabilities();
    expect(approxEq(p[0]!, 0.5)).toBe(true);
    expect(approxEq(p[1]!, 0.5)).toBe(true);
  });

  test("YY = I (up to global phase, same probabilities)", () => {
    const c = new CpuCircuit(1);
    c.apply(Y, 0);
    c.apply(Y, 0);
    expect(approxEq(c.probabilities()[0]!, 1)).toBe(true);
  });

  test("S² = Z on |+⟩ maintains 50/50", () => {
    const c = new CpuCircuit(1);
    c.apply(H, 0);
    c.apply(S, 0);
    c.apply(S, 0);
    const p = c.probabilities();
    expect(approxEq(p[0]!, 0.5)).toBe(true);
    expect(approxEq(p[1]!, 0.5)).toBe(true);
  });

  test("T⁴ = Z on |+⟩", () => {
    const c = new CpuCircuit(1);
    c.apply(H, 0);
    for (let i = 0; i < 4; i++) c.apply(T, 0);
    const p = c.probabilities();
    expect(approxEq(p[0]!, 0.5)).toBe(true);
    expect(approxEq(p[1]!, 0.5)).toBe(true);
  });

  test("Rx(π) on |0⟩ → |1⟩ (up to phase)", () => {
    const c = new CpuCircuit(1);
    c.apply(Rx(Math.PI), 0);
    expect(approxEq(c.probabilities()[1]!, 1, 1e-10)).toBe(true);
  });

  test("Ry(π/2) on |0⟩ = 50/50", () => {
    const c = new CpuCircuit(1);
    c.apply(Ry(Math.PI / 2), 0);
    const p = c.probabilities();
    expect(approxEq(p[0]!, 0.5, 1e-12)).toBe(true);
    expect(approxEq(p[1]!, 0.5, 1e-12)).toBe(true);
  });

  test("Rz is diagonal → only phases change, probabilities fixed", () => {
    const c = new CpuCircuit(1);
    c.apply(H, 0);
    c.apply(Rz(1.234), 0);
    const p = c.probabilities();
    expect(approxEq(p[0]!, 0.5)).toBe(true);
    expect(approxEq(p[1]!, 0.5)).toBe(true);
  });
});

describe("multi-qubit & entanglement", () => {
  test("Bell state: H(0) CNOT(0,1) → {|00⟩: 0.5, |11⟩: 0.5}", () => {
    const c = new CpuCircuit(2);
    c.apply(H, 0);
    c.cnot(0, 1);
    const p = c.probabilities();
    expect(approxEq(p[0]!, 0.5)).toBe(true);
    expect(approxEq(p[1]!, 0)).toBe(true);
    expect(approxEq(p[2]!, 0)).toBe(true);
    expect(approxEq(p[3]!, 0.5)).toBe(true);
  });

  test("GHZ-3: H(0) CNOT(0,1) CNOT(1,2) → {|000⟩: 0.5, |111⟩: 0.5}", () => {
    const c = new CpuCircuit(3);
    c.apply(H, 0);
    c.cnot(0, 1);
    c.cnot(1, 2);
    const p = c.probabilities();
    expect(approxEq(p[0b000]!, 0.5)).toBe(true);
    expect(approxEq(p[0b111]!, 0.5)).toBe(true);
    // Everything else = 0
    for (let i = 1; i < 7; i++) expect(approxEq(p[i]!, 0)).toBe(true);
  });

  test("GHZ-5 builds the cat state", () => {
    const c = new CpuCircuit(5);
    c.apply(H, 0);
    for (let k = 0; k < 4; k++) c.cnot(k, k + 1);
    const p = c.probabilities();
    expect(approxEq(p[0b00000]!, 0.5)).toBe(true);
    expect(approxEq(p[0b11111]!, 0.5)).toBe(true);
  });

  test("CNOT with control = 1 flips target", () => {
    // Prepare |10⟩, then CNOT(0,1) → |11⟩ (bit 0 is LSB)
    const c = new CpuCircuit(2);
    c.apply(X, 0); // |01⟩ in bit-string (q0=1, q1=0) → index = 1
    c.cnot(0, 1);
    const p = c.probabilities();
    // q0=1, q1=1 → index 0b11 = 3
    expect(approxEq(p[3]!, 1)).toBe(true);
  });

  test("CNOT with control = 0 does nothing", () => {
    const c = new CpuCircuit(2);
    // stay in |00⟩
    c.cnot(0, 1);
    expect(approxEq(c.probabilities()[0]!, 1)).toBe(true);
  });

  test("norm preserved through a mid-size random circuit", () => {
    const c = new CpuCircuit(6);
    const rng = mulberry32(42);
    for (let step = 0; step < 40; step++) {
      const q = Math.floor(rng() * 6);
      const op = Math.floor(rng() * 4);
      if (op === 0) c.apply(H, q);
      else if (op === 1) c.apply(X, q);
      else if (op === 2) c.apply(Rz(rng() * Math.PI), q);
      else {
        const q2 = (q + 1 + Math.floor(rng() * 5)) % 6;
        c.cnot(q, q2);
      }
    }
    expect(approxEq(c.norm(), 1, 1e-9)).toBe(true);
  });
});

// Deterministic RNG for reproducible random-circuit tests.
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
