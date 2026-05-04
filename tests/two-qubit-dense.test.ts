// Verify the brick-wall pair fusion: a single dense 4×4 dispatch
// reproduces the 3-step sequence (single-q + single-q+1 + CNOT)
// to floating-point precision. CPU-only — the GPU agreement is
// further verified by the L3 e2e suite when E10 invokes the same
// kernel against the unfused statevector path.

import { describe, expect, test } from "vitest";
import { CpuCircuit } from "../src/cpu-reference.js";
import { H, Rx, Ry, Rz, X, type Matrix2 } from "../src/gates.js";
import { fuseBrickwallPair, kron4, matmul4, identity4, CNOT_LO_CTRL } from "../src/two-qubit-dense.js";

function fidelity(a: Float64Array, b: Float64Array): number {
  let re = 0, im = 0;
  for (let i = 0; i < a.length; i += 2) {
    const ar = a[i]!, ai = a[i + 1]!;
    const br = b[i]!, bi = b[i + 1]!;
    re += ar * br + ai * bi;
    im += ar * bi - ai * br;
  }
  return re * re + im * im;
}

function randomCircuitPrep(c: CpuCircuit, seed: number): void {
  let s = seed >>> 0;
  const rng = (): number => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let q = 0; q < c.nQubits; q++) {
    c.apply(Rx(rng() * Math.PI), q);
    c.apply(Ry(rng() * Math.PI), q);
  }
}

describe("Matrix4 algebra", () => {
  test("identity4 · M = M", () => {
    const M = kron4(H, X);
    const I = identity4();
    const out = matmul4(I, M);
    for (let i = 0; i < 16; i++) {
      expect(Math.abs(out[i]![0] - M[i]![0])).toBeLessThan(1e-15);
      expect(Math.abs(out[i]![1] - M[i]![1])).toBeLessThan(1e-15);
    }
  });

  test("CNOT_LO_CTRL twice = identity", () => {
    const sq = matmul4(CNOT_LO_CTRL, CNOT_LO_CTRL);
    const I = identity4();
    for (let i = 0; i < 16; i++) {
      expect(Math.abs(sq[i]![0] - I[i]![0])).toBeLessThan(1e-15);
    }
  });

  test("kron4(H, X) is the right tensor product on the (s_qHi, s_qLo) basis", () => {
    // (H ⊗ X)|s_qHi=0, s_qLo=0⟩ = (H|0⟩) ⊗ (X|0⟩) = (1/√2)(|0⟩+|1⟩) ⊗ |1⟩
    //   = (1/√2)(|01⟩ + |11⟩)  — basis index 1 and 3 with amplitude 1/√2.
    const M = kron4(H, X);
    const sqrt1_2 = 1 / Math.SQRT2;
    expect(Math.abs(M[0 * 4 + 0]![0])).toBeLessThan(1e-15);
    expect(Math.abs(M[1 * 4 + 0]![0] - sqrt1_2)).toBeLessThan(1e-15);
    expect(Math.abs(M[3 * 4 + 0]![0] - sqrt1_2)).toBeLessThan(1e-15);
  });
});

describe("fuseBrickwallPair vs 3-step sequence", () => {
  for (const N of [4, 6, 8] as const) {
    test(`N=${N}: fused 4×4 matches single+single+CNOT on every brick pair`, () => {
      const seed = 0xBA1C2A11 + N;

      // Reference path: 3 sequential gates.
      const ref = new CpuCircuit(N);
      randomCircuitPrep(ref, seed);
      const pairs: [number, number, Matrix2, Matrix2][] = [];
      let s = seed >>> 0;
      const rng = (): number => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      for (let q = 0; q + 1 < N; q += 2) {
        const Ulo: Matrix2 = Rz(rng() * 2 * Math.PI);
        const Uhi: Matrix2 = Ry(rng() * 2 * Math.PI);
        pairs.push([q, q + 1, Ulo, Uhi]);
        ref.apply(Ulo, q);
        ref.apply(Uhi, q + 1);
        ref.cnot(q, q + 1);
      }

      // Fused path: one dense 4×4 dispatch per pair.
      const fused = new CpuCircuit(N);
      randomCircuitPrep(fused, seed);
      for (const [qLo, qHi, Ulo, Uhi] of pairs) {
        fused.applyDense4x4(fuseBrickwallPair(Ulo, Uhi, "lo"), qLo, qHi);
      }

      const F = fidelity(ref.psi, fused.psi);
      expect(F).toBeGreaterThan(1 - 1e-12);
    });
  }

  test("CNOT_HI_CTRL fusion matches the swapped CNOT direction", () => {
    const N = 4;
    const ref = new CpuCircuit(N);
    randomCircuitPrep(ref, 0xC0DECAFE);
    // Apply: single on q=2 (qHi), single on q=1 (qLo), CNOT(c=2, t=1).
    ref.apply(H, 1);
    ref.apply(Rx(0.7), 2);
    ref.applyControlled(X, 2, 1);

    const fused = new CpuCircuit(N);
    randomCircuitPrep(fused, 0xC0DECAFE);
    fused.applyDense4x4(fuseBrickwallPair(H, Rx(0.7), "hi"), 1, 2);

    expect(fidelity(ref.psi, fused.psi)).toBeGreaterThan(1 - 1e-12);
  });
});
