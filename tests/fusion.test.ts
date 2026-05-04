import { describe, expect, test } from "vitest";
import { CpuCircuit } from "../src/cpu-reference";
import { fuseMatrices, matmul2 } from "../src/fusion";
import { H, X, Y, Z, S, T, Rx, Ry, Rz, I2 } from "../src/gates";
import type { Matrix2 } from "../src/gates";

// Sum of |Δψ|² between two interleaved [re, im] buffers.
function l2Sq(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    s += d * d;
  }
  return s;
}

describe("fusion — matmul2 + fuseMatrices", () => {
  test("matmul2 identity laws", () => {
    const M: Matrix2 = Rx(0.7);
    const IM = matmul2(I2, M);
    const MI = matmul2(M, I2);
    for (let i = 0; i < 4; i++) {
      expect(IM[i]![0]).toBeCloseTo(M[i]![0], 12);
      expect(IM[i]![1]).toBeCloseTo(M[i]![1], 12);
      expect(MI[i]![0]).toBeCloseTo(M[i]![0], 12);
      expect(MI[i]![1]).toBeCloseTo(M[i]![1], 12);
    }
  });

  test("HH = I", () => {
    const P = fuseMatrices([H, H]);
    expect(P[0]![0]).toBeCloseTo(1, 12); expect(P[0]![1]).toBeCloseTo(0, 12);
    expect(P[1]![0]).toBeCloseTo(0, 12); expect(P[1]![1]).toBeCloseTo(0, 12);
    expect(P[2]![0]).toBeCloseTo(0, 12); expect(P[2]![1]).toBeCloseTo(0, 12);
    expect(P[3]![0]).toBeCloseTo(1, 12); expect(P[3]![1]).toBeCloseTo(0, 12);
  });

  test("XX = I, YY = I, ZZ = I", () => {
    for (const G of [X, Y, Z]) {
      const P = fuseMatrices([G, G]);
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const want = r === c ? 1 : 0;
          expect(P[r * 2 + c]![0]).toBeCloseTo(want, 10);
          expect(P[r * 2 + c]![1]).toBeCloseTo(0, 10);
        }
      }
    }
  });

  test("T⁴ = Z (up to numerical precision)", () => {
    const P = fuseMatrices([T, T, T, T]);
    for (let i = 0; i < 4; i++) {
      expect(P[i]![0]).toBeCloseTo(Z[i]![0], 10);
      expect(P[i]![1]).toBeCloseTo(Z[i]![1], 10);
    }
  });

  test("Rx(a)Rx(b) = Rx(a+b)", () => {
    const P = fuseMatrices([Rx(0.7), Rx(0.3)]);
    const Q = Rx(1.0);
    for (let i = 0; i < 4; i++) {
      expect(P[i]![0]).toBeCloseTo(Q[i]![0], 12);
      expect(P[i]![1]).toBeCloseTo(Q[i]![1], 12);
    }
  });

  test("ordering: fuseMatrices([A, B]) acts like A then B", () => {
    // "A then B" means apply A first, then B — equivalent to B · A · |ψ⟩.
    // A one-qubit state |0⟩; A = Rx(θ_a); B = Ry(θ_b).
    const A = Rx(0.9);
    const B = Ry(0.4);
    const P = fuseMatrices([A, B]);

    // Reference: apply A then B via CpuCircuit on N=1.
    const ref = new CpuCircuit(1);
    ref.apply(A, 0);
    ref.apply(B, 0);

    const test = new CpuCircuit(1);
    test.apply(P, 0);

    expect(l2Sq(ref.psi, test.psi)).toBeLessThan(1e-20);
  });

  test("chain on |0…⟩ N=6: fused product equals k unfused applies", () => {
    const N = 6;
    const q = 2;
    const ks = [1, 2, 3, 8, 16, 32];
    for (const k of ks) {
      const chain: Matrix2[] = [];
      // Deterministic rotation sequence.
      for (let i = 0; i < k; i++) {
        const pick = i % 4;
        if (pick === 0) chain.push(Rx(0.1 + 0.01 * i));
        else if (pick === 1) chain.push(Ry(0.2 + 0.013 * i));
        else if (pick === 2) chain.push(Rz(0.3 + 0.017 * i));
        else chain.push(S);
      }

      const ref = new CpuCircuit(N);
      for (const M of chain) ref.apply(M, q);

      const fused = new CpuCircuit(N);
      fused.apply(fuseMatrices(chain), q);

      expect(l2Sq(ref.psi, fused.psi)).toBeLessThan(1e-18);
    }
  });

  test("empty chain fuses to identity", () => {
    const P = fuseMatrices([]);
    for (let i = 0; i < 4; i++) {
      expect(P[i]![0]).toBeCloseTo(I2[i]![0], 12);
      expect(P[i]![1]).toBeCloseTo(I2[i]![1], 12);
    }
  });

  test("long chain of random gates — unitarity check", () => {
    // Product of unitaries is unitary. Verify P · P^† = I up to ε.
    const chain: Matrix2[] = [];
    let seed = 0xC0FFEE;
    const rng = (): number => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    for (let i = 0; i < 50; i++) {
      const pick = i % 3;
      if (pick === 0) chain.push(Rx(rng() * 2 * Math.PI));
      else if (pick === 1) chain.push(Ry(rng() * 2 * Math.PI));
      else chain.push(Rz(rng() * 2 * Math.PI));
    }
    const P = fuseMatrices(chain);
    const Pdag: Matrix2 = [
      [P[0][0], -P[0][1]],
      [P[2][0], -P[2][1]],  // (P^†)[0][1] = conj(P[1][0])
      [P[1][0], -P[1][1]],
      [P[3][0], -P[3][1]],
    ];
    const PPh = matmul2(P, Pdag);
    expect(PPh[0][0]).toBeCloseTo(1, 10); expect(PPh[0][1]).toBeCloseTo(0, 10);
    expect(PPh[3][0]).toBeCloseTo(1, 10); expect(PPh[3][1]).toBeCloseTo(0, 10);
    expect(PPh[1][0]).toBeCloseTo(0, 10); expect(PPh[1][1]).toBeCloseTo(0, 10);
    expect(PPh[2][0]).toBeCloseTo(0, 10); expect(PPh[2][1]).toBeCloseTo(0, 10);
  });
});
