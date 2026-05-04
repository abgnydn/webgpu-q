// Verify the 4-qubit cascade fusion: a single dense 16×16 dispatch
// reproduces the 7-step sequence (4 singles + 3 cascading CNOTs) to
// floating-point precision. CPU-only; the GPU agreement is further
// verified by E13 in the L3 e2e suite.

import { describe, expect, test } from "vitest";
import { CpuCircuit } from "../src/cpu-reference.js";
import { Rx, Ry, Rz, type Matrix2 } from "../src/gates.js";
import {
  embed4Q_LoMid1,
  embed4Q_Mid2Hi,
  fuseQuadCascade,
  identity16,
  kron16,
  matmul16,
} from "../src/four-qubit-dense.js";
import { CNOT_LO_CTRL } from "../src/two-qubit-dense.js";

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

describe("Matrix16 algebra", () => {
  test("identity16 · M = M", () => {
    const M = kron16(Rx(0.4), Ry(0.7), Rz(1.1), Rx(1.5));
    const I = identity16();
    const out = matmul16(I, M);
    for (let i = 0; i < 256; i++) {
      expect(Math.abs(out[i]![0] - M[i]![0])).toBeLessThan(1e-13);
      expect(Math.abs(out[i]![1] - M[i]![1])).toBeLessThan(1e-13);
    }
  });

  test("embedded CNOT(qLo→qMid1) on |0001⟩ flips qMid1 (4-qubit chain)", () => {
    // |0001⟩ = qLo=1, others 0 → basis index 1.
    const c = new CpuCircuit(4);
    c.psi.fill(0);
    c.psi[1 * 2] = 1;
    c.applyDense16x16(embed4Q_LoMid1(CNOT_LO_CTRL), 0, 1, 2, 3);
    // qLo=1 → qMid1 flips → state qLo=1, qMid1=1, others 0 → idx = 1+2 = 3.
    expect(Math.abs(c.psi[3 * 2]! - 1)).toBeLessThan(1e-14);
    expect(Math.abs(c.psi[1 * 2]!)).toBeLessThan(1e-14);
  });

  test("embed4Q_Mid2Hi on identity 4×4 = identity 16×16", () => {
    const I4 = [
      [1, 0], [0, 0], [0, 0], [0, 0],
      [0, 0], [1, 0], [0, 0], [0, 0],
      [0, 0], [0, 0], [1, 0], [0, 0],
      [0, 0], [0, 0], [0, 0], [1, 0],
    ] as const;
    const I16 = embed4Q_Mid2Hi(I4 as never);
    const I16ref = identity16();
    for (let i = 0; i < 256; i++) {
      expect(Math.abs(I16[i]![0] - I16ref[i]![0])).toBeLessThan(1e-15);
    }
  });
});

describe("Tier D cascade fusion vs 7-step sequence", () => {
  for (const N of [4, 5, 6, 8] as const) {
    test(`N=${N}: fused 16×16 cascade matches 4-singles + 3-CNOTs on every quadruple`, () => {
      const seed = 0xD4CA51 ^ (N * 0x9e3779b1);

      let s = seed >>> 0;
      const rng = (): number => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };

      const tiles: { qLo: number; qMid1: number; qMid2: number; qHi: number;
                     Ulo: Matrix2; Umid1: Matrix2; Umid2: Matrix2; Uhi: Matrix2 }[] = [];
      for (let q = 0; q + 3 < N; q += 4) {
        tiles.push({
          qLo: q, qMid1: q + 1, qMid2: q + 2, qHi: q + 3,
          Ulo: Rz(rng() * 2 * Math.PI),
          Umid1: Ry(rng() * 2 * Math.PI),
          Umid2: Rx(rng() * 2 * Math.PI),
          Uhi: Rz(rng() * 2 * Math.PI),
        });
      }
      if (tiles.length === 0) {
        expect(N).toBeLessThan(4);
        return;
      }

      const ref = new CpuCircuit(N);
      randomCircuitPrep(ref, seed);
      for (const t of tiles) {
        ref.apply(t.Ulo, t.qLo);
        ref.apply(t.Umid1, t.qMid1);
        ref.apply(t.Umid2, t.qMid2);
        ref.apply(t.Uhi, t.qHi);
        ref.cnot(t.qLo, t.qMid1);
        ref.cnot(t.qMid1, t.qMid2);
        ref.cnot(t.qMid2, t.qHi);
      }

      const fused = new CpuCircuit(N);
      randomCircuitPrep(fused, seed);
      for (const t of tiles) {
        fused.applyDense16x16(
          fuseQuadCascade(t.Ulo, t.Umid1, t.Umid2, t.Uhi),
          t.qLo, t.qMid1, t.qMid2, t.qHi,
        );
      }

      const F = fidelity(ref.psi, fused.psi);
      expect(F).toBeGreaterThan(1 - 1e-12);
    });
  }
});
