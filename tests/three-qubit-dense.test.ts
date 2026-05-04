// Verify the 3-qubit cascade fusion: a single dense 8×8 dispatch
// reproduces the 5-step sequence (3 singles + CNOT + CNOT) to
// floating-point precision. CPU-only; the GPU agreement is
// further verified by E12 in the L3 e2e suite.

import { describe, expect, test } from "vitest";
import { CpuCircuit } from "../src/cpu-reference.js";
import { H, Rx, Ry, Rz, type Matrix2, X } from "../src/gates.js";
import {
  embedLowMid,
  embedMidHigh,
  fuseTriCascade,
  identity8,
  kron8,
  matmul8,
} from "../src/three-qubit-dense.js";
import { CNOT_LO_CTRL, kron4 } from "../src/two-qubit-dense.js";

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

describe("Matrix8 algebra", () => {
  test("identity8 · M = M", () => {
    const M = kron8(H, X, H);
    const I = identity8();
    const out = matmul8(I, M);
    for (let i = 0; i < 64; i++) {
      expect(Math.abs(out[i]![0] - M[i]![0])).toBeLessThan(1e-15);
      expect(Math.abs(out[i]![1] - M[i]![1])).toBeLessThan(1e-15);
    }
  });

  test("kron8 = kron(I_2, kron4) for the (qMid, qLo) sub-pair via embedLowMid", () => {
    // kron8(I, U_qMid, U_qLo) should equal embedLowMid(kron4(U_qMid, U_qLo)).
    const Umid = Rx(0.7);
    const Ulo = Ry(1.3);
    const Iq: Matrix2 = [[1, 0], [0, 0], [0, 0], [1, 0]];
    const direct = kron8(Iq, Umid, Ulo);
    const embedded = embedLowMid(kron4(Umid, Ulo));
    for (let i = 0; i < 64; i++) {
      expect(Math.abs(direct[i]![0] - embedded[i]![0])).toBeLessThan(1e-13);
      expect(Math.abs(direct[i]![1] - embedded[i]![1])).toBeLessThan(1e-13);
    }
  });

  test("embedMidHigh on identity 4×4 = identity 8×8", () => {
    const I4 = kron4(
      [[1, 0], [0, 0], [0, 0], [1, 0]],
      [[1, 0], [0, 0], [0, 0], [1, 0]],
    );
    const I8 = embedMidHigh(I4);
    const I8ref = identity8();
    for (let i = 0; i < 64; i++) {
      expect(Math.abs(I8[i]![0] - I8ref[i]![0])).toBeLessThan(1e-15);
      expect(Math.abs(I8[i]![1] - I8ref[i]![1])).toBeLessThan(1e-15);
    }
  });
});

describe("Tier C cascade fusion vs 5-step sequence", () => {
  for (const N of [3, 4, 5, 6, 8] as const) {
    test(`N=${N}: fused 8×8 cascade matches single+single+single+CNOT+CNOT on every triple`, () => {
      const seed = 0x71E3CA5C ^ (N * 0x9e3779b1);

      // Build a deterministic list of (qLo, qMid, qHi) triples + per-qubit single gates.
      let s = seed >>> 0;
      const rng = (): number => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };

      const triples: { qLo: number; qMid: number; qHi: number; Ulo: Matrix2; Umid: Matrix2; Uhi: Matrix2 }[] = [];
      for (let q = 0; q + 2 < N; q += 3) {
        triples.push({
          qLo: q,
          qMid: q + 1,
          qHi: q + 2,
          Ulo: Rz(rng() * 2 * Math.PI),
          Umid: Ry(rng() * 2 * Math.PI),
          Uhi: Rx(rng() * 2 * Math.PI),
        });
      }
      // Skip if no triples fit (N<3 cases).
      if (triples.length === 0) {
        expect(N).toBeLessThan(3);
        return;
      }

      // Reference path: 5 sequential gates per triple.
      const ref = new CpuCircuit(N);
      randomCircuitPrep(ref, seed);
      for (const t of triples) {
        ref.apply(t.Ulo, t.qLo);
        ref.apply(t.Umid, t.qMid);
        ref.apply(t.Uhi, t.qHi);
        ref.cnot(t.qLo, t.qMid);
        ref.cnot(t.qMid, t.qHi);
      }

      // Fused path: one dense 8×8 dispatch per triple.
      const fused = new CpuCircuit(N);
      randomCircuitPrep(fused, seed);
      for (const t of triples) {
        fused.applyDense8x8(fuseTriCascade(t.Ulo, t.Umid, t.Uhi), t.qLo, t.qMid, t.qHi);
      }

      const F = fidelity(ref.psi, fused.psi);
      expect(F).toBeGreaterThan(1 - 1e-12);
    });
  }

  test("embedded CNOT(qLo→qMid) on the |010⟩ basis state flips qMid only when qLo=1", () => {
    // Sanity check: prepare |001⟩ on a 3-qubit chain (qLo=1, others 0), apply
    // embedLowMid(CNOT_LO_CTRL). Expect qMid to flip → |011⟩.
    const c = new CpuCircuit(3);
    // |001⟩ = qLo=1, qMid=0, qHi=0 → basis index 1 (LSB convention).
    c.psi.fill(0);
    c.psi[1 * 2] = 1;

    c.applyDense8x8(embedLowMid(CNOT_LO_CTRL), 0, 1, 2);

    // After CNOT(qLo→qMid): qLo=1 → qMid flips to 1. State is qLo=1, qMid=1, qHi=0
    // → basis index 1 + 2 = 3.
    expect(Math.abs(c.psi[3 * 2]! - 1)).toBeLessThan(1e-14);
    expect(Math.abs(c.psi[1 * 2]!)).toBeLessThan(1e-14);
  });
});
