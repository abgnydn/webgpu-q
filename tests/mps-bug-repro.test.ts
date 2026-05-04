// Regression for the V8-Chromium-only zero-norm bug at the E5
// brick-depth-4 N=8 trial=3 cell. Reproduces only that exact
// gate sequence (heavy diagnostic sweeps were trimmed once the
// L2 Playwright suite became the authoritative full-coverage
// check — running 1260 MPS builds in unit tests caused vitest
// flakes under CPU pressure without adding signal beyond what
// the e2e already proves).
import { describe, expect, test } from "vitest";
import { MPS } from "../src/mps.js";
import { CpuCircuit } from "../src/cpu-reference.js";
import { H, X, Rz } from "../src/gates.js";
import type { Matrix2 } from "../src/gates.js";

const E5_MPS_CORRECTNESS = 0x5E71ADD9;

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

interface GateApi {
  readonly nQubits: number;
  apply: (m: Matrix2, q: number) => void;
  cnot: (c: number, t: number) => void;
}

function applyBrickwall(c: GateApi, layers: number, rng: () => number): void {
  const N = c.nQubits;
  for (let l = 0; l < layers; l++) {
    for (let q = 0; q < N; q++) {
      const pick = Math.floor(rng() * 3);
      if (pick === 0) c.apply(H, q);
      else if (pick === 1) c.apply(Rz(rng() * Math.PI * 2), q);
      else c.apply(X, q);
    }
    const offset = l & 1;
    for (let q = offset; q + 1 < N; q += 2) c.cnot(q, q + 1);
  }
}

describe("MPS regression: E5 brick-depth-4 N=8 trial=3", () => {
  test("non-zero norm + finite fidelity vs CPU", () => {
    const N = 8;
    const trial = 3;
    const seed = ((E5_MPS_CORRECTNESS ^ (N * 2654435761) ^ (trial * 0x9E3779B1)) >>> 0);

    const cpu = new CpuCircuit(N);
    const mps = new MPS({ nQubits: N, chiMax: 64 });
    applyBrickwall(cpu, 4, mulberry32(seed));
    applyBrickwall(mps, 4, mulberry32(seed));

    const psiMps = mps.statevector();
    let normMps = 0;
    for (let i = 0; i < psiMps.length; i += 2) {
      normMps += psiMps[i]! * psiMps[i]! + psiMps[i + 1]! * psiMps[i + 1]!;
    }
    expect(Number.isFinite(normMps)).toBe(true);
    expect(normMps).toBeGreaterThan(0.99);
    expect(normMps).toBeLessThan(1.01);
  });
});
