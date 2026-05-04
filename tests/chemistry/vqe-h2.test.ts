// VQE on H₂ — stack-end verification. If these pass, every
// downstream chemistry experiment can trust:
//   • the optimizer reduces a cost function (Rosenbrock smoke test)
//   • the Givens 1-parameter ansatz reaches FCI on H₂ (because H₂'s
//     ground state lives entirely in the |0011⟩ ↔ |1100⟩ subspace)
//   • the HEA ansatz reaches chemical accuracy when given enough
//     layers and a reasonable seed
import { describe, expect, test } from "vitest";
import { H2_R0_7414, E_FCI_H2_R0_7414, E_HF_H2_R0_7414 } from "../../src/chemistry/h2.js";
import { runVQE_HEA, runVQE_Custom } from "../../src/chemistry/vqe.js";
import { buildH2GivensState, heaParamCount } from "../../src/chemistry/ansatz.js";
import { nelderMead } from "../../src/chemistry/optimizer.js";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("Nelder-Mead optimizer", () => {
  test("converges on the Rosenbrock function from a far init", () => {
    const rosenbrock = (x: Float64Array): number => {
      const a = 1, b = 100;
      return (a - x[0]!) ** 2 + b * (x[1]! - x[0]! * x[0]!) ** 2;
    };
    const x0 = new Float64Array([-1.2, 1.0]);
    const r = nelderMead(rosenbrock, x0, { maxIter: 4000, fTol: 1e-9, xTol: 1e-7 });
    expect(r.bestF).toBeLessThan(1e-6);
    expect(Math.abs(r.bestX[0]! - 1)).toBeLessThan(1e-3);
    expect(Math.abs(r.bestX[1]! - 1)).toBeLessThan(1e-3);
  });
});

describe("VQE on H₂ — Givens 1-parameter ansatz", () => {
  test("reaches FCI within Jacobi tolerance", () => {
    // 1-parameter ansatz that mixes |0011⟩ ↔ |1100⟩. The H₂ ground state
    // is in this subspace, so VQE *must* hit FCI to within optimizer noise.
    const init = new Float64Array([0.1]);
    const r = runVQE_Custom(
      H2_R0_7414,
      (theta: Float64Array) => buildH2GivensState(theta[0]!),
      init,
      { maxIter: 500, fTol: 1e-12, xTol: 1e-9 },
    );
    expect(Math.abs(r.energy - E_FCI_H2_R0_7414)).toBeLessThan(1e-7);
    expect(r.termination).toBe("converged");
  });
});

describe("VQE on H₂ — Hardware-Efficient Ansatz", () => {
  test("HEA(L=3) hits chemical accuracy from a random init", () => {
    const nQubits = 4;
    const nLayers = 3;
    const nParams = heaParamCount(nQubits, nLayers);
    const rng = mulberry32(0xE15);
    const init = new Float64Array(nParams);
    for (let i = 0; i < nParams; i++) init[i] = (rng() - 0.5) * 0.4;

    const r = runVQE_HEA(H2_R0_7414, nQubits, nLayers, init, {
      maxIter: 4000,
      fTol: 1e-8,
      xTol: 1e-6,
      initialStep: 0.3,
      seed: 0x4242,
    });

    // Chemical accuracy = 1.6 mHa.
    expect(r.energy - E_FCI_H2_R0_7414).toBeLessThan(1.6e-3);
    // VQE must beat HF (otherwise we found a stationary point above |HF⟩).
    expect(r.energy).toBeLessThan(E_HF_H2_R0_7414);
  });
});
