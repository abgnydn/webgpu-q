import { describe, expect, test } from "vitest";
import { MPS } from "../src/mps";
import { CpuCircuit } from "../src/cpu-reference";
import { H, X, Y, Z, S, T, Rx, Ry, Rz } from "../src/gates";

function fidelity(psiRef: Float64Array | Float32Array, psiTest: Float64Array | Float32Array): number {
  // |⟨ref|test⟩|² with no normalization — both should be unit-norm here.
  let re = 0, im = 0;
  const n = psiRef.length / 2;
  for (let i = 0; i < n; i++) {
    const ar = psiRef[i * 2]!, ai = psiRef[i * 2 + 1]!;
    const br = psiTest[i * 2]!, bi = psiTest[i * 2 + 1]!;
    re += ar * br + ai * bi;       // conj(ref)·test = (ar − i·ai)(br + i·bi)
    im += ar * bi - ai * br;
  }
  return re * re + im * im;
}

describe("MPS — initial state & single-qubit gates", () => {
  test("|000…⟩ initial state: statevector is e_0", () => {
    const mps = new MPS({ nQubits: 4 });
    const psi = mps.statevector();
    expect(psi[0]).toBeCloseTo(1, 12);
    expect(psi[1]).toBeCloseTo(0, 12);
    for (let i = 2; i < psi.length; i++) expect(psi[i]).toBeCloseTo(0, 12);
  });

  test("H|0⟩ on qubit 0 gives (|0⟩+|1⟩)/√2", () => {
    const mps = new MPS({ nQubits: 1 });
    mps.apply(H, 0);
    const psi = mps.statevector();
    expect(psi[0]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(psi[2]).toBeCloseTo(Math.SQRT1_2, 12);
  });

  test("X flips qubit 0 on |0⟩", () => {
    const mps = new MPS({ nQubits: 3 });
    mps.apply(X, 0);
    const psi = mps.statevector();
    // After X on q0: index 1 (q0=1) = 1.
    expect(psi[1 * 2]).toBeCloseTo(1, 12);
  });

  test("X on qubit 2 of a 3-qubit state", () => {
    const mps = new MPS({ nQubits: 3 });
    mps.apply(X, 2);
    const psi = mps.statevector();
    // After X on q2: index 4 (q2=1) = 1.
    expect(psi[4 * 2]).toBeCloseTo(1, 12);
  });

  test("HH = I on any qubit", () => {
    const mps = new MPS({ nQubits: 5 });
    mps.apply(H, 2);
    mps.apply(H, 2);
    const psi = mps.statevector();
    expect(psi[0]).toBeCloseTo(1, 10);
    for (let i = 1; i < 32; i++) expect(psi[i * 2]).toBeCloseTo(0, 10);
  });

  test("rotations match CpuCircuit on 1-qubit sequences", () => {
    for (const angle of [0.3, 1.1, 2.4, -0.8]) {
      const mps = new MPS({ nQubits: 1 });
      const cpu = new CpuCircuit(1);
      mps.apply(Rx(angle), 0);
      mps.apply(Rz(angle / 2), 0);
      mps.apply(Ry(-angle), 0);
      cpu.apply(Rx(angle), 0);
      cpu.apply(Rz(angle / 2), 0);
      cpu.apply(Ry(-angle), 0);
      const F = fidelity(cpu.psi, mps.statevector());
      expect(F).toBeGreaterThan(1 - 1e-10);
    }
  });

  test("YZS on q1 of a 4-qubit state — fidelity vs CPU", () => {
    const mps = new MPS({ nQubits: 4 });
    const cpu = new CpuCircuit(4);
    mps.apply(Y, 1);
    cpu.apply(Y, 1);
    mps.apply(Z, 1);
    cpu.apply(Z, 1);
    mps.apply(S, 1);
    cpu.apply(S, 1);
    mps.apply(T, 1);
    cpu.apply(T, 1);
    const F = fidelity(cpu.psi, mps.statevector());
    expect(F).toBeGreaterThan(1 - 1e-10);
  });
});

describe("MPS — two-site gates, Bell & GHZ", () => {
  test("Bell state via H(0) + CNOT(0,1)", () => {
    const mps = new MPS({ nQubits: 2 });
    mps.apply(H, 0);
    mps.cnot(0, 1);
    const psi = mps.statevector();
    // (|00⟩ + |11⟩)/√2 → indices 0 and 3.
    expect(psi[0]).toBeCloseTo(Math.SQRT1_2, 10);
    expect(psi[6]).toBeCloseTo(Math.SQRT1_2, 10);  // index 3 → offset 6 (re)
    expect(psi[2]).toBeCloseTo(0, 10);
    expect(psi[4]).toBeCloseTo(0, 10);
  });

  test("Bell state via H(1) + CNOT(1,0) — c > t variant", () => {
    const mps = new MPS({ nQubits: 2 });
    const cpu = new CpuCircuit(2);
    mps.apply(H, 1);
    cpu.apply(H, 1);
    mps.cnot(1, 0);
    cpu.cnot(1, 0);
    const F = fidelity(cpu.psi, mps.statevector());
    expect(F).toBeGreaterThan(1 - 1e-10);
  });

  test("GHZ-5 via adjacent CNOT ladder", () => {
    const mps = new MPS({ nQubits: 5 });
    mps.apply(H, 0);
    for (let k = 0; k < 4; k++) mps.cnot(k, k + 1);
    const psi = mps.statevector();
    // |00000⟩ index 0, |11111⟩ index 31.
    expect(psi[0]).toBeCloseTo(Math.SQRT1_2, 9);
    expect(psi[31 * 2]).toBeCloseTo(Math.SQRT1_2, 9);
    // Everything else near zero.
    for (let i = 1; i < 31; i++) {
      const re = psi[i * 2]!, im = psi[i * 2 + 1]!;
      expect(re * re + im * im).toBeLessThan(1e-18);
    }
  });

  test("GHZ-8 fidelity vs CpuCircuit", () => {
    const N = 8;
    const mps = new MPS({ nQubits: N });
    const cpu = new CpuCircuit(N);
    mps.apply(H, 0);
    cpu.apply(H, 0);
    for (let k = 0; k < N - 1; k++) {
      mps.cnot(k, k + 1);
      cpu.cnot(k, k + 1);
    }
    const F = fidelity(cpu.psi, mps.statevector());
    expect(F).toBeGreaterThan(1 - 1e-9);
  });

  test("controlled-Rz(θ) on adjacent pair matches CPU", () => {
    for (const theta of [0.7, 1.8, -1.3]) {
      const mps = new MPS({ nQubits: 4 });
      const cpu = new CpuCircuit(4);
      mps.apply(H, 0);
      cpu.apply(H, 0);
      mps.apply(H, 1);
      cpu.apply(H, 1);
      mps.applyControlled(Rz(theta), 1, 2);
      cpu.applyControlled(Rz(theta), 1, 2);
      const F = fidelity(cpu.psi, mps.statevector());
      expect(F).toBeGreaterThan(1 - 1e-10);
    }
  });
});

describe("MPS — brick-wall random circuit vs CPU ground truth", () => {
  const rng = (seed: number) => {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  function brickwall(mps: MPS, cpu: CpuCircuit, layers: number, seed: number): void {
    const N = mps.nQubits;
    const r = rng(seed);
    for (let l = 0; l < layers; l++) {
      for (let q = 0; q < N; q++) {
        const pick = Math.floor(r() * 3);
        if (pick === 0) { mps.apply(H, q); cpu.apply(H, q); }
        else if (pick === 1) {
          const th = r() * Math.PI * 2;
          mps.apply(Rz(th), q); cpu.apply(Rz(th), q);
        } else { mps.apply(X, q); cpu.apply(X, q); }
      }
      const offset = l & 1;
      for (let q = offset; q + 1 < N; q += 2) {
        mps.cnot(q, q + 1);
        cpu.cnot(q, q + 1);
      }
    }
  }

  for (const [N, depth] of [[4, 2], [6, 3], [8, 2], [10, 2]] as const) {
    test(`N=${N} depth=${depth} brick-wall fidelity ≥ 1 − 1e-8 at χ=64`, () => {
      const mps = new MPS({ nQubits: N, chiMax: 64 });
      const cpu = new CpuCircuit(N);
      brickwall(mps, cpu, depth, 0x5EEDBEEF + N);
      const F = fidelity(cpu.psi, mps.statevector());
      expect(F).toBeGreaterThan(1 - 1e-8);
      expect(Math.abs(mps.norm() - 1)).toBeLessThan(1e-9);
    });
  }
});

describe("MPS — truncation & bond dimensions", () => {
  test("bond dims grow as entanglement spreads", () => {
    const mps = new MPS({ nQubits: 6, chiMax: 64 });
    // Starts at all chi=1
    expect(mps.bondDimensions()).toEqual([1, 1, 1, 1, 1]);
    mps.apply(H, 0);
    mps.cnot(0, 1);
    expect(mps.bondDimensions()[0]).toBe(2);
    mps.cnot(1, 2);
    expect(mps.bondDimensions()[1]).toBe(2);
  });

  test("truncation to χ=2 on a GHZ stays exact (entropy ≤ 1)", () => {
    // GHZ has Schmidt rank 2 at every bond → χ=2 suffices.
    const N = 6;
    const mps = new MPS({ nQubits: N, chiMax: 2 });
    const cpu = new CpuCircuit(N);
    mps.apply(H, 0);
    cpu.apply(H, 0);
    for (let k = 0; k < N - 1; k++) {
      mps.cnot(k, k + 1);
      cpu.cnot(k, k + 1);
    }
    const F = fidelity(cpu.psi, mps.statevector());
    expect(F).toBeGreaterThan(1 - 1e-9);
    expect(Math.max(...mps.bondDimensions())).toBe(2);
    // No truncation loss — the exact Schmidt rank fit under the budget.
    expect(mps.truncationLoss()).toBeLessThan(1e-18);
  });

  test("norm stays near 1 across many truncations", () => {
    const mps = new MPS({ nQubits: 8, chiMax: 4 });  // aggressive truncation
    const rngFn = (() => {
      let a = 0x1234 >>> 0;
      return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    })();
    for (let l = 0; l < 6; l++) {
      for (let q = 0; q < 8; q++) mps.apply(Rx(rngFn() * Math.PI), q);
      for (let q = 0; q + 1 < 8; q += 2) mps.cnot(q, q + 1);
      for (let q = 1; q + 1 < 8; q += 2) mps.cnot(q, q + 1);
    }
    expect(Math.abs(mps.norm() - 1)).toBeLessThan(1e-10);
  });
});

describe("MPS — Schmidt spectrum", () => {
  test("|0…⟩ product state: all cuts have σ = [1]", () => {
    const mps = new MPS({ nQubits: 5 });
    for (let cut = 0; cut < 4; cut++) {
      const sigma = mps.schmidtSpectrum(cut);
      expect(sigma[0]).toBeCloseTo(1, 12);
      for (let i = 1; i < sigma.length; i++) expect(sigma[i] ?? 0).toBeCloseTo(0, 12);
    }
  });

  test("GHZ state: mid-cut σ = [1/√2, 1/√2] (entropy = 1 bit)", () => {
    const N = 6;
    const mps = new MPS({ nQubits: N });
    mps.apply(H, 0);
    for (let k = 0; k < N - 1; k++) mps.cnot(k, k + 1);
    const sigma = mps.schmidtSpectrum(Math.floor(N / 2) - 1);
    expect(sigma[0]).toBeCloseTo(Math.SQRT1_2, 10);
    expect(sigma[1]).toBeCloseTo(Math.SQRT1_2, 10);
    // Entropy S = −Σ p log₂ p = 1.
    let S = 0;
    for (const s of sigma) {
      const p = s * s;
      if (p > 0) S -= p * Math.log2(p);
    }
    expect(S).toBeCloseTo(1, 9);
  });

  test("product state after local gates: every cut keeps σ ≈ [1]", () => {
    const mps = new MPS({ nQubits: 4 });
    mps.apply(Rx(0.7), 0);
    mps.apply(Ry(1.3), 2);
    mps.apply(Rz(-0.4), 3);
    for (let cut = 0; cut < 3; cut++) {
      const sigma = mps.schmidtSpectrum(cut);
      expect(sigma[0]).toBeCloseTo(1, 10);
      for (let i = 1; i < sigma.length; i++) expect(sigma[i] ?? 0).toBeCloseTo(0, 9);
    }
  });

  test("schmidt spectrum leaves represented state invariant (fidelity ≈ 1)", () => {
    const N = 6;
    const mps = new MPS({ nQubits: N, chiMax: 16 });
    const cpu = new CpuCircuit(N);
    mps.apply(H, 0);
    cpu.apply(H, 0);
    for (let k = 0; k + 1 < N; k++) {
      mps.cnot(k, k + 1);
      cpu.applyControlled(X, k, k + 1);
    }
    // Call schmidtSpectrum then read the statevector.
    mps.schmidtSpectrum(2);
    const F = fidelity(cpu.psi, mps.statevector());
    expect(F).toBeGreaterThan(1 - 1e-10);
  });
});
