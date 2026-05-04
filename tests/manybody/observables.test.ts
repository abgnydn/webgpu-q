import { describe, expect, test } from "vitest";
import { siteExpectations } from "../../src/manybody/observables.js";

describe("siteExpectations", () => {
  test("|0…0⟩ has ⟨σ_z⟩ = 1, ⟨σ_x⟩ = ⟨σ_y⟩ = 0", () => {
    const N = 4;
    const psi = new Float64Array(2 * (1 << N));
    psi[0] = 1; // |0000⟩
    const e = siteExpectations(psi, N);
    for (let i = 0; i < N; i++) {
      expect(e.sz[i]).toBeCloseTo(1, 12);
      expect(e.sx[i]).toBeCloseTo(0, 12);
      expect(e.sy[i]).toBeCloseTo(0, 12);
    }
  });

  test("|+⟩ on a single qubit has ⟨σ_x⟩ = 1, ⟨σ_z⟩ = 0", () => {
    const N = 1;
    const psi = new Float64Array([1 / Math.SQRT2, 0, 1 / Math.SQRT2, 0]);
    const e = siteExpectations(psi, N);
    expect(e.sx[0]).toBeCloseTo(1, 12);
    expect(e.sz[0]).toBeCloseTo(0, 12);
    expect(e.sy[0]).toBeCloseTo(0, 12);
  });

  test("|+i⟩ = (|0⟩ + i|1⟩)/√2 has ⟨σ_y⟩ = 1", () => {
    const psi = new Float64Array([1 / Math.SQRT2, 0, 0, 1 / Math.SQRT2]);
    const e = siteExpectations(psi, 1);
    expect(e.sy[0]).toBeCloseTo(1, 12);
    expect(e.sx[0]).toBeCloseTo(0, 12);
    expect(e.sz[0]).toBeCloseTo(0, 12);
  });

  test("Bell state has zero local magnetization on every site", () => {
    // |Bell⟩ = (|00⟩ + |11⟩)/√2: maximally mixed reduced density matrices,
    // every Pauli expectation is zero on each individual site.
    const N = 2;
    const psi = new Float64Array(2 * (1 << N));
    psi[0 * 2] = 1 / Math.SQRT2;
    psi[3 * 2] = 1 / Math.SQRT2;
    const e = siteExpectations(psi, N);
    for (let i = 0; i < N; i++) {
      expect(Math.abs(e.sx[i]!)).toBeLessThan(1e-12);
      expect(Math.abs(e.sy[i]!)).toBeLessThan(1e-12);
      expect(Math.abs(e.sz[i]!)).toBeLessThan(1e-12);
    }
  });
});
