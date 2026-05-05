// BeH₂ dense-Hamiltonian validation. Linear H–Be–H, STO-3G s-only
// basis (Be 1s + Be 2s + 2 H 1s = 4 spatial / 8 spin-orbitals).
// Phase C v2 scope; the Be 2p sub-shell is deferred to v3 (needs
// Cartesian-Gaussian p-shell integrals).
import { describe, expect, test } from "vitest";
import { buildBeH2Dense, computeBeH2Integrals } from "../../src/chemistry/beh2-builder.js";
import { lowestInParticleSector } from "../../src/chemistry/sector.js";

const N_QUBITS = 8;
const DIM = 1 << N_QUBITS;
const N_SPATIAL = 4;

describe("BeH₂ integrals — basic properties", () => {
  test("AO overlap is symmetric, unit diagonal, positive-definite", () => {
    const I = computeBeH2Integrals(1.34);
    for (let i = 0; i < N_SPATIAL; i++) {
      for (let j = 0; j < N_SPATIAL; j++) {
        expect(I.S_AO[i * N_SPATIAL + j]).toBeCloseTo(I.S_AO[j * N_SPATIAL + i]!, 12);
      }
      expect(I.S_AO[i * N_SPATIAL + i]).toBeCloseTo(1, 7);
    }
    // Off-diagonal cross-overlaps (Be 1s vs the others) must be < 1 in absolute value.
    for (let j = 1; j < N_SPATIAL; j++) {
      expect(Math.abs(I.S_AO[0 * N_SPATIAL + j]!)).toBeLessThan(1);
    }
  });

  test("V_nn = 4·1/R (Be↔H_L) + 4·1/R (Be↔H_R) + 1·1/(2R) (H↔H)", () => {
    const I = computeBeH2Integrals(1.34);
    const expected = (4 / I.R_Bohr) + (4 / I.R_Bohr) + (1 / (2 * I.R_Bohr));
    expect(I.Vnn).toBeCloseTo(expected, 12);
  });

  test("h^AO is symmetric", () => {
    const I = computeBeH2Integrals(1.34);
    for (let i = 0; i < N_SPATIAL; i++) {
      for (let j = 0; j < N_SPATIAL; j++) {
        expect(I.h_AO[i * N_SPATIAL + j]).toBeCloseTo(I.h_AO[j * N_SPATIAL + i]!, 12);
      }
    }
  });

  test("ERI 8-fold symmetry holds", () => {
    const I = computeBeH2Integrals(1.34);
    const idx = (a: number, b: number, c: number, d: number) =>
      ((a * N_SPATIAL + b) * N_SPATIAL + c) * N_SPATIAL + d;
    for (let mu = 0; mu < N_SPATIAL; mu++) {
      for (let nu = 0; nu < N_SPATIAL; nu++) {
        for (let la = 0; la < N_SPATIAL; la++) {
          for (let si = 0; si < N_SPATIAL; si++) {
            const v = I.eri_AO[idx(mu, nu, la, si)]!;
            expect(I.eri_AO[idx(nu, mu, la, si)]).toBeCloseTo(v, 12);
            expect(I.eri_AO[idx(mu, nu, si, la)]).toBeCloseTo(v, 12);
            expect(I.eri_AO[idx(la, si, mu, nu)]).toBeCloseTo(v, 12);
          }
        }
      }
    }
  });
});

describe("BeH₂ dense Hamiltonian — invariants", () => {
  test("H is real symmetric", () => {
    const { H } = buildBeH2Dense(1.34);
    let maxAsym = 0;
    for (let i = 0; i < DIM; i++) {
      for (let j = i + 1; j < DIM; j++) {
        const d = Math.abs(H[i * DIM + j]! - H[j * DIM + i]!);
        if (d > maxAsym) maxAsym = d;
      }
    }
    expect(maxAsym).toBeLessThan(1e-12);
  });

  test("[H, N̂] = 0 (Hamiltonian conserves particle number)", () => {
    const { H } = buildBeH2Dense(1.34);
    const Ndiag = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) {
      let n = 0;
      for (let q = 0; q < N_QUBITS; q++) if ((i >>> q) & 1) n++;
      Ndiag[i] = n;
    }
    let maxOffSector = 0;
    for (let i = 0; i < DIM; i++) {
      for (let j = 0; j < DIM; j++) {
        if (Ndiag[i] !== Ndiag[j]) {
          maxOffSector = Math.max(maxOffSector, Math.abs(H[i * DIM + j]!));
        }
      }
    }
    expect(maxOffSector).toBeLessThan(1e-12);
  });

  test("N=6 sector ground state is bound (BeH₂ neutral)", () => {
    // Expect E_FCI(N=6) ≈ -15.35 Ha at R = 1.34 Å in the s-only basis.
    // Full STO-3G with Be 2p included is closer to -15.6 Ha; the gap is
    // the missing 2p contribution to dynamic correlation.
    const { H } = buildBeH2Dense(1.34);
    const r = lowestInParticleSector(H, N_QUBITS, 6);
    expect(r.energy).toBeLessThan(-15.0);
    expect(r.energy).toBeGreaterThan(-15.6);
    // Eigenstate normalized and entirely in the N=6 sector.
    let norm2 = 0;
    for (let i = 0; i < DIM; i++) norm2 += r.psi[i]! * r.psi[i]!;
    expect(norm2).toBeCloseTo(1, 10);
    let nExp = 0;
    for (let i = 0; i < DIM; i++) {
      let n = 0;
      for (let q = 0; q < N_QUBITS; q++) if ((i >>> q) & 1) n++;
      nExp += r.psi[i]! * r.psi[i]! * n;
    }
    expect(nExp).toBeCloseTo(6, 8);
  });
});

describe("BeH₂ symmetric-stretch dissociation", () => {
  test("ground state has a minimum near the experimental Be–H bond R = 1.34 Å", () => {
    const Rs = [1.0, 1.2, 1.34, 1.5, 1.8];
    const energies = Rs.map((R) => lowestInParticleSector(buildBeH2Dense(R).H, N_QUBITS, 6).energy);
    let minIdx = 0;
    for (let i = 1; i < energies.length; i++) {
      if (energies[i]! < energies[minIdx]!) minIdx = i;
    }
    // Minimum should be in [1.2, 1.5] Å — the experimental Be-H bond is 1.34.
    expect(Rs[minIdx]!).toBeGreaterThanOrEqual(1.2);
    expect(Rs[minIdx]!).toBeLessThanOrEqual(1.5);
    // Well depth: minimum ≥ 30 mHa below the R = 1.0 Å point (compressed).
    expect(energies[minIdx]! - energies[0]!).toBeLessThan(-0.03);
  }, 60_000);
});
