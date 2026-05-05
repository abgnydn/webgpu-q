// LiH dense-Hamiltonian validation. Ground-state energy in the
// STO-3G s-only basis (Li 1s, 2s + H 1s) is HIGHER than the full
// STO-3G LiH FCI energy (~-7.882 Ha) because the 2p orbitals are
// missing — they contribute ~10 mHa of correlation. The s-only
// reference range is well-defined though, and the basic Hamiltonian
// invariants (Hermiticity, particle conservation, dissociation)
// are basis-independent and provide the strongest correctness bar.
import { describe, expect, test } from "vitest";
import {
  buildLiHDense, computeLiHIntegrals, lowestInParticleSector,
} from "../../src/chemistry/lih-builder.js";
import { eigsymmetric } from "../../src/manybody/dense-eig.js";

const DIM = 64;

function n4GroundStateEnergy(H: Float64Array): number {
  return lowestInParticleSector(H, 6, 4).energy;
}

/** Project state vector onto its particle-number sectors. */
function particleNumberDistribution(psi: Float64Array, nQubits: number): Map<number, number> {
  const dim = 1 << nQubits;
  const out = new Map<number, number>();
  for (let i = 0; i < dim; i++) {
    let n = 0;
    for (let q = 0; q < nQubits; q++) if ((i >>> q) & 1) n++;
    const w = psi[i]! * psi[i]!;
    out.set(n, (out.get(n) ?? 0) + w);
  }
  return out;
}

describe("LiH integrals — basic properties", () => {
  test("AO overlap is symmetric positive-definite", () => {
    const I = computeLiHIntegrals(1.595);
    // Symmetric.
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(I.S_AO[i * 3 + j]).toBeCloseTo(I.S_AO[j * 3 + i]!, 12);
      }
    }
    // Diagonal entries = 1 (each shell normalized).
    for (let i = 0; i < 3; i++) {
      expect(I.S_AO[i * 3 + i]).toBeCloseTo(1, 7);
    }
    // Eigenvalues all positive (positive-definite).
    const eig = eigsymmetric(I.S_AO, 3);
    for (let k = 0; k < 3; k++) {
      expect(eig.values[k]!).toBeGreaterThan(0);
    }
  });

  test("h^AO is symmetric", () => {
    const I = computeLiHIntegrals(1.595);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(I.h_AO[i * 3 + j]).toBeCloseTo(I.h_AO[j * 3 + i]!, 12);
      }
    }
  });

  test("V_nn = 3/R (Z_Li · Z_H = 3)", () => {
    const I = computeLiHIntegrals(1.595);
    expect(I.Vnn).toBeCloseTo(3 / I.R_Bohr, 12);
  });

  test("ERI 8-fold symmetry: (μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν)", () => {
    const I = computeLiHIntegrals(1.595);
    const idx = (a: number, b: number, c: number, d: number) =>
      ((a * 3 + b) * 3 + c) * 3 + d;
    for (let mu = 0; mu < 3; mu++) {
      for (let nu = 0; nu < 3; nu++) {
        for (let la = 0; la < 3; la++) {
          for (let si = 0; si < 3; si++) {
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

describe("LiH dense Hamiltonian — invariants", () => {
  test("H is real symmetric (Hermiticity)", () => {
    const { H } = buildLiHDense(1.595);
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
    // Direct test of the underlying invariant. Avoids the eigenvector-
    // basis ambiguity that arises when H has degenerate eigenvalues
    // spanning multiple particle-number sectors (e.g. our N=3, N=4, N=5
    // degeneracy at E ≈ -7.716 Ha) — the diagonalizer can pick any
    // linear combination within the degenerate subspace.
    const { H } = buildLiHDense(1.595);
    // N̂ as a diagonal matrix in the computational basis: N̂[i,i] = popcount(i).
    const Ndiag = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) {
      let n = 0;
      for (let q = 0; q < 6; q++) if ((i >>> q) & 1) n++;
      Ndiag[i] = n;
    }
    // (HN − NH)[i,j] = H[i,j] · (N[j] − N[i]).  Since N is diagonal,
    // every off-N-sector entry of H must be zero.
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

  test("N=4 sector ground state is bound (LiH neutral)", () => {
    // The full second-quantized H spans every particle-number sector
    // from N=0 to N=6. We project to N=4 to get the physical neutral
    // LiH. In STO-3G s-only this lands at E ≈ -7.8434 Ha, ≈39 mHa above
    // the published full STO-3G FCI of -7.8823 Ha (the gap is the
    // missing Li 2p contribution to dynamic correlation).
    const { H } = buildLiHDense(1.595);
    const r = lowestInParticleSector(H, 6, 4);
    expect(r.energy).toBeLessThan(-7.84);
    expect(r.energy).toBeGreaterThan(-7.85);
    // The expanded statevector should sum to ‖ψ‖² = 1 and live entirely
    // in the N=4 sector.
    let norm2 = 0;
    for (let i = 0; i < DIM; i++) norm2 += r.psi[i]! * r.psi[i]!;
    expect(norm2).toBeCloseTo(1, 10);
    const dist = particleNumberDistribution(r.psi, 6);
    expect(dist.get(4) ?? 0).toBeCloseTo(1, 10);
  });
});

describe("LiH dissociation behavior (N=4 sector)", () => {
  test("ground state energy has a minimum between 1.4 Å and 1.8 Å", () => {
    // Sample the dissociation curve in the physical N=4 sector.
    // Experimental LiH equilibrium is 1.595 Å — STO-3G s-only should
    // come close (typically slightly stretched in minimal basis).
    const Rs = [1.0, 1.4, 1.6, 1.8, 2.5];
    const energies = Rs.map((R) => n4GroundStateEnergy(buildLiHDense(R).H));
    let minIdx = 0;
    for (let i = 1; i < energies.length; i++) {
      if (energies[i]! < energies[minIdx]!) minIdx = i;
    }
    expect(Rs[minIdx]!).toBeGreaterThanOrEqual(1.4);
    expect(Rs[minIdx]!).toBeLessThanOrEqual(1.8);
    // Well depth: E(R_eq) − E(R = 2.5 Å) should be at least 10 mHa
    // (chemical accuracy ~ 1.6 mHa, well depth of LiH ~ 60 mHa
    // experimentally).
    expect(energies[minIdx]! - energies[energies.length - 1]!).toBeLessThan(-0.01);
  });

  test("R = 50 Å and R = 100 Å give the same energy (saturated dissociation)", () => {
    // At large R the LiH ground state separates into Li atom + H atom
    // (no overlap, no Coulomb attraction). The energy plateaus.
    const E_50 = n4GroundStateEnergy(buildLiHDense(50).H);
    const E_100 = n4GroundStateEnergy(buildLiHDense(100).H);
    expect(Math.abs(E_50 - E_100)).toBeLessThan(0.001);
    // And: dissociated energy is below H atom alone (Li atom binds the
    // remaining 3 electrons).
    const E_H_atom = -0.4666;  // H STO-3G 1s
    expect(E_50).toBeLessThan(E_H_atom);
  }, 30_000);
});
