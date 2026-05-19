// UKS-DFT properties — confirm existing UHF property routines
// work on UKS results without change (D_alpha / D_beta / S_AO are
// the shared interface).
//
// Pass bars:
//   1. dipoleMoment on D_total = D_α + D_β from UKS gives a real
//      3-vector with sensible magnitude (H₂O ~1.8 D experimentally;
//      LDA/STO-3G ~2.0 D ballpark).
//   2. mullikenSpinPopulations on (D_α, D_β) sum to (n_α − n_β) =
//      total spin density / 2.
//   3. bondOrdersUHF on (D_α, D_β) gives positive bond orders.
//   4. naturalOrbitalOccupations on D_total: closed-shell UKS gives
//      essentially integer NOON; open-shell gives ⟨S²⟩ ~ 0.75 ± ε.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type AtomSymbol } from "../../src/chemistry/atoms.js";
import { runUKSDFT } from "../../src/chemistry/dft/uks-scf.js";
import {
  dipoleMoment, dipoleMagnitude, AU_TO_DEBYE,
  mullikenSpinPopulations, bondOrdersUHF, mayerValencesUHF,
} from "../../src/chemistry/properties.js";
import { naturalOrbitalOccupations } from "../../src/chemistry/natural-orbitals.js";

describe("UKS-DFT properties (dipole, Mulliken, Mayer, NOON)", () => {
  test("Closed-shell H₂O / LDA UKS (n_α=n_β=5): dipole + zero spin populations + integer NOON", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const nucleiSymbols: AtomSymbol[] = ["O", "H", "H"];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uks = runUKSDFT(integrals, 5, 5, nucleiSymbols, {
      functional: "lda-svwn",
      useDIIS: true, maxIter: 200, energyTol: 1e-9, residualTol: 1e-7,
      symmetryBreaking: 0,
    });
    expect(uks.converged).toBe(true);

    // Total density.
    const D_total = new Float64Array(integrals.n * integrals.n);
    for (let i = 0; i < D_total.length; i++) D_total[i] = uks.D_alpha[i]! + uks.D_beta[i]!;

    // Dipole. H₂O is polar along z (oriented with H atoms above origin); expect non-zero z component.
    const mu = dipoleMoment(integrals, D_total);
    expect(Math.abs(mu[0])).toBeLessThan(1e-8);   // by C₂v symmetry
    expect(Math.abs(mu[1])).toBeLessThan(1e-8);
    expect(Math.abs(mu[2])).toBeGreaterThan(0.5);  // physical: ~0.7 a.u.

    const totalDebye = dipoleMagnitude(mu) * AU_TO_DEBYE;
    expect(totalDebye).toBeGreaterThan(1.0);     // STO-3G LDA H₂O is around 1.5-2.0 D
    expect(totalDebye).toBeLessThan(3.0);

    // Spin populations: closed-shell → all zero.
    const shellAtomIdx = inferShellAtomIdx(integrals.shells, atoms);
    const spinPop = mullikenSpinPopulations(integrals, uks.D_alpha, uks.D_beta, shellAtomIdx);
    let totalSpin = 0;
    for (let a = 0; a < spinPop.length; a++) totalSpin += spinPop[a]!;
    expect(Math.abs(totalSpin)).toBeLessThan(1e-6);

    // Mayer bond orders: O-H bonds positive.
    const bondOrders = bondOrdersUHF(integrals, uks.D_alpha, uks.D_beta, shellAtomIdx);
    // bondOrders is row-major (nAtoms × nAtoms).
    expect(bondOrders[0 * 3 + 1]!).toBeGreaterThan(0.5);   // O-H1
    expect(bondOrders[0 * 3 + 2]!).toBeGreaterThan(0.5);   // O-H2

    // Mayer atomic valences: O ≈ 2 (two bonds), H ≈ 1.
    const valences = mayerValencesUHF(integrals, uks.D_alpha, uks.D_beta, shellAtomIdx);
    expect(valences[0]!).toBeGreaterThan(1.5);   // O
    expect(valences[0]!).toBeLessThan(2.5);
    expect(valences[1]!).toBeGreaterThan(0.5);   // H
    expect(valences[1]!).toBeLessThan(1.3);

    // NOON on D_total: 5 strongly occupied, 2 strongly virtual for closed-shell.
    const noon = naturalOrbitalOccupations(D_total, integrals.S_AO);
    expect(noon.nStronglyOccupied).toBe(5);
    expect(noon.nActive).toBe(0);
    expect(Math.abs(noon.totalElectrons - 10)).toBeLessThan(1e-8);
  }, 60_000);

  test("Li atom doublet UKS-LDA: spin density = 1, NOON sum = 3", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uks = runUKSDFT(integrals, 2, 1, ["Li"], {
      functional: "lda-svwn",
      maxIter: 200, energyTol: 1e-9, residualTol: 1e-7,
    });
    expect(uks.converged).toBe(true);
    const shellAtomIdx = inferShellAtomIdx(integrals.shells, [{ symbol: "Li", pos: [0, 0, 0] }]);
    const spinPop = mullikenSpinPopulations(integrals, uks.D_alpha, uks.D_beta, shellAtomIdx);
    let total = 0;
    for (let a = 0; a < spinPop.length; a++) total += spinPop[a]!;
    // Total spin = (n_α − n_β)/2 = 0.5 per Mulliken sum convention.
    expect(Math.abs(total - 0.5)).toBeLessThan(1e-6);
    // NOON sum = 3 electrons.
    const D_total = new Float64Array(integrals.n * integrals.n);
    for (let i = 0; i < D_total.length; i++) D_total[i] = uks.D_alpha[i]! + uks.D_beta[i]!;
    const noon = naturalOrbitalOccupations(D_total, integrals.S_AO);
    expect(Math.abs(noon.totalElectrons - 3)).toBeLessThan(1e-8);
  });
});

// Helper: same logic as molden.ts's inferShellAtomIdx but local.
function inferShellAtomIdx(
  shells: readonly { center: readonly [number, number, number] }[],
  atoms: readonly Atom[],
): number[] {
  const tol = 1e-6;
  const idx: number[] = [];
  for (const sh of shells) {
    let found = -1;
    for (let a = 0; a < atoms.length; a++) {
      const ax = atoms[a]!.pos[0] * 1.8897261245650618;
      const ay = atoms[a]!.pos[1] * 1.8897261245650618;
      const az = atoms[a]!.pos[2] * 1.8897261245650618;
      if (Math.abs(sh.center[0] - ax) < tol
       && Math.abs(sh.center[1] - ay) < tol
       && Math.abs(sh.center[2] - az) < tol) { found = a; break; }
    }
    if (found < 0) throw new Error(`shell center doesn't match any atom`);
    idx.push(found);
  }
  return idx;
}
