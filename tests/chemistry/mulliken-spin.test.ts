// Mulliken atomic spin populations (Tier 3 — open-shell extension).
//
// Pass bars:
//   1. Sum of spin populations equals (n_α − n_β)/2 for any UHF
//      converged determinant — the molecular ⟨Ŝ_z⟩ exactly.
//   2. Closed-shell UHF (n_α = n_β) gives all zeros.
//   3. For Li atom doublet, the unpaired α electron sits on the Li
//      atom → spin population near +0.5.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import { mullikenSpinPopulations } from "../../src/chemistry/properties.js";

describe("Mulliken atomic spin populations", () => {
  test("Closed-shell UHF on H₂O: all spin populations are zero", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei, shellAtomIdx } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 5, 5, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      symmetryBreaking: 0,
    });
    expect(uhf.converged).toBe(true);
    const sp = mullikenSpinPopulations(integrals, uhf.D_alpha, uhf.D_beta, shellAtomIdx);
    expect(sp.length).toBe(3);
    for (let A = 0; A < 3; A++) {
      expect(Math.abs(sp[A]!)).toBeLessThan(1e-8);
    }
  });

  test("Li atom doublet: spin population sums to +½ (one unpaired α)", () => {
    const { shells, nuclei, shellAtomIdx } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 1, {
      maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(uhf.converged).toBe(true);
    const sp = mullikenSpinPopulations(integrals, uhf.D_alpha, uhf.D_beta, shellAtomIdx);
    expect(sp.length).toBe(1);
    // Total ⟨S_z⟩ = (nα − nβ)/2 = 0.5; on a single atom that's all on Li.
    expect(Math.abs(sp[0]! - 0.5)).toBeLessThan(1e-8);
  });

  test("Sum of spin populations equals (n_α − n_β)/2 exactly", () => {
    // BH radical (BH⁺ would also work — but B isn't in our basis at
    // cc-pVDZ; use a simpler hydrogen system). H atom = 1 electron.
    // n_α = 1, n_β = 0, ⟨S_z⟩ = 0.5.
    const { shells, nuclei, shellAtomIdx } = moleculeToShellsNuclei([
      { symbol: "H", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 1, 0, {
      maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(uhf.converged).toBe(true);
    const sp = mullikenSpinPopulations(integrals, uhf.D_alpha, uhf.D_beta, shellAtomIdx);
    let total = 0;
    for (let A = 0; A < sp.length; A++) total += sp[A]!;
    expect(Math.abs(total - 0.5)).toBeLessThan(1e-8);
  });
});
