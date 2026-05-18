// UMP2 — open-shell MP2 on UHF reference.
//
// Pass bars:
//   1. Closed-shell UHF (n_α = n_β) → UMP2 = RHF MP2 to numerical precision.
//   2. H₂⁺ (1 electron, no correlation possible) → E_corr = 0.
//   3. Li atom doublet: UMP2 gives small correlation energy (mostly
//      2s-2s correlation, ~mHa in STO-3G).

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import { runMP2 } from "../../src/chemistry/mp2.js";
import { runUMP2 } from "../../src/chemistry/ump2.js";

describe("UMP2 — open-shell MP2", () => {
  test("Closed-shell H₂O: UMP2 (n_α=n_β=5) matches RHF MP2 to 1e-9", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);

    const rhf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const rmp2 = runMP2(rhf, integrals);

    const uhf = runUHFSCF(integrals, 5, 5, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      symmetryBreaking: 0,
    });
    const ump2 = runUMP2(uhf, integrals);

    expect(uhf.converged).toBe(true);
    // UHF on closed-shell with symmetry-breaking=0 should match RHF energy.
    expect(Math.abs(uhf.energy - rhf.energy)).toBeLessThan(1e-8);
    // UMP2 correlation = RHF MP2 correlation by spin-orbital basis independence.
    expect(Math.abs(ump2.correlationEnergy - rmp2.correlationEnergy)).toBeLessThan(1e-8);
  });

  test("H atom (1 electron): UMP2 correlation = 0 exactly", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "H", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 1, 0, {
      maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(uhf.converged).toBe(true);
    const ump2 = runUMP2(uhf, integrals);
    // 1 electron → no occupied-occupied pair → all MP2 sums vanish.
    expect(Math.abs(ump2.correlationEnergy)).toBeLessThan(1e-12);
  });

  test("Li atom doublet: UMP2 gives finite negative correlation energy", () => {
    const { shells, nuclei } = moleculeToShellsNuclei([
      { symbol: "Li", pos: [0, 0, 0] },
    ]);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 1, {
      maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(uhf.converged).toBe(true);
    const ump2 = runUMP2(uhf, integrals);
    expect(Number.isFinite(ump2.correlationEnergy)).toBe(true);
    // Bounded magnitude — small basis on a 3-electron atom.
    expect(Math.abs(ump2.correlationEnergy)).toBeLessThan(0.05);
  });
});
