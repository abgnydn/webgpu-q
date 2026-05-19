// DFT SCF level-shift — open-shell-stabilization technique for KS SCF.
//
// Pass bars:
//   1. Energy invariance: At convergence, total E with levelShift > 0
//      equals E without level-shift (to ≤ 1e-7). The shift only lifts
//      virtual eigenvalues during SCF; at fix-point eigenvectors are
//      unchanged, so the energy functional value is identical.
//   2. Orbital-energy strip: orbitalEnergies returned with
//      levelShift > 0 match the unshifted run to ≤ 1e-7 per orbital
//      (the strip step subtracts the shift from virtual eps).
//   3. Smoke-test convergence: level-shift doesn't break convergence
//      on a well-behaved closed-shell case.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type AtomSymbol } from "../../src/chemistry/atoms.js";
import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";

describe("DFT SCF — virtual-orbital level shift", () => {
  test("H₂O / B3LYP5 / STO-3G: energy + eps invariant under levelShift", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const nucleiSymbols: AtomSymbol[] = ["O", "H", "H"];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);

    const ksBase = runRKSDFT(integrals, nElectrons, nucleiSymbols, {
      functional: "b3lyp5",
      useDIIS: true, maxIter: 200, energyTol: 1e-8, residualTol: 1e-6,
    });
    expect(ksBase.converged).toBe(true);

    const ksShift = runRKSDFT(integrals, nElectrons, nucleiSymbols, {
      functional: "b3lyp5",
      useDIIS: true, maxIter: 200, energyTol: 1e-8, residualTol: 1e-6,
      levelShift: 0.5,                              // 0.5 Ha virtual lift
    });
    expect(ksShift.converged).toBe(true);

    // Energy unchanged: the level shift doesn't alter the converged
    // density or the energy functional, only the SCF iteration path.
    expect(Math.abs(ksShift.energy - ksBase.energy)).toBeLessThan(1e-7);

    // Orbital energies: strip removes the shift from virtuals on return,
    // so all orbital energies should match. (Note: eigenvectors of a
    // shifted operator are identical to the unshifted; the strip is
    // a bookkeeping fix to the returned virtual-eps values.)
    for (let i = 0; i < ksBase.orbitalEnergies.length; i++) {
      expect(Math.abs(ksShift.orbitalEnergies[i]! - ksBase.orbitalEnergies[i]!)).toBeLessThan(1e-7);
    }
  }, 60_000);

  test("BeH₂ / LDA / STO-3G: level-shift converges to the same energy", () => {
    // Linear BeH₂ as a benign closed-shell second test case across a
    // different functional + element.
    const atoms: Atom[] = [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0,  1.33] },
      { symbol: "H",  pos: [0, 0, -1.33] },
    ];
    const nucleiSymbols: AtomSymbol[] = ["Be", "H", "H"];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);

    const ksBase = runRKSDFT(integrals, nElectrons, nucleiSymbols, {
      functional: "lda-svwn",
      useDIIS: true, maxIter: 200, energyTol: 1e-8, residualTol: 1e-6,
    });
    const ksShift = runRKSDFT(integrals, nElectrons, nucleiSymbols, {
      functional: "lda-svwn",
      useDIIS: true, maxIter: 200, energyTol: 1e-8, residualTol: 1e-6,
      levelShift: 1.0,
    });
    expect(ksBase.converged).toBe(true);
    expect(ksShift.converged).toBe(true);
    expect(Math.abs(ksShift.energy - ksBase.energy)).toBeLessThan(1e-7);
  }, 60_000);
});
