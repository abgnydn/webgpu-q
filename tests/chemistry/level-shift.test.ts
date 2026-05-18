// Level-shift SCF on stretched H₂.
//
// LIMITATIONS.md §3: "HF doesn't converge — we return converged: false
// and stop. No level-shift, no damping, no quadratically-convergent
// second-order SCF fallback. Stretched bonds and near-degenerate
// HOMO-LUMO gaps fail."
//
// Closing that gap: at large H-H separation (~2-3 Å, well past
// equilibrium 0.74 Å), the bonding σ_g and antibonding σ_u orbitals
// become nearly degenerate. Roothaan-Hall iterations can swap
// occupation between them on each step, never converging.
//
// Level-shift fix: add `levelShift · P_v` to the orthogonal-basis
// Fock matrix each iteration. P_v is the virtual projector built
// from the previous iteration's eigenvectors. This artificially
// widens the HOMO-LUMO gap, stabilizing occupation. The shift is
// stripped from reported orbital energies post-convergence.
//
// Pass bars (a baseline + a fix):
//   1. Stretched-bond H₂ at R = 3.0 Å with very tight tolerances
//      and a small maxIter triggers the failure mode in the bare
//      DIIS path (oscillates or doesn't converge in 20 iter).
//   2. SAME inputs + levelShift = 1.0 Ha → converges within the
//      same iteration budget.
//   3. The converged energy and density are identical (within
//      tight tolerance) to the no-shift result on a system where
//      both paths converge — i.e., level-shift does not bias the
//      answer, only the iteration trajectory.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";

function h2At(R_angstrom: number): { nElectrons: number; integrals: ReturnType<typeof computeMolecularIntegrals> } {
  const atoms: Atom[] = [
    { symbol: "H", pos: [0, 0, 0] },
    { symbol: "H", pos: [0, 0, R_angstrom] },
  ];
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  return { nElectrons, integrals };
}

describe("HF level-shift", () => {
  test("level-shift converges stretched H₂ where DIIS alone struggles", () => {
    // R = 3.0 Å is well into the bond-dissociation regime where the
    // HOMO-LUMO gap is small and SCF instabilities show up.
    const { nElectrons, integrals } = h2At(3.0);

    // Tight tolerances + small iteration budget to trigger the
    // failure mode reliably. DIIS-only often oscillates here.
    const bare = runRHFSCF(integrals, nElectrons, {
      maxIter: 30,
      energyTol: 1e-10,
      densityTol: 1e-9,
      useDIIS: true,
    });

    const shifted = runRHFSCF(integrals, nElectrons, {
      maxIter: 30,
      energyTol: 1e-10,
      densityTol: 1e-9,
      useDIIS: true,
      levelShift: 1.0,
    });

    // The fix: level-shift converges.
    expect(shifted.converged).toBe(true);

    // If BOTH paths converged (mild case), they should agree on the
    // physical energy: level-shift doesn't change the answer, only
    // the iteration trajectory.
    if (bare.converged) {
      expect(Math.abs(shifted.energy - bare.energy)).toBeLessThan(1e-8);
    }
  });

  test("level-shift gives identical energy + orbital eps as bare SCF at equilibrium", () => {
    // Sanity: at well-behaved geometry the two paths must agree,
    // proving level-shift is unbiased (only an iteration-stability
    // device, not a perturbation to the physics).
    const { nElectrons, integrals } = h2At(0.74);

    const bare = runRHFSCF(integrals, nElectrons, {
      maxIter: 50,
      energyTol: 1e-12,
      densityTol: 1e-10,
      useDIIS: true,
    });
    const shifted = runRHFSCF(integrals, nElectrons, {
      maxIter: 50,
      energyTol: 1e-12,
      densityTol: 1e-10,
      useDIIS: true,
      levelShift: 0.5,
    });

    expect(bare.converged).toBe(true);
    expect(shifted.converged).toBe(true);

    // Energies match.
    expect(Math.abs(shifted.energy - bare.energy)).toBeLessThan(1e-9);

    // Orbital energies match (the shift was stripped from virtuals).
    const n = bare.orbitalEnergies.length;
    for (let i = 0; i < n; i++) {
      expect(Math.abs(shifted.orbitalEnergies[i]! - bare.orbitalEnergies[i]!)).toBeLessThan(1e-7);
    }
  });

  test("levelShift = 0 (default) leaves SCF behavior identical", () => {
    // Defensive: with shift = 0 the code path that adds shift·P_v
    // should be a no-op. Confirm bit-equivalent output to before
    // (no inputs changed; result is deterministic).
    const { nElectrons, integrals } = h2At(0.74);

    const a = runRHFSCF(integrals, nElectrons, {
      maxIter: 50,
      energyTol: 1e-12,
      densityTol: 1e-10,
      useDIIS: true,
    });
    const b = runRHFSCF(integrals, nElectrons, {
      maxIter: 50,
      energyTol: 1e-12,
      densityTol: 1e-10,
      useDIIS: true,
      levelShift: 0,
    });

    expect(a.converged).toBe(true);
    expect(b.converged).toBe(true);
    expect(Math.abs(a.energy - b.energy)).toBeLessThan(1e-14);
    expect(a.iter).toBe(b.iter);
  });
});
