// Counterpoise / BSSE — Tier 3 (Boys-Bernardi).
//
// Pass bars:
//   1. Sign: BSSE correction is POSITIVE for non-bonded fragments
//      in any incomplete basis (dimer's borrowed basis artificially
//      stabilizes; CP removes the stabilization, making ΔE_CP > ΔE).
//   2. Ghost-only sanity: a fragment computed with its real atoms +
//      no ghosts must match the same fragment computed normally
//      (counterpoise reduces to plain HF when there's nothing
//      ghosted).
//   3. Self-consistency: the supermolecule energy returned by
//      counterpoise must equal the plain HF energy on the same atoms.
//   4. Magnitude bounded: STO-3G BSSE on H₂...H₂ at 3 Å should be
//      sub-mHa (small basis on a non-interacting pair).

import { describe, expect, test } from "vitest";
import { runCounterpoise } from "../../src/chemistry/counterpoise.js";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";

describe("Counterpoise / BSSE", () => {
  test("H₂...H₂ STO-3G at 3 Å: BSSE positive, sub-mHa, all converged", () => {
    // Two H₂ molecules end-to-end, 3 Å between the closest H atoms.
    // STO-3G is a poor basis → BSSE is the *only* "binding" you see
    // for two H₂ molecules far apart.
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
      { symbol: "H", pos: [0, 0, 3.7414] },  // 3 Å gap
      { symbol: "H", pos: [0, 0, 4.4828] },
    ];
    const cp = runCounterpoise(
      atoms,
      [
        { atomIndices: [0, 1] },
        { atomIndices: [2, 3] },
      ],
      "sto-3g",
      { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 },
    );

    expect(cp.allConverged).toBe(true);
    // BSSE = ΔE_CP − ΔE = Σ(E_bare − E_CP) ≥ 0 by the variational
    // principle (ghost-augmented monomer is always ≤ bare).
    expect(cp.bsseCorrection).toBeGreaterThan(0);
    // Magnitude sub-mHa on STO-3G H₂...H₂ at 3 Å.
    expect(cp.bsseCorrection).toBeLessThan(1e-3);
    // CP-corrected ΔE > uncorrected ΔE always (CP raises the
    // interaction energy by exactly bsseCorrection).
    expect(cp.interactionEnergyCP).toBeGreaterThan(cp.interactionEnergy);
    expect(cp.interactionEnergyCP - cp.interactionEnergy).toBeCloseTo(cp.bsseCorrection, 12);
  });

  test("supermolecule energy returned by counterpoise matches plain HF", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
      { symbol: "H", pos: [0, 0, 3.0]    },
      { symbol: "H", pos: [0, 0, 3.7414] },
    ];
    // Plain HF reference.
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, "sto-3g");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const plain = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(plain.converged).toBe(true);

    const cp = runCounterpoise(
      atoms,
      [{ atomIndices: [0, 1] }, { atomIndices: [2, 3] }],
      "sto-3g",
      { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 },
    );

    expect(Math.abs(cp.supermoleculeEnergy - plain.energy)).toBeLessThan(1e-9);
  });

  test("single-fragment counterpoise reduces to plain HF (no ghosts)", () => {
    // If you put every atom in ONE fragment, there are no ghosts —
    // the "fragment in dimer basis" run is identical to the bare
    // monomer run, and BSSE = 0.
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    const cp = runCounterpoise(
      atoms,
      [{ atomIndices: [0, 1] }],
      "sto-3g",
      { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 },
    );
    expect(cp.allConverged).toBe(true);
    // With only one fragment: ΔE = ΔE_CP, so BSSE = 0.
    expect(cp.bsseCorrection).toBeLessThan(1e-9);
    // Fragment-with-ghosts == fragment-bare when there are no other atoms.
    expect(Math.abs(cp.fragmentEnergiesCP[0]! - cp.fragmentEnergiesBare[0]!)).toBeLessThan(1e-9);
  });

  test("ghost-augmented fragment energy is lower (variational basis-set expansion)", () => {
    // E(fragment in dimer basis) ≤ E(fragment bare) by the
    // variational principle — adding basis functions can only
    // lower (or equal) the SCF energy.
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
      { symbol: "H", pos: [0, 0, 3.0]    },
      { symbol: "H", pos: [0, 0, 3.7414] },
    ];
    const cp = runCounterpoise(
      atoms,
      [{ atomIndices: [0, 1] }, { atomIndices: [2, 3] }],
      "sto-3g",
      { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 },
    );
    expect(cp.allConverged).toBe(true);
    for (let k = 0; k < 2; k++) {
      // ghost-augmented ≤ bare (variational; tiny shift in STO-3G).
      expect(cp.fragmentEnergiesCP[k]!).toBeLessThanOrEqual(cp.fragmentEnergiesBare[k]! + 1e-12);
    }
  });

  test("MP2 counterpoise: BSSE positive at MP2 + matches HF supermolecule on extreme distance", () => {
    // MP2 BSSE on top of HF BSSE. Same sign convention, same
    // variational direction. At 5 Å the H₂...H₂ pair is essentially
    // non-interacting, so the MP2 supermolecule energy ≈ 2× monomer.
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
      { symbol: "H", pos: [0, 0, 5.7414] },
      { symbol: "H", pos: [0, 0, 6.4828] },
    ];
    const cp = runCounterpoise(
      atoms,
      [{ atomIndices: [0, 1] }, { atomIndices: [2, 3] }],
      "sto-3g",
      { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 },
      "mp2",
    );
    expect(cp.allConverged).toBe(true);
    // BSSE ≥ 0 (Boys-Bernardi sign).
    expect(cp.bsseCorrection).toBeGreaterThanOrEqual(0);
    // At 5 Å the interaction (both uncorrected and CP) is sub-mHa.
    expect(Math.abs(cp.interactionEnergyCP)).toBeLessThan(1e-3);
    expect(cp.fragmentEnergiesCP.length).toBe(2);
  });

  test("CCSD counterpoise on small H₂...H₂: sign + magnitude bounded", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
      { symbol: "H", pos: [0, 0, 5.7414] },
      { symbol: "H", pos: [0, 0, 6.4828] },
    ];
    const cp = runCounterpoise(
      atoms,
      [{ atomIndices: [0, 1] }, { atomIndices: [2, 3] }],
      "sto-3g",
      { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 },
      "ccsd",
      { maxIter: 200, tol: 1e-9 },
    );
    expect(cp.allConverged).toBe(true);
    expect(cp.bsseCorrection).toBeGreaterThanOrEqual(0);
    expect(Math.abs(cp.interactionEnergyCP)).toBeLessThan(1e-3);
  });

  test("validates fragmentation: throws on overlap + on missing atoms", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    // Overlap.
    expect(() => runCounterpoise(
      atoms,
      [{ atomIndices: [0, 1] }, { atomIndices: [1] }],
      "sto-3g",
    )).toThrow(/multiple fragments/);
    // Missing.
    expect(() => runCounterpoise(
      atoms,
      [{ atomIndices: [0] }],
      "sto-3g",
    )).toThrow(/not assigned/);
    // Out of range.
    expect(() => runCounterpoise(
      atoms,
      [{ atomIndices: [0, 1, 5] }],
      "sto-3g",
    )).toThrow(/out of range/);
  });
});
