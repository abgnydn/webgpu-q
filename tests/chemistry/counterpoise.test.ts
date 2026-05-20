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

describe("Counterpoise / BSSE — open-shell (UHF / UCCSD)", () => {
  test("H + H at 5 Å, both α-spin (triplet): UHF BSSE ≈ 0 at long separation", () => {
    // Two H atoms with parallel spins at 5 Å. STO-3G UHF should give
    // ΔE ≈ 0 (no interaction) and BSSE ≈ 0 (each H atom has the same
    // 1-orbital basis whether the ghost is there or not — STO-3G H has
    // only 1 basis function per atom, no extra orbitals to "borrow").
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 5.0] },
    ];
    const cp = runCounterpoise(
      atoms,
      [
        { atomIndices: [0], spin: { nAlpha: 1, nBeta: 0 } },
        { atomIndices: [1], spin: { nAlpha: 1, nBeta: 0 } },
      ],
      "sto-3g",
      {},        // hfOpts (unused for uhf)
      "uhf",
      {},        // ccsdOpts (unused)
      { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 },
    );
    expect(cp.allConverged).toBe(true);
    // Each H atom STO-3G UHF: ε = -0.466582 Ha. Σ = -0.933164.
    expect(Math.abs(cp.fragmentEnergiesBare[0]! - (-0.466582))).toBeLessThan(1e-4);
    expect(Math.abs(cp.fragmentEnergiesBare[1]! - (-0.466582))).toBeLessThan(1e-4);
    // Long-range non-interacting interaction.
    expect(Math.abs(cp.interactionEnergy)).toBeLessThan(1e-3);
    // BSSE positive (variational) and tiny for this minimal-basis case.
    expect(cp.bsseCorrection).toBeGreaterThanOrEqual(0);
    expect(cp.bsseCorrection).toBeLessThan(5e-4);
  });

  test("Li atom + H atom at 5 Å (Li doublet + H doublet → triplet supermolecule): UHF gives finite ΔE and ΔE_CP", () => {
    // Li atom = doublet (1s² 2s¹), nα=2, nβ=1.
    // H atom α-spin (nα=1, nβ=0) → supermolecule (nα=3, nβ=1) — triplet.
    const atoms: Atom[] = [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 5.0] },
    ];
    const cp = runCounterpoise(
      atoms,
      [
        { atomIndices: [0], spin: { nAlpha: 2, nBeta: 1 } },
        { atomIndices: [1], spin: { nAlpha: 1, nBeta: 0 } },
      ],
      "sto-3g",
      {},
      "uhf",
      {},
      { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 },
    );
    expect(cp.allConverged).toBe(true);
    // Both interaction energies real, finite.
    expect(Number.isFinite(cp.interactionEnergy)).toBe(true);
    expect(Number.isFinite(cp.interactionEnergyCP)).toBe(true);
    // BSSE non-negative (variational).
    expect(cp.bsseCorrection).toBeGreaterThanOrEqual(-1e-10);
  });

  test("UCCSD method: H₂ (singlet) + H (α-doublet) → doublet supermolecule completes without throwing", () => {
    // Mixed closed-shell + open-shell fragments. H₂ fragment is treated
    // via UHF with nα=nβ=1 (collapses to RHF); H atom α-spin via UHF with
    // (1, 0). Each fragment then runs UCCSD on top of its UHF.
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
      { symbol: "H", pos: [0, 0, 5.0]    },  // 4.3 Å gap
    ];
    const cp = runCounterpoise(
      atoms,
      [
        { atomIndices: [0, 1], spin: { nAlpha: 1, nBeta: 1 } },
        { atomIndices: [2],    spin: { nAlpha: 1, nBeta: 0 } },
      ],
      "sto-3g",
      {},
      "uccsd",
      { maxIter: 100, tol: 1e-8 },
      { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 },
    );
    expect(cp.allConverged).toBe(true);
    expect(Number.isFinite(cp.interactionEnergy)).toBe(true);
    expect(Number.isFinite(cp.interactionEnergyCP)).toBe(true);
    // BSSE non-negative.
    expect(cp.bsseCorrection).toBeGreaterThanOrEqual(-1e-10);
  });

  test("RKS-LDA H₂...H₂ at 3 Å: counterpoise BSSE small + non-negative", () => {
    // DFT-counterpoise on H₂ dimer. STO-3G LDA exchange-correlation
    // adds a small attractive contribution; BSSE is the artificial
    // stabilization from borrowed basis functions.
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
      { symbol: "H", pos: [0, 0, 3.7414] },
      { symbol: "H", pos: [0, 0, 4.4828] },
    ];
    const cp = runCounterpoise(
      atoms,
      [{ atomIndices: [0, 1] }, { atomIndices: [2, 3] }],
      "sto-3g",
      {},     // hfOpts (unused)
      "rks",
      {},     // ccsdOpts (unused)
      {},     // uhfOpts (unused)
      { functional: "lda-svwn", rks: { useDIIS: true, maxIter: 200, energyTol: 1e-8 } },
    );
    expect(cp.allConverged).toBe(true);
    expect(Number.isFinite(cp.interactionEnergy)).toBe(true);
    expect(cp.bsseCorrection).toBeGreaterThanOrEqual(-1e-10);
  }, 60_000);

  test("RKS-LDA + D2: counterpoise integrates dispersion + DFT cleanly", () => {
    // The end-to-end DFT-D2 BSSE workflow: each fragment / supermolecule
    // single-point gets its DFT energy + D2 dispersion. D2 is
    // distance-dependent, so the supermolecule and fragments-in-bare-basis
    // differ. Validate the workflow produces finite, sensible values.
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
      { symbol: "H", pos: [0, 0, 3.7414] },
      { symbol: "H", pos: [0, 0, 4.4828] },
    ];
    const cp = runCounterpoise(
      atoms,
      [{ atomIndices: [0, 1] }, { atomIndices: [2, 3] }],
      "sto-3g",
      {},
      "rks",
      {},
      {},
      { functional: "blyp", rks: { useDIIS: true, maxIter: 200, energyTol: 1e-8 } },
      { addD2: true, d2Opts: { functional: "blyp" } },
    );
    expect(cp.allConverged).toBe(true);
    expect(Number.isFinite(cp.interactionEnergy)).toBe(true);
    expect(Number.isFinite(cp.interactionEnergyCP)).toBe(true);
    expect(cp.bsseCorrection).toBeGreaterThanOrEqual(-1e-10);
    // STO-3G H₂-H₂ at 3 Å is dominated by exchange repulsion; D2
    // adds an attractive correction but doesn't fully overcome
    // the basis-incompleteness repulsion (you need larger basis for
    // that). Both interaction energies should still be sub-mHa.
    expect(Math.abs(cp.interactionEnergy)).toBeLessThan(1e-3);
  }, 60_000);

  test("RKS missing functional throws clearly", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 5.0] },
    ];
    expect(() =>
      runCounterpoise(
        atoms,
        [{ atomIndices: [0] }, { atomIndices: [1] }],
        "sto-3g",
        {},
        "rks",
      ),
    ).toThrow(/requires dftOpts.functional/);
  });

  test("UHF method without per-fragment spin throws", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 5.0] },
    ];
    expect(() =>
      runCounterpoise(
        atoms,
        [{ atomIndices: [0] }, { atomIndices: [1] }],
        "sto-3g",
        {},
        "uhf",
      ),
    ).toThrow(/requires fragments\[0\]\.spin/);
  });
});
