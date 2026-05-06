// Spherical-harmonic d-shell basis. Tier 1 / Phase E stage 6 work
// (Tier 1 bundle from CLAUDE.md roadmap).
//
// Pass bars:
//   • H₂O cc-pVDZ HF spherical matches PySCF to ≤ 0.1 mHa (PySCF ref
//     uses spherical d). Cartesian basis, by contrast, sits ~0.4 mHa
//     below PySCF because the (xx+yy+zz)/√3 redundant component makes
//     the basis variationally larger — a documented basis-set
//     convention difference, not a code bug.
//   • Spherical n = Cartesian n − (#d-shells). For H₂O cc-pVDZ:
//     25 Cart → 24 Sph (one O d-shell drops 1 redundant function).
//   • The 5 spherical-d AOs at a single center must be mutually
//     orthonormal in the AO overlap matrix (within FP precision).
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";

const half = (104.52 / 2) * Math.PI / 180;
const xH = 0.9572 * Math.sin(half);
const zH = 0.9572 * Math.cos(half);
const h2o: Atom[] = [
  { symbol: "O", pos: [0, 0, 0] },
  { symbol: "H", pos: [ xH, 0, zH] },
  { symbol: "H", pos: [-xH, 0, zH] },
];

describe("Spherical-harmonic d-shell basis (cc-pVDZ)", () => {
  test("H₂O Cartesian: n=25, Spherical: n=24 (drops 1 d redundancy)", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "cc-pvdz");
    const cart = computeMolecularIntegrals(shells, nuclei);
    const sph  = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    expect(cart.n).toBe(25);
    expect(sph.n).toBe(24);
  });

  test("Spherical-d AOs are mutually orthonormal at the d-shell center", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "cc-pvdz");
    const sph = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    // O d-shell occupies indices 9..13 in the spherical basis:
    //   0..2   O 1s, 2s, 2s'         (3 s-shells)
    //   3..8   O 2p_{xyz}, 2p'_{xyz} (6 p-shells)
    //   9..13  O 3d (5 spherical d, was 6 Cartesian)
    //   14..18 H1: 1s, 2s, 2p_{xyz}
    //   19..23 H2: 1s, 2s, 2p_{xyz}
    // The 5×5 d-d overlap block at indices 9..13 must be the identity
    // — they all sit at the same center with the same primitives, so
    // they decompose into orthonormal real spherical harmonics.
    const n = sph.n;
    for (let i = 9; i <= 13; i++) {
      for (let j = 9; j <= 13; j++) {
        const expected = i === j ? 1 : 0;
        const got = sph.S_AO[i * n + j]!;
        expect(Math.abs(got - expected)).toBeLessThan(1e-12);
      }
    }
  });

  test("H₂O HF/cc-pVDZ spherical matches PySCF to ≤ 0.1 mHa", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "cc-pvdz");
    const sph = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    const hf = runRHFSCF(sph, 10, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(hf.converged).toBe(true);
    // PySCF cc-pVDZ HF/H₂O at experimental geometry = -76.026765 Ha.
    expect(Math.abs(hf.energy - (-76.026765))).toBeLessThan(1e-4);
  }, 30_000);

  test("Spherical HF lies above Cartesian HF (variational principle: Cart basis ⊃ Sph basis)", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(h2o, "cc-pvdz");
    const cart = computeMolecularIntegrals(shells, nuclei);
    const sph  = computeMolecularIntegrals(shells, nuclei, { spherical: true });
    const hfC = runRHFSCF(cart, 10, { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
    const hfS = runRHFSCF(sph, 10,  { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
    expect(hfC.energy).toBeLessThan(hfS.energy);
    expect((hfS.energy - hfC.energy) * 1000).toBeLessThan(1.0);  // diff < 1 mHa
  }, 30_000);
});
