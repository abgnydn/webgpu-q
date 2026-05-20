// Geometry utility tests — bond length, angle, dihedral, findBonds.

import { describe, expect, test } from "vitest";
import {
  bondLength, bondAngle, dihedralAngle, findBonds,
  centerOfMass, totalMass, principalMomentsOfInertia, rotationalConstants,
  translateMolecule, rotateMolecule, standardOrientation,
  rmsd, rmsdAligned, molecularGraph, coordinationNumbers, distanceMatrix,
  molecularFormula, planarity, lewisStructure, extractFragments,
} from "../../src/chemistry/geometry.js";
import type { Atom } from "../../src/chemistry/atoms.js";

describe("Geometry utilities", () => {
  test("H₂O bond length: O-H ≈ 0.9572 Å", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    expect(bondLength(atoms, 0, 1)).toBeCloseTo(0.9572, 6);
    expect(bondLength(atoms, 0, 2)).toBeCloseTo(0.9572, 6);
    expect(bondLength(atoms, 1, 2)).toBeCloseTo(2 * xH, 6);
  });

  test("H₂O bond angle: H-O-H = 104.52°", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    expect(bondAngle(atoms, 1, 0, 2)).toBeCloseTo(104.52, 4);
  });

  test("Dihedral on flat (s-trans) hyrocarbon-like setup: ±180°", () => {
    // 4 atoms in a plane (z=0): i, j, k, l arranged in trans.
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 1, 0] },
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [1, 0, 0] },
      { symbol: "C", pos: [1, -1, 0] },
    ];
    expect(Math.abs(dihedralAngle(atoms, 0, 1, 2, 3))).toBeCloseTo(180, 4);
  });

  test("Dihedral on staggered: 90°", () => {
    // i at +z, l at +x; planes perpendicular → dihedral = ±90°.
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 1] },
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [1, 0, 0] },
      { symbol: "C", pos: [1, 1, 0] },
    ];
    const phi = dihedralAngle(atoms, 0, 1, 2, 3);
    expect(Math.abs(Math.abs(phi) - 90)).toBeLessThan(1e-6);
  });

  test("Dihedral on collinear b2: NaN", () => {
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [1, 0, 0] },
      { symbol: "C", pos: [2, 0, 0] },        // collinear with b1
      { symbol: "C", pos: [3, 1, 0] },
    ];
    const phi = dihedralAngle(atoms, 0, 1, 2, 3);
    expect(Number.isNaN(phi)).toBe(true);
  });

  test("findBonds on H₂O: finds O-H1 and O-H2, NOT H-H", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const bonds = findBonds(atoms);
    // Two O-H bonds, no H-H.
    expect(bonds.length).toBe(2);
    expect(bonds[0]!.i).toBe(0); expect(bonds[0]!.j).toBe(1);
    expect(bonds[1]!.i).toBe(0); expect(bonds[1]!.j).toBe(2);
    expect(bonds[0]!.length).toBeCloseTo(0.9572, 6);
  });

  test("findBonds on isolated atoms (very far apart): no bonds", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]   },
      { symbol: "H", pos: [0, 0, 10]  },
    ];
    expect(findBonds(atoms).length).toBe(0);
  });

  test("Total mass + center of mass of H₂O", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    // m_H = 1.008, m_O = 16.0 (approximately) → total ≈ 18.0 amu.
    expect(totalMass(atoms)).toBeCloseTo(18.011, 2);
    const com = centerOfMass(atoms);
    // By C₂v symmetry, x and y components are zero.
    expect(Math.abs(com[0])).toBeLessThan(1e-12);
    expect(Math.abs(com[1])).toBeLessThan(1e-12);
    // z component slightly above O (since H atoms are above).
    expect(com[2]).toBeGreaterThan(0);
    expect(com[2]).toBeLessThan(zH);
  });

  test("Rotational constants of H₂O: 27.88 / 14.52 / 9.28 cm⁻¹ literature", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const r = rotationalConstants(atoms);
    // Literature equilibrium A / B / C (cm⁻¹): 27.88 / 14.52 / 9.28
    // (Mathur et al. JMS 1971; CCCBDB). Our values within a few % of
    // experimental — we use the exact-geometry r_e, not vibrationally
    // averaged. Sanity range:
    expect(r.A_cm1).toBeGreaterThan(20);
    expect(r.A_cm1).toBeLessThan(35);
    expect(r.B_cm1).toBeGreaterThan(10);
    expect(r.B_cm1).toBeLessThan(18);
    expect(r.C_cm1).toBeGreaterThan(7);
    expect(r.C_cm1).toBeLessThan(12);
    // A > B > C ordering.
    expect(r.A_cm1).toBeGreaterThan(r.B_cm1);
    expect(r.B_cm1).toBeGreaterThan(r.C_cm1);
    // GHz conversion: 1 cm⁻¹ = 29.98 GHz.
    expect(r.A_GHz / r.A_cm1).toBeCloseTo(29.9792458, 5);
  });

  test("Translate molecule: bond lengths preserved", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    const moved = translateMolecule(atoms, [10, -5, 2]);
    expect(moved[0]!.pos[0]).toBe(10);
    expect(moved[1]!.pos[2]).toBeCloseTo(2.7414, 6);
    expect(bondLength(moved, 0, 1)).toBeCloseTo(0.7414, 6);
  });

  test("Rotate molecule (90° about z): x → y, y → -x", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [1, 0, 0] },
      { symbol: "H", pos: [0, 1, 0] },
    ];
    // R_z(90°) = [[0,-1,0],[1,0,0],[0,0,1]]
    const Rz90 = [0, -1, 0, 1, 0, 0, 0, 0, 1];
    const rotated = rotateMolecule(atoms, Rz90);
    expect(rotated[0]!.pos[0]).toBeCloseTo(0, 10);
    expect(rotated[0]!.pos[1]).toBeCloseTo(1, 10);
    expect(rotated[1]!.pos[0]).toBeCloseTo(-1, 10);
    expect(rotated[1]!.pos[1]).toBeCloseTo(0, 10);
  });

  test("RMSD: identical geometries → 0", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    expect(rmsd(atoms, atoms)).toBeCloseTo(0, 14);
  });

  test("RMSD: 1 Å shift on 1 atom → RMSD = √(1/2) ≈ 0.707", () => {
    const atoms1: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 1] },
    ];
    const atoms2: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 2] },        // shifted by 1 Å
    ];
    // RMSD = √((0² + 1²) / 2) = √0.5
    expect(rmsd(atoms1, atoms2)).toBeCloseTo(Math.sqrt(0.5), 10);
  });

  test("rmsdAligned: same molecule in different orientation → aligned RMSD ≈ 0", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms1: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    // Same H₂O rotated 90° about x and translated.
    const Rx90 = [1, 0, 0, 0, 0, -1, 0, 1, 0];
    const rotated = rotateMolecule(atoms1, Rx90);
    const translated = translateMolecule(rotated, [3, -2, 1]);

    const result = rmsdAligned(atoms1, translated);
    expect(result.rmsd).toBeLessThan(1e-9);     // exact match after alignment
  });

  test("Standard orientation: H₂O has COM at origin", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [5, 5, 5] },       // offset from origin
      { symbol: "H", pos: [5 + xH, 5, 5 + zH] },
      { symbol: "H", pos: [5 - xH, 5, 5 + zH] },
    ];
    const std = standardOrientation(atoms);
    const com = centerOfMass(std);
    expect(Math.abs(com[0])).toBeLessThan(1e-10);
    expect(Math.abs(com[1])).toBeLessThan(1e-10);
    expect(Math.abs(com[2])).toBeLessThan(1e-10);
    // Bond lengths preserved.
    expect(bondLength(std, 0, 1)).toBeCloseTo(0.9572, 6);
    expect(bondAngle(std, 1, 0, 2)).toBeCloseTo(104.52, 4);
  });

  test("Principal moments of inertia: H₂O has 3 non-zero moments (nonlinear)", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { Ia, Ib, Ic } = principalMomentsOfInertia(atoms);
    expect(Ia).toBeGreaterThan(0);
    expect(Ib).toBeGreaterThan(Ia);
    expect(Ic).toBeGreaterThan(Ib);
    // I_c ≈ I_a + I_b (planar molecule identity).
    expect(Math.abs(Ic - Ia - Ib)).toBeLessThan(1e-6);
  });

  test("molecularGraph on CH₄: 1 component, 4 bonds, C has degree 4", () => {
    const r = 1.0870;
    const s = r / Math.sqrt(3);
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [ s,  s,  s] },
      { symbol: "H", pos: [-s, -s,  s] },
      { symbol: "H", pos: [ s, -s, -s] },
      { symbol: "H", pos: [-s,  s, -s] },
    ];
    const g = molecularGraph(atoms);
    expect(g.nBonds).toBe(4);
    expect(g.degree[0]).toBe(4);    // C is bonded to all 4 H
    for (let i = 1; i <= 4; i++) expect(g.degree[i]).toBe(1);  // each H bonded only to C
    expect(g.connectedComponents.length).toBe(1);
    expect(g.connectedComponents[0]).toEqual([0, 1, 2, 3, 4]);
  });

  test("molecularGraph on two separated H₂: 2 components", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
      { symbol: "H", pos: [0, 0, 5.0]    },
      { symbol: "H", pos: [0, 0, 5.7414] },
    ];
    const g = molecularGraph(atoms);
    expect(g.nBonds).toBe(2);
    expect(g.connectedComponents.length).toBe(2);
    expect(g.connectedComponents[0]).toEqual([0, 1]);
    expect(g.connectedComponents[1]).toEqual([2, 3]);
  });

  test("coordinationNumbers on CH₄: C has CN ≈ 4, each H has CN ≈ 1", () => {
    const r = 1.0870;
    const s = r / Math.sqrt(3);
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [ s,  s,  s] },
      { symbol: "H", pos: [-s, -s,  s] },
      { symbol: "H", pos: [ s, -s, -s] },
      { symbol: "H", pos: [-s,  s, -s] },
    ];
    const cn = coordinationNumbers(atoms);
    expect(cn[0]).toBeGreaterThan(3.5);   // C should see ~4 neighbors
    expect(cn[0]).toBeLessThan(4.5);
    for (let i = 1; i <= 4; i++) {
      expect(cn[i]).toBeGreaterThan(0.7); // H sees ~1 neighbor (C)
      expect(cn[i]).toBeLessThan(1.3);
    }
  });

  test("extractFragments: H₂ + H₂ dimer → 2 fragments of 2 atoms each", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
      { symbol: "H", pos: [0, 0, 5.0]    },
      { symbol: "H", pos: [0, 0, 5.7414] },
    ];
    const { fragments, fragmentAtomIndices } = extractFragments(atoms);
    expect(fragments.length).toBe(2);
    expect(fragments[0]!.length).toBe(2);
    expect(fragments[1]!.length).toBe(2);
    expect(fragmentAtomIndices[0]).toEqual([0, 1]);
    expect(fragmentAtomIndices[1]).toEqual([2, 3]);
    // Atom symbols preserved.
    expect(fragments[0]![0]!.symbol).toBe("H");
    expect(fragments[0]![0]!.pos[2]).toBe(0);
  });

  test("extractFragments: single connected molecule → 1 fragment", () => {
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [0.95, 0, 0] },
      { symbol: "H", pos: [-0.95, 0, 0] },
    ];
    const { fragments } = extractFragments(atoms);
    expect(fragments.length).toBe(1);
    expect(fragments[0]!.length).toBe(3);
  });

  test("lewisStructure: bond orders → multiplicities (single/double/triple)", () => {
    // Hand-crafted bond order matrix for a 4-atom fake molecule:
    // 0-1: single (BO=0.95), 1-2: double (BO=1.9), 2-3: triple (BO=2.8),
    // 0-3: no bond (BO=0.2).
    const N = 4;
    const bo = new Float64Array(N * N);
    bo[0 * N + 1] = 0.95; bo[1 * N + 0] = 0.95;
    bo[1 * N + 2] = 1.9;  bo[2 * N + 1] = 1.9;
    bo[2 * N + 3] = 2.8;  bo[3 * N + 2] = 2.8;
    bo[0 * N + 3] = 0.2;  bo[3 * N + 0] = 0.2;
    const bonds = lewisStructure(bo, N);
    expect(bonds.length).toBe(3);
    expect(bonds[0]).toEqual({ i: 0, j: 1, multiplicity: 1, bondOrder: 0.95 });
    expect(bonds[1]).toEqual({ i: 1, j: 2, multiplicity: 2, bondOrder: 1.9 });
    expect(bonds[2]).toEqual({ i: 2, j: 3, multiplicity: 3, bondOrder: 2.8 });
  });

  test("lewisStructure: cutoff parameter filters weak bonds", () => {
    const N = 2;
    const bo = new Float64Array(N * N);
    bo[0 * N + 1] = 0.3; bo[1 * N + 0] = 0.3;
    expect(lewisStructure(bo, N).length).toBe(0);             // default cutoff 0.4
    expect(lewisStructure(bo, N, 0.2).length).toBe(1);         // lower cutoff
  });

  test("planarity: H₂O is planar (3 heavy/all atoms in a plane)", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    // heavyOnly=false to include H atoms (otherwise just 1 atom).
    const r = planarity(atoms, { heavyOnly: false });
    expect(r.isPlanar).toBe(true);
    expect(r.rmsDeviation).toBeLessThan(1e-10);
  });

  test("planarity: CH₄ is non-planar (tetrahedral)", () => {
    const r = 1.0870;
    const s = r / Math.sqrt(3);
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [ s,  s,  s] },
      { symbol: "H", pos: [-s, -s,  s] },
      { symbol: "H", pos: [ s, -s, -s] },
      { symbol: "H", pos: [-s,  s, -s] },
    ];
    const res = planarity(atoms, { heavyOnly: false });
    expect(res.isPlanar).toBe(false);
    expect(res.rmsDeviation).toBeGreaterThan(0.2);
  });

  test("planarity: <3 atoms → trivially planar", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]      },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    const r = planarity(atoms, { heavyOnly: false });
    expect(r.isPlanar).toBe(true);
    expect(r.rmsDeviation).toBe(0);
  });

  test("molecularFormula: H₂O = 'H2O', CH₄ = 'CH4', ethanol = 'C2H6O'", () => {
    const h2o: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [0.95, 0, 0] },
      { symbol: "H", pos: [-0.95, 0, 0] },
    ];
    expect(molecularFormula(h2o)).toBe("H2O");

    const ch4: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [0.6, 0.6, 0.6] },
      { symbol: "H", pos: [-0.6, -0.6, 0.6] },
      { symbol: "H", pos: [0.6, -0.6, -0.6] },
      { symbol: "H", pos: [-0.6, 0.6, -0.6] },
    ];
    expect(molecularFormula(ch4)).toBe("CH4");

    const ethanol: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "C", pos: [1.5, 0, 0] },
      { symbol: "O", pos: [-1.4, 0, 0] },
      { symbol: "H", pos: [0, 1, 0] },
      { symbol: "H", pos: [0, -1, 0] },
      { symbol: "H", pos: [1.5, 1, 0] },
      { symbol: "H", pos: [1.5, -1, 0] },
      { symbol: "H", pos: [2.5, 0, 0] },
      { symbol: "H", pos: [-2, 0, 0] },
    ];
    expect(molecularFormula(ethanol)).toBe("C2H6O");
  });

  test("molecularFormula: NH₃ uses alphabetical when no carbon", () => {
    const nh3: Atom[] = [
      { symbol: "N", pos: [0, 0, 0] },
      { symbol: "H", pos: [1, 0, 0] },
      { symbol: "H", pos: [-1, 0, 0] },
      { symbol: "H", pos: [0, 1, 0] },
    ];
    // Hill convention with no carbon: alphabetical → H comes before N.
    expect(molecularFormula(nh3)).toBe("H3N");
  });

  test("molecularFormula: ghost atoms skipped", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 1] },
      { symbol: "O", pos: [5, 0, 0], ghost: true },
    ];
    expect(molecularFormula(atoms)).toBe("H2");
  });

  test("distanceMatrix on H₂O: symmetric, zero diagonal, correct pairs", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const M = distanceMatrix(atoms);
    expect(M.length).toBe(9);
    // Diagonal zero.
    expect(M[0]).toBe(0); expect(M[4]).toBe(0); expect(M[8]).toBe(0);
    // Symmetric.
    expect(M[1]).toBe(M[3]);   // (0,1) = (1,0)
    expect(M[2]).toBe(M[6]);   // (0,2) = (2,0)
    expect(M[5]).toBe(M[7]);   // (1,2) = (2,1)
    // Values.
    expect(M[1]).toBeCloseTo(0.9572, 6);   // O-H1
    expect(M[5]).toBeCloseTo(2 * xH, 6);    // H1-H2
  });

  test("coordinationNumbers on isolated atoms: CN ≈ 0", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0]   },
      { symbol: "H", pos: [0, 0, 10]  },
    ];
    const cn = coordinationNumbers(atoms);
    expect(cn[0]).toBeLessThan(0.01);
    expect(cn[1]).toBeLessThan(0.01);
  });

  test("molecularGraph adjacency is sorted per atom", () => {
    // H₂O: O bonded to both H1 and H2.
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [0.95, 0, 0] },
      { symbol: "H", pos: [-0.95, 0, 0] },
    ];
    const g = molecularGraph(atoms);
    expect(g.adjacency[0]).toEqual([1, 2]);   // O bonded to H1, H2 (sorted)
    expect(g.adjacency[1]).toEqual([0]);
    expect(g.adjacency[2]).toEqual([0]);
  });
});
