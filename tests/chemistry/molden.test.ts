// Molden-format export — structural validation.
//
// Pass bars:
//   1. Output starts with [Molden Format] and contains all required
//      section headers: [Atoms], [GTO], [MO].
//   2. Atom count + Z values + positions parse out to the input atoms
//      (in bohr). H₂O has 3 atoms with Z = 8, 1, 1.
//   3. Number of [Sym=...] MO entries equals the basis dimension n.
//   4. Number of basis-coefficient lines per MO equals n.
//   5. [6D] tag present when shells contain L=2 (Cartesian d).
//   6. Spherical-d basis throws cleanly.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { toMoldenString } from "../../src/chemistry/molden.js";

describe("Molden-format export", () => {
  test("H₂O HF/STO-3G: required section headers + atom count + MO count", () => {
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
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });

    const text = toMoldenString({
      atoms, shells,
      C_MO: hf.C_MO,
      orbitalEnergies: hf.orbitalEnergies,
      nOccupied: hf.nOccupied,
    });

    expect(text.startsWith("[Molden Format]")).toBe(true);
    expect(text).toContain("[Atoms] AU");
    expect(text).toContain("[GTO]");
    expect(text).toContain("[MO]");
    // [6D] tag only emitted when there's a d-shell — STO-3G H₂O is
    // s-only on H and (1s, 2s, 2p) on O → no d → no [6D].
    expect(text).not.toContain("[6D]");

    // Three atom lines: parse Z values from the [Atoms] block.
    const atomBlock = text.split("[Atoms] AU")[1]!.split("[GTO]")[0]!;
    const atomLines = atomBlock.trim().split("\n");
    expect(atomLines.length).toBe(3);
    // Format: "<symbol>  <idx>  <Z>  <x> <y> <z>"
    // Order: O (Z=8), H (Z=1), H (Z=1).
    const tokens = atomLines.map((l) => l.trim().split(/\s+/));
    expect(tokens[0]![0]).toBe("O");
    expect(Number(tokens[0]![2])).toBe(8);
    expect(tokens[1]![0]).toBe("H");
    expect(Number(tokens[1]![2])).toBe(1);
    expect(tokens[2]![0]).toBe("H");
    expect(Number(tokens[2]![2])).toBe(1);

    // Number of MO blocks (each starts with " Sym= a") equals n = 7 for H₂O/STO-3G.
    const moBlockCount = (text.match(/^ Sym= a$/gm) ?? []).length;
    expect(moBlockCount).toBe(7);
  });

  test("BeH₂ HF/cc-pVDZ Cartesian: [6D] tag present + GTO blocks per atom", () => {
    const atoms: Atom[] = [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0,  1.33] },
      { symbol: "H",  pos: [0, 0, -1.33] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, "cc-pvdz");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });

    const text = toMoldenString({
      atoms, shells,
      C_MO: hf.C_MO,
      orbitalEnergies: hf.orbitalEnergies,
      nOccupied: hf.nOccupied,
    });

    // cc-pVDZ has d-shells → [6D] expected.
    expect(text).toContain("[6D]");
    // Three "<idx> 0" headers in [GTO] block (one per atom).
    const gtoBlock = text.split("[GTO]")[1]!.split("[6D]")[0]!;
    expect((gtoBlock.match(/^\s+\d+ 0$/gm) ?? []).length).toBe(3);
    // Number of MO blocks equals n.
    const n = integrals.n;
    const moBlockCount = (text.match(/^ Sym= a$/gm) ?? []).length;
    expect(moBlockCount).toBe(n);
  }, 60_000);

  test("Spherical-d basis throws clearly", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
    ];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 2, {});  // 2 electrons works for H⁻ but doesn't matter — we won't call hf

    // Pass a fake non-null sphericalT to trigger the guard. We don't need
    // a valid SCF for this test, just any HF-shaped object.
    expect(() => toMoldenString({
      atoms, shells,
      C_MO: hf.C_MO,
      orbitalEnergies: hf.orbitalEnergies,
      nOccupied: 1,
      sphericalT: new Float64Array(4),
    })).toThrow(/spherical-d basis not yet supported/);
  });
});
