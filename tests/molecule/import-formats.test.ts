import { describe, expect, test } from "vitest";
import { parseGeometry, detectFormat } from "../../src/molecule/import-formats.js";

describe("Geometry import — XYZ / PDB / MOL / SDF", () => {
  test("detectFormat by extension", () => {
    expect(detectFormat("h2o.xyz")).toBe("xyz");
    expect(detectFormat("H2O.PDB")).toBe("pdb");
    expect(detectFormat("foo.mol")).toBe("mol");
    expect(detectFormat("foo.sdf")).toBe("sdf");
    expect(detectFormat("foo.cube")).toBe(null);
  });

  test("XYZ — H₂O standard sample", () => {
    const xyz = `3
water
O      0.000000     0.000000     0.000000
H      0.757160     0.586260     0.000000
H     -0.757160     0.586260     0.000000`;
    const r = parseGeometry(xyz, "xyz");
    expect(r.atoms.length).toBe(3);
    expect(r.atoms[0]!.symbol).toBe("O");
    expect(r.atoms[1]!.symbol).toBe("H");
    expect(r.atoms[0]!.pos[1]).toBe(0);
    expect(r.atoms[1]!.pos[0]).toBeCloseTo(0.757160, 5);
    expect(r.title).toBe("water");
  });

  test("XYZ — rejects malformed", () => {
    expect(() => parseGeometry("1", "xyz")).toThrow();
    expect(() => parseGeometry("not a number\n\n", "xyz")).toThrow();
    expect(() => parseGeometry("1\ncomment\nbad row", "xyz")).toThrow();
  });

  test("XYZ — rejects unsupported atom", () => {
    const xyz = `1\nsi atom\nSi 0 0 0`;
    expect(() => parseGeometry(xyz, "xyz")).toThrow(/Unsupported atom symbol/);
  });

  test("PDB — H₂O from cols", () => {
    // Standard PDB atom-record format. Element field at cols 77-78.
    const pdb =
`HEADER    TEST
TITLE     Water Molecule
HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.757   0.586   0.000  1.00  0.00           H
HETATM    3  H2  HOH A   1      -0.757   0.586   0.000  1.00  0.00           H
END`;
    const r = parseGeometry(pdb, "pdb");
    expect(r.atoms.length).toBe(3);
    expect(r.atoms[0]!.symbol).toBe("O");
    expect(r.atoms[1]!.symbol).toBe("H");
    expect(r.atoms[1]!.pos[0]).toBeCloseTo(0.757, 3);
    expect(r.title).toBe("Water Molecule");
  });

  test("MOL/SDF — H₂O V2000", () => {
    const mol =
` water
  Empty

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.7572    0.5863    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7572    0.5863    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
M  END`;
    const r = parseGeometry(mol, "mol");
    expect(r.atoms.length).toBe(3);
    expect(r.atoms[0]!.symbol).toBe("O");
    expect(r.atoms[1]!.pos[0]).toBeCloseTo(0.7572, 4);
    expect(r.title).toBe("water");
  });

  test("Symbol normalization tolerates case + isotope prefix", () => {
    const xyz = `2\nmix\n13C 0 0 0\nh 1 0 0`;
    const r = parseGeometry(xyz, "xyz");
    expect(r.atoms[0]!.symbol).toBe("C");
    expect(r.atoms[1]!.symbol).toBe("H");
  });
});
