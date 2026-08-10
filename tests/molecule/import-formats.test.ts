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
    // Titanium: period 4, far outside anything the basis tables cover. This
    // test previously used Si, which became supported when period 3 landed.
    // The negative example must ALSO not begin with a supported one-letter
    // symbol — canonicalSymbol falls back to the 1-char form, so "Fe" would
    // silently parse as F and "Sc"/"Pd" as S/P. "T" is not a symbol, so Ti
    // reaches the throw. (tests/chemistry/xyz.test.ts uses Kr for the same
    // role; a distinct element keeps the two failures distinguishable.)
    const xyz = `1\nti atom\nTi 0 0 0`;
    expect(() => parseGeometry(xyz, "xyz")).toThrow(/Unsupported atom symbol/);
  });

  test("XYZ — an unsupported 2-char symbol never truncates to a supported 1-char one", () => {
    // Regression guard. canonicalSymbol used to fall back from the 2-char
    // form to the 1-char form unconditionally, so "Fe" imported as
    // FLUORINE — a real, wrong, silent element substitution. The hazard
    // grew when period 3 made P and S supported single letters. The
    // fallback is now opt-in and reserved for PDB atom names.
    for (const sym of ["Fe", "Pd", "Pt", "Sc", "Se", "Sn", "Sr", "Ni"]) {
      const xyz = `1\ntest\n${sym} 0 0 0`;
      expect(() => parseGeometry(xyz, "xyz"), `${sym} must throw, not truncate`)
        .toThrow(/Unsupported atom symbol/);
    }
  });

  test("PDB — atom-name fallback still resolves CA to carbon", () => {
    // The one place the 1-char truncation IS correct: PDB cols 13-16 carry
    // atom names, where "CA" means alpha-carbon rather than calcium. The
    // element column (77-78) is left blank here to force that path.
    const pdb =
`ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00`;
    const r = parseGeometry(pdb, "pdb");
    expect(r.atoms).toHaveLength(1);
    expect(r.atoms[0]!.symbol).toBe("C");
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

  test("XYZ — CRLF line endings parse cleanly", () => {
    const xyz = "2\r\nH₂\r\nH 0 0 0\r\nH 0.7414 0 0\r\n";
    const r = parseGeometry(xyz, "xyz");
    expect(r.atoms.length).toBe(2);
    expect(r.atoms[0]!.symbol).toBe("H");
    expect(r.atoms[1]!.pos[0]).toBeCloseTo(0.7414, 4);
  });

  test("XYZ — extra whitespace and trailing blank lines tolerated", () => {
    const xyz = `  2   \nspaced\n  O   0.0   0.0   0.0  \n  H    0.95   0.0  0.0   \n\n\n`;
    const r = parseGeometry(xyz, "xyz");
    expect(r.atoms.length).toBe(2);
    expect(r.atoms[0]!.symbol).toBe("O");
    expect(r.atoms[1]!.pos[0]).toBeCloseTo(0.95, 4);
  });

  test("XYZ — non-finite coordinate rejected with named error", () => {
    const xyz = `1\nbad\nC 0 NaN 0`;
    expect(() => parseGeometry(xyz, "xyz")).toThrow(/non-finite coordinate/);
  });

  test("PDB — file with no ATOM/HETATM records throws", () => {
    const pdb = "HEADER    EMPTY\nTITLE     Nothing\nEND\n";
    expect(() => parseGeometry(pdb, "pdb")).toThrow(/no ATOM/);
  });

  test("PDB — element field inferred from atom name when col 77-78 blank", () => {
    // Element columns (77-78) intentionally blank; element should come
    // from the atom-name field (cols 13-16).
    const pdb =
`HETATM    1  OW  HOH A   1       0.000   0.000   0.000  1.00  0.00
HETATM    2  HW1 HOH A   1       0.757   0.586   0.000  1.00  0.00
END`;
    const r = parseGeometry(pdb, "pdb");
    expect(r.atoms.length).toBe(2);
    expect(r.atoms[0]!.symbol).toBe("O");
    expect(r.atoms[1]!.symbol).toBe("H");
  });

  test("MOL — too-short file throws", () => {
    expect(() => parseGeometry("name\nempty\n\n", "mol")).toThrow(/too short/);
  });
});
