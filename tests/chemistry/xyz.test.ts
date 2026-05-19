// XYZ parser / emitter tests.

import { describe, expect, test } from "vitest";
import { parseXYZ, toXYZ } from "../../src/chemistry/xyz.js";

describe("XYZ format", () => {
  test("Parse standard H₂O XYZ", () => {
    const text = `3
water at experimental geometry
O    0.0000000    0.0000000    0.0000000
H    0.7570000    0.0000000    0.5860000
H   -0.7570000    0.0000000    0.5860000
`;
    const { atoms, comment } = parseXYZ(text);
    expect(atoms.length).toBe(3);
    expect(atoms[0]!.symbol).toBe("O");
    expect(atoms[0]!.pos).toEqual([0, 0, 0]);
    expect(atoms[1]!.symbol).toBe("H");
    expect(atoms[1]!.pos[0]).toBeCloseTo(0.757, 6);
    expect(atoms[2]!.pos[0]).toBeCloseTo(-0.757, 6);
    expect(comment).toBe("water at experimental geometry");
  });

  test("Parse XYZ with atomic numbers instead of symbols", () => {
    const text = `2
H₂
1 0.0 0.0  0.371
1 0.0 0.0 -0.371
`;
    const { atoms } = parseXYZ(text);
    expect(atoms.length).toBe(2);
    expect(atoms[0]!.symbol).toBe("H");
    expect(atoms[1]!.symbol).toBe("H");
    expect(atoms[0]!.pos[2]).toBeCloseTo(0.371, 6);
  });

  test("Parse XYZ with Windows line endings", () => {
    const text = "1\r\ncomment\r\nH 0 0 0\r\n";
    const { atoms } = parseXYZ(text);
    expect(atoms.length).toBe(1);
    expect(atoms[0]!.symbol).toBe("H");
  });

  test("Round-trip toXYZ → parseXYZ", () => {
    const original = [
      { symbol: "O" as const, pos: [0, 0, 0] as [number, number, number] },
      { symbol: "H" as const, pos: [0.757, 0, 0.586] as [number, number, number] },
      { symbol: "H" as const, pos: [-0.757, 0, 0.586] as [number, number, number] },
    ];
    const text = toXYZ(original, "test");
    const { atoms } = parseXYZ(text);
    expect(atoms.length).toBe(3);
    for (let i = 0; i < 3; i++) {
      expect(atoms[i]!.symbol).toBe(original[i]!.symbol);
      for (let k = 0; k < 3; k++) {
        expect(atoms[i]!.pos[k]).toBeCloseTo(original[i]!.pos[k]!, 6);
      }
    }
  });

  test("Unsupported element throws clearly", () => {
    const text = `1
helium not supported in current AtomSymbol union
He 0 0 0
`;
    expect(() => parseXYZ(text)).toThrow(/not supported/);
  });

  test("Too few atom lines throws", () => {
    const text = `3
short
H 0 0 0
H 0 0 1`;       // no trailing newline → only 4 lines total
    expect(() => parseXYZ(text)).toThrow(/declared 3 atoms/);
  });

  test("Atomic number out of range throws", () => {
    const text = `1
gold not in table
79 0 0 0
`;
    expect(() => parseXYZ(text)).toThrow(/atomic number 79/);
  });
});
