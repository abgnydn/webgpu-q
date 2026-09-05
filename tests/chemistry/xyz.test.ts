// XYZ parser / emitter tests.

import { describe, expect, test } from "vitest";
import { parseXYZ, toXYZ, toMultiFrameXYZ, parseMultiFrameXYZ } from "../../src/chemistry/xyz.js";

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
    // Krypton: genuinely absent from the AtomSymbol union and from the basis
    // tables. This test previously used He (a supported AtomSymbol all along),
    // then B — which became supported when periods 1-2 were completed. Kr is
    // period 4, far outside anything the basis tables cover.
    const text = `1
krypton not supported in current AtomSymbol union
Kr 0 0 0
`;
    expect(() => parseXYZ(text)).toThrow(/not supported/);
  });

  test("Boron round-trips: symbol, Z=5, and toXYZ identity", () => {
    const text = `1
boron atom
B 0.00000000 0.00000000 0.00000000
`;
    const { atoms } = parseXYZ(text);
    expect(atoms).toHaveLength(1);
    expect(atoms[0]!.symbol).toBe("B");
    expect(parseXYZ("1\nby Z\n5 0 0 0\n").atoms[0]!.symbol).toBe("B");
    expect(parseXYZ(toXYZ(atoms)).atoms[0]!.symbol).toBe("B");
  });

  test("Neon round-trips: symbol, Z=10, and toXYZ identity", () => {
    const text = `1
neon atom
Ne 0.00000000 0.00000000 0.00000000
`;
    const { atoms } = parseXYZ(text);
    expect(atoms).toHaveLength(1);
    expect(atoms[0]!.symbol).toBe("Ne");
    expect(parseXYZ("1\nby Z\n10 0 0 0\n").atoms[0]!.symbol).toBe("Ne");
    expect(parseXYZ(toXYZ(atoms)).atoms[0]!.symbol).toBe("Ne");
  });

  test("He round-trips: it is a real AtomSymbol the library ships a molecule for", () => {
    const text = `1
helium atom
He 0.00000000 0.00000000 0.00000000
`;
    const { atoms } = parseXYZ(text);
    expect(atoms).toHaveLength(1);
    expect(atoms[0]!.symbol).toBe("He");
    // Atomic-number form must resolve too (Z=2 was missing from SYMBOL_BY_Z).
    expect(parseXYZ("1\nby Z\n2 0 0 0\n").atoms[0]!.symbol).toBe("He");
    // toXYZ → parseXYZ is an identity for a molecule this library provides.
    expect(parseXYZ(toXYZ(atoms)).atoms[0]!.symbol).toBe("He");
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

  test("Multi-frame trajectory round-trip: 3 frames with energies", () => {
    const frames = [
      { atoms: [{ symbol: "H" as const, pos: [0, 0, 0] as [number, number, number] }],
        energy: -0.5 },
      { atoms: [{ symbol: "H" as const, pos: [0, 0, 0.01] as [number, number, number] }],
        energy: -0.4999 },
      { atoms: [{ symbol: "H" as const, pos: [0, 0, 0.02] as [number, number, number] }],
        energy: -0.49998 },
    ];
    const text = toMultiFrameXYZ(frames);
    const parsed = parseMultiFrameXYZ(text);
    expect(parsed.length).toBe(3);
    for (let i = 0; i < 3; i++) {
      expect(parsed[i]!.atoms.length).toBe(1);
      expect(parsed[i]!.atoms[0]!.symbol).toBe("H");
      expect(parsed[i]!.atoms[0]!.pos[2]).toBeCloseTo(frames[i]!.atoms[0]!.pos[2]!, 6);
      expect(parsed[i]!.energy).toBeCloseTo(frames[i]!.energy, 6);
    }
  });

  test("Multi-frame XYZ produces concatenated standard XYZ blocks", () => {
    const frames = [
      { atoms: [
        { symbol: "H" as const, pos: [0, 0, 0] as [number, number, number] },
        { symbol: "O" as const, pos: [0, 0, 1] as [number, number, number] },
      ] },
      { atoms: [
        { symbol: "H" as const, pos: [0, 0, 0.1] as [number, number, number] },
        { symbol: "O" as const, pos: [0, 0, 1.1] as [number, number, number] },
      ] },
    ];
    const text = toMultiFrameXYZ(frames);
    const lines = text.trim().split("\n");
    // Frame 1: 2 + 2 lines = 4. Frame 2: another 4. Total 8.
    expect(lines.length).toBe(8);
    expect(lines[0]).toBe("2");
    expect(lines[4]).toBe("2");        // second N_atoms header
  });
});
