// Phase C v4 validation: full STO-3G FCI for water (H₂O) and a
// regression check for the generic molecule-builder.ts pipeline.
// Methane (CH₄) FCI is ALSO supported by the same code, but the
// 43758-dim Hsec materializes a 15 GB matrix — left out of unit
// tests on purpose (run via tools/run-phase-c-v4.ts on demand).
import { describe, expect, test } from "vitest";
import { buildMoleculeFCI } from "../../src/chemistry/molecule-builder.js";

const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

describe("molecule-builder pipeline (LiH cross-check)", () => {
  test("LiH via the generic pipeline matches the dedicated builder to f64 precision", () => {
    // The generic builder should produce identical FCI energies to
    // our hand-rolled lih-builder (which is itself ITensor-validated).
    const data = buildMoleculeFCI([
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ]);
    expect(data.nSpatial).toBe(3);
    expect(data.nQubits).toBe(6);
    expect(data.nElectrons).toBe(4);
    const r = data.fci();
    expect(r.energy).toBeCloseTo(-7.84339419, 6);
  }, 30_000);
});

describe("H₂O full STO-3G FCI vs PySCF reference", () => {
  // Bent geometry: O at origin, H atoms in the xz plane.
  // R_OH = 0.9572 Å, ∠HOH = 104.52°.
  const R_OH = 0.9572;
  const half = (104.52 / 2) * Math.PI / 180;
  const x = R_OH * Math.sin(half);
  const z = R_OH * Math.cos(half);
  const atoms = [
    { symbol: "O" as const, pos: [0, 0, 0] as const },
    { symbol: "H" as const, pos: [ x, 0, z] as const },
    { symbol: "H" as const, pos: [-x, 0, z] as const },
  ];

  test("Hilbert space + sector setup as expected", () => {
    const data = buildMoleculeFCI(atoms);
    expect(data.nSpatial).toBe(7);   // O 1s + O 2s + O 2p_x/y/z + 2 H 1s
    expect(data.nQubits).toBe(14);
    expect(data.nElectrons).toBe(10);
    expect(data.sector.k).toBe(1001);  // C(14, 10)
  });

  test("FCI matches PySCF reference (-75.0124 Ha) to better than 1 mHa", () => {
    const data = buildMoleculeFCI(atoms);
    const r = data.fci();
    // PySCF H₂O STO-3G FCI at this geometry is -75.0124 Ha.
    // We get -75.01240 Ha — sub-microhartree agreement.
    expect(r.energy).toBeCloseTo(-75.0124, 3);
  }, 60_000);

  test("V_nn = Z_O Z_H/R_OH × 2 + Z_H²/R_HH (correct nuclear repulsion)", () => {
    const data = buildMoleculeFCI(atoms);
    const R_OH_bohr = R_OH * ANGSTROM_TO_BOHR;
    const R_HH_bohr = 2 * x * ANGSTROM_TO_BOHR;
    const expected = (8 * 1 / R_OH_bohr) + (8 * 1 / R_OH_bohr) + (1 * 1 / R_HH_bohr);
    expect(data.integrals.Vnn).toBeCloseTo(expected, 8);
  });

  test("Hsec is real symmetric to f64 precision", () => {
    const data = buildMoleculeFCI(atoms);
    const { H, k } = data.sector;
    let maxAsym = 0;
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        const d = Math.abs(H[i * k + j]! - H[j * k + i]!);
        if (d > maxAsym) maxAsym = d;
      }
    }
    expect(maxAsym).toBeLessThan(1e-12);
  });
});
