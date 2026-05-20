// TRK (Thomas-Reiche-Kuhn) sum rule tests.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type AtomSymbol } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runTDDFT } from "../../src/chemistry/tda-dft.js";
import { trkSumRule } from "../../src/chemistry/trk-sum-rule.js";

describe("TRK sum rule for oscillator strengths", () => {
  test("Hand-picked f's: sum = N_e exactly → fraction = 1, status ok", () => {
    const r = trkSumRule([0.2, 0.5, 0.3, 1.0], 2);
    expect(r.sum).toBeCloseTo(2.0, 10);
    expect(r.fraction).toBeCloseTo(1.0, 10);
    expect(r.status).toBe("ok");
  });

  test("Under-bound: only 10% of N_e captured → status under-bound", () => {
    const r = trkSumRule([0.5], 10);
    expect(r.fraction).toBeCloseTo(0.05, 10);
    expect(r.status).toBe("under-bound");
  });

  test("Over-bound (sign error / double counting): fraction > 1.01 → over-bound", () => {
    const r = trkSumRule([3.0, 4.0, 5.0], 2);
    expect(r.fraction).toBe(6.0);
    expect(r.status).toBe("over-bound");
  });

  test("Real H₂O TDDFT/RPA: TRK fraction is in finite-basis-typical range", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const nucleiSymbols: AtomSymbol[] = ["O", "H", "H"];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const tddft = runTDDFT(integrals, hf, { method: "hf", spin: "singlet", nucleiSymbols });

    const r = trkSumRule(tddft.oscillatorStrengths, nElectrons);
    // STO-3G H₂O has 10 electrons; minimal basis captures ~15-50% of TRK.
    // No over-bounding ever (rigorous variational bound).
    expect(r.status).not.toBe("over-bound");
    expect(r.fraction).toBeGreaterThan(0.05);
    expect(r.fraction).toBeLessThan(1.01);
  }, 60_000);

  test("Throws on invalid N_e", () => {
    expect(() => trkSumRule([0.1, 0.2], 0)).toThrow(/positive/);
    expect(() => trkSumRule([0.1, 0.2], -1)).toThrow(/positive/);
  });
});
