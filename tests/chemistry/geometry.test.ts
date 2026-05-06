// Geometry optimization. Tier 2 stage 1: BFGS minimization of E_HF
// over atomic positions, with central finite-difference gradients.
//
// Pass bars:
//   • H₂ / STO-3G:  R_HH  ≈ 0.713 Å  (PySCF: 0.7126)
//   • H₂O / STO-3G: R_OH  ≈ 0.989 Å,  ∠HOH ≈ 100.0°  (PySCF: 0.9893, 100.04)
//   • BeH₂ / STO-3G: R_BeH ≈ 1.291 Å, ∠HBeH = 180°  (PySCF: ~1.291)
// All within 5 mÅ / 1° of PySCF — generous since FD gradients limit
// precision to ~1e-4 Ha/Å, which translates to ~1 mÅ in atom positions.
//
// Plus a sanity check: starting away from optimum, the optimizer must
// drive the energy strictly below the initial energy.
import { describe, expect, test } from "vitest";
import { optimizeGeometry, bondLength, bondAngle } from "../../src/chemistry/geometry.js";
import type { Atom } from "../../src/chemistry/atoms.js";

describe("Geometry optimization — STO-3G molecule set", () => {
  test("H₂: optimum bond ≈ 0.713 Å", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.9] },   // start far from optimum
    ];
    const r = optimizeGeometry(atoms);
    expect(r.optimizer.termination).toBe("converged");
    const R = bondLength(r.atoms, 0, 1);
    expect(Math.abs(R - 0.7126)).toBeLessThan(5e-3);
    // Energy at PySCF-optimized H₂ STO-3G is -1.11751 Ha.
    expect(Math.abs(r.energy - (-1.11751))).toBeLessThan(1e-3);
  }, 30_000);

  test("H₂O: optimum R_OH ≈ 0.989 Å, ∠HOH ≈ 100°", () => {
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ 0.7, 0, 0.6] },
      { symbol: "H", pos: [-0.7, 0, 0.6] },
    ];
    const r = optimizeGeometry(atoms);
    expect(r.optimizer.termination).toBe("converged");
    const R1 = bondLength(r.atoms, 0, 1);
    const R2 = bondLength(r.atoms, 0, 2);
    const ang = bondAngle(r.atoms, 1, 0, 2);
    expect(Math.abs(R1 - 0.9893)).toBeLessThan(5e-3);
    expect(Math.abs(R2 - 0.9893)).toBeLessThan(5e-3);
    expect(Math.abs(ang - 100.04)).toBeLessThan(1.0);
    // C₂v symmetry should be preserved.
    expect(Math.abs(R1 - R2)).toBeLessThan(1e-3);
  }, 60_000);

  test("BeH₂: optimum R_BeH ≈ 1.291 Å, linear", () => {
    const atoms: Atom[] = [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.5] },
      { symbol: "H",  pos: [0, 0, -1.5] },
    ];
    const r = optimizeGeometry(atoms);
    expect(r.optimizer.termination).toBe("converged");
    const R1 = bondLength(r.atoms, 0, 1);
    const R2 = bondLength(r.atoms, 0, 2);
    const ang = bondAngle(r.atoms, 1, 0, 2);
    expect(Math.abs(R1 - 1.291)).toBeLessThan(5e-3);
    expect(Math.abs(R2 - 1.291)).toBeLessThan(5e-3);
    expect(Math.abs(ang - 180)).toBeLessThan(1.0);
    expect(Math.abs(R1 - R2)).toBeLessThan(1e-3);  // D∞h symmetry
  }, 60_000);

  test("Optimizer drives energy strictly downhill", () => {
    // Start at a deliberately bad H₂ geometry; energy must drop.
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 1.3] },  // way too long
    ];
    const r = optimizeGeometry(atoms, { lbfgs: { maxIter: 50 } });
    // Initial energy is the first sample in the L-BFGS history.
    const E_initial = r.optimizer.history[0]!;
    expect(r.energy).toBeLessThan(E_initial);
  }, 30_000);
});
