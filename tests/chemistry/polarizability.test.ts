// Tier 2 stage 17: static dipole polarizability via finite-field.
//
// Pass bars:
//   1. H₂O HF/STO-3G isotropic α: ~3-5 a.u. The STO-3G basis
//      DRAMATICALLY underestimates polarizability (small basis can't
//      describe outer-shell polarization), so this is much smaller
//      than the experimental ~9.79 a.u. With aug-cc-pVDZ HF gives
//      ~7-8 a.u., still below experiment by ~10%, since HF lacks
//      correlation-driven polarization. STO-3G is a sanity-only test.
//   2. H₂O α tensor is symmetric (diff < 1e-6).
//   3. He atom (He has no built-in symbol — use H₂ as a closed-shell
//      symmetric proxy): polarizability is positive and isotropic by
//      symmetry up to FD noise.
//   4. H₂ at equilibrium: α tensor non-zero, α_∥ > α_⊥ (parallel along
//      bond axis is larger — known textbook result).
import { describe, expect, test } from "vitest";
import { polarizabilityFiniteField, AU_TO_ANGSTROM3 } from "../../src/chemistry/polarizability.js";
import { optimizeGeometry } from "../../src/chemistry/geometry.js";
import type { Atom } from "../../src/chemistry/atoms.js";

describe("Static polarizability — H₂O HF/STO-3G", () => {
  test("H₂O α is positive, symmetric, isotropic value in a sensible range", { timeout: 30000 }, () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const x = 0.9572 * Math.sin(half);
    const z = 0.9572 * Math.cos(half);
    const start: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ x, 0, z] },
      { symbol: "H", pos: [-x, 0, z] },
    ];
    const opt = optimizeGeometry(start, { method: "hf" });
    const pol = polarizabilityFiniteField(opt.atoms, { method: "hf" });

    // All principal components positive (closed-shell ground state ⇒ α > 0).
    for (let i = 0; i < 3; i++) {
      expect(pol.principalComponents[i]!).toBeGreaterThan(0);
    }
    // Tensor is symmetric.
    for (let i = 0; i < 3; i++) {
      for (let j = i + 1; j < 3; j++) {
        const diff = Math.abs(pol.tensor[i * 3 + j]! - pol.tensor[j * 3 + i]!);
        expect(diff).toBeLessThan(1e-6);
      }
    }
    // STO-3G H₂O α_iso is much smaller than experiment (~9.79 a.u.) —
    // STO-3G can't describe diffuse polarization. Sanity range:
    // 1 a.u. < α_iso < 10 a.u.
    expect(pol.isotropic).toBeGreaterThan(1);
    expect(pol.isotropic).toBeLessThan(10);
    // Non-trivial anisotropy for a planar non-isotropic molecule.
    expect(pol.anisotropy).toBeGreaterThan(0);
  });
});

describe("Static polarizability — H₂ HF/STO-3G", () => {
  test("H₂ α_∥ > α_⊥ (parallel-axis polarizability dominates)", { timeout: 20000 }, () => {
    const start: Atom[] = [
      { symbol: "H", pos: [0, 0, -0.37] },
      { symbol: "H", pos: [0, 0,  0.37] },
    ];
    const opt = optimizeGeometry(start, { method: "hf" });
    const pol = polarizabilityFiniteField(opt.atoms, { method: "hf" });

    // Bond is along z; α_zz is the parallel component.
    // α_xx, α_yy are perpendicular and exactly zero in STO-3G —
    // there are no p polarization functions on H, so no basis can
    // describe the perpendicular polarization. (Adding aug-cc-pVDZ
    // would lift α_⊥ from 0 to ~3 a.u.) The test confirms the
    // STO-3G consequence: large parallel α, vanishing perpendicular.
    const aPerp1 = pol.tensor[0]!;     // α_xx
    const aPerp2 = pol.tensor[4]!;     // α_yy
    const aParallel = pol.tensor[8]!;  // α_zz
    expect(aParallel).toBeGreaterThan(0);
    expect(Math.abs(aPerp1)).toBeLessThan(0.01);    // ~zero, no p basis
    expect(Math.abs(aPerp2)).toBeLessThan(0.01);
    expect(aParallel).toBeGreaterThan(Math.abs(aPerp1));
    expect(aParallel).toBeGreaterThan(Math.abs(aPerp2));
    // Off-diagonals vanish to FD noise.
    expect(Math.abs(pol.tensor[1]!)).toBeLessThan(0.01);
    expect(Math.abs(pol.tensor[2]!)).toBeLessThan(0.01);
    expect(Math.abs(pol.tensor[5]!)).toBeLessThan(0.01);
  });
});

describe("Polarizability units conversion", () => {
  test("AU_TO_ANGSTROM3 ≈ 0.1482", () => {
    // 1 a.u. of polarizability = 0.148184711 Å³ (CODATA).
    expect(AU_TO_ANGSTROM3).toBeGreaterThan(0.148);
    expect(AU_TO_ANGSTROM3).toBeLessThan(0.149);
  });
});
