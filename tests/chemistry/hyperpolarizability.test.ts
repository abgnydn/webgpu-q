// Tier 2 stage 20: first hyperpolarizability β via finite-field.
//
// Pass bars:
//   1. β tensor for H₂O HF/STO-3G is FINITE and reasonable
//      (no NaN, vector invariant in the ~0–500 a.u. range).
//   2. H₂ symmetric centrosymmetric (D∞h) ⇒ β = 0 by symmetry.
//      (Centrosymmetric molecules have all β_ijk = 0; a clean
//      smoking-gun symmetry test.)
//   3. β tensor satisfies Kleinman symmetry to FD-noise level
//      (post-symmetrization, the asymmetry should be 0).
import { describe, expect, test } from "vitest";
import { hyperpolarizabilityFiniteField } from "../../src/chemistry/hyperpolarizability.js";
import { optimizeGeometry } from "../../src/chemistry/geometry.js";
import type { Atom } from "../../src/chemistry/atoms.js";

describe("First hyperpolarizability — H₂O HF/STO-3G", () => {
  // Per-test timeout raised to match vitest.config.ts's global 360 s. The old
  // value (30000 ms) was tight enough to false-red under ordinary background
  // load — measured on an idle M2 Pro this cell takes ~12 s, but under a
  // loaded machine it took 56 s and timed out with correct chemistry,
  // which is exactly the landmine the global ceiling was raised to remove.
  test("β tensor is finite, no NaN, vector invariant in a sensible range", { timeout: 360_000 }, () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const x = 0.9572 * Math.sin(half);
    const z = 0.9572 * Math.cos(half);
    const start: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ x, 0, z] },
      { symbol: "H", pos: [-x, 0, z] },
    ];
    const opt = optimizeGeometry(start, { method: "hf" });
    const β = hyperpolarizabilityFiniteField(opt.atoms, { method: "hf" });
    // No NaN, no infinity in any of the 27 components.
    for (let i = 0; i < 27; i++) {
      expect(Number.isFinite(β.tensor[i]!)).toBe(true);
    }
    // Sensible range: β in atomic units, |β_vec| typically 0–500 a.u.
    // for small molecules at HF/STO-3G. STO-3G dramatically
    // underestimates β versus larger basis (β is even more sensitive
    // to basis quality than α).
    expect(β.vectorInvariant).toBeGreaterThanOrEqual(0);
    expect(β.vectorInvariant).toBeLessThan(2000);
  });
});

describe("Centrosymmetry forces β = 0 — H₂", () => {
  test("H₂ has all β_ijk ≈ 0 (D∞h is centrosymmetric)", { timeout: 360_000 }, () => {
    const start: Atom[] = [
      { symbol: "H", pos: [0, 0, -0.37] },
      { symbol: "H", pos: [0, 0,  0.37] },
    ];
    const opt = optimizeGeometry(start, { method: "hf" });
    const β = hyperpolarizabilityFiniteField(opt.atoms, { method: "hf" });
    // Inversion symmetry forces every β_ijk to zero. The largest
    // residual we should see is FD noise — bound at 1e-2 a.u. per
    // component is generous (typical noise is < 1e-4 at the 5e-3
    // field strength).
    for (let i = 0; i < 27; i++) {
      expect(Math.abs(β.tensor[i]!)).toBeLessThan(1e-2);
    }
    // Vector invariant also vanishes.
    expect(β.vectorInvariant).toBeLessThan(1e-2);
  });
});

describe("Kleinman symmetry of β tensor", () => {
  test("Post-symmetrization the tensor is invariant under index permutation", { timeout: 360_000 }, () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const x = 0.9572 * Math.sin(half);
    const z = 0.9572 * Math.cos(half);
    const start: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ x, 0, z] },
      { symbol: "H", pos: [-x, 0, z] },
    ];
    const opt = optimizeGeometry(start, { method: "hf" });
    const β = hyperpolarizabilityFiniteField(opt.atoms, { method: "hf" });
    // β_ijk = β_ikj = β_jik = β_jki = β_kij = β_kji (Kleinman).
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 3; k++) {
          const refs = [
            β.tensor[i * 9 + j * 3 + k]!,
            β.tensor[i * 9 + k * 3 + j]!,
            β.tensor[j * 9 + i * 3 + k]!,
            β.tensor[j * 9 + k * 3 + i]!,
            β.tensor[k * 9 + i * 3 + j]!,
            β.tensor[k * 9 + j * 3 + i]!,
          ];
          for (const v of refs) {
            // Post-symmetrization, all 6 permutations equal.
            expect(Math.abs(v - refs[0]!)).toBeLessThan(1e-12);
          }
        }
      }
    }
  });
});
