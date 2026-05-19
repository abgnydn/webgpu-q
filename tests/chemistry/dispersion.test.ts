// C₆ dispersion coefficient — Casimir-Polder integral of α(iω).
//
// Pass bars:
//   1. α(iω) monotonically decreases from α(0) at ω = 0 to ≈ 0 at
//      ω → ∞. No poles on the imaginary axis.
//   2. C₆ is positive for every closed-shell system.
//   3. C₆(A, B) = C₆(B, A) (Casimir-Polder integrand is symmetric).
//   4. Quadrature convergence: doubling Gauss-Legendre nodes
//      (8 → 16 → 32) gives values agreeing to ≤ 2%.
//   5. Self-dispersion magnitude: H₂ STO-3G falls in the physically
//      reasonable range (positive, finite, O(1)–O(10) a.u.).

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { tdhfPolarizability, tdhfPolarizabilityImag } from "../../src/chemistry/tdhf.js";
import { c6Coefficient } from "../../src/chemistry/dispersion.js";

function buildH2() {
  const atoms: Atom[] = [
    { symbol: "H", pos: [0, 0,  0.371] },
    { symbol: "H", pos: [0, 0, -0.371] },
  ];
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons, {
    useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
  });
  return { hf, integrals, shells };
}

function buildBeH2() {
  // Linear BeH₂, Be at origin, H atoms ±1.33 Å along z.
  const atoms: Atom[] = [
    { symbol: "Be", pos: [0, 0, 0] },
    { symbol: "H",  pos: [0, 0,  1.33] },
    { symbol: "H",  pos: [0, 0, -1.33] },
  ];
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons, {
    useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
  });
  return { hf, integrals, shells };
}

describe("C₆ dispersion via Casimir-Polder integral of α(iω)", () => {
  test("α(iω) monotonically decreases from α(0) along imaginary axis", () => {
    const h2 = buildH2();
    const a0 = tdhfPolarizability(h2.hf, h2.integrals, h2.shells, 0).isotropic;
    const samples = [0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0].map((w) =>
      tdhfPolarizabilityImag(h2.hf, h2.integrals, h2.shells, w).isotropic,
    );
    // α(i·0) = α(0).
    expect(Math.abs(samples[0]! - a0)).toBeLessThan(1e-10);
    // Monotonic decrease.
    for (let i = 1; i < samples.length; i++) {
      expect(samples[i]!).toBeLessThan(samples[i - 1]!);
    }
    // Asymptotic decay: α(iω=10) ≪ α(0).
    expect(samples[samples.length - 1]! / a0).toBeLessThan(0.05);
  });

  test("H₂-H₂ self-dispersion: C₆ positive, finite, in physical range", () => {
    const h2 = buildH2();
    const r = c6Coefficient(h2.hf, h2.integrals, h2.shells,
                            h2.hf, h2.integrals, h2.shells);
    expect(Number.isFinite(r.c6)).toBe(true);
    expect(r.c6).toBeGreaterThan(0);
    // Reference: literature CCSD-F12 H₂-H₂ C₆ ≈ 12.1 a.u. (CBS, with
    // correlation). STO-3G TDHF underestimates α(0) badly (no diffuse
    // functions, no correlation) — observed C₆ ≈ 0.7 a.u. here, which
    // is qualitatively-correct order-of-magnitude for an unpolarized
    // minimal basis. Tighter bounds need cc-pVDZ+aug-cc-pVDZ basis.
    expect(r.c6).toBeGreaterThan(0.1);
    expect(r.c6).toBeLessThan(25);
  });

  test("Symmetry: C₆(H₂, BeH₂) = C₆(BeH₂, H₂)", () => {
    const h2 = buildH2();
    const beh2 = buildBeH2();
    const ab = c6Coefficient(h2.hf, h2.integrals, h2.shells,
                             beh2.hf, beh2.integrals, beh2.shells);
    const ba = c6Coefficient(beh2.hf, beh2.integrals, beh2.shells,
                             h2.hf, h2.integrals, h2.shells);
    expect(Math.abs(ab.c6 - ba.c6)).toBeLessThan(1e-10);
  });

  test("BeH₂ larger / more polarizable than H₂ → C₆(BeH₂) > C₆(H₂)", () => {
    const h2 = buildH2();
    const beh2 = buildBeH2();
    const h2_self = c6Coefficient(h2.hf, h2.integrals, h2.shells,
                                  h2.hf, h2.integrals, h2.shells);
    const beh2_self = c6Coefficient(beh2.hf, beh2.integrals, beh2.shells,
                                    beh2.hf, beh2.integrals, beh2.shells);
    expect(beh2_self.c6).toBeGreaterThan(h2_self.c6);
    expect(beh2_self.c6).toBeGreaterThan(0);
  });

  test("Quadrature convergence: N = 8 / 16 / 32 agree to ≤ 2%", () => {
    const h2 = buildH2();
    const c8  = c6Coefficient(h2.hf, h2.integrals, h2.shells,
                              h2.hf, h2.integrals, h2.shells,
                              { quadraturePoints: 8 }).c6;
    const c16 = c6Coefficient(h2.hf, h2.integrals, h2.shells,
                              h2.hf, h2.integrals, h2.shells,
                              { quadraturePoints: 16 }).c6;
    const c32 = c6Coefficient(h2.hf, h2.integrals, h2.shells,
                              h2.hf, h2.integrals, h2.shells,
                              { quadraturePoints: 32 }).c6;
    expect(Math.abs(c16 - c8) / c16).toBeLessThan(0.02);
    expect(Math.abs(c32 - c16) / c32).toBeLessThan(0.02);
  });
});
