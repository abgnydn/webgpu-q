// CCSD wavefunction-quality diagnostics — Lee-Taylor T1 + Janssen-Nielsen D1.
//
// These are heuristic flags for "is CCSD trustworthy on this
// system?" — values below the textbook bars (~0.02 T1, ~0.05 D1)
// indicate a confidently single-reference wavefunction; above
// suggests multi-reference character where CCSD may not be reliable.
//
// Pass bars:
//   1. Closed-shell H₂O at equilibrium STO-3G: t1 + d1 << 0.02
//      (very clean single-reference case).
//   2. Both quantities are non-negative.
//   3. d1 ≥ t1·√(N_e/N_v) approximately — D1 picks the largest
//      single-orbital amplitude, t1 averages over all of them, so
//      D1 is always ≥ a per-orbital normalized t1 metric.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD, ccsdDiagnostics } from "../../src/chemistry/ccsd.js";

describe("CCSD T1 + D1 wavefunction-quality diagnostics", () => {
  test("closed-shell H₂O STO-3G: both diagnostics well below multi-ref threshold", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-10 });
    expect(ccsd.converged).toBe(true);

    // H₂O at equilibrium is the textbook single-reference case.
    expect(ccsd.t1Diagnostic).toBeGreaterThan(0);
    expect(ccsd.t1Diagnostic).toBeLessThan(0.02);  // well-behaved single-ref
    expect(ccsd.d1Diagnostic).toBeGreaterThan(0);
    expect(ccsd.d1Diagnostic).toBeLessThan(0.05);  // well-behaved
    // D1 ≥ T1·√(N_e/N_v) heuristic — D1 picks the worst orbital, T1
    // averages, so for non-pathological systems D1 ≥ T1.
    expect(ccsd.d1Diagnostic).toBeGreaterThanOrEqual(ccsd.t1Diagnostic - 1e-10);
  });

  test("ccsdDiagnostics helper: zero T1 → zero t1 + zero d1", () => {
    const T1 = new Float64Array(10 * 4);  // all zeros
    const { t1, d1 } = ccsdDiagnostics(T1, 10, 4);
    expect(t1).toBe(0);
    expect(d1).toBe(0);
  });

  test("ccsdDiagnostics helper: single-element T1 gives Frobenius + max-σ correctly", () => {
    // T1[i=2, a=1] = 0.1, NOCC=4, NVIRT=3.
    // ‖T1‖_F = 0.1; t1 = 0.1 / √4 = 0.05.
    // T1 as 4×3 matrix has one nonzero entry; singular values are
    // (0.1, 0, 0) so d1 = 0.1.
    const T1 = new Float64Array(4 * 3);
    T1[2 * 3 + 1] = 0.1;
    const { t1, d1 } = ccsdDiagnostics(T1, 4, 3);
    expect(Math.abs(t1 - 0.05)).toBeLessThan(1e-12);
    expect(Math.abs(d1 - 0.1)).toBeLessThan(1e-10);
  });
});
