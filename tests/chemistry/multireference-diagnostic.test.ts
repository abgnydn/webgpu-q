// Multireference diagnostic aggregator tests.
//
// Pass bars:
//   1. All-safe input gives "single-reference" with no flags.
//   2. Any one MR-cutoff trigger gives "multi-reference".
//   3. Borderline inputs give "borderline" verdict.
//   4. Real H₂O CCSD: T1 ≈ 0.008, D1 ≈ 0.02 → "single-reference".
//   5. Closed-shell HF: NOON = {2, 2, …, 0, 0} → no NOON deviation.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { naturalOrbitalOccupations } from "../../src/chemistry/natural-orbitals.js";
import { multireferenceDiagnostic } from "../../src/chemistry/multireference-diagnostic.js";

describe("Multireference diagnostic aggregator", () => {
  test("All-safe input → single-reference, no flags", () => {
    const r = multireferenceDiagnostic({
      t1: 0.005, d1: 0.01,
      s2: 0, s2Exact: 0,
    });
    expect(r.severity).toBe("single-reference");
    expect(r.flags.length).toBe(0);
  });

  test("T1 above MR cutoff → multi-reference", () => {
    const r = multireferenceDiagnostic({ t1: 0.05, d1: 0.01 });
    expect(r.severity).toBe("multi-reference");
    expect(r.flags[0]).toContain("T1");
    expect(r.flags[0]).toContain("multi-reference cutoff");
  });

  test("Borderline T1, low D1 → borderline overall", () => {
    const r = multireferenceDiagnostic({ t1: 0.03, d1: 0.02 });
    expect(r.severity).toBe("borderline");
  });

  test("Severe S2 contamination → multi-reference", () => {
    // Triplet S=1: s2Exact = 2; got 2.5 (contaminated up).
    const r = multireferenceDiagnostic({ s2: 2.5, s2Exact: 2.0 });
    expect(r.severity).toBe("multi-reference");
    expect(r.flags.some((f) => f.includes("contamination"))).toBe(true);
  });

  test("Large NOON deviation → multi-reference", () => {
    const noon = new Float64Array([1.6, 0.4, 0, 0]);  // active-orbital diradical
    const r = multireferenceDiagnostic({ noon, nOccupied: 1 });
    expect(r.severity).toBe("multi-reference");
    expect(r.flags.some((f) => f.includes("NOON"))).toBe(true);
  });

  test("Real H₂O HF/STO-3G CCSD: single-reference", () => {
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
    const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });
    const noon = naturalOrbitalOccupations(hf.D, integrals.S_AO);

    const mr = multireferenceDiagnostic({
      t1: ccsd.t1Diagnostic,
      d1: ccsd.d1Diagnostic,
      s2: 0, s2Exact: 0,
      noon: noon.occupations,
      nOccupied: hf.nOccupied,
    });
    expect(mr.severity).toBe("single-reference");
    expect(mr.flags.length).toBe(0);
  });
});
