// Federated CCSD(T) slice — partition invariance + reducer guard.
//
// The (T) correction E_(T) = Σ_i (independent outer occupied spin-orbital sum),
// so slicing i into disjoint ranges and summing the partials must reproduce the
// single-shot E_(T) to float noise. This is the correctness floor the
// distributed-(T) swarm rests on — the (T) analogue of tests/chemistry/mp2-slice.
//
// Why (T) and not MP2: the distributed-MP2 honest negative showed single-molecule
// distribution barely helps because the redundant SCF+DF setup dwarfs MP2's O(N⁵)
// grind. (T) is O(N⁷), non-iterative, parallel over i — so for systems where it
// dominates, distributing this loop is where single-molecule speedup actually
// lives. Correctness first; the speedup crossover is the e2e's job.
//
// H₂O STO-3G (NOCC=10, NVIRT=4) — (T) is nonzero (needs NVIRT≥3; LiH STO-3G's
// NVIRT=2 makes it identically zero) and runs in ~0.3 s.
import { describe, test, expect } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runCCSDT } from "../../src/chemistry/ccsd-t.js";
import {
  ccsdtSliceTiles,
  runCCSDTSliceTile,
  reduceCCSDTSlices,
  type CCSDTSliceResult,
} from "../../src/swarm/chemistry-kernel.js";

const H2O: Atom[] = [
  { symbol: "O", pos: [0, 0, 0] },
  { symbol: "H", pos: [0, -0.757, 0.587] },
  { symbol: "H", pos: [0, 0.757, 0.587] },
];

describe("ccsdtFromSO iRange — partition invariance", () => {
  test("H₂O STO-3G: Σ i-slice (T) partials == single-shot E_(T) to < 1e-9", () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O, "sto-3g");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { energyTol: 1e-12, densityTol: 1e-12 });
    const ccsd = runCCSD(hf, integrals, { tol: 1e-11, maxIter: 200 });

    // Same converged amplitudes feed every call → only the iRange differs, so
    // this isolates the slicing from any SCF/CCSD convergence drift.
    const full = runCCSDT(ccsd, hf, integrals).tripleCorrection;
    expect(Number.isFinite(full)).toBe(true);
    expect(full).toBeLessThan(0); // (T) is stabilizing (negative) near equilibrium

    const NOCC = nElectrons; // closed-shell: occupied spin-orbitals = nElectrons
    const ranges: [number, number][] = [[0, 3], [3, 7], [7, NOCC]]; // disjoint cover of [0, NOCC)
    let sum = 0;
    for (const r of ranges) sum += runCCSDT(ccsd, hf, integrals, { iRange: r }).tripleCorrection;
    expect(Math.abs(sum - full)).toBeLessThan(1e-9);

    // A slice past the occupied range is a safe empty sum (clamped), contributing 0.
    expect(runCCSDT(ccsd, hf, integrals, { iRange: [NOCC, NOCC + 4] }).tripleCorrection).toBe(0);
  });
});

describe("federated (T): ccsdtSliceTiles + reduceCCSDTSlices == single-shot", () => {
  test("H₂O STO-3G: distributed (T) tiles reproduce runCCSDT", async () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O, "sto-3g");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    // Reference computed with the SAME opts runCCSDTSliceTile uses internally, so
    // the comparison tests the SLICING, not a tolerance mismatch.
    const hf = runRHFSCF(integrals, nElectrons, { maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
    const ccsd = runCCSD(hf, integrals, { tol: 1e-9, maxIter: 200 });
    const single = runCCSDT(ccsd, hf, integrals, { nFrozenCore: 0 });
    expect(single.tripleCorrection).toBeLessThan(0);

    const tiles = ccsdtSliceTiles({ label: "H2O", atoms: H2O, basis: "sto-3g", nSlices: 4, nFrozenCore: 0 });
    expect(tiles.length).toBeGreaterThan(1); // genuinely partitioned
    const slices: CCSDTSliceResult[] = [];
    for (const t of tiles) slices.push(await runCCSDTSliceTile(t));

    const reduced = reduceCCSDTSlices(slices);
    expect(Math.abs(reduced.tripleCorrection - single.tripleCorrection)).toBeLessThan(1e-9);
    expect(Math.abs(reduced.totalEnergy - single.totalEnergy)).toBeLessThan(1e-9);
  });

  test("reducer sums partials and guards divergent CCSD references", () => {
    const mk = (partial: number, eCCSD: number): CCSDTSliceResult => ({
      label: "x", partial, iLo: 0, iHi: 1, eCCSD, nOccupiedSO: 10, durationMs: 0,
    });
    expect(reduceCCSDTSlices([mk(-0.01, -76.1), mk(-0.02, -76.1)]).tripleCorrection).toBeCloseTo(-0.03, 12);
    // Workers that converged to different CCSD references must not be summed.
    expect(() => reduceCCSDTSlices([mk(-0.01, -76.1), mk(-0.02, -76.2)])).toThrow(/CCSD references disagree/);
    expect(() => reduceCCSDTSlices([])).toThrow(/no slices/);
  });
});
