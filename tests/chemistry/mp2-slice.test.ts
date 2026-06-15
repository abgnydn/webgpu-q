// Distributed-MP2 partition invariant — the acceptance gate for the swarm's
// first collaborative single-molecule reduction.
//
// The DF-MP2 correlation energy E_corr = Σ_i Σ_j Σ_ab … partitions over the
// outer occupied index i. If a worker computes only its [iLo,iHi) slice and a
// reducer sums the partials, the result MUST equal the single-machine DF-MP2
// to floating-point reassociation noise (~1e-12). This is the same partition-
// sum-vs-single-slab invariant the swarm already holds for distributed J/K.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, defaultFrozenCore, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { mp2EnergyDF } from "../../src/chemistry/mp2.js";
import { generateAutoAux, buildAuxBasisDFStreaming } from "../../src/chemistry/df-aux.js";
import { reduceMP2Slices, type MP2SliceResult } from "../../src/swarm/chemistry-kernel.js";

const H2O: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half);
  const z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
})();

describe("distributed DF-MP2 partition invariant", () => {
  test("Σ slices == single-machine DF-MP2 (≤ 1e-12), all-electron and frozen-core", async () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O, "sto-3g");
    const n = shells.length;
    const nOcc = nElectrons / 2;
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-11, densityTol: 1e-9, maxIter: 200 });
    const aux = generateAutoAux(shells, 1);
    const df = await buildAuxBasisDFStreaming(shells, aux);

    for (const nFrozen of [0, defaultFrozenCore(H2O)]) {
      const full = mp2EnergyDF(hf.C_MO, hf.orbitalEnergies, nOcc, n, nFrozen, df);
      expect(full).toBeLessThan(0); // correlation energy is negative near equilibrium

      // Slice the active occupied range [nFrozen, nOcc) every way from 1..active.
      const active = nOcc - nFrozen;
      for (let k = 1; k <= active; k++) {
        let sum = 0;
        for (let s = 0; s < k; s++) {
          const iLo = nFrozen + Math.floor((s * active) / k);
          const iHi = nFrozen + Math.floor(((s + 1) * active) / k);
          sum += mp2EnergyDF(hf.C_MO, hf.orbitalEnergies, nOcc, n, nFrozen, df, [iLo, iHi]);
        }
        expect(Math.abs(sum - full)).toBeLessThan(1e-12);
      }
    }
  });

  test("an out-of-range slice contributes exactly zero", async () => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(H2O, "sto-3g");
    const n = shells.length;
    const nOcc = nElectrons / 2;
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, energyTol: 1e-11, densityTol: 1e-9, maxIter: 200 });
    const df = await buildAuxBasisDFStreaming(shells, generateAutoAux(shells, 1));
    // [nOcc, nOcc+5) is entirely virtual → no occupied i → zero.
    expect(mp2EnergyDF(hf.C_MO, hf.orbitalEnergies, nOcc, n, 0, df, [nOcc, nOcc + 5])).toBe(0);
  });
});

describe("reduceMP2Slices — reduction + heterogeneity guard", () => {
  const slice = (partial: number, eHF: number): MP2SliceResult => ({
    label: "s", partial, iLo: 0, iHi: 1, eHF, nOccupied: 5, nBasisFunctions: 24,
    durationMs: 1, engine: "wasm",
  });

  test("sums partials and adds the shared HF energy", () => {
    const r = reduceMP2Slices([slice(-0.1, -76), slice(-0.05, -76), slice(-0.05, -76)]);
    expect(r.correlationEnergy).toBeCloseTo(-0.2, 12);
    expect(r.hfEnergy).toBe(-76);
    expect(r.totalEnergy).toBeCloseTo(-76.2, 12);
  });

  test("throws if workers disagree on the HF reference (mixed engine/path)", () => {
    // One worker ran a different reference → its partial is over different MOs.
    expect(() => reduceMP2Slices([slice(-0.1, -76.0), slice(-0.05, -76.03)])).toThrow(/HF references disagree/);
  });

  test("tolerates sub-1e-7 reference jitter (deterministic SCF noise)", () => {
    expect(() => reduceMP2Slices([slice(-0.1, -76.0), slice(-0.05, -76.0 + 1e-9)])).not.toThrow();
  });
});
