// WebGPU-integral feasibility probe (step 0 of the GPU integral build).
//
// WebGPU/WGSL is f32-only (no f64). Before writing any WGSL for the 3-index
// integral build, answer the make-or-break: can a single-precision fitting
// tensor still hold chemical accuracy? This rounds the f64 B-tensor to f32 and
// runs HF — an OPTIMISTIC lower bound (it captures output rounding, not the
// accumulation error of evaluating Boys/Hermite recurrences in f32), but if even
// this blows past chemical accuracy, the GPU-integral path is capped and we stop.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { buildAuxBasisDFStreaming, generateAutoAux } from "../../src/chemistry/df-aux.js";
import type { DFResult } from "../../src/chemistry/df.js";

const CHEM_ACC = 1.5936e-3;

const H2O: Atom[] = (() => {
  const h = (104.52 / 2) * Math.PI / 180, x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
  return [{ symbol: "O", pos: [0, 0, 0] }, { symbol: "H", pos: [x, 0, z] }, { symbol: "H", pos: [-x, 0, z] }];
})();
const CH4: Atom[] = (() => {
  const d = 0.629;
  return [
    { symbol: "C", pos: [0, 0, 0] }, { symbol: "H", pos: [d, d, d] }, { symbol: "H", pos: [-d, -d, d] },
    { symbol: "H", pos: [-d, d, -d] }, { symbol: "H", pos: [d, -d, -d] },
  ];
})();

describe("f32 DF feasibility for the WebGPU integral build", () => {
  for (const [name, atoms, nElec] of [["H2O", H2O, 10], ["CH4", CH4, 10]] as const) {
    test(`${name} cc-pVDZ — f32-rounded B vs f64 B and exact HF`, async () => {
      const { shells, nuclei } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const exact = runRHFSCF(integrals, nElec, { energyTol: 1e-10, densityTol: 1e-8, maxIter: 100 }).energy;

      const aux = generateAutoAux(shells, 1);
      const df = await buildAuxBasisDFStreaming(shells, aux, 1e-10);
      const eF64 = runRHFSCF(integrals, nElec, { energyTol: 1e-10, densityTol: 1e-8, maxIter: 100, useDF: df }).energy;

      // Round every B element through f32 (Math.fround) — simulate single-precision.
      const Bf32 = new Float64Array(df.B.length);
      for (let i = 0; i < df.B.length; i++) Bf32[i] = Math.fround(df.B[i]!);
      const dfF32: DFResult = { B: Bf32, nAux: df.nAux, threshold: df.threshold, n: df.n };
      const eF32 = runRHFSCF(integrals, nElec, { energyTol: 1e-10, densityTol: 1e-8, maxIter: 100, useDF: dfF32 }).energy;

      const f32_vs_f64 = Math.abs(eF32 - eF64);
      const f32_vs_exact = Math.abs(eF32 - exact);
      /* eslint-disable no-console */
      console.log(`\n[f32-feas ${name}] exact=${exact.toFixed(8)}  f64-DF=${eF64.toFixed(8)}  f32-DF=${eF32.toFixed(8)}`);
      console.log(`[f32-feas ${name}] f32 rounding cost = ${f32_vs_f64.toExponential(2)} Ha (${(f32_vs_f64 / CHEM_ACC).toExponential(2)}× chem-acc)`);
      console.log(`[f32-feas ${name}] f32-DF vs exact   = ${f32_vs_exact.toExponential(2)} Ha (${(f32_vs_exact / CHEM_ACC).toFixed(2)}× chem-acc)`);
      /* eslint-enable no-console */

      // Optimistic lower bound: f32-rounded B should still be inside chemical
      // accuracy vs exact. If not, the GPU f32 integral path is a dead end.
      expect(f32_vs_exact).toBeLessThan(CHEM_ACC);
    });
  }
});
