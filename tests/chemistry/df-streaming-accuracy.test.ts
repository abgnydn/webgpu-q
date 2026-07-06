// Accuracy study: how good is the streaming-DF fit vs EXACT HF, and how does it
// improve with a richer auxiliary basis? Small molecules where the full 4-index
// ERI is feasible, so we have an exact reference. The DF fitting error is a
// per-method property that transfers to larger systems.
//
// This is a diagnostic, not a pass/fail gate — it prints the error ladder so we
// can decide whether auto-aux level 1 is good enough or we need a real RI basis.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { buildAuxBasisDFStreaming, generateAutoAux } from "../../src/chemistry/df-aux.js";

const CHEM_ACC = 1.5936e-3; // 1 kcal/mol in Hartree

const H2O: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half), z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
})();

const CH4: Atom[] = (() => {
  const d = 0.629; // C-H / sqrt(3) component
  return [
    { symbol: "C", pos: [0, 0, 0] },
    { symbol: "H", pos: [d, d, d] },
    { symbol: "H", pos: [-d, -d, d] },
    { symbol: "H", pos: [-d, d, -d] },
    { symbol: "H", pos: [d, -d, -d] },
  ];
})();

describe("streaming-DF accuracy vs exact HF, by aux richness", () => {
  for (const [name, atoms, nElec] of [["H2O", H2O, 10], ["CH4", CH4, 10]] as const) {
    test(`${name} cc-pVDZ — DF error ladder (auto-aux levels 1/2/3)`, async () => {
      const { shells, nuclei } = moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei); // full ERI
      const exact = runRHFSCF(integrals, nElec, { energyTol: 1e-10, densityTol: 1e-8, maxIter: 100 }).energy;
      const n = shells.length;

      const rows: Array<{ level: number; nAux: number; nKept: number; err: number }> = [];
      for (const level of [1, 2, 3]) {
        const aux = generateAutoAux(shells, level);
        const df = await buildAuxBasisDFStreaming(shells, aux, 1e-10);
        const eDF = runRHFSCF(integrals, nElec, { energyTol: 1e-10, densityTol: 1e-8, maxIter: 100, useDF: df }).energy;
        rows.push({ level, nAux: aux.length, nKept: df.nAux, err: Math.abs(eDF - exact) });
      }

       
      console.log(`\n[df-acc ${name}] exact HF = ${exact.toFixed(8)} Ha (n=${n})`);
      for (const r of rows)
        console.log(`  auto-aux L${r.level}: nAux=${String(r.nAux).padStart(4)} nKept=${String(r.nKept).padStart(4)}  |E_DF−E_exact| = ${r.err.toExponential(3)} Ha  (${(r.err / CHEM_ACC).toFixed(2)}× chem-acc)`);
       

      // Sanity: richer aux must not make the fit worse, and L3 should be well
      // inside chemical accuracy.
      expect(rows[2]!.err).toBeLessThanOrEqual(rows[0]!.err + 1e-9);
      expect(rows[2]!.err).toBeLessThan(CHEM_ACC);
      // CH₄'s L3 ladder (824 aux fns) needs ~115-120 s at best on an idle
      // M2 Pro — exactly the global 120 s ceiling, so any background load
      // produced a spurious timeout with correct chemistry. Long-running
      // test, long-running timeout.
    }, 360_000);
  }
});
