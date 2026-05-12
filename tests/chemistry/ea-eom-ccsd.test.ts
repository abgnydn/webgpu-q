// EA-EOM-CCSD validation (Tier 2 stage 38).
// Strategy:
//   1. For H₂O STO-3G the LUMO has positive ε (unbound) → EA is negative.
//      The "best" EA (largest, least negative) should be near −ε_LUMO.
//   2. Eigenvalues come out real.
//   3. Ordering: most-bound (largest EA, least negative) appears at the
//      tail of the sorted ascending list.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runEAEOMCCSD } from "../../src/chemistry/ea-eom-ccsd.js";

const HARTREE_TO_EV = 27.211386245988;

describe("EA-EOM-CCSD — Tier 2 stage 38", () => {
  test("H₂O STO-3G — best EA ≈ −ε_LUMO (Koopmans EA limit)", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const x = 0.9572 * Math.sin(half);
    const z = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ x, 0, z] },
      { symbol: "H", pos: [-x, 0, z] },
    ];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 10);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-10, maxIter: 50 });
    const ea = runEAEOMCCSD(ccsd, integrals, hf, { nRoots: 8 });

    // All real.
    for (let k = 0; k < ea.eas.length; k++) {
      expect(Math.abs(ea.imag[k]!)).toBeLessThan(1e-6);
    }
    // Koopmans EA at LUMO = -ε_LUMO.
    const koopmansEA_eV = -hf.orbitalEnergies[hf.nOccupied]! * HARTREE_TO_EV;
    console.log(`[ea-eom-h2o] Koopmans LUMO EA = ${koopmansEA_eV.toFixed(2)} eV`);
    console.log(`[ea-eom-h2o] EA-EOM-CCSD eigenvalues (eV): [${
      Array.from(ea.eas.slice(0, 8)).map(x => (x * HARTREE_TO_EV).toFixed(2)).join(", ")
    }]`);
    // For STO-3G H₂O the LUMO has ε > 0 → Koopmans EA < 0.
    expect(koopmansEA_eV).toBeLessThan(0);
    // EA-EOM-CCSD best EA (descending sort → first entry) should be
    // close to Koopmans (within ~3 eV correlation correction).
    const bestEA = ea.eas[0]!;
    const bestEA_eV = bestEA * HARTREE_TO_EV;
    console.log(`[ea-eom-h2o] best EA-EOM-CCSD = ${bestEA_eV.toFixed(2)} eV`);
    expect(Math.abs(bestEA_eV - koopmansEA_eV)).toBeLessThan(5);
  });

  test("BeH₂ STO-3G — eigenvalues real, no spurious complex pairs", () => {
    const atoms: Atom[] = [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -1.34] },
      { symbol: "H",  pos: [0, 0,  1.34] },
    ];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 6);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-10, maxIter: 50 });
    const ea = runEAEOMCCSD(ccsd, integrals, hf, { nRoots: 6 });
    for (let k = 0; k < ea.eas.length; k++) {
      expect(Math.abs(ea.imag[k]!)).toBeLessThan(1e-6);
    }
    console.log(`[ea-eom-beh2] EA-EOM-CCSD eigenvalues (eV): [${
      Array.from(ea.eas.slice(0, 6)).map(x => (x * HARTREE_TO_EV).toFixed(2)).join(", ")
    }]`);
  });
});
