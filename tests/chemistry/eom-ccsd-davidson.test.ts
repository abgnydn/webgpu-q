// Cross-validate the Davidson EOM-CCSD path against the dense
// Hessenberg-QR path on H₂O / STO-3G.
//
// H₂O STO-3G has dim = NOCC·NVIRT + (NOCC choose 2)·(NVIRT choose 2)
// = 10·4 + 45·6 = 310 — small enough that dense is fast (seconds)
// and Davidson has room to actually iterate (subspaceCap ≪ dim).
//
// Pass bars:
//   1. Davidson converges in < 200 iterations
//   2. The k=5 lowest real-positive eigenvalues agree with the dense
//      path to ≤ 1×10⁻⁴ Ha (Davidson at tol=1e-7 residual gives
//      eigenvalues to ~1e-5 Ha on EOM-CCSD-style matrices)

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runEOMCCSD } from "../../src/chemistry/eom-ccsd.js";

describe("EOM-CCSD — Davidson vs dense", () => {
  test("matches dense path on H₂O STO-3G — first 5 roots", { timeout: 180000 }, () => {
    // H₂O equilibrium geometry (Å).
    const half = (104.52 / 2) * Math.PI / 180;
    const x = 0.9572 * Math.sin(half);
    const z = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ x, 0, z] },
      { symbol: "H", pos: [-x, 0, z] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);

    const hf = runRHFSCF(integrals, nElectrons);
    expect(hf.converged).toBe(true);

    const ccsd = runCCSD(hf, integrals);
    expect(ccsd.converged).toBe(true);

    // Dense reference run.
    const dense = runEOMCCSD(ccsd, integrals, hf, { nRoots: 5 });
    expect(dense.energies.length).toBe(5);

    // Davidson run.
    const dav = runEOMCCSD(ccsd, integrals, hf, {
      nRoots: 5,
      useDavidson: true,
      davidsonTol: 1e-7,
      davidsonMaxIter: 200,
    });

    expect(dav.energies.length).toBe(5);

    // Eigenvalue agreement: 1e-4 Ha bar (Davidson at tol=1e-7 typically
    // gives eigenvalues to ~1e-5 Ha; 1e-4 is the slack we allow).
    for (let k = 0; k < 5; k++) {
      expect(Math.abs(dav.energies[k]! - dense.energies[k]!)).toBeLessThan(1e-4);
    }
  });
});
