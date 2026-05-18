// IP-EOM-CCSD + EA-EOM-CCSD frozen-core + Davidson cross-checks.
//
// Closes the last frozen-core row in LIMITATIONS.md (IP-EOM, EA-EOM)
// and wires Davidson into both modules. Pattern mirrors the
// EE-EOM-CCSD frozen-core + Davidson work (commits 6b9a96f / 0a33884):
//   - Restrict packed (R_1 + antisym R_2) basis to occupied indices
//     ≥ 2·nFrozenCore. R_1/R_2 are zero at frozen indices, so the
//     σ-equation's inner sums over m, n, k automatically drop
//     frozen-index contributions.
//   - Add `useDavidson: boolean` option behind the existing dense
//     path (default false → bit-identical to pre-change behavior).
//
// Pass bars:
//   1. With nFrozenCore=0 + useDavidson=false (defaults), the
//      pre-change spectrum is reproduced (baseline regression).
//   2. With nFrozenCore=1 on H₂O, the lowest IPs / EAs shift but
//      remain finite, real, and bounded — frozen-core only
//      excludes core-ionization satellites.
//   3. With useDavidson=true on H₂O, the dense + Davidson paths
//      agree on the k=3 (IP) or k=3 (EA) extremal eigenvalues to
//      ≤ 1×10⁻⁴ Ha.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runIPEOMCCSD } from "../../src/chemistry/ip-eom-ccsd.js";
import { runEAEOMCCSD } from "../../src/chemistry/ea-eom-ccsd.js";

function h2o(): { atoms: Atom[] } {
  const half = (104.52 / 2) * Math.PI / 180;
  const xH = 0.9572 * Math.sin(half);
  const zH = 0.9572 * Math.cos(half);
  return {
    atoms: [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ],
  };
}

describe("IP-EOM-CCSD — frozen-core + Davidson", () => {
  test("frozen-1s H₂O lowest IP is real, positive, bounded shift from all-electron", () => {
    const { atoms } = h2o();
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });

    const ccsdAll = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });
    const ipAll   = runIPEOMCCSD(ccsdAll, integrals, hf, { nRoots: 3 });

    const ccsdFC = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9, nFrozenCore: 1 });
    const ipFC   = runIPEOMCCSD(ccsdFC, integrals, hf, { nRoots: 3, nFrozenCore: 1 });

    expect(ipFC.ips.length).toBe(3);

    for (let k = 0; k < 3; k++) {
      expect(ipFC.ips[k]!).toBeGreaterThan(0);
      expect(Math.abs(ipFC.imag[k]!)).toBeLessThan(1e-5);
      if (k > 0) expect(ipFC.ips[k]!).toBeGreaterThanOrEqual(ipFC.ips[k - 1]!);
    }

    // The lowest (physical) IP shouldn't shift by more than ~100 mHa
    // when freezing the 1s core — core electrons don't participate
    // in the HOMO-derived ionization.
    expect(Math.abs(ipFC.ips[0]! - ipAll.ips[0]!)).toBeLessThan(0.1);

    // Frozen-core reduces the manifold dim.
    expect(ipFC.dim).toBeLessThan(ipAll.dim);
  });

  test("Davidson IP-EOM on H₂O agrees with dense to ≤ 1e-4 Ha for k=3", () => {
    const { atoms } = h2o();
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });

    const dense = runIPEOMCCSD(ccsd, integrals, hf, { nRoots: 3 });
    const dav   = runIPEOMCCSD(ccsd, integrals, hf, {
      nRoots: 3, useDavidson: true, davidsonTol: 1e-7,
    });

    expect(dav.ips.length).toBe(3);
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(dav.ips[k]! - dense.ips[k]!)).toBeLessThan(1e-4);
    }
  });
});

describe("EA-EOM-CCSD — frozen-core + Davidson", () => {
  test("frozen-1s H₂O lowest EA is real, bounded shift from all-electron", () => {
    const { atoms } = h2o();
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });

    const ccsdAll = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });
    const eaAll   = runEAEOMCCSD(ccsdAll, integrals, hf, { nRoots: 3 });

    const ccsdFC = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9, nFrozenCore: 1 });
    const eaFC   = runEAEOMCCSD(ccsdFC, integrals, hf, { nRoots: 3, nFrozenCore: 1 });

    expect(eaFC.eas.length).toBe(3);
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(eaFC.imag[k]!)).toBeLessThan(1e-5);
    }
    // Most-bound EA shouldn't move by more than ~100 mHa with frozen-1s.
    expect(Math.abs(eaFC.eas[0]! - eaAll.eas[0]!)).toBeLessThan(0.1);
    expect(eaFC.dim).toBeLessThan(eaAll.dim);
  });

  test("Davidson EA-EOM on H₂O agrees with dense to ≤ 1e-4 Ha for k=3", () => {
    const { atoms } = h2o();
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });

    const dense = runEAEOMCCSD(ccsd, integrals, hf, { nRoots: 3 });
    const dav   = runEAEOMCCSD(ccsd, integrals, hf, {
      nRoots: 3, useDavidson: true, davidsonTol: 1e-7,
    });

    expect(dav.eas.length).toBe(3);
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(dav.eas[k]! - dense.eas[k]!)).toBeLessThan(1e-4);
    }
  });
});
