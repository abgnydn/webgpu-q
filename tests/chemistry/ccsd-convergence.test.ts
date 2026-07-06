// CCSD convergence hardening (DIIS + residual-gated stop).
//
// Two pre-existing latent issues this locks down:
//   1. The stop test used |ΔE| < tol ALONE. Energy is stationary to first
//      order in the amplitude error, so |ΔE| can fall below tol while ‖ΔT‖ is
//      still large — a false "converged". We now require BOTH |ΔE| < tol and
//      the free-amplitude residual ‖ΔT‖ < tol.
//   2. Frozen-core CCSD looked non-convergent because the residual was being
//      measured over the core amplitudes too — directions zeroCoreAmplitudes
//      constrains to zero, so their Jacobi push is discarded every step and
//      pins a permanent ~1e-4 "residual". Measuring the residual over the FREE
//      amplitudes only lets frozen-core converge to tol like the all-electron
//      case. Guards against a regression that reintroduces either.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";

const r3 = 0.629;
const CH4: Atom[] = [
  { symbol: "C", pos: [0, 0, 0] },
  { symbol: "H", pos: [r3, r3, r3] },
  { symbol: "H", pos: [r3, -r3, -r3] },
  { symbol: "H", pos: [-r3, r3, -r3] },
  { symbol: "H", pos: [-r3, -r3, r3] },
];

function hfOf(atoms: Atom[]) {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons, {
    useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
  });
  return { hf, integrals };
}

describe("CCSD convergence — DIIS + residual gate", () => {
  const { hf, integrals } = hfOf(CH4);
  const tol = 1e-9;

  test("all-electron: converges with BOTH ΔE and ‖ΔT‖ below tol", () => {
    const cc = runCCSD(hf, integrals, { maxIter: 200, tol });
    expect(cc.converged).toBe(true);
    // The residual gate's contract: a reported convergence means the free
    // amplitudes really settled, not just the energy.
    expect(cc.residualNorm).toBeLessThan(tol);
  });

  test("frozen-core: genuinely converges (residual measured over free amplitudes)", () => {
    const cc = runCCSD(hf, integrals, { maxIter: 200, tol, nFrozenCore: 1 });
    expect(cc.converged).toBe(true);
    expect(cc.residualNorm).toBeLessThan(tol);
    // Sanity: frozen-core captures less correlation than all-electron.
    const ccAll = runCCSD(hf, integrals, { maxIter: 200, tol });
    expect(cc.totalEnergy).toBeGreaterThan(ccAll.totalEnergy);
  });

  test("DIIS accelerates: fewer iterations than the raw iteration", () => {
    const withDIIS = runCCSD(hf, integrals, { maxIter: 200, tol, useDIIS: true });
    const noDIIS = runCCSD(hf, integrals, { maxIter: 200, tol, useDIIS: false });
    expect(withDIIS.converged).toBe(true);
    expect(noDIIS.converged).toBe(true);
    expect(withDIIS.iter).toBeLessThanOrEqual(noDIIS.iter);
    // Same fixed point regardless of accelerator.
    expect(Math.abs(withDIIS.correlationEnergy - noDIIS.correlationEnergy)).toBeLessThan(1e-8);
  });
});
