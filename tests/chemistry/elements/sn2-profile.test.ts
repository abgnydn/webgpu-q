// SN2 reaction profile — the calculation period 3 was added for.
//
// Identity reaction  Cl⁻ + CH₃Cl → ClCH₃ + Cl⁻, the canonical gas-phase
// SN2 benchmark and the mechanism every introductory organic course
// teaches. Chlorine was the single element blocking it.
//
// The path is a LINEAR INTERPOLATION, not an optimized reaction path:
//   s ∈ [−1, 1];  r₁ = r_mid + s·d (C···Cl_a),  r₂ = r_mid − s·d (C···Cl_b)
//   umbrella angle φ(s) = 90° + s·18°, so CH₃ flips through planar.
// That is enough to show the barrier and the Walden inversion; it is not
// a substitute for an optimized transition state.
//
// System is an ANION: 43 electrons from the neutral atoms, +1 for the
// charge = 44, closed shell.
//
// References below are PySCF 2.13.1, mol.charge = -1, mol.cart = True,
// conv_tol 1e-11, at the identical geometries this file constructs.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei } from "../../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../../src/chemistry/hf-scf.js";
import {
  sn2Point, SN2_EXTRA_ELECTRONS, HARTREE_TO_KCAL,
  SN2_CCPVDZ_BARRIER_KCAL, SN2_LITERATURE_BARRIER_KCAL,
} from "../../../src/labs/sn2-geometry.js";

// Geometry comes from src/labs/sn2-geometry.ts — the SAME module the labs
// page uses, so the page and this validation cannot describe different
// molecules.
/** Anion: neutral electron count + 1. */
function sn2Energy(s: number, basis: "sto-3g" | "cc-pvdz") {
  const atoms = sn2Point(s).atoms;
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons + SN2_EXTRA_ELECTRONS, {
    useDIIS: true, maxIter: 400, energyTol: 1e-11, densityTol: 1e-9,
  });
  return { energy: hf.energy, converged: hf.converged, n: integrals.n };
}

describe("SN2: Cl⁻ + CH₃Cl identity reaction", () => {
  test("STO-3G endpoint and transition state match PySCF", () => {
    const end = sn2Energy(-1, "sto-3g");
    const ts = sn2Energy(0, "sto-3g");
    expect(end.converged).toBe(true);
    expect(ts.converged).toBe(true);
    expect(end.n).toBe(26);
    // PySCF, charge=-1, cart=True
    expect(Math.abs((end.energy - -948.21772145) * 1000)).toBeLessThan(0.1);
    expect(Math.abs((ts.energy - -948.16446439) * 1000)).toBeLessThan(0.1);
  }, 300_000);

  test("the profile is symmetric — an identity reaction must be", () => {
    // Free correctness check: E(s) and E(−s) describe the same system with
    // the two chlorines swapped, so any asymmetry is a bug in the integrals,
    // the geometry construction, or the SCF — not chemistry.
    for (const s of [0.2, 0.6, 1.0]) {
      const plus = sn2Energy(s, "sto-3g");
      const minus = sn2Energy(-s, "sto-3g");
      expect(plus.converged && minus.converged).toBe(true);
      expect(Math.abs(plus.energy - minus.energy)).toBeLessThan(1e-9);
    }
  }, 600_000);

  test("minimal basis overestimates the barrier by more than 2×", () => {
    // The teaching point, and the reason a methods course does not stop at
    // STO-3G. Literature is SN2_LITERATURE_BARRIER_KCAL relative to the
    // ion-dipole complex — see that constant for why the qualifier matters.
    const barrier = (basis: "sto-3g" | "cc-pvdz") => {
      const end = sn2Energy(-1, basis);
      const ts = sn2Energy(0, basis);
      expect(end.converged && ts.converged).toBe(true);
      return (ts.energy - end.energy) * HARTREE_TO_KCAL;
    };
    const sto3g = barrier("sto-3g");
    const ccpvdz = barrier("cc-pvdz");
    expect(sto3g).toBeGreaterThan(32);
    expect(sto3g).toBeLessThan(35);
    // Pins the exact number the labs page quotes in its prose. The page
    // computes STO-3G only, so without this assertion its cc-pVDZ claim
    // would be an unchecked literal — which is how it originally shipped.
    expect(Math.abs(ccpvdz - SN2_CCPVDZ_BARRIER_KCAL)).toBeLessThan(1.0);
    expect(SN2_LITERATURE_BARRIER_KCAL).toBeLessThan(ccpvdz);
    expect(sto3g / ccpvdz).toBeGreaterThan(2);
  }, 900_000);
});
