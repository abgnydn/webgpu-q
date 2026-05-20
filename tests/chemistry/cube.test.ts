// Cube-file export — structural validation + integration sanity.
//
// Pass bars:
//   1. Output structure: title lines + N_atoms/origin line + 3 axis lines
//      + atom records + numeric volumetric block.
//   2. Atom count + Z values + bounding-box derivation.
//   3. Density Cube: ∫ρ dV ≈ N_electrons (validates that integrating
//      the cube approximates the total electron count — typically
//      within a few % for the default 0.3-bohr step on a small molecule
//      where the bounding box doesn't capture all density tails).
//   4. MO Cube: signed amplitudes; ∫|φ|² dV ≈ 1 (normalization
//      approximate, again limited by step size + bounding box).

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { densityCube, moCube, spinDensityCube } from "../../src/chemistry/cube.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";

describe("Gaussian Cube export", () => {
  test("H₂ HF/STO-3G: density cube has correct structure and integrates ≈ 2 e⁻", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0,  0.371] },
      { symbol: "H", pos: [0, 0, -0.371] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });

    const text = densityCube(shells, hf.D, atoms, { padding: 5.0, stepSize: 0.4 });
    const lines = text.trim().split("\n");

    // First two lines are titles (any text).
    expect(lines.length).toBeGreaterThan(7);
    // Third line: N_atoms = 2, origin (3 floats).
    const tokens3 = lines[2]!.trim().split(/\s+/);
    expect(Number(tokens3[0])).toBe(2);
    // Lines 4-6: axis vectors. First token = grid count > 0.
    const Nx = Number(lines[3]!.trim().split(/\s+/)[0]);
    const Ny = Number(lines[4]!.trim().split(/\s+/)[0]);
    const Nz = Number(lines[5]!.trim().split(/\s+/)[0]);
    expect(Nx).toBeGreaterThan(0);
    expect(Ny).toBeGreaterThan(0);
    expect(Nz).toBeGreaterThan(0);
    // Lines 7-8: atom records with Z=1 each.
    const atomLine1 = lines[6]!.trim().split(/\s+/);
    expect(Number(atomLine1[0])).toBe(1);
    const atomLine2 = lines[7]!.trim().split(/\s+/);
    expect(Number(atomLine2[0])).toBe(1);

    // Parse the volumetric block and integrate.
    const dx = 0.4, dy = 0.4, dz = 0.4;
    const dV = dx * dy * dz;
    const startIdx = 8;        // after 2 titles + N_atoms + 3 axis + 2 atom lines
    const values: number[] = [];
    for (let i = startIdx; i < lines.length; i++) {
      const toks = lines[i]!.trim().split(/\s+/).filter((t) => t.length > 0);
      for (const t of toks) {
        const v = Number(t);
        if (!Number.isNaN(v)) values.push(v);
      }
    }
    const totalValues = Nx * Ny * Nz;
    expect(values.length).toBe(totalValues);
    let integral = 0;
    for (const v of values) integral += v * dV;
    // ∫ρ ≈ N_electrons = 2 within ~10% for the cube box / step size.
    expect(integral).toBeGreaterThan(1.5);
    expect(integral).toBeLessThan(2.3);
  }, 60_000);

  test("H₂ MO Cube: signed amplitudes; ∫|φ|² ≈ 1 (HOMO normalized)", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0,  0.371] },
      { symbol: "H", pos: [0, 0, -0.371] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });

    // HOMO is index nOccupied-1 = 0 for H₂.
    const text = moCube(shells, hf.C_MO, 0, atoms, { padding: 5.0, stepSize: 0.4 });
    const lines = text.trim().split("\n");

    // Parse volumetric values.
    const Nx = Number(lines[3]!.trim().split(/\s+/)[0]);
    const Ny = Number(lines[4]!.trim().split(/\s+/)[0]);
    const Nz = Number(lines[5]!.trim().split(/\s+/)[0]);
    const dV = 0.4 ** 3;
    const startIdx = 8;
    const values: number[] = [];
    for (let i = startIdx; i < lines.length; i++) {
      const toks = lines[i]!.trim().split(/\s+/).filter((t) => t.length > 0);
      for (const t of toks) {
        const v = Number(t);
        if (!Number.isNaN(v)) values.push(v);
      }
    }
    expect(values.length).toBe(Nx * Ny * Nz);
    // ∫|φ|² ≈ 1 ± box truncation.
    let normSq = 0;
    let hasPositive = false;
    let hasNegative = false;
    for (const v of values) {
      normSq += v * v * dV;
      if (v > 0) hasPositive = true;
      if (v < 0) hasNegative = true;
    }
    // HOMO of H₂ (bonding σ_g) is positive everywhere — should NOT have
    // both signs in the bounding box.
    expect(hasPositive).toBe(true);
    // Norm ≈ 1 within truncation error (typically 0.85-0.99 for 0.4 bohr).
    expect(normSq).toBeGreaterThan(0.7);
    expect(normSq).toBeLessThan(1.1);
    // For H₂ σ_g, ensure the test logically references the sign check
    // (else the variable would be dead).
    expect(hasNegative).toBe(false);
  }, 60_000);

  test("Li doublet spin density Cube: ∫(ρ_α − ρ_β) dV ≈ +1 (one unpaired α electron)", () => {
    const atoms: Atom[] = [{ symbol: "Li", pos: [0, 0, 0] }];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, 2, 1, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const text = spinDensityCube(shells, uhf.D_alpha, uhf.D_beta, atoms,
      { padding: 6.0, stepSize: 0.4 });
    const lines = text.trim().split("\n");
    expect(lines[0]!).toContain("Spin density");
    // Parse and integrate.
    const Nx = Number(lines[3]!.trim().split(/\s+/)[0]);
    const Ny = Number(lines[4]!.trim().split(/\s+/)[0]);
    const Nz = Number(lines[5]!.trim().split(/\s+/)[0]);
    const dV = 0.4 ** 3;
    const values: number[] = [];
    for (let i = 7; i < lines.length; i++) {
      const toks = lines[i]!.trim().split(/\s+/).filter((t) => t.length > 0);
      for (const t of toks) {
        const v = Number(t);
        if (!Number.isNaN(v)) values.push(v);
      }
    }
    expect(values.length).toBe(Nx * Ny * Nz);
    let integral = 0;
    for (const v of values) integral += v * dV;
    // Total spin = n_α − n_β = 1. Integration with 0.4-bohr step
    // captures most of the SOMO density; box-truncation drops some
    // tail, so expect ~0.8–1.1.
    expect(integral).toBeGreaterThan(0.7);
    expect(integral).toBeLessThan(1.2);
  }, 60_000);
});
