// ─────────────────────────────────────────────────────────────
// redox.ts — vertical ionization potentials and electron
// affinities. Tier 2 stage 22.
//
// Two routes are supported per property:
//
//   • Koopmans' theorem (frozen-orbital approximation):
//       IP_Koopmans = −ε_HOMO         (lowest unbound orbital)
//       EA_Koopmans = −ε_LUMO         (lowest unoccupied orbital)
//     Free from the closed-shell SCF: just read off orbital
//     energies. Systematically OVER-estimates IP by ~5–15 % at
//     HF level (orbital relaxation in the cation lowers the
//     true IP) and is even worse for EA at small basis.
//
//   • ΔSCF (relaxed orbitals):
//       IP_ΔSCF = E(cation, N−1 e⁻) − E(neutral, N e⁻)
//       EA_ΔSCF = E(neutral, N e⁻) − E(anion,   N+1 e⁻)
//     More accurate than Koopmans because the cation/anion
//     orbitals relax. Requires UHF on the open-shell ion.
//
// Both methods are "vertical" — geometry held fixed at the
// neutral's optimized structure. Adiabatic IPs/EAs would
// optimize the ion separately; that's a follow-up.
//
// Returns IP/EA in:
//   • Hartree (the native energy unit of this codebase)
//   • eV     (multiply by HA_TO_EV ≈ 27.2114)
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "./atoms.js";
import { moleculeToShellsNuclei } from "./atoms.js";
import { computeMolecularIntegrals, type IntegralOpts } from "./cg-molecular.js";
import { runRHFSCF, type HFOpts } from "./hf-scf.js";
import { runUHFSCF, type UHFOpts } from "./uhf-scf.js";

/** Hartree → eV conversion. */
export const HA_TO_EV = 27.211386245988;

export interface IPOpts {
  readonly basis?: BasisName;
  readonly spherical?: boolean;
  readonly hf?: HFOpts;
  readonly uhf?: UHFOpts;
}

export interface IPResult {
  /** Koopmans IP (−ε_HOMO from the neutral SCF) in Hartree. */
  readonly koopmansHa: number;
  readonly koopmansEv: number;
  /** ΔSCF IP (E_cation − E_neutral) in Hartree. */
  readonly deltaScfHa: number;
  readonly deltaScfEv: number;
  /** Electronic energy of the neutral N-electron species. */
  readonly neutralEnergy: number;
  /** Electronic energy of the cation (N − 1 electrons, doublet). */
  readonly cationEnergy: number;
  /** ⟨S²⟩ of the cation UHF determinant. */
  readonly cationS2: number;
}

/**
 * Vertical ionization potential at the given (neutral) geometry.
 *
 * Removes one electron from α (creating a doublet open-shell
 * cation) and runs UHF; reports the energy difference and the
 * Koopmans' value side-by-side. Both are reported in Hartree
 * AND eV — eV is the chemistry-paper standard for IPs.
 */
export function ionizationPotential(
  atoms: readonly Atom[],
  opts: IPOpts = {},
): IPResult {
  const integralOpts: IntegralOpts = { spherical: opts.spherical ?? false };
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(
    atoms, opts.basis ?? "sto-3g",
  );
  if (nElectrons % 2 !== 0) {
    throw new Error(
      `ionizationPotential: starting molecule must be closed-shell (got ${nElectrons} electrons). ` +
      `Use UHF directly for open-shell starting points.`,
    );
  }
  const integrals = computeMolecularIntegrals(shells, nuclei, { skipOAO: true, ...integralOpts });
  const hfOpts: HFOpts = {
    useDIIS: true, energyTol: 1e-10, densityTol: 1e-8, maxIter: 200,
    ...opts.hf,
  };
  const uhfOpts: UHFOpts = {
    useDIIS: true, energyTol: 1e-10, densityTol: 1e-8, maxIter: 200,
    symmetryBreaking: 0.01,
    ...opts.uhf,
  };
  // Neutral SCF.
  const hf = runRHFSCF(integrals, nElectrons, hfOpts);
  const epsHomo = hf.orbitalEnergies[hf.nOccupied - 1]!;
  const koopmansHa = -epsHomo;

  // Cation: nElectrons − 1, with α holding the extra electron.
  const nAlpha = Math.ceil((nElectrons - 1) / 2);
  const nBeta  = Math.floor((nElectrons - 1) / 2);
  const uhf = runUHFSCF(integrals, nAlpha, nBeta, uhfOpts);
  if (!uhf.converged) {
    throw new Error(
      `ionizationPotential: cation UHF did not converge in ${uhfOpts.maxIter} iterations. ` +
      `Try a larger maxIter or a different symmetryBreaking.`,
    );
  }
  const deltaScfHa = uhf.energy - hf.energy;

  return {
    koopmansHa, koopmansEv: koopmansHa * HA_TO_EV,
    deltaScfHa, deltaScfEv: deltaScfHa * HA_TO_EV,
    neutralEnergy: hf.energy,
    cationEnergy: uhf.energy,
    cationS2: uhf.s2,
  };
}

export interface EAOpts extends IPOpts {}

export interface EAResult {
  readonly koopmansHa: number;
  readonly koopmansEv: number;
  readonly deltaScfHa: number;
  readonly deltaScfEv: number;
  readonly neutralEnergy: number;
  readonly anionEnergy: number;
  readonly anionS2: number;
}

/**
 * Vertical electron affinity at the given (neutral) geometry.
 *
 * Adds one electron to β (creating a doublet open-shell anion)
 * and runs UHF. Reports both Koopmans' EA (= −ε_LUMO) and the
 * ΔSCF EA. Note that small basis sets (STO-3G in particular)
 * typically can't BIND the extra electron — Koopmans' EA gets
 * a positive value (the LUMO is unbound), and ΔSCF gets a
 * NEGATIVE EA (the anion is higher in energy than the neutral).
 * Diffuse functions (aug-cc-pVDZ) are usually required to get
 * a positive EA for atoms / molecules with experimentally bound
 * anions.
 */
export function electronAffinity(
  atoms: readonly Atom[],
  opts: EAOpts = {},
): EAResult {
  const integralOpts: IntegralOpts = { spherical: opts.spherical ?? false };
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(
    atoms, opts.basis ?? "sto-3g",
  );
  if (nElectrons % 2 !== 0) {
    throw new Error(
      `electronAffinity: starting molecule must be closed-shell (got ${nElectrons} electrons).`,
    );
  }
  const integrals = computeMolecularIntegrals(shells, nuclei, { skipOAO: true, ...integralOpts });
  const hfOpts: HFOpts = {
    useDIIS: true, energyTol: 1e-10, densityTol: 1e-8, maxIter: 200,
    ...opts.hf,
  };
  const uhfOpts: UHFOpts = {
    useDIIS: true, energyTol: 1e-10, densityTol: 1e-8, maxIter: 200,
    symmetryBreaking: 0.01,
    ...opts.uhf,
  };
  const hf = runRHFSCF(integrals, nElectrons, hfOpts);
  const epsLumo = hf.orbitalEnergies[hf.nOccupied]!;
  const koopmansHa = -epsLumo;

  // Anion: nElectrons + 1, extra electron in α.
  const nAlpha = Math.ceil((nElectrons + 1) / 2);
  const nBeta  = Math.floor((nElectrons + 1) / 2);
  const uhf = runUHFSCF(integrals, nAlpha, nBeta, uhfOpts);
  if (!uhf.converged) {
    throw new Error(
      `electronAffinity: anion UHF did not converge in ${uhfOpts.maxIter} iterations.`,
    );
  }
  const deltaScfHa = hf.energy - uhf.energy;

  return {
    koopmansHa, koopmansEv: koopmansHa * HA_TO_EV,
    deltaScfHa, deltaScfEv: deltaScfHa * HA_TO_EV,
    neutralEnergy: hf.energy,
    anionEnergy: uhf.energy,
    anionS2: uhf.s2,
  };
}
