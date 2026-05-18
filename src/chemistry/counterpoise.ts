// ─────────────────────────────────────────────────────────────
// counterpoise.ts — Boys-Bernardi counterpoise correction for
// the basis-set superposition error (BSSE) in supermolecular
// interaction energies.
//
// Why BSSE exists:
//   When you compute ΔE = E(AB) − E(A) − E(B) for a dimer A···B,
//   in the dimer calculation A's electrons can "borrow" B's basis
//   functions (and vice-versa), giving A a better effective basis
//   than it has alone. The monomer A calculation uses only A's
//   own basis, so its energy is described in a worse basis. Result:
//   the dimer looks artificially more stable than it should.
//
// Boys-Bernardi fix (CP):
//   ΔE_CP = E(AB) − E(A in dimer basis) − E(B in dimer basis)
//
//   where "X in dimer basis" means: run HF on X's real atoms,
//   keeping every OTHER atom's basis functions present at their
//   normal positions but with NO nuclear charge (Z=0) and NO
//   electrons. The ghost atom provides AO basis only.
//
// API:
//   `runCounterpoise(atoms, fragments, basis, opts)`
//   - `atoms`: the full supermolecule.
//   - `fragments`: array of arrays-of-atom-indices. Each fragment
//     is one monomer. The union must cover every atom index
//     exactly once.
//   - Returns `CounterpoiseResult` with:
//       supermolecule energy,
//       per-fragment energy in the full dimer basis (CP),
//       per-fragment energy in the bare monomer basis (no CP),
//       uncorrected interaction energy ΔE,
//       counterpoise-corrected ΔE_CP,
//       BSSE = ΔE − ΔE_CP   (positive when the dimer was
//       artificially stabilized by basis-set borrowing).
//
// Limitations:
//   - HF only at this entry point. MP2 / CCSD counterpoise is a
//     mechanical follow-up (same pattern — pass ghost atoms
//     through to the post-HF correlation step).
//   - Assumes every fragment is closed-shell. Open-shell (UHF)
//     counterpoise is also mechanical to add.
//   - Per-fragment fragmentation must be a strict partition; we
//     don't validate physical sensibleness (you can put atoms
//     in arbitrary groups, but the BSSE interpretation only
//     makes sense for non-bonded fragments).
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "./atoms.js";
import { moleculeToShellsNuclei } from "./atoms.js";
import { computeMolecularIntegrals } from "./cg-molecular.js";
import { runRHFSCF, type HFOpts } from "./hf-scf.js";

export interface CounterpoiseFragment {
  /** Indices into the `atoms` array that belong to this fragment. */
  readonly atomIndices: readonly number[];
}

export interface CounterpoiseResult {
  /** E(AB) — full supermolecule with all real atoms, Hartree. */
  readonly supermoleculeEnergy: number;
  /** Per-fragment energies in the full dimer basis (ghost-augmented). */
  readonly fragmentEnergiesCP: readonly number[];
  /** Per-fragment energies in the bare monomer basis (no ghosts). */
  readonly fragmentEnergiesBare: readonly number[];
  /** Uncorrected ΔE = E(AB) − Σ E(fragment bare). */
  readonly interactionEnergy: number;
  /** Counterpoise-corrected ΔE_CP = E(AB) − Σ E(fragment in dimer basis). */
  readonly interactionEnergyCP: number;
  /** BSSE = ΔE_CP − ΔE = Σ_k (E_bare(k) − E_CP(k)). Always ≥ 0 by
   *  the variational principle: ghost-augmented monomer energies
   *  are always ≤ bare-basis energies, so CP raises the interaction
   *  energy (makes the dimer look less attractive / more repulsive).
   *  In small basis sets (STO-3G, cc-pVDZ) this can be 10-50% of the
   *  true interaction energy for hydrogen-bonded systems. */
  readonly bsseCorrection: number;
  /** True iff every HF run (supermolecule + 2·N fragment runs)
   *  converged. If false, the energies are returned but should
   *  not be trusted. */
  readonly allConverged: boolean;
}

/**
 * Run the Boys-Bernardi counterpoise correction on a closed-shell
 * supermolecule partitioned into fragments. Performs (1 + 2·N) HF
 * runs where N is the number of fragments:
 *   1× E(AB) on the full supermolecule
 *   N× E(fragment in dimer basis) — fragment real, others ghost
 *   N× E(fragment bare) — fragment real, others absent
 */
export function runCounterpoise(
  atoms: readonly Atom[],
  fragments: readonly CounterpoiseFragment[],
  basis: BasisName = "sto-3g",
  hfOpts: HFOpts = {},
): CounterpoiseResult {
  // ── Validate fragmentation: every atom in exactly one fragment. ──
  const claimed = new Array<boolean>(atoms.length).fill(false);
  for (const f of fragments) {
    for (const idx of f.atomIndices) {
      if (idx < 0 || idx >= atoms.length) {
        throw new Error(`runCounterpoise: fragment atom index ${idx} out of range [0, ${atoms.length})`);
      }
      if (claimed[idx]) {
        throw new Error(`runCounterpoise: atom ${idx} appears in multiple fragments`);
      }
      claimed[idx] = true;
    }
  }
  for (let i = 0; i < atoms.length; i++) {
    if (!claimed[i]) {
      throw new Error(`runCounterpoise: atom ${i} is not assigned to any fragment`);
    }
  }

  let allConverged = true;

  // ── Supermolecule HF. ──
  const superSh = moleculeToShellsNuclei(atoms, basis);
  const superInt = computeMolecularIntegrals(superSh.shells, superSh.nuclei);
  const superHF = runRHFSCF(superInt, superSh.nElectrons, hfOpts);
  if (!superHF.converged) allConverged = false;

  // ── Per-fragment runs. ──
  const fragmentEnergiesCP: number[] = [];
  const fragmentEnergiesBare: number[] = [];

  for (const f of fragments) {
    const fragmentSet = new Set(f.atomIndices);
    // a) Fragment in dimer basis: all atoms present, but only this
    //    fragment's atoms keep their nuclei + electrons. Others are
    //    ghosts (Z=0, no electrons, basis functions present).
    const dimerBasisAtoms: Atom[] = atoms.map((a, i) =>
      fragmentSet.has(i) ? a : { ...a, ghost: true },
    );
    const cpSh = moleculeToShellsNuclei(dimerBasisAtoms, basis);
    const cpInt = computeMolecularIntegrals(cpSh.shells, cpSh.nuclei);
    const cpHF = runRHFSCF(cpInt, cpSh.nElectrons, hfOpts);
    if (!cpHF.converged) allConverged = false;
    fragmentEnergiesCP.push(cpHF.energy);

    // b) Fragment in bare monomer basis: only this fragment's atoms exist.
    const bareAtoms = atoms.filter((_, i) => fragmentSet.has(i));
    const bareSh = moleculeToShellsNuclei(bareAtoms, basis);
    const bareInt = computeMolecularIntegrals(bareSh.shells, bareSh.nuclei);
    const bareHF = runRHFSCF(bareInt, bareSh.nElectrons, hfOpts);
    if (!bareHF.converged) allConverged = false;
    fragmentEnergiesBare.push(bareHF.energy);
  }

  let sumCP = 0;
  let sumBare = 0;
  for (let k = 0; k < fragments.length; k++) {
    sumCP += fragmentEnergiesCP[k]!;
    sumBare += fragmentEnergiesBare[k]!;
  }
  const interactionEnergy = superHF.energy - sumBare;
  const interactionEnergyCP = superHF.energy - sumCP;
  // BSSE = ΔE_CP − ΔE = Σ (E_bare − E_CP). Variationally ≥ 0.
  const bsseCorrection = interactionEnergyCP - interactionEnergy;

  return {
    supermoleculeEnergy: superHF.energy,
    fragmentEnergiesCP,
    fragmentEnergiesBare,
    interactionEnergy,
    interactionEnergyCP,
    bsseCorrection,
    allConverged,
  };
}
