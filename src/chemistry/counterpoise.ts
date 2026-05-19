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
import { runUHFSCF, type UHFOpts } from "./uhf-scf.js";
import { runMP2 } from "./mp2.js";
import { runCCSD, type CCSDOpts } from "./ccsd.js";
import { runUCCSD } from "./uccsd.js";

export type CounterpoiseMethod = "hf" | "mp2" | "ccsd" | "uhf" | "uccsd";

export interface CounterpoiseFragment {
  /** Indices into the `atoms` array that belong to this fragment. */
  readonly atomIndices: readonly number[];
  /** For open-shell methods ("uhf", "uccsd"): explicit (n_α, n_β)
   *  for this fragment. Required iff `method` is open-shell. The
   *  supermolecule spin is the sum across fragments. For closed-shell
   *  methods this field is ignored.
   */
  readonly spin?: { readonly nAlpha: number; readonly nBeta: number };
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
 * supermolecule partitioned into fragments. Performs (1 + 2·N)
 * single-point energy calculations where N is the number of
 * fragments:
 *   1× E(AB) on the full supermolecule
 *   N× E(fragment in dimer basis) — fragment real, others ghost
 *   N× E(fragment bare) — fragment real, others absent
 *
 * `method` controls the energy at each step: "hf" (default) runs
 * RHF; "mp2" runs HF + MP2 and returns the total energy
 * (E_HF + E_corr_MP2); "ccsd" runs HF + CCSD and returns
 * (E_HF + E_corr_CCSD).
 */
export function runCounterpoise(
  atoms: readonly Atom[],
  fragments: readonly CounterpoiseFragment[],
  basis: BasisName = "sto-3g",
  hfOpts: HFOpts = {},
  method: CounterpoiseMethod = "hf",
  ccsdOpts: CCSDOpts = {},
  uhfOpts: UHFOpts = {},
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

  const isOpenShell = method === "uhf" || method === "uccsd";

  // ── For open-shell methods, validate per-fragment spin and sum
  //    to get the supermolecule (n_α, n_β). ─────────────────────
  let superAlpha = 0, superBeta = 0;
  if (isOpenShell) {
    for (let k = 0; k < fragments.length; k++) {
      const sp = fragments[k]!.spin;
      if (!sp) {
        throw new Error(
          `runCounterpoise: method "${method}" requires fragments[${k}].spin = {nAlpha, nBeta}.`,
        );
      }
      if (sp.nAlpha < 0 || sp.nBeta < 0 || !Number.isInteger(sp.nAlpha) || !Number.isInteger(sp.nBeta)) {
        throw new Error(
          `runCounterpoise: fragments[${k}].spin must be non-negative integers, got (${sp.nAlpha}, ${sp.nBeta}).`,
        );
      }
      superAlpha += sp.nAlpha;
      superBeta += sp.nBeta;
    }
  }

  let allConverged = true;

  /** Run the chosen method on a set of atoms and return the total
   *  energy (E_HF [+ correlation] for closed-shell; E_UHF [+ UCCSD]
   *  for open-shell). For open-shell methods, `nAlpha` / `nBeta`
   *  override the auto-computed neutral closed-shell count.
   *  Sets allConverged = false on any failure. */
  function singlePoint(
    atomsIn: readonly Atom[],
    nAlpha?: number,
    nBeta?: number,
  ): number {
    const sh = moleculeToShellsNuclei(atomsIn, basis);
    const integ = computeMolecularIntegrals(sh.shells, sh.nuclei);

    if (isOpenShell) {
      if (nAlpha === undefined || nBeta === undefined) {
        throw new Error(`runCounterpoise: open-shell singlePoint requires explicit (nAlpha, nBeta)`);
      }
      const uhf = runUHFSCF(integ, nAlpha, nBeta, uhfOpts);
      if (!uhf.converged) allConverged = false;
      if (method === "uhf") return uhf.energy;
      // method === "uccsd". UCCSD with 0 occupied SOs (trivial fragment
      // = no electrons left) or NVIRT_per_spin = 0 has no correlation;
      // runUCCSD throws on those degenerate cases. Skip in that case.
      const totalOcc = uhf.nAlpha + uhf.nBeta;
      const totalVirt = 2 * integ.n - totalOcc;
      if (totalOcc === 0 || totalVirt === 0) return uhf.energy;
      const uccsd = runUCCSD(uhf, integ, ccsdOpts);
      if (!uccsd.converged) allConverged = false;
      return uhf.energy + uccsd.correlationEnergy;
    }

    const hf = runRHFSCF(integ, sh.nElectrons, hfOpts);
    if (!hf.converged) allConverged = false;
    if (method === "hf") return hf.energy;
    if (method === "mp2") {
      const mp2 = runMP2(hf, integ);
      return hf.energy + mp2.correlationEnergy;
    }
    // method === "ccsd"
    const ccsd = runCCSD(hf, integ, ccsdOpts);
    if (!ccsd.converged) allConverged = false;
    return hf.energy + ccsd.correlationEnergy;
  }

  // ── Supermolecule. ──
  const superEnergy = isOpenShell
    ? singlePoint(atoms, superAlpha, superBeta)
    : singlePoint(atoms);

  // ── Per-fragment runs. ──
  const fragmentEnergiesCP: number[] = [];
  const fragmentEnergiesBare: number[] = [];

  for (const f of fragments) {
    const fragmentSet = new Set(f.atomIndices);
    const nA = f.spin?.nAlpha;
    const nB = f.spin?.nBeta;
    // a) Fragment in dimer basis: all atoms present, but only this
    //    fragment's atoms keep their nuclei + electrons. Others are
    //    ghosts (Z=0, no electrons, basis functions present).
    const dimerBasisAtoms: Atom[] = atoms.map((a, i) =>
      fragmentSet.has(i) ? a : { ...a, ghost: true },
    );
    fragmentEnergiesCP.push(singlePoint(dimerBasisAtoms, nA, nB));
    // b) Fragment in bare monomer basis: only this fragment's atoms exist.
    fragmentEnergiesBare.push(singlePoint(atoms.filter((_, i) => fragmentSet.has(i)), nA, nB));
  }

  let sumCP = 0;
  let sumBare = 0;
  for (let k = 0; k < fragments.length; k++) {
    sumCP += fragmentEnergiesCP[k]!;
    sumBare += fragmentEnergiesBare[k]!;
  }
  const interactionEnergy = superEnergy - sumBare;
  const interactionEnergyCP = superEnergy - sumCP;
  // BSSE = ΔE_CP − ΔE = Σ (E_bare − E_CP). Variationally ≥ 0 for HF;
  // for MP2/CCSD the same sign holds asymptotically (variational bound
  // on E_HF dominates the small post-HF correction).
  const bsseCorrection = interactionEnergyCP - interactionEnergy;

  return {
    supermoleculeEnergy: superEnergy,
    fragmentEnergiesCP,
    fragmentEnergiesBare,
    interactionEnergy,
    interactionEnergyCP,
    bsseCorrection,
    allConverged,
  };
}
