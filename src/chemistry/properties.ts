// ─────────────────────────────────────────────────────────────
// properties.ts — molecular property evaluations on top of an
// SCF result. Tier 2 stage 11.
//
// Scope: ground-state observables that follow from the converged
// AO density matrix P + the nuclear positions:
//
//   • Electric dipole moment (closed-shell, atomic units).
//
// Every routine is reference-agnostic — `P` is whatever AO
// density the caller wants the property of (HF, DFT, post-HF,
// CCSD relaxed, ...). For HF / DFT, just pass `result.D` from
// `runRHFSCF` / `runRKSDFT`. The closed-shell convention here
// is P_μν = 2·Σ_{i ∈ occ} C_μi C_νi (i.e. the density matrix
// already carries the factor of 2 for double occupation, which
// matches what `runRHFSCF` returns).
// ─────────────────────────────────────────────────────────────

import { dipole_cg } from "./integrals-cg.js";
import type { MolecularIntegrals } from "./cg-molecular.js";

/** 1 atomic unit (e·Bohr) of dipole = this many Debye. */
export const AU_TO_DEBYE = 2.541746229;

/**
 * Ground-state electric dipole moment in atomic units (e·Bohr):
 *
 *   μ = − ⟨ψ | r̂ | ψ⟩ + Σ_A Z_A · R_A
 *     = − Σ_{μν} P_μν · ⟨χ_μ | r̂ | χ_ν⟩ + Σ_A Z_A · R_A
 *
 * Sign convention: the electronic density carries negative charge,
 * so the electronic contribution is `−Σ_μν P_μν · μ_AO_μν`, and
 * the nuclear contribution is `+Σ_A Z_A · R_A`. Returns the
 * 3-vector `[μ_x, μ_y, μ_z]`. Magnitude is `√(μ_x² + μ_y² + μ_z²)`.
 *
 * Convert to Debye by multiplying by `AU_TO_DEBYE`.
 */
export function dipoleMoment(
  integrals: MolecularIntegrals,
  P: Float64Array,
): [number, number, number] {
  const n = integrals.n;
  const shells = integrals.shells;
  const nuclei = integrals.nuclei;
  const out: [number, number, number] = [0, 0, 0];

  // Electronic contribution: −Σ P_μν · ⟨μ|r_axis|ν⟩.
  // Build the dipole AO integral on the fly per pair (small n).
  for (let axis = 0 as 0 | 1 | 2; axis < 3; axis++) {
    let sumE = 0;
    for (let mu = 0; mu < n; mu++) {
      for (let nu = 0; nu < n; nu++) {
        sumE += P[mu * n + nu]! * dipole_cg(shells[mu]!, shells[nu]!, axis);
      }
    }
    out[axis] = -sumE;
  }

  // Nuclear contribution: +Σ_A Z_A · R_A.
  for (const { Z, pos } of nuclei) {
    out[0] += Z * pos[0];
    out[1] += Z * pos[1];
    out[2] += Z * pos[2];
  }

  return out;
}

/** Magnitude of a 3-vector dipole in the input units. */
export function dipoleMagnitude(mu: readonly [number, number, number]): number {
  return Math.sqrt(mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]);
}

// ─────────────────────────────────────────────────────────────
// Mulliken population analysis (Mulliken 1955).
//
// Partition the total electron count into per-atom populations
// using the AO basis set's "natural" assignment (shells anchored
// to atoms):
//
//   gross population on A:  n_A = Σ_{μ on A} (P · S)_μμ
//   Mulliken charge on A:   q_A = Z_A − n_A
//
// where P is the AO density matrix and S is the AO overlap.
// Σ_A q_A = total molecular charge (= 0 for neutrals) by trace
// invariance: Σ_μ (P·S)_μμ = tr(P·S) = N_electrons.
//
// Mulliken's well-known limitations:
//   • Basis-dependent: charges shift when you change basis sets,
//     sometimes drastically (especially with diffuse functions).
//   • For 50/50-shared overlap regions, the equal split is
//     somewhat arbitrary.
//   • Other schemes (NPA, CHELPG, RESP, Bader) address these but
//     need substantial new infrastructure. Mulliken is the
//     historically-canonical first cut and matches what every
//     introductory textbook reports.
// ─────────────────────────────────────────────────────────────

/**
 * Mulliken atomic charges (atomic units of e). Returns a
 * Float64Array of length nAtoms. Sum of charges equals the total
 * molecular charge (0 for neutrals).
 *
 * @param shellAtomIdx Atom index that owns each shell — same array
 *   returned by `moleculeToShellsNuclei`. Required to attribute
 *   AO populations to atoms.
 */
export function mullikenCharges(
  integrals: MolecularIntegrals,
  P: Float64Array,
  shellAtomIdx: readonly number[],
): Float64Array {
  const n = integrals.n;
  const S = integrals.S_AO;
  const nuclei = integrals.nuclei;
  const nAtoms = nuclei.length;
  if (shellAtomIdx.length !== n) {
    throw new Error(
      `mullikenCharges: shellAtomIdx length ${shellAtomIdx.length} ≠ n ${n}`,
    );
  }

  // (P·S)_μμ = Σ_ν P_μν · S_νμ = Σ_ν P_μν · S_μν  (S symmetric).
  const charges = new Float64Array(nAtoms);
  for (let mu = 0; mu < n; mu++) {
    let pop = 0;
    for (let nu = 0; nu < n; nu++) {
      pop += P[mu * n + nu]! * S[nu * n + mu]!;
    }
    charges[shellAtomIdx[mu]!]! += pop;
  }
  // Convert population → charge: q_A = Z_A − n_A.
  for (let A = 0; A < nAtoms; A++) {
    charges[A] = nuclei[A]!.Z - charges[A]!;
  }
  return charges;
}
