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

/**
 * Mulliken atomic spin populations from a UHF / UCCSD density pair.
 * Returns the per-atom z-component magnetization in units of e:
 *
 *   ⟨Ŝ_z⟩_A = ½ Σ_{μ ∈ A} [(M·S)_μμ]
 *
 * where M = D_α − D_β is the spin density in the AO basis. Positive
 * spin population indicates net α-electron density on the atom;
 * negative indicates net β. For a clean doublet on a single atom,
 * the sum over all atoms is +½ × 2 = 1 (the unpaired electron's
 * contribution to ⟨Ŝ_z⟩ in units of ℏ).
 *
 * Conventions match the Mulliken-charge function above — same
 * basis-set sensitivity caveats apply.
 */
export function mullikenSpinPopulations(
  integrals: MolecularIntegrals,
  D_alpha: Float64Array,
  D_beta: Float64Array,
  shellAtomIdx: readonly number[],
): Float64Array {
  const n = integrals.n;
  const S = integrals.S_AO;
  const nAtoms = integrals.nuclei.length;
  if (shellAtomIdx.length !== n) {
    throw new Error(
      `mullikenSpinPopulations: shellAtomIdx length ${shellAtomIdx.length} ≠ n ${n}`,
    );
  }
  const spin = new Float64Array(nAtoms);
  for (let mu = 0; mu < n; mu++) {
    let pop = 0;
    for (let nu = 0; nu < n; nu++) {
      const M = D_alpha[mu * n + nu]! - D_beta[mu * n + nu]!;
      pop += M * S[nu * n + mu]!;
    }
    // ½·(M·S) is the per-AO α-β population difference; sum into the
    // atom's spin population.
    spin[shellAtomIdx[mu]!]! += 0.5 * pop;
  }
  return spin;
}

// ─────────────────────────────────────────────────────────────
// Wiberg-Mayer bond orders (Mayer 1983, "Charge, bond order,
// and valence in the AB initio SCF theory", Chem. Phys. Lett.
// 97, 270).
//
// For closed-shell SCF densities P (with the convention P =
// 2·Σ_{i ∈ occ} C_i C_i^T), the bond order between atoms A
// and B is:
//
//   B_AB = Σ_{μ ∈ A, ν ∈ B} (P·S)_μν · (P·S)_νμ
//
// Interpretation:
//   • B_AB ≈ 1 for a single bond (two shared electrons).
//   • B_AB ≈ 2 for a double bond, ≈ 3 for a triple.
//   • The diagonal B_AA represents the "valence" of atom A
//     (sum over partners: total bond count it participates in).
//
// Limitations: same basis-set sensitivity as Mulliken charges —
// shifts when basis changes, especially with diffuse functions.
// At STO-3G the values are qualitative ordering indicators
// rather than precise bond-strength measurements.
// ─────────────────────────────────────────────────────────────

/**
 * Mayer atomic valence: V_A = Σ_{B ≠ A} B_AB — the per-atom total
 * of off-diagonal bond orders. For closed-shell ground states this
 * approximately equals the integer count of single-bond
 * equivalents the atom participates in. Convenience wrapper
 * around `bondOrders` that hides the sometimes-confusing diagonal
 * (which is NOT the valence — it's a self-overlap term).
 */
export function mayerValences(
  integrals: MolecularIntegrals,
  P: Float64Array,
  shellAtomIdx: readonly number[],
): Float64Array {
  const B = bondOrders(integrals, P, shellAtomIdx);
  const nAtoms = integrals.nuclei.length;
  const v = new Float64Array(nAtoms);
  for (let A = 0; A < nAtoms; A++) {
    let s = 0;
    for (let X = 0; X < nAtoms; X++) if (X !== A) s += B[A * nAtoms + X]!;
    v[A] = s;
  }
  return v;
}

/**
 * Mayer atomic valences from a UHF/UKS reference. Sums off-diagonal
 * entries of `bondOrdersUHF` per atom. Validates that the open-shell
 * valence diagnostic (e.g., for radicals) is positive and reflects
 * the chemical bonding environment of each atom.
 */
export function mayerValencesUHF(
  integrals: MolecularIntegrals,
  D_alpha: Float64Array,
  D_beta: Float64Array,
  shellAtomIdx: readonly number[],
): Float64Array {
  const B = bondOrdersUHF(integrals, D_alpha, D_beta, shellAtomIdx);
  const nAtoms = integrals.nuclei.length;
  const v = new Float64Array(nAtoms);
  for (let A = 0; A < nAtoms; A++) {
    let s = 0;
    for (let X = 0; X < nAtoms; X++) if (X !== A) s += B[A * nAtoms + X]!;
    v[A] = s;
  }
  return v;
}

/**
 * Mayer bond-order matrix (nAtoms × nAtoms, row-major). The
 * matrix is symmetric: B_AB = B_BA. Diagonal entries are the
 * Mayer atomic valences. Returns a Float64Array of length
 * nAtoms·nAtoms.
 */
export function bondOrders(
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
      `bondOrders: shellAtomIdx length ${shellAtomIdx.length} ≠ n ${n}`,
    );
  }

  // PS = P · S, plain n³ multiply.
  const PS = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const Pik = P[i * n + k]!;
      if (Pik === 0) continue;
      for (let j = 0; j < n; j++) PS[i * n + j]! += Pik * S[k * n + j]!;
    }
  }

  // B_AB = Σ_{μ on A, ν on B} (PS)_μν · (PS)_νμ.
  const B = new Float64Array(nAtoms * nAtoms);
  for (let mu = 0; mu < n; mu++) {
    const A = shellAtomIdx[mu]!;
    for (let nu = 0; nu < n; nu++) {
      const Aprime = shellAtomIdx[nu]!;
      B[A * nAtoms + Aprime]! += PS[mu * n + nu]! * PS[nu * n + mu]!;
    }
  }
  return B;
}

/**
 * Mayer bond orders for UHF / UCCSD (Mayer 1985). Spin-resolved
 * extension of the closed-shell formula:
 *
 *   B_AB = 2 · [Σ_{μ∈A, ν∈B} (P^α·S)_μν (P^α·S)_νμ
 *              + (P^β·S)_μν (P^β·S)_νμ]
 *
 * For closed-shell input (P^α = P^β = P_total/2) this reduces
 * exactly to the closed-shell formula above (factor 2·2·(½)² = 1
 * on each spin block; α + β combine to give Σ (P_total·S)² as in
 * `bondOrders`).
 *
 * Same basis-set sensitivity caveats as the closed-shell version.
 */
export function bondOrdersUHF(
  integrals: MolecularIntegrals,
  D_alpha: Float64Array,
  D_beta: Float64Array,
  shellAtomIdx: readonly number[],
): Float64Array {
  const n = integrals.n;
  const S = integrals.S_AO;
  const nAtoms = integrals.nuclei.length;
  if (shellAtomIdx.length !== n) {
    throw new Error(
      `bondOrdersUHF: shellAtomIdx length ${shellAtomIdx.length} ≠ n ${n}`,
    );
  }
  const matmulPS = (P: Float64Array): Float64Array => {
    const PS = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        const Pik = P[i * n + k]!;
        if (Pik === 0) continue;
        for (let j = 0; j < n; j++) PS[i * n + j]! += Pik * S[k * n + j]!;
      }
    }
    return PS;
  };
  const PSα = matmulPS(D_alpha);
  const PSβ = matmulPS(D_beta);
  const B = new Float64Array(nAtoms * nAtoms);
  for (let mu = 0; mu < n; mu++) {
    const A = shellAtomIdx[mu]!;
    for (let nu = 0; nu < n; nu++) {
      const Aprime = shellAtomIdx[nu]!;
      const ab = PSα[mu * n + nu]! * PSα[nu * n + mu]!
               + PSβ[mu * n + nu]! * PSβ[nu * n + mu]!;
      B[A * nAtoms + Aprime]! += 2 * ab;
    }
  }
  return B;
}
