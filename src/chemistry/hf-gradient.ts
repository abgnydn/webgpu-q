// ─────────────────────────────────────────────────────────────
// hf-gradient.ts — analytical Hartree-Fock energy gradient
// ∂E_HF/∂R_N for every nucleus N in the molecule. Tier 2 stage 5.
//
// Pulay 1969 (CP 17, 197): with the SCF density P held fixed and
// the orbital response captured via the energy-weighted density
// matrix W, the gradient is
//
//   ∂E/∂R_N =   Σ_μν P_μν · ∂h_μν/∂R_N
//             + (1/2) Σ_μνλσ P_μν P_λσ · ∂(μν|λσ)/∂R_N
//             − (1/4) Σ_μνλσ P_μν P_λσ · ∂(μλ|νσ)/∂R_N
//             − Σ_μν W_μν · ∂S_μν/∂R_N
//             + ∂V_NN/∂R_N
//
// with h = T + Σ_C V_C, P = 2·Σ_{i ∈ occ} C_μi C_νi (closed-shell
// AO density), and W = 2·Σ_{i ∈ occ} ε_i C_μi C_νi (energy-weighted
// AO density in the AO basis).
//
// Each AO χ_μ is anchored on a single atom. When nucleus N moves:
//   • Shell μ contributes only if `shellAtomIdx[μ] = N` (then its
//     center derivative is "∂/∂A" of the integral).
//   • Nuclear-attraction operator C contributes only if the
//     Hellmann-Feynman site C is itself the moving nucleus N
//     (then "∂/∂C" of V_C — recovered from translational
//     invariance: ∂A + ∂B + ∂C = 0).
//   • The two-particle ERI has four AO centers; one comes from
//     translational invariance.
//
// Gradient is returned in atomic units (Hartree per Bohr). Caller
// can convert to Hartree/Å by multiplying by `ANGSTROM_TO_BOHR`
// (since dE/d(R_in_Å) = dE/d(R_in_Bohr) · dR_Bohr/dR_Å).
// ─────────────────────────────────────────────────────────────

import {
  type CGShell,
  S_cg, T_cg, V_cg, ERI_cg,
  dS_cg_dA, dT_cg_dA, dV_cg_dA, dERI_cg_dX,
} from "./integrals-cg.js";
import type { MolecularIntegrals, Nucleus } from "./cg-molecular.js";

export interface HFGradientInputs {
  /** Same shells / nuclei used to build the SCF integrals. */
  readonly shells: readonly CGShell[];
  readonly nuclei: readonly Nucleus[];
  /** Atom index for each shell — see `moleculeToShellsNuclei`. */
  readonly shellAtomIdx: readonly number[];
  /** SCF AO density matrix (n × n, row-major). For closed-shell RHF:
   *  P_μν = 2·Σ_{i ∈ occ} C_μi C_νi. */
  readonly P: Float64Array;
  /** Energy-weighted AO density matrix (n × n, row-major):
   *  W_μν = 2·Σ_{i ∈ occ} ε_i C_μi C_νi. */
  readonly W: Float64Array;
}

/**
 * Build the energy-weighted AO density matrix
 *   W_μν = 2·Σ_{i ∈ occ} ε_i · C_μi C_νi
 * from MO coefficients (column-MO row-AO convention) and orbital
 * energies. `nOcc` is the number of doubly-occupied spatial MOs.
 */
export function buildEnergyWeightedDensity(
  C_MO: Float64Array, eps: Float64Array, nOcc: number, n: number,
): Float64Array {
  const W = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let i = 0; i < nOcc; i++) {
        s += eps[i]! * C_MO[mu * n + i]! * C_MO[nu * n + i]!;
      }
      W[mu * n + nu] = 2 * s;
    }
  }
  return W;
}

/**
 * Compute the analytical HF energy gradient (Hartree / Bohr) for every
 * atom, returned as a (3·nAtoms) Float64Array, ordered as
 *   [dE/dR_0_x, dE/dR_0_y, dE/dR_0_z, dE/dR_1_x, …].
 *
 * Cost: dominated by ERI derivative loop, O(n^4) shell quartets each
 * with three primitive-quartet derivative integrals (∂A, ∂B, ∂C);
 * ∂D recovered from translational invariance. For STO-3G H₂O
 * (n=7) this is ~10× the cost of one HF energy build.
 */
export function hfGradient(inp: HFGradientInputs): Float64Array {
  const { shells, nuclei, shellAtomIdx, P, W } = inp;
  const n = shells.length;
  const nAtoms = nuclei.length;
  const grad = new Float64Array(3 * nAtoms);

  // ── 1-electron contributions: T, V, S ──────────────────────
  // Loop over UNIQUE pairs (μ, ν), using S, h symmetry: contribution
  // is (P_μν · ∂h_μν − W_μν · ∂S_μν) summed; the (μ ≠ ν) factor of 2
  // is absorbed by upper-triangular iteration with a 2× multiplier.
  for (let mu = 0; mu < n; mu++) {
    const aMu = shellAtomIdx[mu]!;
    for (let nu = 0; nu < n; nu++) {
      const aNu = shellAtomIdx[nu]!;
      const Pmn = P[mu * n + nu]!;
      const Wmn = W[mu * n + nu]!;
      if (Pmn === 0 && Wmn === 0) continue;

      // dS/dA, dT/dA per axis. Translational invariance gives
      // dS/dB = −dS/dA, dT/dB = −dT/dA — atoms move independently
      // so we apply both contributions separately to grad[aMu] and
      // grad[aNu].
      for (let axis = 0; axis < 3; axis++) {
        const dS_dA = dS_cg_dA(shells[mu]!, shells[nu]!, axis as 0 | 1 | 2);
        const dT_dA = dT_cg_dA(shells[mu]!, shells[nu]!, axis as 0 | 1 | 2);

        // h-derivative bra-side: ∂T_μν/∂R(aMu) + Σ_C ∂V_μν^C/∂R(aMu).
        let dh_dA = dT_dA;
        // V_C derivative: bra-side contribution to atom owning shell μ.
        for (const { Z, pos } of nuclei) {
          dh_dA += dV_cg_dA(shells[mu]!, shells[nu]!, axis as 0 | 1 | 2, Z, pos);
        }
        grad[aMu * 3 + axis]! += Pmn * dh_dA - Wmn * dS_dA;
        // Ket-side contribution: dS/dB = −dS/dA, dT/dB = −dT/dA, but
        // V_μν^C(B) is computed independently by swapping bra/ket roles
        // (i.e. dV/dB of (μν;C) = dV/dA of (νμ;C)). With S, T this is
        // also = bra(νμ): so we just use the same routines on swapped
        // shells, which automatically accounts for sign.
        const dS_dB = dS_cg_dA(shells[nu]!, shells[mu]!, axis as 0 | 1 | 2);
        const dT_dB = dT_cg_dA(shells[nu]!, shells[mu]!, axis as 0 | 1 | 2);
        let dh_dB = dT_dB;
        for (const { Z, pos } of nuclei) {
          dh_dB += dV_cg_dA(shells[nu]!, shells[mu]!, axis as 0 | 1 | 2, Z, pos);
        }
        grad[aNu * 3 + axis]! += Pmn * dh_dB - Wmn * dS_dB;

        // V_C operator-side derivative: when C is itself a moving
        // nucleus, the integral changes through the Hellmann-Feynman
        // site. Recovered via translational invariance:
        //   ∂V/∂C = − (∂V/∂A + ∂V/∂B)
        // for each (μ, ν, C) triple, attributed to atom owning C.
        for (let cIdx = 0; cIdx < nAtoms; cIdx++) {
          const { Z, pos } = nuclei[cIdx]!;
          const dV_dA = dV_cg_dA(shells[mu]!, shells[nu]!, axis as 0 | 1 | 2, Z, pos);
          const dV_dB = dV_cg_dA(shells[nu]!, shells[mu]!, axis as 0 | 1 | 2, Z, pos);
          const dV_dC = -(dV_dA + dV_dB);
          grad[cIdx * 3 + axis]! += Pmn * dV_dC;
        }
      }
    }
  }

  // ── 2-electron ERI contributions ───────────────────────────
  // E_J = (1/2) Σ_μνλσ P_μν P_λσ (μν|λσ)
  // E_K = (1/4) Σ_μνλσ P_μν P_λσ (μλ|νσ)
  // d/dR(N) acts via shell μ (=A) and λ (=C in chemist notation
  // (μν|λσ), where the "C" of dERI_cg_dX is the bra of the second
  // electron — DON'T confuse with V_C). Each shell µ in the
  // 4-center ERI quartet picks up its own atom's gradient slot.
  for (let mu = 0; mu < n; mu++) {
    const aMu = shellAtomIdx[mu]!;
    for (let nu = 0; nu < n; nu++) {
      const aNu = shellAtomIdx[nu]!;
      for (let la = 0; la < n; la++) {
        const aLa = shellAtomIdx[la]!;
        for (let si = 0; si < n; si++) {
          const aSi = shellAtomIdx[si]!;
          // Two ERIs needed per quartet: J piece (μν|λσ) and K piece (μλ|νσ).
          // P_μν P_λσ for J; P_μν P_λσ for K (same product structure but
          // index permutation on the integral). Build coefficients first.
          const Pmn = P[mu * n + nu]!;
          const Pls = P[la * n + si]!;
          const cJ = 0.5 * Pmn * Pls;

          // Exchange uses (μλ|νσ) — same density product but index-
          // permuted on the integral side. Coefficient is −1/4·P_μν·P_λσ.
          const cK = -0.25 * Pmn * Pls;

          if (cJ === 0 && cK === 0) continue;

          for (let axis = 0; axis < 3; axis++) {
            // J-piece derivatives (μν|λσ).
            const dJ_dA = dERI_cg_dX(shells[mu]!, shells[nu]!, shells[la]!, shells[si]!, axis as 0 | 1 | 2, "A");
            const dJ_dB = dERI_cg_dX(shells[mu]!, shells[nu]!, shells[la]!, shells[si]!, axis as 0 | 1 | 2, "B");
            const dJ_dC = dERI_cg_dX(shells[mu]!, shells[nu]!, shells[la]!, shells[si]!, axis as 0 | 1 | 2, "C");
            const dJ_dD = -(dJ_dA + dJ_dB + dJ_dC);
            grad[aMu * 3 + axis]! += cJ * dJ_dA;
            grad[aNu * 3 + axis]! += cJ * dJ_dB;
            grad[aLa * 3 + axis]! += cJ * dJ_dC;
            grad[aSi * 3 + axis]! += cJ * dJ_dD;

            // K-piece derivatives (μλ|νσ). Note shell ordering changes:
            // μ → A, λ → B, ν → C, σ → D. Atom assignments follow.
            const dK_dA = dERI_cg_dX(shells[mu]!, shells[la]!, shells[nu]!, shells[si]!, axis as 0 | 1 | 2, "A");
            const dK_dB = dERI_cg_dX(shells[mu]!, shells[la]!, shells[nu]!, shells[si]!, axis as 0 | 1 | 2, "B");
            const dK_dC = dERI_cg_dX(shells[mu]!, shells[la]!, shells[nu]!, shells[si]!, axis as 0 | 1 | 2, "C");
            const dK_dD = -(dK_dA + dK_dB + dK_dC);
            grad[aMu * 3 + axis]! += cK * dK_dA;
            grad[aLa * 3 + axis]! += cK * dK_dB;
            grad[aNu * 3 + axis]! += cK * dK_dC;
            grad[aSi * 3 + axis]! += cK * dK_dD;
          }
        }
      }
    }
  }

  // ── Nuclear-nuclear repulsion derivative ───────────────────
  // V_NN = Σ_{i<j} Z_i Z_j / |R_i − R_j|
  // dV_NN/dR_N = Σ_{j ≠ N} Z_N Z_j · (R_j − R_N) / |R_N − R_j|³
  for (let i = 0; i < nAtoms; i++) {
    for (let j = 0; j < nAtoms; j++) {
      if (i === j) continue;
      const Ri = nuclei[i]!.pos, Rj = nuclei[j]!.pos;
      const Zi = nuclei[i]!.Z,   Zj = nuclei[j]!.Z;
      const dx = Ri[0] - Rj[0], dy = Ri[1] - Rj[1], dz = Ri[2] - Rj[2];
      const r2 = dx * dx + dy * dy + dz * dz;
      const r3 = r2 * Math.sqrt(r2);
      const k = -Zi * Zj / r3;
      grad[i * 3 + 0]! += k * dx;
      grad[i * 3 + 1]! += k * dy;
      grad[i * 3 + 2]! += k * dz;
    }
  }

  return grad;
}

/**
 * Convenience: silence the unused-import warning for the un-derivative
 * versions that the gradient tests cross-check against.
 */
void S_cg; void T_cg; void V_cg; void ERI_cg;
void ({} as MolecularIntegrals);
