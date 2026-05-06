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

/** Schwarz screening tolerance — same as `cg-molecular.ts`. */
const SCHWARZ_TOL = 1e-10;

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
  /** Fraction of HF exact exchange in the energy expression.
   *  - 1.0 for pure HF (default)
   *  - 0.0 for pure DFT (LDA, GGA)
   *  - 0.20 for B3LYP-style hybrids
   *  Scales the K-coupling −(1/4)·hfMix·P_μλ·P_νσ inside the
   *  ERI 8-fold canonical loop. */
  readonly kFactor?: number;
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
 * Optimizations (Tier 2 stage 5b):
 *   • 1-electron loop: μ ≤ ν only, ×2 for off-diagonal.
 *   • ERI 8-fold canonical loop: iterate (μ ≥ ν, λ ≥ σ, (μν) ≥ (λσ)),
 *     applying the J + K coupling sum across all 8 permutations as
 *     one Γ-coefficient. Each canonical quartet costs ONE set of
 *     three derivative ERIs (∂A, ∂B, ∂C; ∂D from invariance) instead
 *     of six (J + K paths separately).
 *   • Schwarz screening: skip canonical quartets with
 *     Q_μν · Q_λσ · |coupling| below SCHWARZ_TOL.
 */
export function hfGradient(inp: HFGradientInputs): Float64Array {
  const { shells, nuclei, shellAtomIdx, P, W } = inp;
  const kFactor = inp.kFactor ?? 1.0;
  const n = shells.length;
  const nAtoms = nuclei.length;
  const grad = new Float64Array(3 * nAtoms);

  // ── 1-electron contributions: T, V, S ──────────────────────
  // Iterate over UNIQUE pairs (μ ≥ ν) and absorb the (μ, ν) ↔ (ν, μ)
  // duplication into a 2× multiplier on the off-diagonal pairs. On
  // the diagonal both bra-side and ket-side derivatives go to the
  // SAME atom (atom of shell μ), so we still add BOTH contributions
  // — there is no "ket-side that would have come from the (ν, μ)
  // iteration" double-count to worry about, because (μ, μ) is its
  // own swap.
  for (let mu = 0; mu < n; mu++) {
    const aMu = shellAtomIdx[mu]!;
    for (let nu = 0; nu <= mu; nu++) {
      const aNu = shellAtomIdx[nu]!;
      const sym = mu === nu ? 1 : 2;
      const Pmn = P[mu * n + nu]! * sym;
      const Wmn = W[mu * n + nu]! * sym;
      if (Pmn === 0 && Wmn === 0) continue;

      for (let axis = 0; axis < 3; axis++) {
        // Bra-side derivative: ∂/∂R(atom of shell μ) = ∂/∂A.
        const dS_dA = dS_cg_dA(shells[mu]!, shells[nu]!, axis as 0 | 1 | 2);
        const dT_dA = dT_cg_dA(shells[mu]!, shells[nu]!, axis as 0 | 1 | 2);
        let dh_dA = dT_dA;
        for (const { Z, pos } of nuclei) {
          dh_dA += dV_cg_dA(shells[mu]!, shells[nu]!, axis as 0 | 1 | 2, Z, pos);
        }
        grad[aMu * 3 + axis]! += Pmn * dh_dA - Wmn * dS_dA;

        // Ket-side derivative: ∂/∂R(atom of shell ν) = ∂/∂B.
        // Computed by swapping bra/ket (dS/dB(μ,ν) = dS/dA(ν,μ)).
        // Goes to atom(ν) — which on the diagonal IS atom(μ), so the
        // contribution simply sums into the same gradient slot.
        const dS_dB = dS_cg_dA(shells[nu]!, shells[mu]!, axis as 0 | 1 | 2);
        const dT_dB = dT_cg_dA(shells[nu]!, shells[mu]!, axis as 0 | 1 | 2);
        let dh_dB = dT_dB;
        for (const { Z, pos } of nuclei) {
          dh_dB += dV_cg_dA(shells[nu]!, shells[mu]!, axis as 0 | 1 | 2, Z, pos);
        }
        grad[aNu * 3 + axis]! += Pmn * dh_dB - Wmn * dS_dB;

        // V_C operator-side derivative: ∂V/∂C = − (∂V/∂A + ∂V/∂B)
        // by translational invariance, attributed to atom owning C.
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

  // ── 2-electron ERI contributions: 8-fold canonical loop ─────
  // For each canonical (μ ≥ ν, λ ≥ σ, (μν) ≥ (λσ)), the 8
  // ERI-symmetric permutations all share the same integral value
  // and share the SAME slot-by-slot derivative ∂(μν|λσ)/∂R_atom(slot).
  // What varies across permutations is the P-coupling. We sum the
  // 8 couplings (deduplicated for low-multiplicity canonicals) into
  // a single Γ-coefficient and then apply ONE set of derivative
  // contributions per atom slot.
  //
  // The coupling per ordered (a, b, c, d) tuple is
  //    (1/2)·P_ab·P_cd  (J)  −  (1/4)·P_ac·P_bd  (K).
  //
  // Cost: O(n^4 / 8) canonical quartets, each with 3 derivative ERIs
  // (∂A, ∂B, ∂C), instead of n^4 quartets with 6 derivative ERIs in
  // the naive version — 16× fewer integral derivative evaluations
  // before Schwarz screening.

  // Schwarz Q_μν = √|⟨μν|μν⟩| precomputed once (O(n²) ERI evals).
  const Q = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu <= mu; nu++) {
      const q = Math.sqrt(Math.abs(ERI_cg(shells[mu]!, shells[nu]!, shells[mu]!, shells[nu]!)));
      Q[mu * n + nu] = q;
      Q[nu * n + mu] = q;
    }
  }

  // Scratch for the 8 permutations of (μ, ν, λ, σ).
  const permBuf: [number, number, number, number][] = new Array(8) as never;
  for (let i = 0; i < 8; i++) permBuf[i] = [0, 0, 0, 0];

  for (let mu = 0; mu < n; mu++) {
    const aMu = shellAtomIdx[mu]!;
    for (let nu = 0; nu <= mu; nu++) {
      const aNu = shellAtomIdx[nu]!;
      const munu = mu * n + nu;
      const Qmn = Q[munu]!;
      for (let la = 0; la < n; la++) {
        const aLa = shellAtomIdx[la]!;
        for (let si = 0; si <= la; si++) {
          const aSi = shellAtomIdx[si]!;
          const lasi = la * n + si;
          if (lasi > munu) continue;          // (μν) ≥ (λσ) canonical bra-ket
          const Qls = Q[lasi]!;

          // Build the Γ-coupling: sum over the (deduplicated) 8
          // permutations of (μ,ν,λ,σ) of (1/2)·P_ab P_cd − (1/4)·P_ac P_bd.
          // For symmetric P this collapses to a closed form by mult, but
          // the explicit sum over deduped tuples handles every special
          // case (μ=ν, λ=σ, (μν)=(λσ), and combinations) without case
          // analysis.
          permBuf[0]![0] = mu; permBuf[0]![1] = nu; permBuf[0]![2] = la; permBuf[0]![3] = si;
          permBuf[1]![0] = nu; permBuf[1]![1] = mu; permBuf[1]![2] = la; permBuf[1]![3] = si;
          permBuf[2]![0] = mu; permBuf[2]![1] = nu; permBuf[2]![2] = si; permBuf[2]![3] = la;
          permBuf[3]![0] = nu; permBuf[3]![1] = mu; permBuf[3]![2] = si; permBuf[3]![3] = la;
          permBuf[4]![0] = la; permBuf[4]![1] = si; permBuf[4]![2] = mu; permBuf[4]![3] = nu;
          permBuf[5]![0] = si; permBuf[5]![1] = la; permBuf[5]![2] = mu; permBuf[5]![3] = nu;
          permBuf[6]![0] = la; permBuf[6]![1] = si; permBuf[6]![2] = nu; permBuf[6]![3] = mu;
          permBuf[7]![0] = si; permBuf[7]![1] = la; permBuf[7]![2] = nu; permBuf[7]![3] = mu;
          let coef = 0;
          for (let i = 0; i < 8; i++) {
            // Deduplicate against previously seen tuples.
            let dup = false;
            const pi = permBuf[i]!;
            for (let j = 0; j < i; j++) {
              const pj = permBuf[j]!;
              if (pi[0] === pj[0] && pi[1] === pj[1] && pi[2] === pj[2] && pi[3] === pj[3]) {
                dup = true; break;
              }
            }
            if (dup) continue;
            const a = pi[0], b = pi[1], c = pi[2], d = pi[3];
            coef += 0.5 * P[a * n + b]! * P[c * n + d]!
                  - 0.25 * kFactor * P[a * n + c]! * P[b * n + d]!;
          }
          if (coef === 0) continue;
          // Schwarz: skip if Q_μν·Q_λσ·|coef| is below the integral-
          // precision floor. The 1e-10 threshold matches the energy
          // build's screening.
          if (Qmn * Qls * Math.abs(coef) < SCHWARZ_TOL) continue;

          for (let axis = 0; axis < 3; axis++) {
            const dA = dERI_cg_dX(shells[mu]!, shells[nu]!, shells[la]!, shells[si]!, axis as 0 | 1 | 2, "A");
            const dB = dERI_cg_dX(shells[mu]!, shells[nu]!, shells[la]!, shells[si]!, axis as 0 | 1 | 2, "B");
            const dC = dERI_cg_dX(shells[mu]!, shells[nu]!, shells[la]!, shells[si]!, axis as 0 | 1 | 2, "C");
            const dD = -(dA + dB + dC);
            grad[aMu * 3 + axis]! += coef * dA;
            grad[aNu * 3 + axis]! += coef * dB;
            grad[aLa * 3 + axis]! += coef * dC;
            grad[aSi * 3 + axis]! += coef * dD;
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
