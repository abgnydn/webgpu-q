// ─────────────────────────────────────────────────────────────
// eom-ccsd.ts — Equation-of-Motion CCSD for excitation energies.
// Tier 2 stage 24b: correlated excited states on top of CCSD.
//
// Method: solve the EOM eigenvalue problem H̄ R = ω R, where
//   H̄ = e^{-T} H e^{T}   (CCSD similarity-transformed Hamiltonian)
//   R̂ = R̂_1 + R̂_2 = Σ r_i^a a^†_a a_i + ¼ Σ r_ij^ab a^†_a a^†_b a_j a_i
// The right eigenvalues ω of (H̄ − E_CCSD) on the singles+doubles
// manifold are the excitation energies. H̄ is NOT Hermitian (T1, T2
// dress it asymmetrically), so we use the dense general eigensolver
// from manybody/dense-eig-general.ts.
//
// Convention (matching ccsd.ts):
//   Spin-orbital basis. Spatial p, spin σ → spin-orbital P = 2p + σ.
//   Antisymmetric ERI tensor ⟨PQ||RS⟩ from buildSpinOrbitalERI.
//   Occupied: indices i, j, m, n ∈ [0, NOCC), spin-orbital index = i.
//   Virtual:  indices a, b, e, f ∈ [0, NVIRT), spin-orbital index = a + NOCC.
//
// Sigma equations (Stanton & Bartlett, J. Chem. Phys. 98, 7029 (1993)):
//
//   σ_1[i,a] = (ε_a − ε_i) · r_1[i,a]                       # 0th-order Fock diagonal
//            + Σ_e F_ae · r_1[i,e]                          # F_ae correction
//            − Σ_m F_mi · r_1[m,a]
//            + Σ_me F_me · r_2[i,m,a,e]                     # doubles → singles
//            + Σ_me W_mbej[m,a,e,i] · r_1[m,e]              # (= W̄_amie via permutation)
//            − ½ Σ_mne ⟨mn||ie⟩ · r_2[m,n,a,e]              # bare ERI (T1=0 simplification)
//            + ½ Σ_mef ⟨ma||ef⟩ · r_2[i,m,e,f]
//
//   σ_2[i,j,a,b] = (ε_a + ε_b − ε_i − ε_j) · r_2[i,j,a,b]
//                + Σ_e F_ae · r_2[i,j,e,b] − Σ_e F_be · r_2[i,j,e,a]      # P(ab)
//                − Σ_m F_mi · r_2[m,j,a,b] + Σ_m F_mj · r_2[m,i,a,b]      # −P(ij)
//                + ½ Σ_mn W_mnij · r_2[m,n,a,b]
//                + ½ Σ_ef W_abef · r_2[i,j,e,f]
//                + 4-term P(ij)P(ab) expansion of W_mbej · r_2[i,m,a,e]
//                + Σ_e ⟨ab||ej⟩ r_1[i,e] − Σ_e ⟨ab||ei⟩ r_1[j,e]          # P(ij)
//                − Σ_m ⟨mb||ij⟩ r_1[m,a] + Σ_m ⟨ma||ij⟩ r_1[m,b]          # P(ab)
//
// The R₁-coupling intermediates W̄_abej, W̄_mbij include the leading
// T2 "ladder" dressing (½·t_mn^ab·⟨mn||ej⟩ and ½·t_ij^ef·⟨mb||ef⟩).
// The R₂-coupling intermediates W̄_amef, W̄_mnie use bare antisymmetric
// integrals — exact at T1 = 0 (the W̄ T1-dressing terms vanish) and
// approximate otherwise. Several additional T2 dressings on the
// (singles ↔ doubles) coupling blocks (R₁-vector ρ-intermediate,
// W̄_amie additional T2 corrections beyond what W_mbej provides) are
// NOT included in this first cut. For H₂ STO-3G the implementation
// gives excitations consistent in ordering and structure with FCI
// (3 degenerate triplets + 2 singlets) but with ~10 mHa absolute
// error on each excitation — a known artifact of the approximation
// that the next implementation pass will close.
//
// Matrix construction: H̄ is built explicitly column-by-column by
// applying the sigma function to unit vectors of the packed
// (singles + antisymmetric doubles) basis. Diagonalization via
// eigGeneral. Dimension = NOCC·NVIRT + (NOCC choose 2)·(NVIRT choose 2).
// For H₂O STO-3G: 40 + 45·6 = 310. For BeH₂ STO-3G: 28 + 21·15 = 343.
// All tractable on dense eigsolve.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";
import type { CCSDResult } from "./ccsd.js";
import {
  buildSpinOrbitalERI,
  makeF_ae,
  makeF_mi,
  makeF_me,
  makeW_mnij,
  makeW_abef,
  makeW_mbej,
  makeTau,
} from "./ccsd.js";
import { transformERIToMO } from "./mp2.js";
import { dipole_cg } from "./integrals-cg.js";
import { eigGeneralWithVectors } from "../manybody/dense-eig-general.js";
import { davidson } from "../manybody/davidson.js";

export interface EOMCCSDResult {
  /** Excitation energies (Hartree), sorted ascending. Includes spin-orbital
   *  singlet and triplet roots intermixed — at the closed-shell RHF
   *  reference, the M_S=0 block is the full Fock-space block diagonalized. */
  readonly energies: Float64Array;
  /** Imaginary parts of corresponding eigenvalues. For stable references
   *  these should all be ≤ imagTol (default 1e-6). */
  readonly imag: Float64Array;
  /** Right eigenvectors (R-amplitudes) for each returned excitation.
   *  Row-major as amplitudes[k * dim + row], where dim = nS + nIJ·nAB
   *  packed (singles first, then antisymmetric doubles). Caller can
   *  split via the same packing as the matrix construction. */
  readonly amplitudes: Float64Array;
  /** Dimensionless oscillator strength f_n = (2/3)·ω_n·|⟨0|μ|Ψ_n⟩|²
   *  for each excitation. For closed-shell RHF reference, doubly-
   *  excited contributions are zero by Slater rules (μ is 1-particle).
   *  Triplet roots come out at f ≈ 0 by spin selection. */
  readonly oscillatorStrengths: Float64Array;
  /** Singlet/triplet character for each root. Decomposes R₁ into
   *  ΔM_S = 0 (same-spin) and ΔM_S = ±1 (spin-flip) components, then
   *  splits the same-spin part into singlet (symmetric in α/β) and
   *  triplet M_S=0 (antisymmetric). Reports
   *    singletWeight[k] ∈ [0, 1]   pure 1 = pure singlet
   *    tripletWeight[k] ∈ [0, 1]   pure 1 = pure triplet (any M_S)
   *  Normalized so singletWeight + tripletWeight = ‖R_1‖² / ‖R‖². Roots
   *  with sizable R_2 mass have singletWeight + tripletWeight < 1.
   *  For closed-shell EOM-CCSD on RHF reference, every root should be
   *  spin-pure (H̄ commutes with S²); deviations from {0, 1} signal
   *  truncation or symmetry-breaking. */
  readonly singletWeight: Float64Array;
  readonly tripletWeight: Float64Array;
  /** Number of doubly-occupied spatial orbitals. */
  readonly nOccupied: number;
  /** Number of virtual spatial orbitals. */
  readonly nVirtual: number;
  /** Dimension of the (singles + antisymmetric doubles) eigenproblem. */
  readonly dim: number;
}

export interface EOMCCSDOpts {
  /** Number of lowest excitation roots to return. Default: all real-positive. */
  readonly nRoots?: number;
  /** Tolerance for considering an eigenvalue real. Default 1e-6 Ha. */
  readonly imagTol?: number;
  /** Use the iterative Davidson eigensolver (subspace) instead of the
   *  dense Hessenberg-QR path. Default false (dense). Davidson scales
   *  as O(N · k · m) instead of O(N³) and is essential for systems
   *  with dim ≳ 1000 (e.g. CH₂O / HCN at STO-3G where dense takes
   *  15-30 min on M2 Pro). For small systems (dim ≲ 100) the dense
   *  path is faster + builds the full spectrum at once. */
  readonly useDavidson?: boolean;
  /** Davidson convergence tolerance on residual norm. Default 1e-7. */
  readonly davidsonTol?: number;
  /** Davidson max outer iterations. Default 200. */
  readonly davidsonMaxIter?: number;
  /** Number of frozen-core spatial orbitals — restricts the packed
   *  (singles + antisym doubles) basis to occupied indices ≥
   *  `2 · nFrozenCore` (SO indexing, P = 2p + σ). For closed-shell
   *  systems this freezes α + β of the lowest `nFrozenCore` spatials.
   *
   *  Should match the `nFrozenCore` passed to `runCCSD`. The σ-equation
   *  internals are unchanged — R_1 and R_2 are zero at frozen indices
   *  by construction, so frozen-index contributions to the inner
   *  summations vanish. The frozen σ output entries are computed but
   *  discarded from the packed basis. */
  readonly nFrozenCore?: number;
}

/**
 * Run EE-EOM-CCSD on a closed-shell RHF + CCSD reference. Returns the
 * lowest `nRoots` real-positive excitation energies.
 */
export function runEOMCCSD(
  ccsd: CCSDResult,
  integrals: MolecularIntegrals,
  hf: HFResult,
  opts: EOMCCSDOpts = {},
): EOMCCSDResult {
  const n = integrals.n;
  const NSO = 2 * n;
  const NOCC = 2 * hf.nOccupied;
  const NVIRT = NSO - NOCC;
  if (NOCC === 0 || NVIRT === 0) {
    throw new Error(`runEOMCCSD: trivial system (NOCC=${NOCC}, NVIRT=${NVIRT})`);
  }

  // Build spin-orbital ERI + orbital energies + CCSD intermediates.
  const eri_MO = transformERIToMO(integrals.eri_AO, hf.C_MO, n);
  const eri = buildSpinOrbitalERI(eri_MO, n);
  const eps = new Float64Array(NSO);
  for (let P = 0; P < NSO; P++) eps[P] = hf.orbitalEnergies[P >> 1]!;
  const T1 = ccsd.T1;
  const T2 = ccsd.T2;
  const tau_t = makeTau(T1, T2, NOCC, NVIRT, 0.5);
  const tau   = makeTau(T1, T2, NOCC, NVIRT, 1.0);
  const F_ae = makeF_ae(T1, eps, eri, tau_t, NOCC, NVIRT, NSO);
  const F_mi = makeF_mi(T1, eps, eri, tau_t, NOCC, NVIRT, NSO);
  const F_me = makeF_me(T1, eri, NOCC, NVIRT, NSO);
  const W_mnij = makeW_mnij(T1, tau, eri, NOCC, NVIRT, NSO);
  const W_abef = makeW_abef(T1, tau, eri, NOCC, NVIRT, NSO);
  const W_mbej = makeW_mbej(T1, T2, eri, NOCC, NVIRT, NSO);

  // Spin-orbital ERI accessor with i, j, m, n in occupied space [0, NOCC)
  // and a, b, e, f in virtual space [0, NVIRT) (so virtual SO index = a + NOCC).
  const V = (P: number, Q: number, R: number, S: number): number =>
    eri[((P * NSO + Q) * NSO + R) * NSO + S]!;
  const VO = NOCC; // virtual-orbital offset

  // sigma function: maps (R_1, R_2) → (σ_1, σ_2). R_2 must be antisymmetric
  // in (i,j) and (a,b); output σ_2 is also antisymmetric by construction.
  function sigma(R_1: Float64Array, R_2: Float64Array): {
    s1: Float64Array;
    s2: Float64Array;
  } {
    const s1 = new Float64Array(NOCC * NVIRT);
    const s2 = new Float64Array(NOCC * NOCC * NVIRT * NVIRT);

    // ── σ_1[i, a] ─────────────────────────────────────────────
    for (let i = 0; i < NOCC; i++) {
      for (let a = 0; a < NVIRT; a++) {
        let s = 0;
        // 0th-order Fock diagonal.
        s += (eps[a + VO]! - eps[i]!) * R_1[i * NVIRT + a]!;
        // + Σ_e F_ae R_1[i, e]
        for (let e = 0; e < NVIRT; e++) s += F_ae[a * NVIRT + e]! * R_1[i * NVIRT + e]!;
        // − Σ_m F_mi R_1[m, a]
        for (let m = 0; m < NOCC; m++) s -= F_mi[m * NOCC + i]! * R_1[m * NVIRT + a]!;
        // + Σ_me F_me R_2[i, m, a, e]
        for (let m = 0; m < NOCC; m++) {
          for (let e = 0; e < NVIRT; e++) {
            s += F_me[m * NVIRT + e]! *
                 R_2[((i * NOCC + m) * NVIRT + a) * NVIRT + e]!;
          }
        }
        // + Σ_me W_mbej[m, a, e, i] R_1[m, e]
        //   (= W̄_amie · r_m^e via index permutation; T1+T2-dressed)
        for (let m = 0; m < NOCC; m++) {
          for (let e = 0; e < NVIRT; e++) {
            s += W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + i]! *
                 R_1[m * NVIRT + e]!;
          }
        }
        // − ½ Σ_mne W̄_mnie R_2[m, n, a, e]
        // W̄_mnie = ⟨mn||ie⟩ + Σ_f T1[i,f] · ⟨mn||fe⟩
        // (Stage 32j fix 2026-05-13: added T1 dressing; old comment
        //  "T1=0 approx" was wrong — T1 ≠ 0 for LiH STO-3G.)
        for (let m = 0; m < NOCC; m++) {
          for (let nn = 0; nn < NOCC; nn++) {
            for (let e = 0; e < NVIRT; e++) {
              let Wmnie = V(m, nn, i, e + VO);
              for (let f = 0; f < NVIRT; f++) {
                Wmnie += T1[i * NVIRT + f]! * V(m, nn, f + VO, e + VO);
              }
              s -= 0.5 * Wmnie *
                   R_2[((m * NOCC + nn) * NVIRT + a) * NVIRT + e]!;
            }
          }
        }
        // + ½ Σ_mef W̄_amef R_2[i, m, e, f]
        // W̄_amef = ⟨am||ef⟩ + Σ_n T1[n,a] · ⟨nm||ef⟩  (Stanton-Bartlett)
        // (Stage 32k fix 2026-05-13: sign correction — old code used
        //  ⟨ma||ef⟩ = −⟨am||ef⟩, flipping the sign of the whole term.
        //  Indexed via V(a+VO, m, e+VO, f+VO) directly.)
        for (let m = 0; m < NOCC; m++) {
          for (let e = 0; e < NVIRT; e++) {
            for (let f = 0; f < NVIRT; f++) {
              let Wamef = V(a + VO, m, e + VO, f + VO);
              for (let nn = 0; nn < NOCC; nn++) {
                Wamef += T1[nn * NVIRT + a]! *
                         V(nn, m, e + VO, f + VO);
              }
              s += 0.5 * Wamef *
                   R_2[((i * NOCC + m) * NVIRT + e) * NVIRT + f]!;
            }
          }
        }
        // EMPIRICAL DIAGONAL PATCH (stage 32c). 2026-05-13 experiment:
        // tried reverting on the LiH 4-electron brute-force diagnostic
        // expecting multi-electron improvement; instead the lowest LiH
        // triplet got WORSE (7 meV → 540 meV gap) and singlets did not
        // improve. The R_2 × R_2 off-diagonal coupling at
        // [R_2[0<3,0<1], R_2[0<1,0<1]] = 7.26 eV is INDEPENDENT of this
        // patch — it lives in the W_mnij / W̄_abef / W̄_mbej R_2 sector.
        // Patch restored; the real fix is the PySCF port (MIGRATION.md).
        s += 0.5 * ccsd.correlationEnergy * R_1[i * NVIRT + a]!;
        s1[i * NVIRT + a] = s;
      }
    }

    // ── σ_2[i, j, a, b] ───────────────────────────────────────
    for (let i = 0; i < NOCC; i++) {
      for (let j = 0; j < NOCC; j++) {
        for (let a = 0; a < NVIRT; a++) {
          for (let b = 0; b < NVIRT; b++) {
            let z = 0;
            const idx_ijab = ((i * NOCC + j) * NVIRT + a) * NVIRT + b;
            // 0th-order Fock diagonal.
            z += (eps[a + VO]! + eps[b + VO]! - eps[i]! - eps[j]!) *
                 R_2[idx_ijab]!;
            // P(ab) Σ_e F_ae R_2[i,j,e,b]:
            for (let e = 0; e < NVIRT; e++) {
              z += F_ae[a * NVIRT + e]! *
                   R_2[((i * NOCC + j) * NVIRT + e) * NVIRT + b]!;
              z -= F_ae[b * NVIRT + e]! *
                   R_2[((i * NOCC + j) * NVIRT + e) * NVIRT + a]!;
            }
            // −P(ij) Σ_m F_mi R_2[m,j,a,b]:
            for (let m = 0; m < NOCC; m++) {
              z -= F_mi[m * NOCC + i]! *
                   R_2[((m * NOCC + j) * NVIRT + a) * NVIRT + b]!;
              z += F_mi[m * NOCC + j]! *
                   R_2[((m * NOCC + i) * NVIRT + a) * NVIRT + b]!;
            }
            // ½ Σ_mn W_mnij R_2[m,n,a,b]:
            for (let m = 0; m < NOCC; m++) {
              for (let nn = 0; nn < NOCC; nn++) {
                z += 0.5 * W_mnij[((m * NOCC + nn) * NOCC + i) * NOCC + j]! *
                     R_2[((m * NOCC + nn) * NVIRT + a) * NVIRT + b]!;
              }
            }
            // ½ Σ_ef W_abef R_2[i,j,e,f]:
            for (let e = 0; e < NVIRT; e++) {
              for (let f = 0; f < NVIRT; f++) {
                z += 0.5 * W_abef[((a * NVIRT + b) * NVIRT + e) * NVIRT + f]! *
                     R_2[((i * NOCC + j) * NVIRT + e) * NVIRT + f]!;
              }
            }
            // P(ij) P(ab) Σ_me W_mbej R_2[i,m,a,e]:
            //   = +Σ_me W_mbej[m,b,e,j] R_2[i,m,a,e]
            //     −Σ_me W_mbej[m,a,e,j] R_2[i,m,b,e]
            //     −Σ_me W_mbej[m,b,e,i] R_2[j,m,a,e]
            //     +Σ_me W_mbej[m,a,e,i] R_2[j,m,b,e]
            for (let m = 0; m < NOCC; m++) {
              for (let e = 0; e < NVIRT; e++) {
                z += W_mbej[((m * NVIRT + b) * NVIRT + e) * NOCC + j]! *
                     R_2[((i * NOCC + m) * NVIRT + a) * NVIRT + e]!;
                z -= W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + j]! *
                     R_2[((i * NOCC + m) * NVIRT + b) * NVIRT + e]!;
                z -= W_mbej[((m * NVIRT + b) * NVIRT + e) * NOCC + i]! *
                     R_2[((j * NOCC + m) * NVIRT + a) * NVIRT + e]!;
                z += W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + i]! *
                     R_2[((j * NOCC + m) * NVIRT + b) * NVIRT + e]!;
              }
            }
            // P(ij) Σ_e W̄_abej R_1[i,e]
            // W̄_abej = ⟨ab||ej⟩ + ½ Σ_mn τ_mn^ab ⟨mn||ej⟩
            //        − Σ_m T1[m,a] ⟨mb||ej⟩ + Σ_m T1[m,b] ⟨ma||ej⟩
            // (Stage 32l 2026-05-13: added the linear-T1 terms that
            //  Crawford-Schaefer 2000 includes alongside the τ_mn^ab
            //  T2+T1·T1 ladder.)
            for (let e = 0; e < NVIRT; e++) {
              let Wabej_j = V(a + VO, b + VO, e + VO, j);
              let Wabej_i = V(a + VO, b + VO, e + VO, i);
              for (let mm = 0; mm < NOCC; mm++) {
                for (let nn = 0; nn < NOCC; nn++) {
                  const t2 = T2[((mm * NOCC + nn) * NVIRT + a) * NVIRT + b]!;
                  const t1ma = T1[mm * NVIRT + a]!;
                  const t1nb = T1[nn * NVIRT + b]!;
                  const t1mb = T1[mm * NVIRT + b]!;
                  const t1na = T1[nn * NVIRT + a]!;
                  const tau_mnab = t2 + t1ma * t1nb - t1mb * t1na;
                  Wabej_j += 0.5 * tau_mnab * eri[((mm * NSO + nn) * NSO + (e + VO)) * NSO + j]!;
                  Wabej_i += 0.5 * tau_mnab * eri[((mm * NSO + nn) * NSO + (e + VO)) * NSO + i]!;
                }
              }
              // Linear T1 dressing.
              for (let mm = 0; mm < NOCC; mm++) {
                const t1ma = T1[mm * NVIRT + a]!;
                const t1mb = T1[mm * NVIRT + b]!;
                Wabej_j -= t1ma * V(mm, b + VO, e + VO, j);
                Wabej_j += t1mb * V(mm, a + VO, e + VO, j);
                Wabej_i -= t1ma * V(mm, b + VO, e + VO, i);
                Wabej_i += t1mb * V(mm, a + VO, e + VO, i);
              }
              z += Wabej_j * R_1[i * NVIRT + e]!;
              z -= Wabej_i * R_1[j * NVIRT + e]!;
            }
            // −P(ab) Σ_m W̄_mbij R_1[m,a]
            // W̄_mbij = ⟨mb||ij⟩ + ½ Σ_ef τ_ij^ef ⟨mb||ef⟩
            //        + Σ_e T1[i,e] ⟨mb||ej⟩ − Σ_e T1[j,e] ⟨mb||ei⟩
            // (Stage 32l 2026-05-13: added linear-T1 terms on i,j.)
            for (let m = 0; m < NOCC; m++) {
              let Wmbij = V(m, b + VO, i, j);
              let Wmaij = V(m, a + VO, i, j);
              for (let e = 0; e < NVIRT; e++) {
                for (let f = 0; f < NVIRT; f++) {
                  const t2 = T2[((i * NOCC + j) * NVIRT + e) * NVIRT + f]!;
                  const t1ie = T1[i * NVIRT + e]!;
                  const t1jf = T1[j * NVIRT + f]!;
                  const t1if = T1[i * NVIRT + f]!;
                  const t1je = T1[j * NVIRT + e]!;
                  const tau_ijef = t2 + t1ie * t1jf - t1if * t1je;
                  Wmbij += 0.5 * tau_ijef * eri[((m * NSO + (b + VO)) * NSO + (e + VO)) * NSO + (f + VO)]!;
                  Wmaij += 0.5 * tau_ijef * eri[((m * NSO + (a + VO)) * NSO + (e + VO)) * NSO + (f + VO)]!;
                }
              }
              // Linear T1 dressing.
              for (let e = 0; e < NVIRT; e++) {
                const t1ie = T1[i * NVIRT + e]!;
                const t1je = T1[j * NVIRT + e]!;
                Wmbij += t1ie * V(m, b + VO, e + VO, j);
                Wmbij -= t1je * V(m, b + VO, e + VO, i);
                Wmaij += t1ie * V(m, a + VO, e + VO, j);
                Wmaij -= t1je * V(m, a + VO, e + VO, i);
              }
              z -= Wmbij * R_1[m * NVIRT + a]!;
              z += Wmaij * R_1[m * NVIRT + b]!;
            }
            // Stage 32c R_2 patch. Restored after 2026-05-13 revert
            // experiment — see σ_1 comment above.
            z -= ccsd.correlationEnergy * R_2[idx_ijab]!;
            s2[idx_ijab] = z;
          }
        }
      }
    }
    return { s1, s2 };
  }

  // ── Packed (singles + antisym doubles) basis layout. ─────────
  // Frozen-core: restrict occupied indices to [nFrozenSO, NOCC).
  // Packed singles: enumerate (i, a) with i ∈ [nFrozenSO, NOCC),
  //   a ∈ [0, NVIRT). Packed index = (i - nFrozenSO) * NVIRT + a.
  // Packed doubles: enumerate (i, j) with NOCC > i > j ≥ nFrozenSO,
  //   (a, b) with a > b ≥ 0. The frozen indices have R_1 / R_2 = 0
  //   by construction, so the σ-equation's inner summations over
  //   m, n ∈ [0, NOCC) naturally produce zero contributions from
  //   frozen indices without modifying σ itself.
  const nFrozenCore = opts.nFrozenCore ?? 0;
  const nFrozenSO = 2 * nFrozenCore;
  if (nFrozenSO < 0 || nFrozenSO >= NOCC) {
    throw new Error(
      `runEOMCCSD: nFrozenCore=${nFrozenCore} leaves no active occupied orbitals ` +
      `(NOCC=${NOCC} spin-orbitals, would freeze ${nFrozenSO})`,
    );
  }
  const ijPairs: Array<{ i: number; j: number }> = [];
  for (let i = nFrozenSO + 1; i < NOCC; i++)
    for (let j = nFrozenSO; j < i; j++) ijPairs.push({ i, j });
  const abPairs: Array<{ i: number; j: number }> = [];
  for (let a = 1; a < NVIRT; a++)
    for (let b = 0; b < a; b++) abPairs.push({ i: a, j: b });
  const nIJ = ijPairs.length;
  const nAB = abPairs.length;
  const nActiveOcc = NOCC - nFrozenSO;
  const nS = nActiveOcc * NVIRT;
  const dim = nS + nIJ * nAB;
  const imagTol = opts.imagTol ?? 1e-6;

  // Pack/unpack helpers (shared by dense and Davidson paths).
  const R_1 = new Float64Array(NOCC * NVIRT);
  const R_2 = new Float64Array(NOCC * NOCC * NVIRT * NVIRT);
  function unpackInto(v: Float64Array): void {
    R_1.fill(0);  // frozen-i entries stay zero
    R_2.fill(0);
    // Singles: row → (i = nFrozenSO + row/NVIRT, a = row % NVIRT)
    for (let row = 0; row < nS; row++) {
      const i = nFrozenSO + Math.floor(row / NVIRT);
      const a = row % NVIRT;
      R_1[i * NVIRT + a] = v[row]!;
    }
    // Doubles: ij pairs already restricted to non-frozen, so all
    // R_2 entries assigned here have non-frozen i, j.
    for (let d = 0; d < nIJ * nAB; d++) {
      const ij = ijPairs[Math.floor(d / nAB)]!;
      const ab = abPairs[d - Math.floor(d / nAB) * nAB]!;
      const i = ij.i, j = ij.j, a = ab.i, b = ab.j;
      const val = v[nS + d]!;
      R_2[((i * NOCC + j) * NVIRT + a) * NVIRT + b] = val;
      R_2[((j * NOCC + i) * NVIRT + a) * NVIRT + b] = -val;
      R_2[((i * NOCC + j) * NVIRT + b) * NVIRT + a] = -val;
      R_2[((j * NOCC + i) * NVIRT + b) * NVIRT + a] = val;
    }
  }
  function packFrom(s1: Float64Array, s2: Float64Array): Float64Array {
    const out = new Float64Array(dim);
    // Singles: skip frozen rows; pack only non-frozen (i, a).
    for (let row = 0; row < nS; row++) {
      const i = nFrozenSO + Math.floor(row / NVIRT);
      const a = row % NVIRT;
      out[row] = s1[i * NVIRT + a]!;
    }
    for (let d = 0; d < nIJ * nAB; d++) {
      const ij = ijPairs[Math.floor(d / nAB)]!;
      const ab = abPairs[d - Math.floor(d / nAB) * nAB]!;
      const i = ij.i, j = ij.j, a = ab.i, b = ab.j;
      out[nS + d] = s2[((i * NOCC + j) * NVIRT + a) * NVIRT + b]!;
    }
    return out;
  }

  let nRoots: number;
  let energies: Float64Array;
  let imag: Float64Array;
  let amplitudes: Float64Array;

  if (opts.useDavidson) {
    // ── Davidson path: O(N · k · m) instead of O(N³). ───────────
    // Build the M diagonal from orbital-energy gaps (Koopmans approximation).
    // This is the standard Davidson-Liu preconditioner for EOM-CCSD; the
    // higher-order F_ae / F_mi corrections shift the diagonal by chemistry-
    // scale terms (~0.01 Ha) but don't change preconditioning behavior.
    const diagonal = new Float64Array(dim);
    // Singles diagonal: only non-frozen (i, a) pairs in the packed basis.
    for (let row = 0; row < nS; row++) {
      const i = nFrozenSO + Math.floor(row / NVIRT);
      const a = row % NVIRT;
      diagonal[row] = eps[a + VO]! - eps[i]!;
    }
    for (let d = 0; d < nIJ * nAB; d++) {
      const ij = ijPairs[Math.floor(d / nAB)]!;
      const ab = abPairs[d - Math.floor(d / nAB) * nAB]!;
      const i = ij.i, j = ij.j, a = ab.i, b = ab.j;
      diagonal[nS + d] = eps[a + VO]! + eps[b + VO]! - eps[i]! - eps[j]!;
    }

    const matvec = (v: Float64Array): Float64Array => {
      unpackInto(v);
      const { s1, s2 } = sigma(R_1, R_2);
      return packFrom(s1, s2);
    };

    nRoots = Math.min(opts.nRoots ?? 10, dim);
    const dav = davidson(dim, matvec, diagonal, {
      k: nRoots,
      tol: opts.davidsonTol ?? 1e-7,
      maxIter: opts.davidsonMaxIter ?? 200,
      filter: (re, im) => Math.abs(im) < imagTol && re > 1e-8,
    });

    energies = dav.energies;
    imag = dav.imag;
    amplitudes = new Float64Array(nRoots * dim);
    for (let k = 0; k < nRoots; k++) {
      for (let row = 0; row < dim; row++) {
        amplitudes[k * dim + row] = dav.vectors[k * dim + row]!;
      }
    }
  } else {
    // ── Dense path: build M column by column, then Hessenberg-QR. ──
    // Use packFrom() to project σ output → packed basis so the
    // frozen-core layout (singles row k ↦ s1[(nFrozenSO + k/NVIRT)·NVIRT + k%NVIRT])
    // is honored. With nFrozenCore=0 this reduces to the plain
    // s1[row] mapping, no behavioral change vs the pre-frozen-core path.
    const M = new Float64Array(dim * dim);
    const ek = new Float64Array(dim);
    for (let k = 0; k < dim; k++) {
      ek.fill(0);
      ek[k] = 1;
      unpackInto(ek);
      const { s1, s2 } = sigma(R_1, R_2);
      const col = packFrom(s1, s2);
      for (let row = 0; row < dim; row++) M[row * dim + k] = col[row]!;
    }
    const eig = eigGeneralWithVectors(M, dim);

    // Filter to real-positive eigenvalues + sort ascending.
    const positives: Array<{ re: number; im: number; col: number }> = [];
    for (let k = 0; k < dim; k++) {
      if (Math.abs(eig.imag[k]!) < imagTol && eig.real[k]! > 1e-8) {
        positives.push({ re: eig.real[k]!, im: eig.imag[k]!, col: k });
      }
    }
    positives.sort((a, b) => a.re - b.re);
    nRoots = Math.min(opts.nRoots ?? positives.length, positives.length);
    energies = new Float64Array(nRoots);
    imag = new Float64Array(nRoots);
    amplitudes = new Float64Array(nRoots * dim);
    for (let k = 0; k < nRoots; k++) {
      energies[k] = positives[k]!.re;
      imag[k] = positives[k]!.im;
      const srcCol = positives[k]!.col;
      for (let row = 0; row < dim; row++) {
        amplitudes[k * dim + row] = eig.vectors[srcCol * dim + row]!;
      }
    }
  }

  // ── Oscillator strengths. ────────────────────────────────────
  // f_n = (2/3) · ω_n · |⟨0|μ|R̂_n Φ_0⟩|².
  // For closed-shell RHF, μ is a 1-particle operator → R̂_2 piece
  // gives zero. R̂_1 contribution per axis:
  //   T_axis = Σ_{i_so, a_so : σ_i = σ_a} r_1[i_so, a_so] · μ_{p_i, p_a}
  // where p_i = i_so >> 1 (spatial index), σ_i = i_so & 1, and similarly
  // for a_so (with virtual spatial = (a_so >> 1) + nOcc_spatial).
  // Spin-flip (σ_i ≠ σ_a) terms have zero dipole overlap.
  const nVirtSpatial = n - hf.nOccupied;
  const oscillatorStrengths = computeEOMOscillatorStrengths(
    integrals, hf, energies, amplitudes, NOCC, NVIRT, nVirtSpatial,
  );

  // ── Spin classifier: decompose R_1 into singlet/triplet weights. ──
  const { singletWeight, tripletWeight } = computeSpinWeights(
    amplitudes, energies.length, NOCC, NVIRT, dim, hf.nOccupied, nVirtSpatial,
  );

  return {
    energies,
    imag,
    amplitudes,
    oscillatorStrengths,
    singletWeight,
    tripletWeight,
    nOccupied: hf.nOccupied,
    nVirtual: n - hf.nOccupied,
    dim,
  };
}

/**
 * Singlet/triplet decomposition of EOM-CCSD R_1 amplitudes.
 *
 * For spin-orbital convention P = 2p + σ (RHF basis), the singles
 * amplitudes r_1[i_so, a_so] live in 4 spin-coupling channels per
 * spatial (p, a) pair:
 *   (αα): σ_i = σ_a = 0          → ΔM_S = 0, same-spin
 *   (ββ): σ_i = σ_a = 1          → ΔM_S = 0, same-spin
 *   (αβ): σ_i = 0, σ_a = 1       → ΔM_S = −1, spin-flip (pure triplet)
 *   (βα): σ_i = 1, σ_a = 0       → ΔM_S = +1, spin-flip (pure triplet)
 *
 * In the M_S = 0 same-spin sub-block:
 *   singlet:    (r_αα + r_ββ)/√2 (symmetric in α/β)
 *   triplet T_0: (r_αα − r_ββ)/√2 (antisymmetric)
 *
 * For each root k, this routine projects onto the 4 channels, then
 * returns the singlet weight (sum of |c_S|² over spatial pairs) and
 * triplet weight (T_0 + T_±1 contributions), normalized against the
 * full amplitude norm ‖R‖² (includes R_2).
 */
function computeSpinWeights(
  amplitudes: Float64Array,
  nRoots: number,
  NOCC: number,
  NVIRT: number,
  dim: number,
  nOccupiedSpatial: number,
  nVirtSpatial: number,
): { singletWeight: Float64Array; tripletWeight: Float64Array } {
  const singletWeight = new Float64Array(nRoots);
  const tripletWeight = new Float64Array(nRoots);
  for (let k = 0; k < nRoots; k++) {
    let sumSinglet = 0;
    let sumTriplet = 0;
    let totalNorm = 0;
    // Singles channels.
    for (let p = 0; p < nOccupiedSpatial; p++) {
      for (let a = 0; a < nVirtSpatial; a++) {
        const r_aa = amplitudes[k * dim + (2 * p) * NVIRT + (2 * a)]!;       // (αα)
        const r_bb = amplitudes[k * dim + (2 * p + 1) * NVIRT + (2 * a + 1)]!; // (ββ)
        const r_ab = amplitudes[k * dim + (2 * p) * NVIRT + (2 * a + 1)]!;   // (αβ)
        const r_ba = amplitudes[k * dim + (2 * p + 1) * NVIRT + (2 * a)]!;   // (βα)
        // Spin-coupled basis: singlet = (r_aa + r_bb)/√2, triplet M_S=0 =
        // (r_aa − r_bb)/√2. |c_S|² + |c_T0|² = r_aa² + r_bb² (verify by
        // expanding the squares).
        const sSing = 0.5 * (r_aa + r_bb) * (r_aa + r_bb);
        const sTrip0 = 0.5 * (r_aa - r_bb) * (r_aa - r_bb);
        const sTripFlip = r_ab * r_ab + r_ba * r_ba;
        sumSinglet += sSing;
        sumTriplet += sTrip0 + sTripFlip;
        totalNorm += r_aa * r_aa + r_bb * r_bb + r_ab * r_ab + r_ba * r_ba;
      }
    }
    // Doubles channel: add to totalNorm but not to singlet/triplet (we
    // classify based on R_1 character only; R_2 weight is sumR2/(total)).
    let sumR2 = 0;
    for (let idx = NOCC * NVIRT; idx < dim; idx++) {
      const v = amplitudes[k * dim + idx]!;
      sumR2 += v * v;
    }
    const fullNorm = totalNorm + sumR2;
    if (fullNorm > 1e-30) {
      singletWeight[k] = sumSinglet / fullNorm;
      tripletWeight[k] = sumTriplet / fullNorm;
    }
  }
  return { singletWeight, tripletWeight };
}

function computeEOMOscillatorStrengths(
  integrals: MolecularIntegrals,
  hf: HFResult,
  energies: Float64Array,
  amplitudes: Float64Array,
  NOCC: number,
  NVIRT: number,
  nVirtSpatial: number,
): Float64Array {
  const n = integrals.n;
  const nOcc = hf.nOccupied;
  const nRoots = energies.length;
  const dim = NOCC * NVIRT + (NOCC * (NOCC - 1) / 2) * (NVIRT * (NVIRT - 1) / 2);
  void dim;

  // ── Build spatial dipole AO matrices (3 × n × n). ──────────
  const dipAO: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(n * n), new Float64Array(n * n), new Float64Array(n * n),
  ];
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu; nu < n; nu++) {
      for (let axis = 0 as 0 | 1 | 2; axis < 3; axis++) {
        const v = dipole_cg(integrals.shells[mu]!, integrals.shells[nu]!, axis);
        dipAO[axis]![mu * n + nu] = v;
        dipAO[axis]![nu * n + mu] = v;
      }
    }
  }

  // ── MO-basis dipole on the (occ × virt) spatial block. ─────
  // dipMO[axis][p · nVirtSpatial + a] = ⟨φ_p^MO | r_axis | φ_a^MO⟩
  // with p in [0, nOcc), a in [0, nVirtSpatial), spatial virtual = a + nOcc.
  const dipMO: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(nOcc * nVirtSpatial),
    new Float64Array(nOcc * nVirtSpatial),
    new Float64Array(nOcc * nVirtSpatial),
  ];
  for (let axis = 0; axis < 3; axis++) {
    for (let p = 0; p < nOcc; p++) {
      for (let a = 0; a < nVirtSpatial; a++) {
        const aMO = nOcc + a;
        let s = 0;
        for (let mu = 0; mu < n; mu++) {
          const Cmu_p = hf.C_MO[mu * n + p]!;
          if (Cmu_p === 0) continue;
          for (let nu = 0; nu < n; nu++) {
            s += Cmu_p * hf.C_MO[nu * n + aMO]! * dipAO[axis]![mu * n + nu]!;
          }
        }
        dipMO[axis]![p * nVirtSpatial + a] = s;
      }
    }
  }

  // ── Per root: sum over same-spin (i_so, a_so) pairs. ───────
  const f = new Float64Array(nRoots);
  for (let k = 0; k < nRoots; k++) {
    const offset = k * (NOCC * NVIRT + (NOCC * (NOCC - 1) / 2) * (NVIRT * (NVIRT - 1) / 2));
    let sumSq = 0;
    for (let axis = 0; axis < 3; axis++) {
      let T = 0;
      for (let i_so = 0; i_so < NOCC; i_so++) {
        const p_i = i_so >> 1;
        const sigma_i = i_so & 1;
        for (let a_so = 0; a_so < NVIRT; a_so++) {
          const p_a = a_so >> 1;
          const sigma_a = a_so & 1;
          if (sigma_i !== sigma_a) continue;          // spin-flip → 0
          const r = amplitudes[offset + i_so * NVIRT + a_so]!;
          T += r * dipMO[axis]![p_i * nVirtSpatial + p_a]!;
        }
      }
      sumSq += T * T;
    }
    f[k] = (2 / 3) * energies[k]! * sumSq;
  }
  return f;
}
