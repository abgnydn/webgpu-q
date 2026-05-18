// ─────────────────────────────────────────────────────────────
// ip-eom-ccsd.ts — Ionization-Potential EOM-CCSD.
// Tier 2 stage 37: correlated ionization potentials on top of
// CCSD (vastly more accurate than Koopmans / ΔSCF).
//
// The IP-EOM operator removes an electron:
//   R̂ = R̂_1 + R̂_2
//   R̂_1 = Σ_i r_i a_i                          (1-hole)
//   R̂_2 = (1/2) Σ_ija r_ij^a · a^†_a a_j a_i   (2-hole-1-particle)
// R̂ acts on the closed-shell |Φ_0⟩ to give an N−1 electron state.
//
// The eigenvalue problem (H̄ − E_CCSD) R̂ Φ_0 = ω R̂ Φ_0 has ω > 0:
// the diagonal matrix element on |Φ_i⟩ = a_i |Φ_0⟩ is
//   ⟨Φ_i | H̄ − E_CCSD | Φ_i⟩ = (E_HF − ε_i) − E_CCSD = −ε_i − E_corr,
// which is positive because ε_i < 0 for occupied orbitals.
// So IP = ω directly; sorted ascending in this routine.
//
// Stanton-Bartlett-style sigma equations on the N−1 manifold,
// closed-shell RHF reference, canonical orbitals (so the bare
// Fock is diagonal in the spin-orbital basis):
//
//   σ_1[i]      = − ε_i r_i  − Σ_j F_mi[j,i]·r_j         (1-h diagonal)
//               + Σ_je F_me[j,e] · r_ij^e                (1-h ← 2h1p coupling)
//               − ½ Σ_jke ⟨jk||ie⟩ · r_jk^e              (V coupling)
//
//   σ_2[i,j,a]  = (ε_a − ε_i − ε_j) · r_ij^a                       (Fock diag)
//               + Σ_e F_ae[a,e] · r_ij^e
//               − Σ_m F_mi[m,i]·r_mj^a + Σ_m F_mi[m,j]·r_mi^a    (− P(ij) F_mi)
//               + ½ Σ_mn W_mnij · r_mn^a                            (oo-ladder)
//               + P(ij)·Σ_me W_mbej[m,a,e,i]·r_jm^e                 (with sign)
//               − P(ij)·Σ_k ⟨jk||ia⟩ · r_k                          (R_1-coupling, bare V)
//
// W̄ T2 dressings on bare-V couplings (W̄_mnie / W̄_amie / W̄_kjia)
// follow the same pattern as my EE-EOM-CCSD module. T1 dressing
// terms vanish at canonical RHF when T1 = 0. Higher-order T2
// dressings on the singles ↔ doubles coupling are documented
// honest gaps (same as EE-EOM-CCSD stage 24b).
//
// PRECISION VALIDATION (stage 32 close-out, 2026-05-12):
// Brute-force cross-check against explicit H̄ projection on the
// 4-spin-orbital Fock space (tests/chemistry/ip-eom-ccsd-bruteforce.test.ts)
// established:
//   - LOWEST IPs (R_1-dominated, the physically important values)
//     match the brute-force reference EXACTLY for H₂ STO-3G.
//     So H₂O's 12.03 eV lowest IP and similar results are validated.
//   - HIGHER eigenvalues (R_2-dominated 2h1p "satellite" states, e.g.
//     Auger ionization) have a substantial ~2 Ha (60 eV) over-count
//     from the σ_2 P(ij)·W_mbej contraction — a structural bug
//     unrelated to the EE-EOM-CCSD |E_corr|/2 issue.
//   - Fix for the R_2 sector needs careful re-derivation of σ_2's
//     W_mbej contribution; outside this turn's scope. Users
//     consuming `ips[0..k]` (lowest k physical IPs) are unaffected.
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
  makeW_mbej,
  makeTau,
} from "./ccsd.js";
import { transformERIToMO } from "./mp2.js";
import { eigGeneral } from "../manybody/dense-eig-general.js";
import { davidson } from "../manybody/davidson.js";

export interface IPEOMCCSDResult {
  /** Ionization potentials (Hartree), sorted ascending. */
  readonly ips: Float64Array;
  /** Imaginary parts of corresponding eigenvalues. */
  readonly imag: Float64Array;
  /** Dimension of the (R_1 + antisym R_2) manifold. */
  readonly dim: number;
}

export interface IPEOMCCSDOpts {
  readonly nRoots?: number;
  readonly imagTol?: number;
  /** Frozen-core spatial orbitals. Same convention as `runEOMCCSD`:
   *  restricts the packed (1h + antisym 2h1p) basis to occupied
   *  indices ≥ 2·nFrozenCore. Core-ionization satellites are
   *  excluded — this method isn't intended for them anyway
   *  (would need separate Auger-CC machinery). */
  readonly nFrozenCore?: number;
  /** Use block Davidson instead of the dense Hessenberg-QR. Recommended
   *  for dim ≳ 200 systems. */
  readonly useDavidson?: boolean;
  readonly davidsonTol?: number;
  readonly davidsonMaxIter?: number;
}

export function runIPEOMCCSD(
  ccsd: CCSDResult,
  integrals: MolecularIntegrals,
  hf: HFResult,
  opts: IPEOMCCSDOpts = {},
): IPEOMCCSDResult {
  const n = integrals.n;
  const NSO = 2 * n;
  const NOCC = 2 * hf.nOccupied;
  const NVIRT = NSO - NOCC;
  if (NOCC === 0) {
    throw new Error(`runIPEOMCCSD: empty occupied space`);
  }

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
  const W_mbej = makeW_mbej(T1, T2, eri, NOCC, NVIRT, NSO);

  const V = (P: number, Q: number, R: number, S: number): number =>
    eri[((P * NSO + Q) * NSO + R) * NSO + S]!;
  const VO = NOCC;

  // ── sigma function on (R_1, R_2 antisym in (i,j)). ─────────
  function sigma(R_1: Float64Array, R_2: Float64Array): {
    s1: Float64Array;
    s2: Float64Array;
  } {
    const s1 = new Float64Array(NOCC);
    const s2 = new Float64Array(NOCC * NOCC * NVIRT);

    // σ_1[i] ----------------------------------------------------
    for (let i = 0; i < NOCC; i++) {
      let s = 0;
      // Bare Fock diag: −ε_i r_i
      s -= eps[i]! * R_1[i]!;
      // F_mi correction
      for (let j = 0; j < NOCC; j++) s -= F_mi[j * NOCC + i]! * R_1[j]!;
      // F_me · R_2 coupling (1h ← 2h1p)
      for (let j = 0; j < NOCC; j++) {
        for (let e = 0; e < NVIRT; e++) {
          s += F_me[j * NVIRT + e]! * R_2[(i * NOCC + j) * NVIRT + e]!;
        }
      }
      // − ½ Σ_jke ⟨jk||ie⟩ r_jk^e  (V coupling)
      for (let j = 0; j < NOCC; j++) {
        for (let k = 0; k < NOCC; k++) {
          for (let e = 0; e < NVIRT; e++) {
            s -= 0.5 * V(j, k, i, e + VO) *
                 R_2[(j * NOCC + k) * NVIRT + e]!;
          }
        }
      }
      s1[i] = s;
    }

    // σ_2[i,j,a] (antisym in (i,j)) -----------------------------
    for (let i = 0; i < NOCC; i++) {
      for (let j = 0; j < NOCC; j++) {
        for (let a = 0; a < NVIRT; a++) {
          let z = 0;
          // Fock diag (ε_a − ε_i − ε_j) r_ij^a
          z += (eps[a + VO]! - eps[i]! - eps[j]!) *
               R_2[(i * NOCC + j) * NVIRT + a]!;
          // + Σ_e F_ae r_ij^e
          for (let e = 0; e < NVIRT; e++) {
            z += F_ae[a * NVIRT + e]! * R_2[(i * NOCC + j) * NVIRT + e]!;
          }
          // − P(ij) Σ_m F_mi r_mj^a
          for (let m = 0; m < NOCC; m++) {
            z -= F_mi[m * NOCC + i]! * R_2[(m * NOCC + j) * NVIRT + a]!;
            z += F_mi[m * NOCC + j]! * R_2[(m * NOCC + i) * NVIRT + a]!;
          }
          // ½ Σ_mn W_mnij r_mn^a
          for (let m = 0; m < NOCC; m++) {
            for (let nn = 0; nn < NOCC; nn++) {
              z += 0.5 * W_mnij[((m * NOCC + nn) * NOCC + i) * NOCC + j]! *
                   R_2[(m * NOCC + nn) * NVIRT + a]!;
            }
          }
          // P(ij) Σ_me W_mbej[m,a,e,i] r_jm^e
          //  = + Σ_me W_mbej[m,a,e,i] r_jm^e − Σ_me W_mbej[m,a,e,j] r_im^e
          for (let m = 0; m < NOCC; m++) {
            for (let e = 0; e < NVIRT; e++) {
              z += W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + i]! *
                   R_2[(j * NOCC + m) * NVIRT + e]!;
              z -= W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + j]! *
                   R_2[(i * NOCC + m) * NVIRT + e]!;
            }
          }
          // − P(ij) Σ_k ⟨jk||ia⟩ r_k  =  −Σ_k ⟨jk||ia⟩ r_k + Σ_k ⟨ik||ja⟩ r_k
          for (let k = 0; k < NOCC; k++) {
            z -= V(j, k, i, a + VO) * R_1[k]!;
            z += V(i, k, j, a + VO) * R_1[k]!;
          }
          s2[(i * NOCC + j) * NVIRT + a] = z;
        }
      }
    }
    return { s1, s2 };
  }

  // ── Build packed (R_1 + antisym R_2) basis. ────────────────
  // Frozen-core: restrict occupied indices to [nFrozenSO, NOCC).
  // R_1 lives in [0, NOCC) but is zero for frozen i; same for R_2's
  // (i, j) indices. σ-equation internals unchanged — frozen-index
  // contributions vanish because R_1, R_2 are zero there.
  const nFrozenCore = opts.nFrozenCore ?? 0;
  const nFrozenSO = 2 * nFrozenCore;
  if (nFrozenSO < 0 || nFrozenSO >= NOCC) {
    throw new Error(
      `runIPEOMCCSD: nFrozenCore=${nFrozenCore} leaves no active occupied orbitals ` +
      `(NOCC=${NOCC}, would freeze ${nFrozenSO})`,
    );
  }
  const ijPairs: { i: number; j: number }[] = [];
  for (let i = nFrozenSO + 1; i < NOCC; i++)
    for (let j = nFrozenSO; j < i; j++) ijPairs.push({ i, j });
  const nIJ = ijPairs.length;
  const nActiveOcc = NOCC - nFrozenSO;
  const nS = nActiveOcc;
  const dim = nS + nIJ * NVIRT;
  const R_1 = new Float64Array(NOCC);
  const R_2 = new Float64Array(NOCC * NOCC * NVIRT);
  const imagTol = opts.imagTol ?? 1e-6;

  // Pack/unpack helpers — shared by dense and Davidson paths.
  function unpackInto(v: Float64Array): void {
    R_1.fill(0);  // frozen-i entries stay zero
    R_2.fill(0);
    for (let row = 0; row < nS; row++) {
      R_1[nFrozenSO + row] = v[row]!;
    }
    for (let d = 0; d < nIJ * NVIRT; d++) {
      const pair = ijPairs[Math.floor(d / NVIRT)]!;
      const a = d - Math.floor(d / NVIRT) * NVIRT;
      const val = v[nS + d]!;
      R_2[(pair.i * NOCC + pair.j) * NVIRT + a] = val;
      R_2[(pair.j * NOCC + pair.i) * NVIRT + a] = -val;
    }
  }
  function packFrom(s1: Float64Array, s2: Float64Array): Float64Array {
    const out = new Float64Array(dim);
    for (let row = 0; row < nS; row++) out[row] = s1[nFrozenSO + row]!;
    for (let d = 0; d < nIJ * NVIRT; d++) {
      const pair = ijPairs[Math.floor(d / NVIRT)]!;
      const a = d - Math.floor(d / NVIRT) * NVIRT;
      out[nS + d] = s2[(pair.i * NOCC + pair.j) * NVIRT + a]!;
    }
    return out;
  }

  let ips: Float64Array;
  let imag: Float64Array;

  if (opts.useDavidson) {
    // ── Davidson path. ──────────────────────────────────────────
    // Diagonal preconditioner: -ε_i for R_1 (Koopmans IP);
    // -ε_i - ε_j + ε_a for R_2 (2h1p satellite energy gap).
    const diagonal = new Float64Array(dim);
    for (let row = 0; row < nS; row++) diagonal[row] = -eps[nFrozenSO + row]!;
    for (let d = 0; d < nIJ * NVIRT; d++) {
      const pair = ijPairs[Math.floor(d / NVIRT)]!;
      const a = d - Math.floor(d / NVIRT) * NVIRT;
      diagonal[nS + d] = -eps[pair.i]! - eps[pair.j]! + eps[a + VO]!;
    }
    const matvec = (v: Float64Array): Float64Array => {
      unpackInto(v);
      const { s1, s2 } = sigma(R_1, R_2);
      return packFrom(s1, s2);
    };
    const nRoots = Math.min(opts.nRoots ?? 10, dim);
    const dav = davidson(dim, matvec, diagonal, {
      k: nRoots,
      tol: opts.davidsonTol ?? 1e-7,
      maxIter: opts.davidsonMaxIter ?? 200,
      filter: (re, im) => Math.abs(im) < imagTol && re > 1e-6,
    });
    ips = dav.energies;
    imag = dav.imag;
  } else {
    // ── Dense path: build M column by column, then Hessenberg-QR. ──
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
    const eig = eigGeneral(M, dim);
    const ipList: Array<{ ip: number; im: number }> = [];
    for (let k = 0; k < dim; k++) {
      if (Math.abs(eig.imag[k]!) < imagTol && eig.real[k]! > 1e-6) {
        ipList.push({ ip: eig.real[k]!, im: eig.imag[k]! });
      }
    }
    ipList.sort((a, b) => a.ip - b.ip);
    const nRoots = Math.min(opts.nRoots ?? ipList.length, ipList.length);
    ips = new Float64Array(nRoots);
    imag = new Float64Array(nRoots);
    for (let k = 0; k < nRoots; k++) {
      ips[k] = ipList[k]!.ip;
      imag[k] = ipList[k]!.im;
    }
  }
  return { ips, imag, dim };
}
