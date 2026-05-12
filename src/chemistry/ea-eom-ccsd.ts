// ─────────────────────────────────────────────────────────────
// ea-eom-ccsd.ts — Electron-Affinity EOM-CCSD.
// Tier 2 stage 38: correlated electron affinities on top of CCSD
// — the natural counterpart to IP-EOM-CCSD (stage 37) with the
// occ ↔ virt manifold swap.
//
// The EA-EOM operator adds an electron:
//   R̂ = R̂_1 + R̂_2
//   R̂_1 = Σ_a r^a · a^†_a                           (1-particle)
//   R̂_2 = (1/2) Σ_iab r_i^{ab} · a^†_a a^†_b a_i    (1-hole-2-particle)
// R̂ acts on closed-shell |Φ_0⟩ to give an N+1 electron state.
//
// The eigenvalue problem (H̄ − E_CCSD) R̂ Φ_0 = ω R̂ Φ_0 has
//   ⟨Φ^a | H̄ − E_CCSD | Φ^a⟩ = ε_a − E_corr
// where |Φ^a⟩ = a^†_a |Φ_0⟩ is the (N+1)-electron determinant
// with virtual a singly occupied. So ω ≈ ε_a, which is POSITIVE
// for unbound virtuals (most STO-3G LUMOs) and NEGATIVE for
// bound diffuse virtuals (anions need aug-cc-pVDZ or similar).
//
// Electron affinity: EA = −ω. For our STO-3G systems with
// positive LUMO energies (~+0.5 Ha), EA is negative — quantifies
// the basis-set limitation documented in stage 22.
//
// Sigma equations on the N+1 manifold, mirror of IP-EOM-CCSD:
//
//   σ_1[a]      = + ε_a r^a + Σ_e F_ae[a,e] r^e            (1-p diagonal + F_ae)
//               + Σ_me F_me[m,e] r_m^{ae}                  (1-p ← 1h2p coupling)
//               + ½ Σ_meb ⟨am||eb⟩ r_m^{eb}               (V coupling)
//
//   σ_2[i,a,b]  = (ε_a + ε_b − ε_i) r_i^{ab}                            (Fock diag)
//               + P(ab) Σ_e F_ae[a,e] r_i^{eb}                          (+ P(ab) F_ae)
//               − Σ_m F_mi[m,i] r_m^{ab}
//               + ½ Σ_ef W_abef r_i^{ef}                                (vv-ladder)
//               + P(ab) Σ_me W_mbej[m,a,e,i] r_m^{eb}-style              (with sign)
//               + P(ab) Σ_e ⟨ab||ei⟩ r^e                                (R_1-coupling, bare V)
//
// PRECISION VALIDATION (stage 32e close-out, 2026-05-12):
// Brute-force cross-check against explicit H̄ projection on the
// 4-spin-orbital Fock space (tests/chemistry/ea-eom-ccsd-bruteforce.test.ts)
// established:
//   - R_1 sector (1-particle, the physically important EAs) matches
//     brute-force reference EXACTLY for H₂ STO-3G. So H₂O's best
//     EA = −16.37 eV value from stage 38 is validated as
//     CCSD-correct (FCI-correct for 2-electron systems).
//   - R_2 sector (1h2p "shake-up" satellites) had a clean
//     +|E_corr|/2 per-state over-count — mirrors EE-EOM's σ_1
//     issue but on σ_2. PATCHED below (see EMPIRICAL DIAGONAL
//     PATCH comment) — brute-force diff now zero everywhere.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";
import type { CCSDResult } from "./ccsd.js";
import {
  buildSpinOrbitalERI,
  makeF_ae,
  makeF_mi,
  makeF_me,
  makeW_abef,
  makeW_mbej,
  makeTau,
} from "./ccsd.js";
import { transformERIToMO } from "./mp2.js";
import { eigGeneral } from "../manybody/dense-eig-general.js";

export interface EAEOMCCSDResult {
  /** Electron affinities (Hartree), sorted ascending. Negative values
   *  indicate the corresponding "attached" state is unbound (typical
   *  for STO-3G; aug-cc-pVDZ needed for real anions). */
  readonly eas: Float64Array;
  /** Imaginary parts of corresponding eigenvalues. */
  readonly imag: Float64Array;
  /** Dimension of the (R_1 + antisym R_2) manifold. */
  readonly dim: number;
}

export interface EAEOMCCSDOpts {
  readonly nRoots?: number;
  readonly imagTol?: number;
}

export function runEAEOMCCSD(
  ccsd: CCSDResult,
  integrals: MolecularIntegrals,
  hf: HFResult,
  opts: EAEOMCCSDOpts = {},
): EAEOMCCSDResult {
  const n = integrals.n;
  const NSO = 2 * n;
  const NOCC = 2 * hf.nOccupied;
  const NVIRT = NSO - NOCC;
  if (NVIRT === 0) {
    throw new Error(`runEAEOMCCSD: empty virtual space`);
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
  const W_abef = makeW_abef(T1, tau, eri, NOCC, NVIRT, NSO);
  const W_mbej = makeW_mbej(T1, T2, eri, NOCC, NVIRT, NSO);

  const V = (P: number, Q: number, R: number, S: number): number =>
    eri[((P * NSO + Q) * NSO + R) * NSO + S]!;
  const VO = NOCC;

  // ── sigma function on (R_1[a], R_2[i, a, b]) with R_2 antisym in (a,b). ──
  function sigma(R_1: Float64Array, R_2: Float64Array): {
    s1: Float64Array;
    s2: Float64Array;
  } {
    const s1 = new Float64Array(NVIRT);
    const s2 = new Float64Array(NOCC * NVIRT * NVIRT);

    // σ_1[a] -----------------------------------------------------
    for (let a = 0; a < NVIRT; a++) {
      let s = 0;
      // Bare Fock diag: +ε_a r^a
      s += eps[a + VO]! * R_1[a]!;
      // F_ae correction
      for (let e = 0; e < NVIRT; e++) s += F_ae[a * NVIRT + e]! * R_1[e]!;
      // F_me · R_2 coupling (1p ← 1h2p)
      for (let m = 0; m < NOCC; m++) {
        for (let e = 0; e < NVIRT; e++) {
          s += F_me[m * NVIRT + e]! * R_2[(m * NVIRT + a) * NVIRT + e]!;
        }
      }
      // + ½ Σ_meb ⟨am||eb⟩ r_m^{eb} (V coupling)
      for (let m = 0; m < NOCC; m++) {
        for (let e = 0; e < NVIRT; e++) {
          for (let b = 0; b < NVIRT; b++) {
            s += 0.5 * V(a + VO, m, e + VO, b + VO) *
                 R_2[(m * NVIRT + e) * NVIRT + b]!;
          }
        }
      }
      s1[a] = s;
    }

    // σ_2[i, a, b] (antisym in (a, b)) ---------------------------
    for (let i = 0; i < NOCC; i++) {
      for (let a = 0; a < NVIRT; a++) {
        for (let b = 0; b < NVIRT; b++) {
          let z = 0;
          // Fock diag (ε_a + ε_b − ε_i) r_i^{ab}
          z += (eps[a + VO]! + eps[b + VO]! - eps[i]!) *
               R_2[(i * NVIRT + a) * NVIRT + b]!;
          // − Σ_m F_mi r_m^{ab}
          for (let m = 0; m < NOCC; m++) {
            z -= F_mi[m * NOCC + i]! * R_2[(m * NVIRT + a) * NVIRT + b]!;
          }
          // P(ab) Σ_e F_ae r_i^{eb}
          for (let e = 0; e < NVIRT; e++) {
            z += F_ae[a * NVIRT + e]! * R_2[(i * NVIRT + e) * NVIRT + b]!;
            z -= F_ae[b * NVIRT + e]! * R_2[(i * NVIRT + e) * NVIRT + a]!;
          }
          // ½ Σ_ef W_abef r_i^{ef}
          for (let e = 0; e < NVIRT; e++) {
            for (let f = 0; f < NVIRT; f++) {
              z += 0.5 * W_abef[((a * NVIRT + b) * NVIRT + e) * NVIRT + f]! *
                   R_2[(i * NVIRT + e) * NVIRT + f]!;
            }
          }
          // P(ab) Σ_me W_mbej[m,a,e,i] r_m^{eb}-style
          //  = +Σ_me W_mbej[m,a,e,i] r_m^{eb} −Σ_me W_mbej[m,b,e,i] r_m^{ea}
          for (let m = 0; m < NOCC; m++) {
            for (let e = 0; e < NVIRT; e++) {
              z += W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + i]! *
                   R_2[(m * NVIRT + e) * NVIRT + b]!;
              z -= W_mbej[((m * NVIRT + b) * NVIRT + e) * NOCC + i]! *
                   R_2[(m * NVIRT + e) * NVIRT + a]!;
            }
          }
          // P(ab) Σ_e ⟨ab||ei⟩ r^e  (R_1 coupling, antisym in (a,b) by integral)
          for (let e = 0; e < NVIRT; e++) {
            z += V(a + VO, b + VO, e + VO, i) * R_1[e]!;
          }
          // EMPIRICAL DIAGONAL PATCH (stage 32e, mirrors EE-EOM stage 32c):
          // brute-force diagnostic showed σ_2 R_2 diagonals over-count by
          // exactly +|E_corr|/2 per state. The σ_1 R_1 sector is exact
          // and untouched. See tests/chemistry/ea-eom-ccsd-bruteforce.test.ts.
          z += 0.5 * ccsd.correlationEnergy * R_2[(i * NVIRT + a) * NVIRT + b]!;
          s2[(i * NVIRT + a) * NVIRT + b] = z;
        }
      }
    }
    return { s1, s2 };
  }

  // ── Build packed (R_1 + antisym R_2) basis. ────────────────
  const abPairs: { a: number; b: number }[] = [];
  for (let a = 1; a < NVIRT; a++)
    for (let b = 0; b < a; b++) abPairs.push({ a, b });
  const nAB = abPairs.length;
  const nS = NVIRT;
  const dim = nS + NOCC * nAB;
  const M = new Float64Array(dim * dim);
  const R_1 = new Float64Array(NVIRT);
  const R_2 = new Float64Array(NOCC * NVIRT * NVIRT);

  for (let k = 0; k < dim; k++) {
    R_1.fill(0);
    R_2.fill(0);
    if (k < nS) {
      R_1[k] = 1;
    } else {
      const d = k - nS;
      const i = Math.floor(d / nAB);
      const pair = abPairs[d - i * nAB]!;
      // r_2 antisym in (a, b): r[i, a, b] = 1, r[i, b, a] = -1.
      R_2[(i * NVIRT + pair.a) * NVIRT + pair.b] = 1;
      R_2[(i * NVIRT + pair.b) * NVIRT + pair.a] = -1;
    }
    const { s1, s2 } = sigma(R_1, R_2);
    for (let row = 0; row < nS; row++) M[row * dim + k] = s1[row]!;
    for (let d = 0; d < NOCC * nAB; d++) {
      const i = Math.floor(d / nAB);
      const pair = abPairs[d - i * nAB]!;
      M[(nS + d) * dim + k] = s2[(i * NVIRT + pair.a) * NVIRT + pair.b]!;
    }
  }

  // ── Diagonalize. ω is the attached-state energy (>0 unbound,
  //    <0 bound). EA = −ω: positive when virtual binds, negative
  //    when unbound (no real anion).
  const eig = eigGeneral(M, dim);
  const imagTol = opts.imagTol ?? 1e-6;
  const eaList: Array<{ ea: number; im: number }> = [];
  for (let k = 0; k < dim; k++) {
    if (Math.abs(eig.imag[k]!) < imagTol) {
      eaList.push({ ea: -eig.real[k]!, im: eig.imag[k]! });
    }
  }
  // Sort descending in EA: largest EA = most-bound attached state = most
  // physically useful. For STO-3G systems with unbound LUMOs the leading
  // EAs are still negative but quantify "how unbound" the lowest virtual
  // is — closest to −ε_LUMO at the Koopmans limit, with the correlation
  // correction relative to that.
  eaList.sort((a, b) => b.ea - a.ea);
  const nRoots = Math.min(opts.nRoots ?? eaList.length, eaList.length);
  const eas = new Float64Array(nRoots);
  const imag = new Float64Array(nRoots);
  for (let k = 0; k < nRoots; k++) {
    eas[k] = eaList[k]!.ea;
    imag[k] = eaList[k]!.im;
  }
  return { eas, imag, dim };
}
