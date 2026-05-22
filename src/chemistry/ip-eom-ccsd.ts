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
// σ-equations follow PySCF eom_gccsd.ipccsd_matvec (Tu, Wang & Li,
// J. Chem. Phys. 136, 174102 (2012) Eqs. 8-9) using the EOM-CCSD
// intermediates from gintermediates.py (Foo, Fov, Fvv with bare canonical
// Fock diagonal included; Woooo / Wvvvv / Wovvo with full τ/t2 dressings;
// Wooov / Wovoo proper PySCF builds).
//
// Verifier: tests/chemistry/ip-eom-ccsd-bruteforce.test.ts builds
// H̄ = e^(-T̂) H e^(T̂) explicitly in the 16-dim 2-electron H₂ Fock space,
// projects onto the (R_1 + antisym R_2) basis, and diffs the σ-equation
// matrix element-by-element. Hard regression assertion at < 1e-10 Ha.
//
// Ported 2026-05-22 from PySCF (Apache 2.0; see LICENSE-PYSCF). Pre-port
// the code re-derived from Stanton-Bartlett and shared the EE-EOM bug
// pattern: CC-T-equation intermediates used as EOM intermediates, plus
// the (t2, r2) σ_2 coupling term `+½ ⟨mn||ef⟩ r_2[m,n,f] T2[i,j,a,e]` was
// missing entirely. The known ~60 eV R_2 satellite over-count documented
// in earlier versions of this header is closed by the port.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";
import type { CCSDResult } from "./ccsd.js";
import {
  buildSpinOrbitalERI,
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

  const V = (P: number, Q: number, R: number, S: number): number =>
    eri[((P * NSO + Q) * NSO + R) * NSO + S]!;
  const VO = NOCC;

  // ── EOM-CCSD intermediates (PySCF gintermediates.py) ──────────
  // Same as in eom-ccsd.ts; IP-EOM only needs the subset used by
  // ipccsd_matvec (Foo, Fov, Fvv, Woooo, Wovvo, Wooov, Wovoo).
  // See feedback_porting_acceptance_gate.md — these EOM intermediates
  // differ from cc_ counterparts in T-dressing prefactors and inclusion
  // of the bare Fock diagonal.

  const Fov = new Float64Array(NOCC * NVIRT);
  for (let m = 0; m < NOCC; m++) {
    for (let e = 0; e < NVIRT; e++) {
      let s = 0;
      for (let nIdx = 0; nIdx < NOCC; nIdx++) {
        for (let f = 0; f < NVIRT; f++) {
          s += T1[nIdx * NVIRT + f]! * V(m, nIdx, e + VO, f + VO);
        }
      }
      Fov[m * NVIRT + e] = s;
    }
  }

  const Fvv = new Float64Array(NVIRT * NVIRT);
  for (let a = 0; a < NVIRT; a++) {
    for (let e = 0; e < NVIRT; e++) {
      let s = (a === e) ? eps[a + VO]! : 0;
      for (let m = 0; m < NOCC; m++) {
        for (let f = 0; f < NVIRT; f++) {
          s += T1[m * NVIRT + f]! * V(a + VO, m, e + VO, f + VO);
        }
      }
      for (let m = 0; m < NOCC; m++) {
        for (let nIdx = 0; nIdx < NOCC; nIdx++) {
          for (let f = 0; f < NVIRT; f++) {
            s -= 0.5 * tau_t[((m * NOCC + nIdx) * NVIRT + a) * NVIRT + f]! *
                       V(m, nIdx, e + VO, f + VO);
          }
        }
      }
      for (let m = 0; m < NOCC; m++) {
        s -= 0.5 * T1[m * NVIRT + a]! * Fov[m * NVIRT + e]!;
      }
      Fvv[a * NVIRT + e] = s;
    }
  }

  const Foo = new Float64Array(NOCC * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let i = 0; i < NOCC; i++) {
      let s = (m === i) ? eps[m]! : 0;
      for (let nIdx = 0; nIdx < NOCC; nIdx++) {
        for (let e = 0; e < NVIRT; e++) {
          s += T1[nIdx * NVIRT + e]! * V(m, nIdx, i, e + VO);
        }
      }
      for (let nIdx = 0; nIdx < NOCC; nIdx++) {
        for (let e = 0; e < NVIRT; e++) {
          for (let f = 0; f < NVIRT; f++) {
            s += 0.5 * tau_t[((i * NOCC + nIdx) * NVIRT + e) * NVIRT + f]! *
                       V(m, nIdx, e + VO, f + VO);
          }
        }
      }
      for (let e = 0; e < NVIRT; e++) {
        s += 0.5 * T1[i * NVIRT + e]! * Fov[m * NVIRT + e]!;
      }
      Foo[m * NOCC + i] = s;
    }
  }

  const Woooo = new Float64Array(NOCC * NOCC * NOCC * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let nIdx = 0; nIdx < NOCC; nIdx++) {
      for (let i = 0; i < NOCC; i++) {
        for (let j = 0; j < NOCC; j++) {
          let s = V(m, nIdx, i, j);
          for (let e = 0; e < NVIRT; e++) {
            s += T1[j * NVIRT + e]! * V(m, nIdx, i, e + VO);
            s -= T1[i * NVIRT + e]! * V(m, nIdx, j, e + VO);
          }
          for (let e = 0; e < NVIRT; e++) {
            for (let f = 0; f < NVIRT; f++) {
              s += 0.5 * tau[((i * NOCC + j) * NVIRT + e) * NVIRT + f]! *
                         V(m, nIdx, e + VO, f + VO);
            }
          }
          Woooo[((m * NOCC + nIdx) * NOCC + i) * NOCC + j] = s;
        }
      }
    }
  }

  const Wovvo = new Float64Array(NOCC * NVIRT * NVIRT * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let b = 0; b < NVIRT; b++) {
      for (let e = 0; e < NVIRT; e++) {
        for (let j = 0; j < NOCC; j++) {
          let s = V(m, b + VO, e + VO, j);
          for (let f = 0; f < NVIRT; f++) {
            s += T1[j * NVIRT + f]! * V(m, b + VO, e + VO, f + VO);
          }
          for (let nIdx = 0; nIdx < NOCC; nIdx++) {
            s -= T1[nIdx * NVIRT + b]! * V(m, nIdx, e + VO, j);
          }
          for (let nIdx = 0; nIdx < NOCC; nIdx++) {
            for (let f = 0; f < NVIRT; f++) {
              const t2v = T2[((j * NOCC + nIdx) * NVIRT + f) * NVIRT + b]!;
              const t1p = T1[j * NVIRT + f]! * T1[nIdx * NVIRT + b]!;
              s -= (t2v + t1p) * V(m, nIdx, e + VO, f + VO);
            }
          }
          Wovvo[((m * NVIRT + b) * NVIRT + e) * NOCC + j] = s;
        }
      }
    }
  }

  const Wooov = new Float64Array(NOCC * NOCC * NOCC * NVIRT);
  for (let m = 0; m < NOCC; m++) {
    for (let nIdx = 0; nIdx < NOCC; nIdx++) {
      for (let i = 0; i < NOCC; i++) {
        for (let e = 0; e < NVIRT; e++) {
          let s = V(m, nIdx, i, e + VO);
          for (let f = 0; f < NVIRT; f++) {
            s += T1[i * NVIRT + f]! * V(m, nIdx, f + VO, e + VO);
          }
          Wooov[((m * NOCC + nIdx) * NOCC + i) * NVIRT + e] = s;
        }
      }
    }
  }

  const Wovoo = new Float64Array(NOCC * NVIRT * NOCC * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let b = 0; b < NVIRT; b++) {
      for (let i = 0; i < NOCC; i++) {
        for (let j = 0; j < NOCC; j++) {
          let s = V(m, b + VO, i, j);
          for (let e = 0; e < NVIRT; e++) {
            s -= Fov[m * NVIRT + e]! *
                 T2[((i * NOCC + j) * NVIRT + b) * NVIRT + e]!;
          }
          for (let nIdx = 0; nIdx < NOCC; nIdx++) {
            s -= T1[nIdx * NVIRT + b]! *
                 Woooo[((m * NOCC + nIdx) * NOCC + i) * NOCC + j]!;
          }
          for (let e = 0; e < NVIRT; e++) {
            for (let f = 0; f < NVIRT; f++) {
              s += 0.5 * V(m, b + VO, e + VO, f + VO) *
                         tau[((i * NOCC + j) * NVIRT + e) * NVIRT + f]!;
            }
          }
          for (let nIdx = 0; nIdx < NOCC; nIdx++) {
            for (let e = 0; e < NVIRT; e++) {
              s += V(m, nIdx, i, e + VO) *
                   T2[((j * NOCC + nIdx) * NVIRT + b) * NVIRT + e]!;
              s -= V(m, nIdx, j, e + VO) *
                   T2[((i * NOCC + nIdx) * NVIRT + b) * NVIRT + e]!;
            }
          }
          for (let e = 0; e < NVIRT; e++) {
            s += T1[i * NVIRT + e]! * V(m, b + VO, e + VO, j);
            s -= T1[j * NVIRT + e]! * V(m, b + VO, e + VO, i);
          }
          for (let nIdx = 0; nIdx < NOCC; nIdx++) {
            for (let e = 0; e < NVIRT; e++) {
              for (let f = 0; f < NVIRT; f++) {
                const tvj = T2[((nIdx * NOCC + j) * NVIRT + b) * NVIRT + f]!;
                const tvi = T2[((nIdx * NOCC + i) * NVIRT + b) * NVIRT + f]!;
                const eV = V(m, nIdx, e + VO, f + VO);
                s -= T1[i * NVIRT + e]! * tvj * eV;
                s += T1[j * NVIRT + e]! * tvi * eV;
              }
            }
          }
          Wovoo[((m * NVIRT + b) * NOCC + i) * NOCC + j] = s;
        }
      }
    }
  }

  // ── σ via PySCF eom_gccsd.ipccsd_matvec, Eqs. (8)-(9). ──────────
  // Ref: Tu, Wang, and Li, J. Chem. Phys. 136, 174102 (2012).
  function sigma(R_1: Float64Array, R_2: Float64Array): {
    s1: Float64Array;
    s2: Float64Array;
  } {
    const s1 = new Float64Array(NOCC);
    const s2 = new Float64Array(NOCC * NOCC * NVIRT);

    // σ_1[i] (Eq. 8):
    //   −Σ_m Foo[m,i] r_1[m]
    //   +Σ_me Fov[m,e] r_2[m,i,e]
    //   −½ Σ_nme Wooov[n,m,i,e] r_2[m,n,e]
    for (let i = 0; i < NOCC; i++) {
      let s = 0;
      for (let m = 0; m < NOCC; m++) {
        s -= Foo[m * NOCC + i]! * R_1[m]!;
      }
      for (let m = 0; m < NOCC; m++) {
        for (let e = 0; e < NVIRT; e++) {
          s += Fov[m * NVIRT + e]! * R_2[(m * NOCC + i) * NVIRT + e]!;
        }
      }
      for (let nIdx = 0; nIdx < NOCC; nIdx++) {
        for (let m = 0; m < NOCC; m++) {
          for (let e = 0; e < NVIRT; e++) {
            s -= 0.5 * Wooov[((nIdx * NOCC + m) * NOCC + i) * NVIRT + e]! *
                       R_2[(m * NOCC + nIdx) * NVIRT + e]!;
          }
        }
      }
      s1[i] = s;
    }

    // σ_2[i,j,a] (Eq. 9):
    //   +Σ_e Fvv[a,e] r_2[i,j,e]
    //   −P(ij) Σ_m Foo[m,i] r_2[m,j,a]
    //   −Σ_m Wovoo[m,a,j,i] r_1[m]
    //   +½ Σ_mn Woooo[m,n,i,j] r_2[m,n,a]
    //   +P(ij) Σ_me Wovvo[m,a,e,i] r_2[m,j,e]
    //   +½ Σ_mnef ⟨mn||ef⟩ r_2[m,n,f] T2[i,j,a,e]
    for (let i = 0; i < NOCC; i++) {
      for (let j = 0; j < NOCC; j++) {
        for (let a = 0; a < NVIRT; a++) {
          let z = 0;
          const idx_ija = (i * NOCC + j) * NVIRT + a;
          // + Σ_e Fvv[a,e] r_2[i,j,e]
          for (let e = 0; e < NVIRT; e++) {
            z += Fvv[a * NVIRT + e]! * R_2[(i * NOCC + j) * NVIRT + e]!;
          }
          // − P(ij) Σ_m Foo[m,i] r_2[m,j,a]
          for (let m = 0; m < NOCC; m++) {
            z -= Foo[m * NOCC + i]! * R_2[(m * NOCC + j) * NVIRT + a]!;
            z += Foo[m * NOCC + j]! * R_2[(m * NOCC + i) * NVIRT + a]!;
          }
          // − Σ_m Wovoo[m,a,j,i] r_1[m]
          for (let m = 0; m < NOCC; m++) {
            z -= Wovoo[((m * NVIRT + a) * NOCC + j) * NOCC + i]! * R_1[m]!;
          }
          // + ½ Σ_mn Woooo[m,n,i,j] r_2[m,n,a]
          for (let m = 0; m < NOCC; m++) {
            for (let nIdx = 0; nIdx < NOCC; nIdx++) {
              z += 0.5 * Woooo[((m * NOCC + nIdx) * NOCC + i) * NOCC + j]! *
                         R_2[(m * NOCC + nIdx) * NVIRT + a]!;
            }
          }
          // + P(ij) Σ_me Wovvo[m,a,e,i] r_2[m,j,e]
          for (let m = 0; m < NOCC; m++) {
            for (let e = 0; e < NVIRT; e++) {
              z += Wovvo[((m * NVIRT + a) * NVIRT + e) * NOCC + i]! *
                   R_2[(m * NOCC + j) * NVIRT + e]!;
              z -= Wovvo[((m * NVIRT + a) * NVIRT + e) * NOCC + j]! *
                   R_2[(m * NOCC + i) * NVIRT + e]!;
            }
          }
          // + ½ Σ_mnef ⟨mn||ef⟩ r_2[m,n,f] T2[i,j,a,e]
          for (let m = 0; m < NOCC; m++) {
            for (let nIdx = 0; nIdx < NOCC; nIdx++) {
              for (let e = 0; e < NVIRT; e++) {
                for (let f = 0; f < NVIRT; f++) {
                  z += 0.5 * V(m, nIdx, e + VO, f + VO) *
                             R_2[(m * NOCC + nIdx) * NVIRT + f]! *
                             T2[((i * NOCC + j) * NVIRT + a) * NVIRT + e]!;
                }
              }
            }
          }
          s2[idx_ija] = z;
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
