// Brute-force IP-EOM-CCSD reference for H₂ STO-3G — companion to
// eom-ccsd-bruteforce.test.ts. Verifies whether IP-EOM has the same
// structural σ-equation error as EE-EOM (stage 32b/32c).
//
// 1-hole basis: |Φ_i⟩ = a_i |Φ_0⟩ for i ∈ occupied SOs.
// 2h1p basis (antisym in i, j with i>j): |Φ_{ij}^a⟩ = a^†_a a_j a_i |Φ_0⟩.
import { describe, test, expect } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import {
  runCCSD,
  buildSpinOrbitalERI,
  makeTau,
} from "../../src/chemistry/ccsd.js";
import { runIPEOMCCSD } from "../../src/chemistry/ip-eom-ccsd.js";
import { transformERIToMO } from "../../src/chemistry/mp2.js";
import { eigGeneral } from "../../src/manybody/dense-eig-general.js";

const H2: Atom[] = [
  { symbol: "H", pos: [0, 0, 0] },
  { symbol: "H", pos: [0, 0, 0.7414] },
];

const NSO = 4;
const DIM = 16;

const occ = (s: number, P: number): number => (s >>> P) & 1;
function annSign(s: number, P: number): number {
  let n = 0;
  for (let q = 0; q < P; q++) n += occ(s, q);
  return n % 2 === 0 ? 1 : -1;
}
function annihilate(s: number, P: number): { newState: number; sign: number } | null {
  if (occ(s, P) === 0) return null;
  return { newState: s & ~(1 << P), sign: annSign(s, P) };
}
function create(s: number, P: number): { newState: number; sign: number } | null {
  if (occ(s, P) === 1) return null;
  return { newState: s | (1 << P), sign: annSign(s, P) };
}
function applyChain(
  s: number,
  ops: { type: "a" | "a+"; P: number }[],
): { newState: number; sign: number } | null {
  let state = s;
  let totalSign = 1;
  for (let k = ops.length - 1; k >= 0; k--) {
    const op = ops[k]!;
    const r = op.type === "a" ? annihilate(state, op.P) : create(state, op.P);
    if (r === null) return null;
    state = r.newState;
    totalSign *= r.sign;
  }
  return { newState: state, sign: totalSign };
}

function buildFockHamiltonian(
  h_SO: Float64Array,
  eri_SO: Float64Array,
  enuc: number,
): Float64Array {
  const H = new Float64Array(DIM * DIM);
  for (let s = 0; s < DIM; s++) H[s * DIM + s] = enuc;
  for (let p = 0; p < NSO; p++) {
    for (let q = 0; q < NSO; q++) {
      const h_pq = h_SO[p * NSO + q]!;
      if (h_pq === 0) continue;
      for (let s = 0; s < DIM; s++) {
        const r = applyChain(s, [{ type: "a+", P: p }, { type: "a", P: q }]);
        if (r === null) continue;
        const idx = r.newState * DIM + s;
        H[idx] = H[idx]! + h_pq * r.sign;
      }
    }
  }
  for (let p = 0; p < NSO; p++) {
    for (let q = 0; q < NSO; q++) {
      for (let r = 0; r < NSO; r++) {
        for (let sIdx = 0; sIdx < NSO; sIdx++) {
          const V = eri_SO[((p * NSO + q) * NSO + r) * NSO + sIdx]!;
          if (V === 0) continue;
          for (let st = 0; st < DIM; st++) {
            const res = applyChain(st, [
              { type: "a+", P: p },
              { type: "a+", P: q },
              { type: "a", P: sIdx },
              { type: "a", P: r },
            ]);
            if (res === null) continue;
            const idx = res.newState * DIM + st;
            H[idx] = H[idx]! + 0.25 * V * res.sign;
          }
        }
      }
    }
  }
  return H;
}

function buildSpinOrbitalH(h_MO: Float64Array, n: number): Float64Array {
  const h_SO = new Float64Array(NSO * NSO);
  for (let P = 0; P < NSO; P++) {
    for (let Q = 0; Q < NSO; Q++) {
      if ((P & 1) !== (Q & 1)) continue;
      const p = P >> 1, q = Q >> 1;
      h_SO[P * NSO + Q] = h_MO[p * n + q]!;
    }
  }
  return h_SO;
}

function buildT2Matrix(T2: Float64Array, NOCC: number, NVIRT: number): Float64Array {
  const T = new Float64Array(DIM * DIM);
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let a = 0; a < NVIRT; a++) {
        for (let b = 0; b < NVIRT; b++) {
          const t = T2[((i * NOCC + j) * NVIRT + a) * NVIRT + b]!;
          if (t === 0) continue;
          const A = a + NOCC, B = b + NOCC;
          for (let s = 0; s < DIM; s++) {
            const r = applyChain(s, [
              { type: "a+", P: A },
              { type: "a+", P: B },
              { type: "a", P: j },
              { type: "a", P: i },
            ]);
            if (r === null) continue;
            const idx = r.newState * DIM + s;
            T[idx] = T[idx]! + 0.25 * t * r.sign;
          }
        }
      }
    }
  }
  return T;
}

function matmul(A: Float64Array, B: Float64Array): Float64Array {
  const C = new Float64Array(DIM * DIM);
  for (let i = 0; i < DIM; i++) {
    for (let k = 0; k < DIM; k++) {
      const aik = A[i * DIM + k]!;
      if (aik === 0) continue;
      for (let j = 0; j < DIM; j++) {
        const idx = i * DIM + j;
        C[idx] = C[idx]! + aik * B[k * DIM + j]!;
      }
    }
  }
  return C;
}

function eyePlusT(T: Float64Array, sign: 1 | -1): Float64Array {
  const out = new Float64Array(T);
  if (sign < 0) for (let i = 0; i < out.length; i++) out[i] = -out[i]!;
  for (let i = 0; i < DIM; i++) {
    const idx = i * DIM + i;
    out[idx] = out[idx]! + 1;
  }
  return out;
}

// 1-hole basis: |Φ_i⟩ = a_i |Φ_0⟩.
function holeVec(i: number, vac: number): Float64Array {
  const v = new Float64Array(DIM);
  const r = applyChain(vac, [{ type: "a", P: i }]);
  if (r !== null) v[r.newState] = r.sign;
  return v;
}

// 2h1p basis: |Φ_{ij}^a⟩ = a^†_(a+NOCC) a_j a_i |Φ_0⟩.
function twohpVec(i: number, j: number, a: number, NOCC: number, vac: number): Float64Array {
  const v = new Float64Array(DIM);
  const r = applyChain(vac, [
    { type: "a+", P: a + NOCC },
    { type: "a", P: j },
    { type: "a", P: i },
  ]);
  if (r !== null) v[r.newState] = r.sign;
  return v;
}

function quadForm(u: Float64Array, M: Float64Array, v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < DIM; i++) {
    const ui = u[i]!;
    if (ui === 0) continue;
    for (let j = 0; j < DIM; j++) s += ui * M[i * DIM + j]! * v[j]!;
  }
  return s;
}

describe("IP-EOM-CCSD brute-force diagnostic (H₂ STO-3G)", () => {
  test("M_exact eigenvalues from explicit H̄ projection vs σ-equation", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const n = integrals.n;
    const hf = runRHFSCF(integrals, 2);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-12 });
    const NOCC = 2, NVIRT = 2;
    void NOCC; void NVIRT;

    const C = hf.C_MO;
    const tmp = new Float64Array(n * n);
    for (let p = 0; p < n; p++) {
      for (let nu = 0; nu < n; nu++) {
        let s = 0;
        for (let mu = 0; mu < n; mu++) s += C[mu * n + p]! * integrals.h_AO[mu * n + nu]!;
        tmp[p * n + nu] = s;
      }
    }
    const h_MO = new Float64Array(n * n);
    for (let p = 0; p < n; p++) {
      for (let q = 0; q < n; q++) {
        let s = 0;
        for (let nu = 0; nu < n; nu++) s += tmp[p * n + nu]! * C[nu * n + q]!;
        h_MO[p * n + q] = s;
      }
    }
    const eri_MO = transformERIToMO(integrals.eri_AO, C, n);
    const eri_SO = buildSpinOrbitalERI(eri_MO, n);
    const h_SO = buildSpinOrbitalH(h_MO, n);
    const H_full = buildFockHamiltonian(h_SO, eri_SO, integrals.Vnn);
    const T_mat = buildT2Matrix(ccsd.T2, NOCC, NVIRT);
    const Hbar = matmul(matmul(eyePlusT(T_mat, -1), H_full), eyePlusT(T_mat, +1));

    // ── IP-EOM basis: 2 R_1 (1-hole) + 2 R_2 (2h1p packed at i>j). ──
    const vac_HF = (1 << 0) | (1 << 1);
    const basisLabels: string[] = [];
    const basisVecs: Float64Array[] = [];
    for (let i = 0; i < NOCC; i++) {
      basisLabels.push(`R₁[${i}]`);
      basisVecs.push(holeVec(i, vac_HF));
    }
    // Packed 2h1p with (i, j) = (1, 0) and a varying. R_2 antisym sign baked into the chain.
    for (let a = 0; a < NVIRT; a++) {
      basisLabels.push(`R₂[1,0,${a}]`);
      basisVecs.push(twohpVec(1, 0, a, NOCC, vac_HF));
    }
    const dim = basisVecs.length;

    // Build M_exact projection of (H̄ - E_CCSD) onto this basis.
    const E_CCSD = ccsd.totalEnergy;
    const M_exact = new Float64Array(dim * dim);
    const S = new Float64Array(dim * dim);
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        let sij = 0;
        for (let k = 0; k < DIM; k++) sij += basisVecs[i]![k]! * basisVecs[j]![k]!;
        S[i * dim + j] = sij;
        const hij = quadForm(basisVecs[i]!, Hbar, basisVecs[j]!);
        M_exact[i * dim + j] = hij - E_CCSD * sij;
      }
    }
    console.log(`[bf-ip-h2] Basis overlap diagonals: [${
      Array.from({ length: dim }, (_, i) => S[i * dim + i]!.toFixed(3)).join(", ")
    }]`);

    // Eigenvalues of M_exact = expected IP-EOM-CCSD eigenvalues for H₂.
    // Use eigGeneral since H̄ is non-Hermitian.
    const eigExact = eigGeneral(M_exact, dim);
    const exactSorted = Array.from(eigExact.real).sort((a, b) => a - b);
    console.log(`[bf-ip-h2] M_exact eigenvalues (Ha): [${
      exactSorted.map(x => x.toFixed(8)).join(", ")
    }]`);

    // Run my IP-EOM-CCSD and compare.
    const ip = runIPEOMCCSD(ccsd, integrals, hf);
    const myIps = Array.from(ip.ips).sort((a, b) => a - b);
    console.log(`[bf-ip-h2] σ-equation IPs (Ha, sorted): [${
      myIps.map(x => x.toFixed(8)).join(", ")
    }]`);

    // ── Build M_mine in the same basis. ──
    const NSO_full = 2 * n;
    const NOCC_full = 2 * hf.nOccupied;
    const NVIRT_full = NSO_full - NOCC_full;
    const eps_full = new Float64Array(NSO_full);
    for (let P = 0; P < NSO_full; P++) eps_full[P] = hf.orbitalEnergies[P >> 1]!;
    const tau_t = makeTau(ccsd.T1, ccsd.T2, NOCC_full, NVIRT_full, 0.5);
    const tau_full = makeTau(ccsd.T1, ccsd.T2, NOCC_full, NVIRT_full, 1.0);
    // ── EOM-CCSD intermediates (PySCF gintermediates.py). ──
    // Mirror of src/chemistry/ip-eom-ccsd.ts post-PySCF port (2026-05-22).
    const V_eri = (P: number, Q: number, R: number, S: number): number =>
      eri_SO[((P * NSO_full + Q) * NSO_full + R) * NSO_full + S]!;
    const VO = NOCC_full;

    const Fov = new Float64Array(NOCC_full * NVIRT_full);
    for (let m = 0; m < NOCC_full; m++) {
      for (let e = 0; e < NVIRT_full; e++) {
        let s = 0;
        for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
          for (let f = 0; f < NVIRT_full; f++) {
            s += ccsd.T1[nIdx * NVIRT_full + f]! * V_eri(m, nIdx, e + VO, f + VO);
          }
        }
        Fov[m * NVIRT_full + e] = s;
      }
    }
    const Fvv = new Float64Array(NVIRT_full * NVIRT_full);
    for (let a = 0; a < NVIRT_full; a++) {
      for (let e = 0; e < NVIRT_full; e++) {
        let s = (a === e) ? eps_full[a + VO]! : 0;
        for (let m = 0; m < NOCC_full; m++) {
          for (let f = 0; f < NVIRT_full; f++) {
            s += ccsd.T1[m * NVIRT_full + f]! * V_eri(a + VO, m, e + VO, f + VO);
          }
        }
        for (let m = 0; m < NOCC_full; m++) {
          for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
            for (let f = 0; f < NVIRT_full; f++) {
              s -= 0.5 * tau_t[((m * NOCC_full + nIdx) * NVIRT_full + a) * NVIRT_full + f]! *
                         V_eri(m, nIdx, e + VO, f + VO);
            }
          }
        }
        for (let m = 0; m < NOCC_full; m++) {
          s -= 0.5 * ccsd.T1[m * NVIRT_full + a]! * Fov[m * NVIRT_full + e]!;
        }
        Fvv[a * NVIRT_full + e] = s;
      }
    }
    const Foo = new Float64Array(NOCC_full * NOCC_full);
    for (let m = 0; m < NOCC_full; m++) {
      for (let i = 0; i < NOCC_full; i++) {
        let s = (m === i) ? eps_full[m]! : 0;
        for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
          for (let e = 0; e < NVIRT_full; e++) {
            s += ccsd.T1[nIdx * NVIRT_full + e]! * V_eri(m, nIdx, i, e + VO);
          }
        }
        for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
          for (let e = 0; e < NVIRT_full; e++) {
            for (let f = 0; f < NVIRT_full; f++) {
              s += 0.5 * tau_t[((i * NOCC_full + nIdx) * NVIRT_full + e) * NVIRT_full + f]! *
                         V_eri(m, nIdx, e + VO, f + VO);
            }
          }
        }
        for (let e = 0; e < NVIRT_full; e++) {
          s += 0.5 * ccsd.T1[i * NVIRT_full + e]! * Fov[m * NVIRT_full + e]!;
        }
        Foo[m * NOCC_full + i] = s;
      }
    }
    const Woooo = new Float64Array(NOCC_full * NOCC_full * NOCC_full * NOCC_full);
    for (let m = 0; m < NOCC_full; m++) {
      for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
        for (let i = 0; i < NOCC_full; i++) {
          for (let j = 0; j < NOCC_full; j++) {
            let s = V_eri(m, nIdx, i, j);
            for (let e = 0; e < NVIRT_full; e++) {
              s += ccsd.T1[j * NVIRT_full + e]! * V_eri(m, nIdx, i, e + VO);
              s -= ccsd.T1[i * NVIRT_full + e]! * V_eri(m, nIdx, j, e + VO);
            }
            for (let e = 0; e < NVIRT_full; e++) {
              for (let f = 0; f < NVIRT_full; f++) {
                s += 0.5 * tau_full[((i * NOCC_full + j) * NVIRT_full + e) * NVIRT_full + f]! *
                           V_eri(m, nIdx, e + VO, f + VO);
              }
            }
            Woooo[((m * NOCC_full + nIdx) * NOCC_full + i) * NOCC_full + j] = s;
          }
        }
      }
    }
    const Wovvo = new Float64Array(NOCC_full * NVIRT_full * NVIRT_full * NOCC_full);
    for (let m = 0; m < NOCC_full; m++) {
      for (let b = 0; b < NVIRT_full; b++) {
        for (let e = 0; e < NVIRT_full; e++) {
          for (let j = 0; j < NOCC_full; j++) {
            let s = V_eri(m, b + VO, e + VO, j);
            for (let f = 0; f < NVIRT_full; f++) {
              s += ccsd.T1[j * NVIRT_full + f]! * V_eri(m, b + VO, e + VO, f + VO);
            }
            for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
              s -= ccsd.T1[nIdx * NVIRT_full + b]! * V_eri(m, nIdx, e + VO, j);
            }
            for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
              for (let f = 0; f < NVIRT_full; f++) {
                const t2v = ccsd.T2[((j * NOCC_full + nIdx) * NVIRT_full + f) * NVIRT_full + b]!;
                const t1p = ccsd.T1[j * NVIRT_full + f]! * ccsd.T1[nIdx * NVIRT_full + b]!;
                s -= (t2v + t1p) * V_eri(m, nIdx, e + VO, f + VO);
              }
            }
            Wovvo[((m * NVIRT_full + b) * NVIRT_full + e) * NOCC_full + j] = s;
          }
        }
      }
    }
    const Wooov = new Float64Array(NOCC_full * NOCC_full * NOCC_full * NVIRT_full);
    for (let m = 0; m < NOCC_full; m++) {
      for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
        for (let i = 0; i < NOCC_full; i++) {
          for (let e = 0; e < NVIRT_full; e++) {
            let s = V_eri(m, nIdx, i, e + VO);
            for (let f = 0; f < NVIRT_full; f++) {
              s += ccsd.T1[i * NVIRT_full + f]! * V_eri(m, nIdx, f + VO, e + VO);
            }
            Wooov[((m * NOCC_full + nIdx) * NOCC_full + i) * NVIRT_full + e] = s;
          }
        }
      }
    }
    const Wovoo = new Float64Array(NOCC_full * NVIRT_full * NOCC_full * NOCC_full);
    for (let m = 0; m < NOCC_full; m++) {
      for (let b = 0; b < NVIRT_full; b++) {
        for (let i = 0; i < NOCC_full; i++) {
          for (let j = 0; j < NOCC_full; j++) {
            let s = V_eri(m, b + VO, i, j);
            for (let e = 0; e < NVIRT_full; e++) {
              s -= Fov[m * NVIRT_full + e]! *
                   ccsd.T2[((i * NOCC_full + j) * NVIRT_full + b) * NVIRT_full + e]!;
            }
            for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
              s -= ccsd.T1[nIdx * NVIRT_full + b]! *
                   Woooo[((m * NOCC_full + nIdx) * NOCC_full + i) * NOCC_full + j]!;
            }
            for (let e = 0; e < NVIRT_full; e++) {
              for (let f = 0; f < NVIRT_full; f++) {
                s += 0.5 * V_eri(m, b + VO, e + VO, f + VO) *
                           tau_full[((i * NOCC_full + j) * NVIRT_full + e) * NVIRT_full + f]!;
              }
            }
            for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
              for (let e = 0; e < NVIRT_full; e++) {
                s += V_eri(m, nIdx, i, e + VO) *
                     ccsd.T2[((j * NOCC_full + nIdx) * NVIRT_full + b) * NVIRT_full + e]!;
                s -= V_eri(m, nIdx, j, e + VO) *
                     ccsd.T2[((i * NOCC_full + nIdx) * NVIRT_full + b) * NVIRT_full + e]!;
              }
            }
            for (let e = 0; e < NVIRT_full; e++) {
              s += ccsd.T1[i * NVIRT_full + e]! * V_eri(m, b + VO, e + VO, j);
              s -= ccsd.T1[j * NVIRT_full + e]! * V_eri(m, b + VO, e + VO, i);
            }
            for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
              for (let e = 0; e < NVIRT_full; e++) {
                for (let f = 0; f < NVIRT_full; f++) {
                  const tvj = ccsd.T2[((nIdx * NOCC_full + j) * NVIRT_full + b) * NVIRT_full + f]!;
                  const tvi = ccsd.T2[((nIdx * NOCC_full + i) * NVIRT_full + b) * NVIRT_full + f]!;
                  const eV = V_eri(m, nIdx, e + VO, f + VO);
                  s -= ccsd.T1[i * NVIRT_full + e]! * tvj * eV;
                  s += ccsd.T1[j * NVIRT_full + e]! * tvi * eV;
                }
              }
            }
            Wovoo[((m * NVIRT_full + b) * NOCC_full + i) * NOCC_full + j] = s;
          }
        }
      }
    }

    // σ via PySCF eom_gccsd.ipccsd_matvec (Tu-Wang-Li 2012 Eqs. 8-9).
    const sigmaFn = (R_1: Float64Array, R_2: Float64Array): {
      s1: Float64Array; s2: Float64Array;
    } => {
      const s1 = new Float64Array(NOCC_full);
      const s2 = new Float64Array(NOCC_full * NOCC_full * NVIRT_full);
      // σ_1[i]
      for (let i = 0; i < NOCC_full; i++) {
        let s = 0;
        for (let m = 0; m < NOCC_full; m++) s -= Foo[m * NOCC_full + i]! * R_1[m]!;
        for (let m = 0; m < NOCC_full; m++) {
          for (let e = 0; e < NVIRT_full; e++) {
            s += Fov[m * NVIRT_full + e]! * R_2[(m * NOCC_full + i) * NVIRT_full + e]!;
          }
        }
        for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
          for (let m = 0; m < NOCC_full; m++) {
            for (let e = 0; e < NVIRT_full; e++) {
              s -= 0.5 * Wooov[((nIdx * NOCC_full + m) * NOCC_full + i) * NVIRT_full + e]! *
                         R_2[(m * NOCC_full + nIdx) * NVIRT_full + e]!;
            }
          }
        }
        s1[i] = s;
      }
      // σ_2[i,j,a]
      for (let i = 0; i < NOCC_full; i++) {
        for (let j = 0; j < NOCC_full; j++) {
          for (let a = 0; a < NVIRT_full; a++) {
            let z = 0;
            const idx_ija = (i * NOCC_full + j) * NVIRT_full + a;
            for (let e = 0; e < NVIRT_full; e++) {
              z += Fvv[a * NVIRT_full + e]! * R_2[(i * NOCC_full + j) * NVIRT_full + e]!;
            }
            for (let m = 0; m < NOCC_full; m++) {
              z -= Foo[m * NOCC_full + i]! * R_2[(m * NOCC_full + j) * NVIRT_full + a]!;
              z += Foo[m * NOCC_full + j]! * R_2[(m * NOCC_full + i) * NVIRT_full + a]!;
            }
            for (let m = 0; m < NOCC_full; m++) {
              z -= Wovoo[((m * NVIRT_full + a) * NOCC_full + j) * NOCC_full + i]! * R_1[m]!;
            }
            for (let m = 0; m < NOCC_full; m++) {
              for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
                z += 0.5 * Woooo[((m * NOCC_full + nIdx) * NOCC_full + i) * NOCC_full + j]! *
                           R_2[(m * NOCC_full + nIdx) * NVIRT_full + a]!;
              }
            }
            for (let m = 0; m < NOCC_full; m++) {
              for (let e = 0; e < NVIRT_full; e++) {
                z += Wovvo[((m * NVIRT_full + a) * NVIRT_full + e) * NOCC_full + i]! *
                     R_2[(m * NOCC_full + j) * NVIRT_full + e]!;
                z -= Wovvo[((m * NVIRT_full + a) * NVIRT_full + e) * NOCC_full + j]! *
                     R_2[(m * NOCC_full + i) * NVIRT_full + e]!;
              }
            }
            for (let m = 0; m < NOCC_full; m++) {
              for (let nIdx = 0; nIdx < NOCC_full; nIdx++) {
                for (let e = 0; e < NVIRT_full; e++) {
                  for (let f = 0; f < NVIRT_full; f++) {
                    z += 0.5 * V_eri(m, nIdx, e + VO, f + VO) *
                               R_2[(m * NOCC_full + nIdx) * NVIRT_full + f]! *
                               ccsd.T2[((i * NOCC_full + j) * NVIRT_full + a) * NVIRT_full + e]!;
                  }
                }
              }
            }
            s2[idx_ija] = z;
          }
        }
      }
      return { s1, s2 };
    };

    // Build M_mine column-by-column. Same packing as runIPEOMCCSD:
    //   columns 0..NOCC-1 = R_1[i]
    //   columns NOCC.. = packed (i>j, a) in order ((1,0), a varying)
    const R_1 = new Float64Array(NOCC_full);
    const R_2 = new Float64Array(NOCC_full * NOCC_full * NVIRT_full);
    const M_mine = new Float64Array(dim * dim);
    for (let col = 0; col < dim; col++) {
      R_1.fill(0); R_2.fill(0);
      if (col < NOCC_full) {
        R_1[col] = 1;
      } else {
        const a_col = col - NOCC_full;
        R_2[(1 * NOCC_full + 0) * NVIRT_full + a_col] = 1;
        R_2[(0 * NOCC_full + 1) * NVIRT_full + a_col] = -1;
      }
      const { s1, s2 } = sigmaFn(R_1, R_2);
      for (let row = 0; row < NOCC_full; row++) M_mine[row * dim + col] = s1[row]!;
      for (let a_row = 0; a_row < NVIRT_full; a_row++) {
        M_mine[(NOCC_full + a_row) * dim + col] = s2[(1 * NOCC_full + 0) * NVIRT_full + a_row]!;
      }
    }

    console.log(`[bf-ip-h2] M_exact (${dim}×${dim}):`);
    for (let i = 0; i < dim; i++) {
      const row = Array.from({ length: dim }, (_, j) =>
        M_exact[i * dim + j]!.toExponential(3).padStart(11),
      ).join(" ");
      console.log(`  ${basisLabels[i]!.padEnd(11)} ${row}`);
    }
    console.log(`[bf-ip-h2] M_mine (${dim}×${dim}):`);
    for (let i = 0; i < dim; i++) {
      const row = Array.from({ length: dim }, (_, j) =>
        M_mine[i * dim + j]!.toExponential(3).padStart(11),
      ).join(" ");
      console.log(`  ${basisLabels[i]!.padEnd(11)} ${row}`);
    }
    console.log(`[bf-ip-h2] M_mine − M_exact:`);
    let maxDiff = 0;
    for (let i = 0; i < dim; i++) {
      const row = Array.from({ length: dim }, (_, j) => {
        const d = M_mine[i * dim + j]! - M_exact[i * dim + j]!;
        if (Math.abs(d) > maxDiff) maxDiff = Math.abs(d);
        return Math.abs(d) > 1e-9 ? d.toExponential(3).padStart(11) : "    .      ";
      }).join(" ");
      console.log(`  ${basisLabels[i]!.padEnd(11)} ${row}`);
    }
    // Hard regression assertion. After the PySCF eom_gccsd.ipccsd_matvec
    // port (2026-05-22, commit pending), the σ-equation matrix matches
    // the brute-force explicit H̄ projection to numerical noise.
    expect(maxDiff).toBeLessThan(1e-10);
  });
});
