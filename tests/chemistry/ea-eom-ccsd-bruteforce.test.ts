// Brute-force EA-EOM-CCSD reference for H₂ STO-3G — companion to
// ip-eom-ccsd-bruteforce.test.ts and eom-ccsd-bruteforce.test.ts.
// Verifies the EA-EOM R_1/R_2 split structure.
//
// 1-particle basis: |Φ^a⟩ = a^†_a |Φ_0⟩ for a ∈ virtual SOs.
// 1h2p basis (antisym in a, b with a>b): |Φ_i^{ab}⟩ = a^†_a a^†_b a_i |Φ_0⟩.
import { describe, test, expect } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import {
  runCCSD,
  buildSpinOrbitalERI,
  makeF_ae,
  makeF_mi,
  makeF_me,
  makeW_abef,
  makeW_mbej,
  makeTau,
} from "../../src/chemistry/ccsd.js";
import { runEAEOMCCSD } from "../../src/chemistry/ea-eom-ccsd.js";
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

// 1-particle basis: |Φ^a⟩ = a^†_(a+NOCC) |Φ_0⟩.
function partVec(a: number, NOCC: number, vac: number): Float64Array {
  const v = new Float64Array(DIM);
  const r = applyChain(vac, [{ type: "a+", P: a + NOCC }]);
  if (r !== null) v[r.newState] = r.sign;
  return v;
}

// 1h2p basis: |Φ_i^{ab}⟩ = a^†_(a+NOCC) a^†_(b+NOCC) a_i |Φ_0⟩.
function oneh2pVec(
  i: number, a: number, b: number, NOCC: number, vac: number,
): Float64Array {
  const v = new Float64Array(DIM);
  const r = applyChain(vac, [
    { type: "a+", P: a + NOCC },
    { type: "a+", P: b + NOCC },
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

describe("EA-EOM-CCSD brute-force diagnostic (H₂ STO-3G)", () => {
  test("M_exact eigenvalues from explicit H̄ projection vs σ-equation", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const n = integrals.n;
    const hf = runRHFSCF(integrals, 2);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-12 });
    const NOCC = 2, NVIRT = 2;

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

    // ── EA-EOM basis: 2 R_1 (1-particle) + 2 R_2 (1h2p packed at a>b). ──
    const vac_HF = (1 << 0) | (1 << 1);
    const basisLabels: string[] = [];
    const basisVecs: Float64Array[] = [];
    for (let a = 0; a < NVIRT; a++) {
      basisLabels.push(`R₁[${a}]`);
      basisVecs.push(partVec(a, NOCC, vac_HF));
    }
    // Packed 1h2p with (a, b) = (1, 0) and i varying.
    for (let i = 0; i < NOCC; i++) {
      basisLabels.push(`R₂[${i},1,0]`);
      basisVecs.push(oneh2pVec(i, 1, 0, NOCC, vac_HF));
    }
    const dim = basisVecs.length;

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
    console.log(`[bf-ea-h2] Basis overlap diagonals: [${
      Array.from({ length: dim }, (_, i) => S[i * dim + i]!.toFixed(3)).join(", ")
    }]`);

    const eigExact = eigGeneral(M_exact, dim);
    const exactSorted = Array.from(eigExact.real).sort((a, b) => a - b);
    console.log(`[bf-ea-h2] M_exact eigenvalues (Ha): [${
      exactSorted.map(x => x.toFixed(8)).join(", ")
    }]`);

    // Run EA-EOM-CCSD (returns EAs sorted descending).
    const ea = runEAEOMCCSD(ccsd, integrals, hf);
    // EA = -ω; sort ω ascending to compare with M_exact eigenvalues.
    const myOmegas = Array.from(ea.eas).map(x => -x).sort((a, b) => a - b);
    console.log(`[bf-ea-h2] σ-equation ω = −EA (Ha, sorted): [${
      myOmegas.map(x => x.toFixed(8)).join(", ")
    }]`);

    // ── Build M_mine in the same basis. ──
    const NSO_full = 2 * n;
    const NOCC_full = 2 * hf.nOccupied;
    const NVIRT_full = NSO_full - NOCC_full;
    const eps_full = new Float64Array(NSO_full);
    for (let P = 0; P < NSO_full; P++) eps_full[P] = hf.orbitalEnergies[P >> 1]!;
    const tau_t = makeTau(ccsd.T1, ccsd.T2, NOCC_full, NVIRT_full, 0.5);
    const tau_full = makeTau(ccsd.T1, ccsd.T2, NOCC_full, NVIRT_full, 1.0);
    const F_ae = makeF_ae(ccsd.T1, eps_full, eri_SO, tau_t, NOCC_full, NVIRT_full, NSO_full);
    const F_mi = makeF_mi(ccsd.T1, eps_full, eri_SO, tau_t, NOCC_full, NVIRT_full, NSO_full);
    const F_me = makeF_me(ccsd.T1, eri_SO, NOCC_full, NVIRT_full, NSO_full);
    const W_abef = makeW_abef(ccsd.T1, tau_full, eri_SO, NOCC_full, NVIRT_full, NSO_full);
    const W_mbej = makeW_mbej(ccsd.T1, ccsd.T2, eri_SO, NOCC_full, NVIRT_full, NSO_full);

    const V_eri = (P: number, Q: number, R: number, S: number): number =>
      eri_SO[((P * NSO_full + Q) * NSO_full + R) * NSO_full + S]!;
    const VO = NOCC_full;

    const sigmaFn = (R_1: Float64Array, R_2: Float64Array): {
      s1: Float64Array; s2: Float64Array;
    } => {
      const s1 = new Float64Array(NVIRT_full);
      const s2 = new Float64Array(NOCC_full * NVIRT_full * NVIRT_full);
      // σ_1[a]
      for (let a = 0; a < NVIRT_full; a++) {
        let s = 0;
        s += eps_full[a + VO]! * R_1[a]!;
        for (let e = 0; e < NVIRT_full; e++) s += F_ae[a * NVIRT_full + e]! * R_1[e]!;
        for (let m = 0; m < NOCC_full; m++) {
          for (let e = 0; e < NVIRT_full; e++) {
            s += F_me[m * NVIRT_full + e]! * R_2[(m * NVIRT_full + a) * NVIRT_full + e]!;
          }
        }
        for (let m = 0; m < NOCC_full; m++) {
          for (let e = 0; e < NVIRT_full; e++) {
            for (let b = 0; b < NVIRT_full; b++) {
              s += 0.5 * V_eri(a + VO, m, e + VO, b + VO) *
                   R_2[(m * NVIRT_full + e) * NVIRT_full + b]!;
            }
          }
        }
        s1[a] = s;
      }
      // σ_2[i, a, b] antisym in (a, b)
      for (let i = 0; i < NOCC_full; i++) {
        for (let a = 0; a < NVIRT_full; a++) {
          for (let b = 0; b < NVIRT_full; b++) {
            let z = 0;
            z += (eps_full[a + VO]! + eps_full[b + VO]! - eps_full[i]!) *
                 R_2[(i * NVIRT_full + a) * NVIRT_full + b]!;
            for (let m = 0; m < NOCC_full; m++) {
              z -= F_mi[m * NOCC_full + i]! *
                   R_2[(m * NVIRT_full + a) * NVIRT_full + b]!;
            }
            for (let e = 0; e < NVIRT_full; e++) {
              z += F_ae[a * NVIRT_full + e]! *
                   R_2[(i * NVIRT_full + e) * NVIRT_full + b]!;
              z -= F_ae[b * NVIRT_full + e]! *
                   R_2[(i * NVIRT_full + e) * NVIRT_full + a]!;
            }
            for (let e = 0; e < NVIRT_full; e++) {
              for (let f = 0; f < NVIRT_full; f++) {
                z += 0.5 * W_abef[((a * NVIRT_full + b) * NVIRT_full + e) * NVIRT_full + f]! *
                     R_2[(i * NVIRT_full + e) * NVIRT_full + f]!;
              }
            }
            for (let m = 0; m < NOCC_full; m++) {
              for (let e = 0; e < NVIRT_full; e++) {
                z += W_mbej[((m * NVIRT_full + a) * NVIRT_full + e) * NOCC_full + i]! *
                     R_2[(m * NVIRT_full + e) * NVIRT_full + b]!;
                z -= W_mbej[((m * NVIRT_full + b) * NVIRT_full + e) * NOCC_full + i]! *
                     R_2[(m * NVIRT_full + e) * NVIRT_full + a]!;
              }
            }
            for (let e = 0; e < NVIRT_full; e++) {
              z += V_eri(a + VO, b + VO, e + VO, i) * R_1[e]!;
            }
            s2[(i * NVIRT_full + a) * NVIRT_full + b] = z;
          }
        }
      }
      return { s1, s2 };
    };

    // Build M_mine column-by-column.
    const R_1 = new Float64Array(NVIRT_full);
    const R_2 = new Float64Array(NOCC_full * NVIRT_full * NVIRT_full);
    const M_mine = new Float64Array(dim * dim);
    for (let col = 0; col < dim; col++) {
      R_1.fill(0); R_2.fill(0);
      if (col < NVIRT_full) {
        R_1[col] = 1;
      } else {
        const i_col = col - NVIRT_full;
        // Packed double (a=1, b=0): r_2[i_col, 1, 0] = 1, r_2[i_col, 0, 1] = -1.
        R_2[(i_col * NVIRT_full + 1) * NVIRT_full + 0] = 1;
        R_2[(i_col * NVIRT_full + 0) * NVIRT_full + 1] = -1;
      }
      const { s1, s2 } = sigmaFn(R_1, R_2);
      for (let row = 0; row < NVIRT_full; row++) M_mine[row * dim + col] = s1[row]!;
      for (let i_row = 0; i_row < NOCC_full; i_row++) {
        M_mine[(NVIRT_full + i_row) * dim + col] = s2[(i_row * NVIRT_full + 1) * NVIRT_full + 0]!;
      }
    }

    console.log(`[bf-ea-h2] M_exact (${dim}×${dim}):`);
    for (let i = 0; i < dim; i++) {
      const row = Array.from({ length: dim }, (_, j) =>
        M_exact[i * dim + j]!.toExponential(3).padStart(11),
      ).join(" ");
      console.log(`  ${basisLabels[i]!.padEnd(12)} ${row}`);
    }
    console.log(`[bf-ea-h2] M_mine (${dim}×${dim}):`);
    for (let i = 0; i < dim; i++) {
      const row = Array.from({ length: dim }, (_, j) =>
        M_mine[i * dim + j]!.toExponential(3).padStart(11),
      ).join(" ");
      console.log(`  ${basisLabels[i]!.padEnd(12)} ${row}`);
    }
    console.log(`[bf-ea-h2] M_mine − M_exact:`);
    for (let i = 0; i < dim; i++) {
      const row = Array.from({ length: dim }, (_, j) => {
        const d = M_mine[i * dim + j]! - M_exact[i * dim + j]!;
        return Math.abs(d) > 1e-9 ? d.toExponential(3).padStart(11) : "    .      ";
      }).join(" ");
      console.log(`  ${basisLabels[i]!.padEnd(12)} ${row}`);
    }

    // ── ASSERTIONS (this is a "permanent verifier" — it must be able to FAIL). ──
    // (1) The R₁ (1-particle) sector of EA-EOM σ_1 is fully DERIVED and exact:
    //     assert the R₁ rows match the brute-force H̄ projection element-wise.
    let r1Diff = 0;
    for (let i = 0; i < NVIRT; i++) {            // first NVIRT rows are the R₁ sector
      for (let j = 0; j < dim; j++) {
        r1Diff = Math.max(r1Diff, Math.abs(M_mine[i * dim + j]! - M_exact[i * dim + j]!));
      }
    }
    console.log(`[bf-ea-h2] max |M_mine − M_exact| over R₁ sector = ${r1Diff.toExponential(3)} Ha`);
    expect(r1Diff).toBeLessThan(1e-9);

    // (2) Eigenvalue check against the brute-force EA-EOM spectrum. As of
    //     2026-06-16 the production EA-EOM σ is a proper PySCF eom_gccsd port
    //     (no empirical patch) and matches here. NOTE: this M_mine is the test's
    //     OWN inline sigma reconstruction (kept as a diagnostic) and still uses
    //     the older bare-integral form, so its R₂ diagonal differs ~10 mHa — that
    //     reflects the inline reconstruction, NOT production. The authoritative
    //     multi-electron σ_2 verifier is ea-eom-ccsd-bruteforce-lih.test.ts
    //     (LiH, T̂²≠0, matches production to ~5e-13 Ha). H₂ alone (T̂²≈0) cannot
    //     probe σ_2.
    let maxEigDiff = 0;
    for (let k = 0; k < exactSorted.length; k++) {
      maxEigDiff = Math.max(maxEigDiff, Math.abs(myOmegas[k]! - exactSorted[k]!));
    }
    console.log(`[bf-ea-h2] max |σ-eig − exact-eig| (H₂) = ${maxEigDiff.toExponential(3)} Ha`);
    expect(maxEigDiff).toBeLessThan(1e-8);
  });
});
