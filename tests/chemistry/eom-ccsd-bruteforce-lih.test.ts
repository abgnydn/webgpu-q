// Brute-force EOM-CCSD reference for LiH STO-3G (4 electrons).
//
// Stage 32 close-out part 2. After E35 surfaced a ~1 eV gap vs PySCF on
// multi-electron systems (with the gap concentrated in the SINGLET
// sector — LiH triplet agrees to 7 meV with PySCF; LiH singlets disagree
// by ~2.5 eV), this test diagnoses which σ-equation term is wrong.
//
// Methodology: build H̄ = e^(-T̂) H e^(T̂) EXPLICITLY in the 6-spin-orbital
// (NSO = 6, DIM = 64) Fock space, project (H̄ − E_CCSD) onto the
// (R_1 + antisym R_2) basis used by runEOMCCSD (dim = 8 + 6 = 14),
// then diagonalize and compare element-wise to the σ-equation matrix.
//
// For 4-electron LiH STO-3G with NVIRT_so = 2, T̂² is non-zero in
// general (T̂_1² generates allowed double substitutions), so we use a
// Taylor series e^T̂ = I + T̂ + ½T̂² + (higher orders that turn out to
// be 0 by orbital-count limits). H₂'s "I ± T̂" shortcut does NOT apply.
//
// SO convention: P = 2p + σ. For LiH STO-3G:
//   SO 0 = MO 0 (HOMO), α        SO 4 = MO 2 (LUMO), α (virtual)
//   SO 1 = MO 0, β                SO 5 = MO 2, β (virtual)
//   SO 2 = MO 1, α
//   SO 3 = MO 1, β
//
// Reference Slater determinant: |Φ_0⟩ = |1111 00⟩ ↔ bitmask 0b001111 = 15.

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
  makeW_mnij,
  makeW_abef,
  makeW_mbej,
  makeTau,
} from "../../src/chemistry/ccsd.js";
import { runEOMCCSD } from "../../src/chemistry/eom-ccsd.js";
import { transformERIToMO } from "../../src/chemistry/mp2.js";
import { eigsymmetric } from "../../src/manybody/dense-eig.js";

const LIH: Atom[] = [
  { symbol: "Li", pos: [0, 0, 0] },
  { symbol: "H",  pos: [0, 0, 1.595] },
];

const NSO = 6;
const DIM = 64; // 2^6
const NOCC = 4;
const NVIRT = 2;
const VAC_HF = 0b001111; // SOs 0–3 occupied

// ── Bit-state helpers (parametric in NSO via the same bit logic). ──
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

// ── Build full FCI Hamiltonian on 64-state Fock space. ─────────────
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
              { type: "a+", P: p }, { type: "a+", P: q },
              { type: "a", P: sIdx }, { type: "a", P: r },
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

// ── Build T̂ = T̂_1 + T̂_2 as a DIM×DIM matrix. ───────────────────
// T̂_1 = Σ_ia t_i^a a^†_(a+NOCC) a_i
// T̂_2 = (1/4) Σ_ijab t_ij^ab a^†_(a+NOCC) a^†_(b+NOCC) a_j a_i  (antisym)
function buildTMatrix(T1: Float64Array, T2: Float64Array): Float64Array {
  const T = new Float64Array(DIM * DIM);
  // T̂_1
  for (let i = 0; i < NOCC; i++) {
    for (let a = 0; a < NVIRT; a++) {
      const t = T1[i * NVIRT + a]!;
      if (t === 0) continue;
      const A = a + NOCC;
      for (let s = 0; s < DIM; s++) {
        const r = applyChain(s, [{ type: "a+", P: A }, { type: "a", P: i }]);
        if (r === null) continue;
        const idx = r.newState * DIM + s;
        T[idx] = T[idx]! + t * r.sign;
      }
    }
  }
  // T̂_2
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let a = 0; a < NVIRT; a++) {
        for (let b = 0; b < NVIRT; b++) {
          const t = T2[((i * NOCC + j) * NVIRT + a) * NVIRT + b]!;
          if (t === 0) continue;
          const A = a + NOCC, B = b + NOCC;
          for (let s = 0; s < DIM; s++) {
            const r = applyChain(s, [
              { type: "a+", P: A }, { type: "a+", P: B },
              { type: "a", P: j }, { type: "a", P: i },
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

function maxAbs(A: Float64Array): number {
  let m = 0;
  for (let i = 0; i < A.length; i++) m = Math.max(m, Math.abs(A[i]!));
  return m;
}

// e^T̂ via Taylor series truncated when ‖T̂^k‖∞ falls below tol.
// Returns the Taylor sum AND the order at which it converged.
function expMatrix(T: Float64Array, sign: 1 | -1, tol = 1e-15): { exp: Float64Array; order: number } {
  const out = new Float64Array(DIM * DIM);
  for (let i = 0; i < DIM; i++) out[i * DIM + i] = 1; // I
  // Signed copy of T (this is T̂ multiplied by ±1 once).
  const Tsigned = new Float64Array(DIM * DIM);
  for (let i = 0; i < Tsigned.length; i++) Tsigned[i] = (sign < 0 ? -1 : 1) * T[i]!;
  let term: Float64Array = new Float64Array(DIM * DIM);
  for (let i = 0; i < term.length; i++) term[i] = Tsigned[i]!;
  let fact = 1;
  let order = 0;
  for (let k = 1; k <= 12; k++) {
    fact *= k;
    if (k > 1) {
      term = matmul(term, Tsigned);
    }
    const norm = maxAbs(term);
    if (norm < tol) break;
    const invFact = 1 / fact;
    for (let i = 0; i < out.length; i++) out[i] = out[i]! + invFact * term[i]!;
    order = k;
  }
  return { exp: out, order };
}

function singlesVec(i: number, a: number): Float64Array {
  const v = new Float64Array(DIM);
  const r = applyChain(VAC_HF, [{ type: "a+", P: a + NOCC }, { type: "a", P: i }]);
  if (r !== null) v[r.newState] = r.sign;
  return v;
}

function doublesVec(i: number, j: number, a: number, b: number): Float64Array {
  const v = new Float64Array(DIM);
  const r = applyChain(VAC_HF, [
    { type: "a+", P: a + NOCC }, { type: "a+", P: b + NOCC },
    { type: "a", P: j }, { type: "a", P: i },
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

describe("Brute-force EOM-CCSD on LiH STO-3G (4 electrons, NSO=6)", () => {
  test("Exact H̄ projection vs σ-equation eigenvalues — singlet sector diagnosis", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(LIH);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const n = integrals.n; // 3 for s-only Li + H 1s
    expect(n).toBe(3);
    const hf = runRHFSCF(integrals, 4, { energyTol: 1e-12, densityTol: 1e-12 });
    expect(hf.converged).toBe(true);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-12, maxIter: 200 });
    expect(ccsd.converged).toBe(true);

    // ── Build h_MO and ERI_MO. ──────────────────────────────
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

    // ── Full FCI Hamiltonian on 64-state Fock space. ───────
    const H_full = buildFockHamiltonian(h_SO, eri_SO, integrals.Vnn);

    // Sanity: ground state on the 4-electron sector should equal FCI.
    // Project H to the 4-electron block: states with popcount(s) = 4.
    const fourElectronStates: number[] = [];
    for (let s = 0; s < DIM; s++) {
      let n4 = 0;
      for (let p = 0; p < NSO; p++) n4 += occ(s, p);
      if (n4 === 4) fourElectronStates.push(s);
    }
    const NFCI = fourElectronStates.length;
    const H_4e = new Float64Array(NFCI * NFCI);
    for (let i = 0; i < NFCI; i++) {
      for (let j = 0; j < NFCI; j++) {
        H_4e[i * NFCI + j] = H_full[fourElectronStates[i]! * DIM + fourElectronStates[j]!]!;
      }
    }
    const Hsym4e = eigsymmetric(H_4e, NFCI);
    const E_FCI_ground = Hsym4e.values[0]!;
    console.log(`[bf-eom-lih] 4-electron Fock-space dim = ${NFCI}`);
    console.log(`[bf-eom-lih] FCI ground (Ha) = ${E_FCI_ground.toFixed(10)}`);
    console.log(`[bf-eom-lih] CCSD     (Ha) = ${ccsd.totalEnergy.toFixed(10)}`);
    console.log(`[bf-eom-lih] CCSD − FCI (mHa) = ${((ccsd.totalEnergy - E_FCI_ground) * 1000).toFixed(4)}`);

    // ── Build T̂ and verify T̂^k convergence. ──────────────
    const T_mat = buildTMatrix(ccsd.T1, ccsd.T2);
    console.log(`[bf-eom-lih] ‖T̂‖∞ = ${maxAbs(T_mat).toExponential(3)}`);
    const T2_mat = matmul(T_mat, T_mat);
    console.log(`[bf-eom-lih] ‖T̂²‖∞ = ${maxAbs(T2_mat).toExponential(3)}`);
    const T3_mat = matmul(T2_mat, T_mat);
    console.log(`[bf-eom-lih] ‖T̂³‖∞ = ${maxAbs(T3_mat).toExponential(3)}`);

    // e^(±T̂) via Taylor.
    const expTPlus = expMatrix(T_mat, +1);
    const expTMinus = expMatrix(T_mat, -1);
    console.log(`[bf-eom-lih] e^T̂ Taylor converged at order ${expTPlus.order}`);
    console.log(`[bf-eom-lih] e^(−T̂) Taylor converged at order ${expTMinus.order}`);

    const Hbar = matmul(matmul(expTMinus.exp, H_full), expTPlus.exp);

    // ── Build (R_1, R_2) basis: 8 singles + 6 antisym doubles = 14. ──
    const basisLabels: string[] = [];
    const basisVecs: Float64Array[] = [];
    for (let i = 0; i < NOCC; i++) {
      for (let a = 0; a < NVIRT; a++) {
        basisLabels.push(`R₁[${i},${a}]`);
        basisVecs.push(singlesVec(i, a));
      }
    }
    for (let i = 0; i < NOCC; i++) {
      for (let j = i + 1; j < NOCC; j++) {
        for (let a = 0; a < NVIRT; a++) {
          for (let b = a + 1; b < NVIRT; b++) {
            basisLabels.push(`R₂[${i}<${j},${a}<${b}]`);
            basisVecs.push(doublesVec(i, j, a, b));
          }
        }
      }
    }
    const bdim = basisVecs.length;
    expect(bdim).toBe(14);

    // ── M_exact = projection of (H̄ − E_CCSD) onto basis. ────
    const E_CCSD = ccsd.totalEnergy;
    const M_exact = new Float64Array(bdim * bdim);
    let maxOffDiagS = 0;
    for (let i = 0; i < bdim; i++) {
      for (let j = 0; j < bdim; j++) {
        let sij = 0;
        for (let k = 0; k < DIM; k++) sij += basisVecs[i]![k]! * basisVecs[j]![k]!;
        if (i !== j) maxOffDiagS = Math.max(maxOffDiagS, Math.abs(sij));
        const hij = quadForm(basisVecs[i]!, Hbar, basisVecs[j]!);
        M_exact[i * bdim + j] = hij - E_CCSD * sij;
      }
    }
    console.log(`[bf-eom-lih] Max basis off-diagonal overlap: ${maxOffDiagS.toExponential(2)} (expect ≈ 0)`);

    // ── Hermiticity check + symmetrize. ────────────────────
    let asym = 0;
    for (let i = 0; i < bdim; i++) {
      for (let j = i + 1; j < bdim; j++) {
        asym = Math.max(asym, Math.abs(M_exact[i * bdim + j]! - M_exact[j * bdim + i]!));
      }
    }
    console.log(`[bf-eom-lih] Max |M_exact − M_exactᵀ| = ${asym.toExponential(2)} (H̄ non-Hermitian — small for closed-shell)`);

    const M_sym = new Float64Array(bdim * bdim);
    for (let i = 0; i < bdim; i++) {
      for (let j = 0; j < bdim; j++) {
        M_sym[i * bdim + j] = 0.5 * (M_exact[i * bdim + j]! + M_exact[j * bdim + i]!);
      }
    }

    // ── Diagonalize. ──────────────────────────────────────
    const exactEig = eigsymmetric(M_sym, bdim);
    const sortedExact = Array.from(exactEig.values).sort((a, b) => a - b);

    const eom = runEOMCCSD(ccsd, integrals, hf, { nRoots: bdim });
    const eomPairs = Array.from(eom.energies).map((e, i) => ({
      energy: e,
      singletWt: eom.singletWeight[i]!,
      tripletWt: eom.tripletWeight[i]!,
    })).sort((a, b) => a.energy - b.energy);
    const sortedMine = eomPairs.map((p) => p.energy);

    console.log(`[bf-eom-lih]`);
    console.log(`[bf-eom-lih] Eigenvalue comparison (Ha · eV):`);
    console.log(`[bf-eom-lih]   root │  exact (Ha)   │  σ-eq (Ha)    │  |Δ| (Ha)    │  |Δ| (eV)  │  σ-eq char (S/T)`);
    console.log(`[bf-eom-lih]   ─────┼───────────────┼───────────────┼──────────────┼────────────┼────────────────────`);
    const HA_TO_EV = 27.211386245988;
    for (let k = 0; k < Math.min(sortedExact.length, sortedMine.length); k++) {
      const eExact = sortedExact[k]!;
      const eMine = sortedMine[k]!;
      const d = Math.abs(eExact - eMine);
      const pair = eomPairs[k]!;
      const char = pair.singletWt > 0.5 ? "S" : pair.tripletWt > 0.5 ? "T" : "?";
      console.log(
        `[bf-eom-lih]   ${(k + 1).toString().padStart(2)}    │  ${eExact.toFixed(9).padStart(11)}  │  ${eMine.toFixed(9).padStart(11)}  │  ${d.toExponential(3).padStart(10)} │  ${(d * HA_TO_EV).toFixed(4).padStart(8)} │  ${char} (${pair.singletWt.toFixed(2)} / ${pair.tripletWt.toFixed(2)})`,
      );
    }

    // ── Build M_mine via σ-equation on unit vectors (replicating
    //    runEOMCCSD's matrix-build EXACTLY, so we can diff it against
    //    M_exact element-by-element). ──────────────────────────
    const T1 = ccsd.T1;
    const T2 = ccsd.T2;
    const tau_t = makeTau(T1, T2, NOCC, NVIRT, 0.5);
    const tau   = makeTau(T1, T2, NOCC, NVIRT, 1.0);
    const eps = new Float64Array(NSO);
    for (let P = 0; P < NSO; P++) eps[P] = hf.orbitalEnergies[P >> 1]!;
    const F_ae = makeF_ae(T1, eps, eri_SO, tau_t, NOCC, NVIRT, NSO);
    const F_mi = makeF_mi(T1, eps, eri_SO, tau_t, NOCC, NVIRT, NSO);
    const F_me = makeF_me(T1, eri_SO, NOCC, NVIRT, NSO);
    const W_mnij = makeW_mnij(T1, tau, eri_SO, NOCC, NVIRT, NSO);
    const W_abef = makeW_abef(T1, tau, eri_SO, NOCC, NVIRT, NSO);
    const W_mbej = makeW_mbej(T1, T2, eri_SO, NOCC, NVIRT, NSO);
    const V = (P: number, Q: number, R: number, S: number): number =>
      eri_SO[((P * NSO + Q) * NSO + R) * NSO + S]!;

    // σ on (R_1, R_2) → (s1, s2), replicating eom-ccsd.ts exactly
    // (including the stage-32c diagonal patch).
    function sigma(R_1_in: Float64Array, R_2_in: Float64Array): { s1: Float64Array; s2: Float64Array } {
      const s1 = new Float64Array(NOCC * NVIRT);
      const s2 = new Float64Array(NOCC * NOCC * NVIRT * NVIRT);
      const VO = NOCC;
      for (let i = 0; i < NOCC; i++) {
        for (let a = 0; a < NVIRT; a++) {
          let s = (eps[a + VO]! - eps[i]!) * R_1_in[i * NVIRT + a]!;
          for (let e = 0; e < NVIRT; e++) s += F_ae[a * NVIRT + e]! * R_1_in[i * NVIRT + e]!;
          for (let m = 0; m < NOCC; m++) s -= F_mi[m * NOCC + i]! * R_1_in[m * NVIRT + a]!;
          for (let m = 0; m < NOCC; m++) {
            for (let e = 0; e < NVIRT; e++) {
              s += F_me[m * NVIRT + e]! * R_2_in[((i * NOCC + m) * NVIRT + a) * NVIRT + e]!;
            }
          }
          for (let m = 0; m < NOCC; m++) {
            for (let e = 0; e < NVIRT; e++) {
              s += W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + i]! * R_1_in[m * NVIRT + e]!;
            }
          }
          // T1-dressed W̄_mnie (Stage 32j 2026-05-13).
          for (let m = 0; m < NOCC; m++) {
            for (let nn = 0; nn < NOCC; nn++) {
              for (let e = 0; e < NVIRT; e++) {
                let Wmnie = V(m, nn, i, e + VO);
                for (let f = 0; f < NVIRT; f++) {
                  Wmnie += T1[i * NVIRT + f]! * V(m, nn, f + VO, e + VO);
                }
                s -= 0.5 * Wmnie * R_2_in[((m * NOCC + nn) * NVIRT + a) * NVIRT + e]!;
              }
            }
          }
          // T1-dressed W̄_amef (Stage 32k 2026-05-13 — sign correction).
          for (let m = 0; m < NOCC; m++) {
            for (let e = 0; e < NVIRT; e++) {
              for (let f = 0; f < NVIRT; f++) {
                let Wamef = V(a + VO, m, e + VO, f + VO);
                for (let nn = 0; nn < NOCC; nn++) {
                  Wamef += T1[nn * NVIRT + a]! * V(nn, m, e + VO, f + VO);
                }
                s += 0.5 * Wamef * R_2_in[((i * NOCC + m) * NVIRT + e) * NVIRT + f]!;
              }
            }
          }
          // Stage 32c patch (restored 2026-05-13 after revert experiment
          // showed it's a net positive on multi-electron triplets).
          s += 0.5 * ccsd.correlationEnergy * R_1_in[i * NVIRT + a]!;
          s1[i * NVIRT + a] = s;
        }
      }
      // σ_2 — replicate eom-ccsd.ts exactly.
      for (let i = 0; i < NOCC; i++) {
        for (let j = 0; j < NOCC; j++) {
          for (let a = 0; a < NVIRT; a++) {
            for (let b = 0; b < NVIRT; b++) {
              let z = 0;
              const idx_ijab = ((i * NOCC + j) * NVIRT + a) * NVIRT + b;
              z += (eps[a + VO]! + eps[b + VO]! - eps[i]! - eps[j]!) * R_2_in[idx_ijab]!;
              for (let e = 0; e < NVIRT; e++) {
                z += F_ae[a * NVIRT + e]! * R_2_in[((i * NOCC + j) * NVIRT + e) * NVIRT + b]!;
                z -= F_ae[b * NVIRT + e]! * R_2_in[((i * NOCC + j) * NVIRT + e) * NVIRT + a]!;
              }
              for (let m = 0; m < NOCC; m++) {
                z -= F_mi[m * NOCC + i]! * R_2_in[((m * NOCC + j) * NVIRT + a) * NVIRT + b]!;
                z += F_mi[m * NOCC + j]! * R_2_in[((m * NOCC + i) * NVIRT + a) * NVIRT + b]!;
              }
              for (let m = 0; m < NOCC; m++) {
                for (let nn = 0; nn < NOCC; nn++) {
                  z += 0.5 * W_mnij[((m * NOCC + nn) * NOCC + i) * NOCC + j]! *
                       R_2_in[((m * NOCC + nn) * NVIRT + a) * NVIRT + b]!;
                }
              }
              for (let e = 0; e < NVIRT; e++) {
                for (let f = 0; f < NVIRT; f++) {
                  z += 0.5 * W_abef[((a * NVIRT + b) * NVIRT + e) * NVIRT + f]! *
                       R_2_in[((i * NOCC + j) * NVIRT + e) * NVIRT + f]!;
                }
              }
              for (let m = 0; m < NOCC; m++) {
                for (let e = 0; e < NVIRT; e++) {
                  z += W_mbej[((m * NVIRT + b) * NVIRT + e) * NOCC + j]! *
                       R_2_in[((i * NOCC + m) * NVIRT + a) * NVIRT + e]!;
                  z -= W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + j]! *
                       R_2_in[((i * NOCC + m) * NVIRT + b) * NVIRT + e]!;
                  z -= W_mbej[((m * NVIRT + b) * NVIRT + e) * NOCC + i]! *
                       R_2_in[((j * NOCC + m) * NVIRT + a) * NVIRT + e]!;
                  z += W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + i]! *
                       R_2_in[((j * NOCC + m) * NVIRT + b) * NVIRT + e]!;
                }
              }
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
                    Wabej_j += 0.5 * tau_mnab * eri_SO[((mm * NSO + nn) * NSO + (e + VO)) * NSO + j]!;
                    Wabej_i += 0.5 * tau_mnab * eri_SO[((mm * NSO + nn) * NSO + (e + VO)) * NSO + i]!;
                  }
                }
                z += Wabej_j * R_1_in[i * NVIRT + e]!;
                z -= Wabej_i * R_1_in[j * NVIRT + e]!;
              }
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
                    Wmbij += 0.5 * tau_ijef * eri_SO[((m * NSO + (b + VO)) * NSO + (e + VO)) * NSO + (f + VO)]!;
                    Wmaij += 0.5 * tau_ijef * eri_SO[((m * NSO + (a + VO)) * NSO + (e + VO)) * NSO + (f + VO)]!;
                  }
                }
                z -= Wmbij * R_1_in[m * NVIRT + a]!;
                z += Wmaij * R_1_in[m * NVIRT + b]!;
              }
              // Stage 32c R_2 patch (restored 2026-05-13).
              z -= ccsd.correlationEnergy * R_2_in[idx_ijab]!;
              s2[idx_ijab] = z;
            }
          }
        }
      }
      return { s1, s2 };
    }

    // Build the FULL 14×14 M_mine by applying σ to each packed basis
    // unit vector (8 singles + 6 antisym doubles). This is the
    // matrix-on-unit-vectors construction runEOMCCSD does internally.
    //
    // CRITICAL: basisVecs above is enumerated with i<j (rows 8–13 =
    // "R₂[0<1]", "R₂[0<2]", "R₂[0<3]", "R₂[1<2]", "R₂[1<3]", "R₂[2<3]"),
    // so to diff M_mine vs M_exact element-by-element we MUST iterate
    // ijPairs with the same i<j ordering. (Previously this test used
    // an i>j enumeration matching production runEOMCCSD's internal
    // packing — a hidden basis permutation that made M_mine appear to
    // have a "sign-flip bug" at indices 10/11 vs the i<j-ordered
    // basisVecs. Stage 32h diagnosis traced to here. The production
    // packing doesn't have a bug; the diff matrix did.)
    const NS = NOCC * NVIRT;
    const ijPairs: Array<{ i: number; j: number }> = [];
    for (let i = 0; i < NOCC; i++) {
      for (let j = i + 1; j < NOCC; j++) ijPairs.push({ i, j });
    }
    const abPairs: Array<{ i: number; j: number }> = [];
    for (let a = 0; a < NVIRT; a++) {
      for (let b = a + 1; b < NVIRT; b++) abPairs.push({ i: a, j: b });
    }
    const nIJ = ijPairs.length;
    const nAB = abPairs.length;
    expect(NS + nIJ * nAB).toBe(bdim);

    const M_mine = new Float64Array(bdim * bdim);
    const R1unit = new Float64Array(NOCC * NVIRT);
    const R2unit = new Float64Array(NOCC * NOCC * NVIRT * NVIRT);
    for (let col = 0; col < bdim; col++) {
      R1unit.fill(0);
      R2unit.fill(0);
      if (col < NS) {
        R1unit[col] = 1;
      } else {
        const d = col - NS;
        const ij = ijPairs[Math.floor(d / nAB)]!;
        const ab = abPairs[d - Math.floor(d / nAB) * nAB]!;
        const i = ij.i, j = ij.j, a = ab.i, b = ab.j;
        R2unit[((i * NOCC + j) * NVIRT + a) * NVIRT + b] = 1;
        R2unit[((j * NOCC + i) * NVIRT + a) * NVIRT + b] = -1;
        R2unit[((i * NOCC + j) * NVIRT + b) * NVIRT + a] = -1;
        R2unit[((j * NOCC + i) * NVIRT + b) * NVIRT + a] = 1;
      }
      const { s1, s2 } = sigma(R1unit, R2unit);
      for (let row = 0; row < NS; row++) M_mine[row * bdim + col] = s1[row]!;
      for (let d = 0; d < nIJ * nAB; d++) {
        const ij = ijPairs[Math.floor(d / nAB)]!;
        const ab = abPairs[d - Math.floor(d / nAB) * nAB]!;
        const i = ij.i, j = ij.j, a = ab.i, b = ab.j;
        M_mine[(NS + d) * bdim + col] = s2[((i * NOCC + j) * NVIRT + a) * NVIRT + b]!;
      }
    }

    // ── Diff M_mine − M_exact full 14×14, broken into 3 blocks. ──
    const blocks: Array<{ name: string; rowFrom: number; rowTo: number; colFrom: number; colTo: number }> = [
      { name: "R₁ × R₁",   rowFrom: 0,  rowTo: NS,    colFrom: 0,  colTo: NS },
      { name: "R₁ × R₂",   rowFrom: 0,  rowTo: NS,    colFrom: NS, colTo: bdim },
      { name: "R₂ × R₁",   rowFrom: NS, rowTo: bdim,  colFrom: 0,  colTo: NS },
      { name: "R₂ × R₂",   rowFrom: NS, rowTo: bdim,  colFrom: NS, colTo: bdim },
    ];
    let globalMaxDiff = 0;
    let globalMaxLoc = "";
    for (const blk of blocks) {
      let blockMax = 0;
      let blockLoc = "";
      for (let i = blk.rowFrom; i < blk.rowTo; i++) {
        for (let j = blk.colFrom; j < blk.colTo; j++) {
          const d = (M_mine[i * bdim + j]! - M_exact[i * bdim + j]!) * HA_TO_EV;
          if (Math.abs(d) > Math.abs(blockMax)) {
            blockMax = d;
            blockLoc = `[${basisLabels[i]}, ${basisLabels[j]}]`;
          }
        }
      }
      console.log(`[bf-eom-lih] block ${blk.name.padEnd(10)} max |Δ| = ${Math.abs(blockMax).toFixed(4)} eV at ${blockLoc}`);
      if (Math.abs(blockMax) > Math.abs(globalMaxDiff)) {
        globalMaxDiff = blockMax;
        globalMaxLoc = `${blk.name}: ${blockLoc}`;
      }
    }
    console.log(`[bf-eom-lih] OVERALL max |M_mine − M_exact| = ${Math.abs(globalMaxDiff).toFixed(4)} eV at ${globalMaxLoc}`);

    // Print full 14×14 diff in eV.
    console.log(`[bf-eom-lih]`);
    console.log(`[bf-eom-lih] Full M_mine − M_exact (14×14, in eV):`);
    const colHdr = basisLabels.map((l) => l.padStart(11)).join(" ");
    console.log(`[bf-eom-lih]             ${colHdr}`);
    for (let i = 0; i < bdim; i++) {
      const row: string[] = [];
      for (let j = 0; j < bdim; j++) {
        const d = (M_mine[i * bdim + j]! - M_exact[i * bdim + j]!) * HA_TO_EV;
        row.push(d.toFixed(3).padStart(11));
      }
      console.log(`[bf-eom-lih]   ${basisLabels[i]!.padEnd(11)} ${row.join(" ")}`);
    }

    // Print full M_exact (14×14) in eV for reference.
    console.log(`[bf-eom-lih]`);
    console.log(`[bf-eom-lih] Full M_exact (14×14, in eV):`);
    console.log(`[bf-eom-lih]             ${colHdr}`);
    for (let i = 0; i < bdim; i++) {
      const row: string[] = [];
      for (let j = 0; j < bdim; j++) {
        const v = M_exact[i * bdim + j]! * HA_TO_EV;
        row.push(v.toFixed(3).padStart(11));
      }
      console.log(`[bf-eom-lih]   ${basisLabels[i]!.padEnd(11)} ${row.join(" ")}`);
    }

    // Antisymmetry sanity check on σ_2. The σ_2 output should satisfy:
    //   σ_2[i,j,a,b] = -σ_2[j,i,a,b]   (antisym in occupied pair)
    //   σ_2[i,j,a,b] = -σ_2[i,j,b,a]   (antisym in virtual pair)
    // If either fails for ANY (i,j,a,b) given an antisym R_2 input, the
    // σ_2 equations themselves violate antisymmetry — that would be the
    // immediate cause of the sign-flip pattern in the diff matrix.
    {
      // Apply σ to one antisym R_2 unit vector and inspect output antisym.
      const R1z = new Float64Array(NOCC * NVIRT);
      const R2u = new Float64Array(NOCC * NOCC * NVIRT * NVIRT);
      // R_2 unit at (i=0, j=3, a=0, b=1):
      R2u[((0 * NOCC + 3) * NVIRT + 0) * NVIRT + 1] = 1;
      R2u[((3 * NOCC + 0) * NVIRT + 0) * NVIRT + 1] = -1;
      R2u[((0 * NOCC + 3) * NVIRT + 1) * NVIRT + 0] = -1;
      R2u[((3 * NOCC + 0) * NVIRT + 1) * NVIRT + 0] = 1;
      const { s2: s2_unit } = sigma(R1z, R2u);
      let maxAsym_ij = 0;
      let maxAsym_ab = 0;
      let loc_ij = "", loc_ab = "";
      for (let i = 0; i < NOCC; i++) {
        for (let j = 0; j < NOCC; j++) {
          for (let a = 0; a < NVIRT; a++) {
            for (let b = 0; b < NVIRT; b++) {
              const v_ij_ab = s2_unit[((i * NOCC + j) * NVIRT + a) * NVIRT + b]!;
              const v_ji_ab = s2_unit[((j * NOCC + i) * NVIRT + a) * NVIRT + b]!;
              const v_ij_ba = s2_unit[((i * NOCC + j) * NVIRT + b) * NVIRT + a]!;
              const dij = Math.abs(v_ij_ab + v_ji_ab);
              const dab = Math.abs(v_ij_ab + v_ij_ba);
              if (dij > maxAsym_ij) { maxAsym_ij = dij; loc_ij = `(${i},${j},${a},${b})`; }
              if (dab > maxAsym_ab) { maxAsym_ab = dab; loc_ab = `(${i},${j},${a},${b})`; }
            }
          }
        }
      }
      console.log(`[bf-eom-lih]`);
      console.log(`[bf-eom-lih] σ_2 antisymmetry check (R_2 unit at [0,3,0,1]):`);
      console.log(`[bf-eom-lih]   max |σ_2[i,j,a,b] + σ_2[j,i,a,b]| = ${maxAsym_ij.toExponential(3)} at ${loc_ij}`);
      console.log(`[bf-eom-lih]   max |σ_2[i,j,a,b] + σ_2[i,j,b,a]| = ${maxAsym_ab.toExponential(3)} at ${loc_ab}`);
      console.log(`[bf-eom-lih]   (both should be ≤ 1e-14 if σ_2 is properly antisymmetric)`);
    }

    // Surgical: report key W_mnij values at the offending indices.
    // The R_2 × R_2 max-diff entry is [R_2[0<3,0<1], R_2[0<1,0<1]] = 7.26 eV.
    // The σ_2 ← R_2 term involved is 0.5 Σ_mn W_mnij[m,n,i,j] · R_2[m,n,a,b].
    // For (i=0,j=3) reading R_2[0,1,0,1]=+1 / R_2[1,0,0,1]=−1 (antisym
    // unit vector), the contribution is ½(W_mnij[0,1,0,3]−W_mnij[1,0,0,3]),
    // which doubles to W_mnij[0,1,0,3] if W is antisym in (m,n).
    console.log(`[bf-eom-lih]`);
    console.log(`[bf-eom-lih] W_mnij surgical sample at indices used by offending coupling:`);
    const sampleIdxs: Array<[number, number, number, number, string]> = [
      [0, 1, 0, 3, "W_mnij[0,1,0,3] — feeds [R₂[0<3,0<1], R₂[0<1,0<1]]"],
      [1, 0, 0, 3, "W_mnij[1,0,0,3] — antisym partner; expect = -W_mnij[0,1,0,3]"],
      [0, 1, 3, 0, "W_mnij[0,1,3,0] — antisym in (i,j); expect = -W_mnij[0,1,0,3]"],
      [0, 3, 0, 1, "W_mnij[0,3,0,1] — column permutation; not obvious"],
    ];
    // Recompute W_mnij here to confirm value (it's already a local in
    // the sigma builder above).
    for (const [m, nn, i, j, lbl] of sampleIdxs) {
      const idx = ((m * NOCC + nn) * NOCC + i) * NOCC + j;
      const wVal = W_mnij[idx]!;
      console.log(
        `[bf-eom-lih]   ${lbl.padEnd(70)} = ${wVal.toFixed(8)} Ha = ${(wVal * HA_TO_EV).toFixed(4)} eV`,
      );
    }

    // No hard assertions — this is a diagnostic, not a regression check.
  }, 60_000);
});
