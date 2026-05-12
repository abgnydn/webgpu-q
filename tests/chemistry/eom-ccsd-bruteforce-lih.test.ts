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
import { runCCSD, buildSpinOrbitalERI } from "../../src/chemistry/ccsd.js";
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

    // ── M_exact preview: top-left 8×8 of the singles block. ──
    console.log(`[bf-eom-lih]`);
    console.log(`[bf-eom-lih] M_exact preview (R₁ × R₁ block, 8×8, in eV):`);
    console.log(`[bf-eom-lih]   ${basisLabels.slice(0, 8).map((l) => l.padStart(10)).join(" ")}`);
    for (let i = 0; i < 8; i++) {
      const row: string[] = [];
      for (let j = 0; j < 8; j++) {
        row.push((M_exact[i * bdim + j]! * HA_TO_EV).toFixed(3).padStart(10));
      }
      console.log(`[bf-eom-lih]   ${basisLabels[i]!.padEnd(10)} ${row.join(" ")}`);
    }

    // No hard assertions — this is a diagnostic, not a regression check.
    // We expect the singlet roots to disagree by ~2-3 eV (the E35 finding);
    // the table above tells us WHICH roots are singlet and whether they
    // shift relative to the exact projection.
  }, 60_000);
});
