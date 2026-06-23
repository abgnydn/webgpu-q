// Brute-force IP-EOM-CCSD reference for LiH STO-3G (4 electrons → 3-electron
// ionization sector). The MULTI-ELECTRON oracle the H₂ IP test could never be.
//
// Why this exists (2026-06-23 audit): IP-EOM-CCSD was the only one of the three
// EOM variants (EE / IP / EA) whose *exact* brute-force verifier ran solely on
// H₂ — a 2-electron system where T̂²≈0, so e^(±T̂) ≈ I ± T̂ and the σ_2 equations
// are never exercised against the full similarity transform. EE and EA each got
// a LiH (NSO=6, T̂²≠0) oracle, and the EA one CAUGHT a real ~1 mHa empirical
// σ_2 patch (see ea-eom-ccsd-bruteforce-lih.test.ts). IP had no equivalent.
// Every other multi-electron IP test (ip-eom-ccsd / ip-ea-eom-extensions) is a
// sanity check — positivity, ordering, Koopmans band, Davidson-vs-dense — none
// of which can tell a correct σ_2 from a patched one. This closes that gap.
//
// Method (mirrors the EA LiH oracle): build H̄ = e^(−T̂) H e^(T̂) in the 64-state
// Fock space via the FULL Taylor exponential (T̂ from T1 AND T2 — not the H₂
// "I±T̂" shortcut), project (H̄ − E_CCSD) onto the IP basis (1-hole a_i|Φ₀⟩ +
// antisymmetric 2h1p a†_a a_j a_i|Φ₀⟩, i>j), diagonalize, and compare the
// SPECTRUM to runIPEOMCCSD. Spectrum comparison is basis-ordering-independent
// and is exactly the test that exposed the EA patch. A correct σ matches to
// numerical noise; an empirical patch curve-fit to H₂ would not generalize.
//
// SO convention: P = 2·p + σ. LiH STO-3G (s-only Li + H 1s ⇒ n=3):
//   SO 0,1 = MO 0 (α,β)   SO 2,3 = MO 1 (α,β)   SO 4,5 = MO 2 (α,β, virtual)
// Reference determinant |Φ₀⟩ = |111100⟩ = 0b001111 = 15.

import { describe, test, expect } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD, buildSpinOrbitalERI } from "../../src/chemistry/ccsd.js";
import { runIPEOMCCSD } from "../../src/chemistry/ip-eom-ccsd.js";
import { transformERIToMO } from "../../src/chemistry/mp2.js";
import { eigGeneral } from "../../src/manybody/dense-eig-general.js";

const LIH: Atom[] = [
  { symbol: "Li", pos: [0, 0, 0] },
  { symbol: "H", pos: [0, 0, 1.595] },
];

const NSO = 6;
const DIM = 64; // 2^6
const NOCC = 4;
const NVIRT = 2;
const VAC_HF = 0b001111;

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
function applyChain(s: number, ops: { type: "a" | "a+"; P: number }[]): { newState: number; sign: number } | null {
  let state = s, totalSign = 1;
  for (let k = ops.length - 1; k >= 0; k--) {
    const op = ops[k]!;
    const r = op.type === "a" ? annihilate(state, op.P) : create(state, op.P);
    if (r === null) return null;
    state = r.newState; totalSign *= r.sign;
  }
  return { newState: state, sign: totalSign };
}

function buildFockHamiltonian(h_SO: Float64Array, eri_SO: Float64Array, enuc: number): Float64Array {
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
      h_SO[P * NSO + Q] = h_MO[(P >> 1) * n + (Q >> 1)]!;
    }
  }
  return h_SO;
}

function buildTMatrix(T1: Float64Array, T2: Float64Array): Float64Array {
  const T = new Float64Array(DIM * DIM);
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
      for (let j = 0; j < DIM; j++) C[i * DIM + j] = C[i * DIM + j]! + aik * B[k * DIM + j]!;
    }
  }
  return C;
}
const maxAbs = (A: Float64Array): number => { let m = 0; for (let i = 0; i < A.length; i++) m = Math.max(m, Math.abs(A[i]!)); return m; };

// e^(±T̂) via Taylor, truncated when ‖T̂^k‖∞/k! is negligible.
function expMatrix(T: Float64Array, sign: 1 | -1): Float64Array {
  const out = new Float64Array(DIM * DIM);
  for (let i = 0; i < DIM; i++) out[i * DIM + i] = 1;
  const Ts = new Float64Array(DIM * DIM);
  for (let i = 0; i < Ts.length; i++) Ts[i] = (sign < 0 ? -1 : 1) * T[i]!;
  let term: Float64Array = new Float64Array(DIM * DIM);
  for (let i = 0; i < term.length; i++) term[i] = Ts[i]!;
  let fact = 1;
  for (let k = 1; k <= 12; k++) {
    fact *= k;
    if (k > 1) term = matmul(term, Ts);
    if (maxAbs(term) < 1e-15) break;
    const inv = 1 / fact;
    for (let i = 0; i < out.length; i++) out[i] = out[i]! + inv * term[i]!;
  }
  return out;
}

// IP basis vectors over the 3-electron (ionized) sector.
// 1-hole: a_i|Φ₀⟩.
const holeVec = (i: number): Float64Array => {
  const v = new Float64Array(DIM);
  const r = applyChain(VAC_HF, [{ type: "a", P: i }]);
  if (r !== null) v[r.newState] = r.sign;
  return v;
};
// 2h1p: a†_(a+NOCC) a_j a_i|Φ₀⟩ (i>j).
const twoH1pVec = (i: number, j: number, a: number): Float64Array => {
  const v = new Float64Array(DIM);
  const r = applyChain(VAC_HF, [{ type: "a+", P: a + NOCC }, { type: "a", P: j }, { type: "a", P: i }]);
  if (r !== null) v[r.newState] = r.sign;
  return v;
};
function quadForm(u: Float64Array, M: Float64Array, v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < DIM; i++) {
    const ui = u[i]!;
    if (ui === 0) continue;
    for (let j = 0; j < DIM; j++) s += ui * M[i * DIM + j]! * v[j]!;
  }
  return s;
}

// For each exact eigenvalue, the distance to the nearest production eigenvalue.
function matchedMaxDiff(exact: number[], mine: number[]): number {
  let worst = 0;
  for (const e of exact) {
    let best = Infinity;
    for (const m of mine) best = Math.min(best, Math.abs(e - m));
    worst = Math.max(worst, best);
  }
  return worst;
}

describe("Brute-force IP-EOM-CCSD on LiH STO-3G (multi-electron oracle, NSO=6)", () => {
  test("explicit H̄ projection spectrum vs runIPEOMCCSD — the missing multi-electron oracle", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(LIH);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const n = integrals.n;
    expect(n).toBe(3);
    const hf = runRHFSCF(integrals, 4, { energyTol: 1e-12, densityTol: 1e-12 });
    expect(hf.converged).toBe(true);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-12, maxIter: 200 });
    expect(ccsd.converged).toBe(true);

    // h_MO, ERI_MO → spin-orbital.
    const C = hf.C_MO;
    const tmp = new Float64Array(n * n);
    for (let p = 0; p < n; p++) for (let nu = 0; nu < n; nu++) {
      let s = 0; for (let mu = 0; mu < n; mu++) s += C[mu * n + p]! * integrals.h_AO[mu * n + nu]!;
      tmp[p * n + nu] = s;
    }
    const h_MO = new Float64Array(n * n);
    for (let p = 0; p < n; p++) for (let q = 0; q < n; q++) {
      let s = 0; for (let nu = 0; nu < n; nu++) s += tmp[p * n + nu]! * C[nu * n + q]!;
      h_MO[p * n + q] = s;
    }
    const eri_MO = transformERIToMO(integrals.eri_AO, C, n);
    const eri_SO = buildSpinOrbitalERI(eri_MO, n);
    const h_SO = buildSpinOrbitalH(h_MO, n);

    // H̄ = e^(−T̂) H e^(T̂) — full Taylor, T̂ from T1 AND T2 (T̂²≠0 on LiH).
    const H_full = buildFockHamiltonian(h_SO, eri_SO, integrals.Vnn);
    const T_mat = buildTMatrix(ccsd.T1, ccsd.T2);
    const Hbar = matmul(matmul(expMatrix(T_mat, -1), H_full), expMatrix(T_mat, +1));

    // IP basis: 1-hole (i=0..3) + 2h1p (i>j, a=0..1).
    const basis: Float64Array[] = [];
    for (let i = 0; i < NOCC; i++) basis.push(holeVec(i));
    for (let i = 0; i < NOCC; i++) {
      for (let j = 0; j < i; j++) {
        for (let a = 0; a < NVIRT; a++) basis.push(twoH1pVec(i, j, a));
      }
    }
    const dim = basis.length; // 4 + C(4,2)·2 = 4 + 12 = 16

    const E = ccsd.totalEnergy;
    const M_exact = new Float64Array(dim * dim);
    let maxOverlapOff = 0;
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        let sij = 0;
        for (let k = 0; k < DIM; k++) sij += basis[i]![k]! * basis[j]![k]!;
        if (i !== j) maxOverlapOff = Math.max(maxOverlapOff, Math.abs(sij));
        M_exact[i * dim + j] = quadForm(basis[i]!, Hbar, basis[j]!) - E * sij;
      }
    }
    expect(maxOverlapOff).toBeLessThan(1e-12); // IP basis is orthonormal ⇒ S = I

    const exactEig = eigGeneral(M_exact, dim);
    const maxImagExact = Math.max(...Array.from(exactEig.imag).map(Math.abs));
    const omegaExact = Array.from(exactEig.real).sort((a, b) => a - b);

    // Production IP-EOM. IP eigenvalues ARE the (H̄ − E_CCSD) eigenvalues
    // directly (positive removal energies — no sign flip, unlike EA).
    const ip = runIPEOMCCSD(ccsd, integrals, hf, { nRoots: 64 });
    const omegaMine = Array.from(ip.ips).sort((a, b) => a - b);

    console.log(`[bf-ip-lih] CCSD (Ha) = ${ccsd.totalEnergy.toFixed(10)}, dim = ${dim}, prod dim = ${ip.dim}`);
    console.log(`[bf-ip-lih] max |Im(λ_exact)| = ${maxImagExact.toExponential(2)} (IP states should be real)`);
    console.log(`[bf-ip-lih] ω_exact (brute force): [${omegaExact.map((x) => x.toFixed(6)).join(", ")}]`);
    console.log(`[bf-ip-lih] ω_mine  (production) : [${omegaMine.map((x) => x.toFixed(6)).join(", ")}]`);
    const diff = matchedMaxDiff(omegaExact, omegaMine);
    console.log(`[bf-ip-lih] matched max |ω_exact − nearest ω_mine| = ${diff.toExponential(3)} Ha`);

    // Production must reproduce the full IP space (same antisym dimension).
    expect(ip.dim).toBe(dim);
    // Every exact IP-EOM eigenvalue must be reproduced by the production solver.
    // A correct σ matches to numerical noise; an H₂-fit empirical patch would not.
    expect(diff).toBeLessThan(1e-7);
  });
});
