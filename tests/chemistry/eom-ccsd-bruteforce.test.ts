// Brute-force EOM-CCSD reference for H₂ STO-3G (Tier 2 stage 32 close-out).
//
// Goal: identify which term in the σ equations is wrong/missing by
// constructing H̄ = e^(-T̂) H e^(T̂) EXPLICITLY in the 4-spin-orbital Fock
// space (16 basis states), projecting onto the 5-dimensional (R_1 +
// antisym R_2) basis used by runEOMCCSD, and comparing element-by-element
// with the σ-equation matrix.
//
// For 2-electron systems, T̂² |Φ_0⟩ = 0 (cannot double-excite a doubly-
// excited state), so e^(±T̂) = I ± T̂ exactly. This makes the brute-force
// path cheap and analytically clean.
//
// SO convention (matches runEOMCCSD): P = 2p + σ. For H₂ STO-3G:
//   SO 0 = MO 0 (HOMO), α
//   SO 1 = MO 0, β
//   SO 2 = MO 1 (LUMO), α
//   SO 3 = MO 1, β
// Fock-space basis: |n_0 n_1 n_2 n_3⟩, stored as integer with bit P = n_P.
//   |Φ_0⟩ = |1100⟩ = 0b0011 = 3   (using bit 0 = SO 0, bit 1 = SO 1, ...)
import { describe, test } from "vitest";
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

const H2: Atom[] = [
  { symbol: "H", pos: [0, 0, 0] },
  { symbol: "H", pos: [0, 0, 0.7414] },
];

const NSO = 4;
const DIM = 16; // 2^NSO

// Bit-occupied helper: returns 1 if SO P is occupied in state s.
const occ = (s: number, P: number): number => (s >>> P) & 1;

// Sign for annihilation a_P on state s: (-1)^(number of occupied SOs at indices < P).
function annSign(s: number, P: number): number {
  let n = 0;
  for (let q = 0; q < P; q++) n += occ(s, q);
  return n % 2 === 0 ? 1 : -1;
}

// a_P |s⟩ → returns { newState, sign } or null if SO P not occupied.
function annihilate(s: number, P: number): { newState: number; sign: number } | null {
  if (occ(s, P) === 0) return null;
  return { newState: s & ~(1 << P), sign: annSign(s, P) };
}

// a^†_P |s⟩ → returns { newState, sign } or null if SO P already occupied.
function create(s: number, P: number): { newState: number; sign: number } | null {
  if (occ(s, P) === 1) return null;
  return { newState: s | (1 << P), sign: annSign(s, P) };
}

// Apply a chain of operators (a^†_a1 a^†_a2 ... a_i1 a_i2) to |s⟩.
// Ops are applied right-to-left. Returns { newState, sign } or null if 0.
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

// Build the full FCI Hamiltonian in spin-orbital basis using Slater rules.
// H = Σ h_pq a^†_p a_q + (1/4) Σ ⟨pq||rs⟩ a^†_p a^†_q a_s a_r
function buildFockHamiltonian(
  h_SO: Float64Array,
  eri_SO: Float64Array,
  enuc: number,
): Float64Array {
  const H = new Float64Array(DIM * DIM);
  // Diagonal: add Vnn so the eigenvalues are total energies.
  for (let s = 0; s < DIM; s++) H[s * DIM + s] = enuc;

  // 1-body: Σ_pq h_pq a^†_p a_q
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
  // 2-body: (1/4) Σ_pqrs ⟨pq||rs⟩ a^†_p a^†_q a_s a_r
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

// Build h_pq in spin-orbital basis from spatial h_MO[p, q].
// For closed-shell RHF (same MO for α and β): h_so[P, Q] = h_spatial[P>>1, Q>>1] if σ_P=σ_Q else 0.
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

// Build T̂_2 as a (DIM × DIM) matrix from T2[i,j,a,b]. For canonical antisym T2:
//   T̂_2 = (1/4) Σ_ijab t_ij^ab · a^†_(a+NOCC) a^†_(b+NOCC) a_j a_i
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

// Multiply two DIM×DIM matrices.
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

// (I ± T): construct the matrix I ± T (since T̂² = 0 on the relevant
// subspace for 2-electron, e^(±T̂) = I ± T̂ exactly).
function eyePlusT(T: Float64Array, sign: 1 | -1): Float64Array {
  const out = new Float64Array(T);
  if (sign < 0) for (let i = 0; i < out.length; i++) out[i] = -out[i]!;
  for (let i = 0; i < DIM; i++) {
    const idx = i * DIM + i;
    out[idx] = out[idx]! + 1;
  }
  return out;
}

// Build a Fock-space basis vector as |Φ_i^a⟩ = a^†_(a+NOCC) a_i |Φ_0⟩.
// Returns a length-DIM vector with the sign baked in.
function singlesVec(i: number, a: number, NOCC: number, vac: number): Float64Array {
  const v = new Float64Array(DIM);
  const r = applyChain(vac, [{ type: "a+", P: a + NOCC }, { type: "a", P: i }]);
  if (r !== null) v[r.newState] = r.sign;
  return v;
}

// Build |Φ_ij^ab⟩ = a^†_(a+NOCC) a^†_(b+NOCC) a_j a_i |Φ_0⟩.
function doublesVec(
  i: number, j: number, a: number, b: number, NOCC: number, vac: number,
): Float64Array {
  const v = new Float64Array(DIM);
  const r = applyChain(vac, [
    { type: "a+", P: a + NOCC },
    { type: "a+", P: b + NOCC },
    { type: "a", P: j },
    { type: "a", P: i },
  ]);
  if (r !== null) v[r.newState] = r.sign;
  return v;
}

// Inner product ⟨u|M|v⟩.
function quadForm(u: Float64Array, M: Float64Array, v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < DIM; i++) {
    const ui = u[i]!;
    if (ui === 0) continue;
    for (let j = 0; j < DIM; j++) s += ui * M[i * DIM + j]! * v[j]!;
  }
  return s;
}

describe("Stage 32 close-out: brute-force EOM-CCSD reference for H₂ STO-3G", () => {
  test("Element-wise diff between σ-equation matrix and explicit H̄ projection", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const n = integrals.n; // 2
    const hf = runRHFSCF(integrals, 2);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-12 });
    const NOCC = 2, NVIRT = 2;

    // ── Build h_MO and ERI_MO in spatial MO basis. ──────────
    const C = hf.C_MO;
    // h_MO = C^T h_AO C
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

    // ── Build full FCI Hamiltonian on 4-qubit Fock space. ───
    const H_full = buildFockHamiltonian(h_SO, eri_SO, integrals.Vnn);

    // Sanity: H ground state eigenvalue should equal CCSD = FCI.
    // (Diagonalize H_full on M_S=0 sector via brute-force eig of full matrix.)
    const Hsym = eigsymmetric(H_full, DIM);
    const E_FCI_ground = Hsym.values[0]!;
    console.log(`[bf-eom-h2] FCI ground = ${E_FCI_ground.toFixed(10)} Ha`);
    console.log(`[bf-eom-h2] CCSD       = ${ccsd.totalEnergy.toFixed(10)} Ha (should equal FCI for 2e)`);

    // ── Build T̂_2 and e^(±T̂). For 2e, T̂² = 0 so e^(±T̂) = I ± T̂. ──
    const T_mat = buildT2Matrix(ccsd.T2, NOCC, NVIRT);
    // Verify T² = 0 (or numerically negligible) for our 2e case.
    const TT = matmul(T_mat, T_mat);
    let ttNorm = 0;
    for (let i = 0; i < TT.length; i++) ttNorm = Math.max(ttNorm, Math.abs(TT[i]!));
    console.log(`[bf-eom-h2] |T²|_max = ${ttNorm.toExponential(2)} (expect ≈ 0 for 2-electron)`);

    const expT = eyePlusT(T_mat, +1);
    const expMinusT = eyePlusT(T_mat, -1);
    const Hbar = matmul(matmul(expMinusT, H_full), expT);

    // ── Singles + antisym doubles basis (5 vectors for H₂). ──
    const vac_HF = (1 << 0) | (1 << 1); // |Φ_0⟩ = SOs 0 and 1 occupied → bits 0,1 set
    const basisLabels: string[] = [];
    const basisVecs: Float64Array[] = [];
    // 4 singles (i, a) with i ∈ {0,1}, a ∈ {0,1}
    for (let i = 0; i < NOCC; i++) {
      for (let a = 0; a < NVIRT; a++) {
        basisLabels.push(`R₁[${i},${a}]`);
        basisVecs.push(singlesVec(i, a, NOCC, vac_HF));
      }
    }
    // 1 packed antisym double: (i=1,j=0,a=1,b=0)
    basisLabels.push("R₂[1,0,1,0]");
    basisVecs.push(doublesVec(1, 0, 1, 0, NOCC, vac_HF));

    const dim = basisVecs.length;

    // ── Build M_exact: project (H̄ − E_CCSD) onto basis via Gram matrix. ──
    // The basis vectors are not orthonormal in general (some |Φ_ij^ab⟩ have
    // signs flipped). Build the overlap S[i,j] = ⟨v_i|v_j⟩ and the raw
    // M_raw[i,j] = ⟨v_i | H̄ − E_CCSD | v_j⟩, then solve M_exact = S^{-1} M_raw.
    // Actually for spin-orbital basis vectors that are themselves Slater
    // determinants (just with possibly +1 or -1 sign), the basis is orthonormal
    // if we restrict to one packed pair per equivalence class. Let's check.
    const S = new Float64Array(dim * dim);
    const M_raw = new Float64Array(dim * dim);
    const E_CCSD = ccsd.totalEnergy;
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        // ⟨v_i|v_j⟩
        let sij = 0;
        for (let k = 0; k < DIM; k++) sij += basisVecs[i]![k]! * basisVecs[j]![k]!;
        S[i * dim + j] = sij;
        // ⟨v_i | H̄ | v_j⟩ minus E_CCSD δ_ij
        const hij = quadForm(basisVecs[i]!, Hbar, basisVecs[j]!);
        M_raw[i * dim + j] = hij - E_CCSD * sij;
      }
    }
    console.log(`[bf-eom-h2] Basis overlap diagonals: [${
      Array.from({ length: dim }, (_, i) => S[i * dim + i]!.toFixed(3)).join(", ")
    }]`);
    // Basis is orthonormal (each vec is ±|determinant⟩, so ⟨v|v⟩ = 1).
    // Off-diagonals: should all be 0 for distinct determinants.
    let maxOffDiag = 0;
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        if (i === j) continue;
        maxOffDiag = Math.max(maxOffDiag, Math.abs(S[i * dim + j]!));
      }
    }
    console.log(`[bf-eom-h2] Max basis off-diagonal overlap: ${maxOffDiag.toExponential(2)}`);

    // M_exact = M_raw (since S is identity on this basis).
    const M_exact = M_raw;

    // ── Build M_mine via runEOMCCSD's matrix construction. ──
    // Easiest path: call runEOMCCSD and reconstruct the matrix from its
    // eigenvectors + eigenvalues: M_mine = V · diag(λ) · V^{-1}. But V may be
    // ill-conditioned for degenerate cases.
    // Cleaner: pull eigenvalues directly and compare to M_exact eigenvalues.
    const eom = runEOMCCSD(ccsd, integrals, hf);
    console.log(`[bf-eom-h2] σ-equation eigenvalues: [${
      Array.from(eom.energies).map(x => x.toFixed(8)).join(", ")
    }]`);

    // Diagonalize M_exact directly (it's NOT symmetric — H̄ is non-Hermitian,
    // but our basis is real and the projection is). Use eigsymmetric since
    // for closed-shell M_S=0 the projection is Hermitian to roundoff.
    // Check Hermiticity:
    let asym = 0;
    for (let i = 0; i < dim; i++) {
      for (let j = i + 1; j < dim; j++) {
        asym = Math.max(asym, Math.abs(M_exact[i * dim + j]! - M_exact[j * dim + i]!));
      }
    }
    console.log(`[bf-eom-h2] Max |M_exact − M_exactᵀ| = ${asym.toExponential(2)}`);

    // Symmetrize for clean eigenvalues (only if asym is small).
    const M_sym = new Float64Array(dim * dim);
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        M_sym[i * dim + j] = 0.5 * (M_exact[i * dim + j]! + M_exact[j * dim + i]!);
      }
    }
    const exactEig = eigsymmetric(M_sym, dim);
    const sortedExact = Array.from(exactEig.values).sort((a, b) => a - b);
    const sortedMine = Array.from(eom.energies).slice().sort((a, b) => a - b);
    console.log(`[bf-eom-h2] M_exact eigenvalues:    [${sortedExact.map(x => x.toFixed(8)).join(", ")}]`);
    console.log(`[bf-eom-h2] σ-equation eigenvalues: [${sortedMine.map(x => x.toFixed(8)).join(", ")}]`);

    // ── Print M_exact for inspection. ───────────────────────
    console.log(`[bf-eom-h2] M_exact (5×5, in basis [R₁_00, R₁_01, R₁_10, R₁_11, R₂_1010]):`);
    for (let i = 0; i < dim; i++) {
      const row = Array.from({ length: dim }, (_, j) =>
        M_exact[i * dim + j]!.toExponential(3).padStart(11),
      ).join(" ");
      console.log(`  ${basisLabels[i]!.padEnd(10)} ${row}`);
    }

    // ── Reconstruct M_mine = V · diag(λ) · V^{-1} from the diagonalized
    // form. Since amplitudes are returned but the sigma matrix isn't, we
    // rebuild M_mine by direct similarity reconstruction. ────────────
    const N = eom.dim;
    const Vmat = new Float64Array(N * N);
    for (let k = 0; k < N; k++) {
      for (let row = 0; row < N; row++) {
        Vmat[row * N + k] = eom.amplitudes[k * N + row]!;
      }
    }
    // For closed-shell H₂ EOM-CCSD with degenerate eigenvalues, V may be
    // ill-conditioned. Instead, since we have M_exact (which differs only
    // from M_mine by the missing T2-dressing term), and we already know
    // the eigenvalues from eom.energies, we extract the wrong-by-δ pattern
    // by inspecting M_mine = M_exact + δH where δH preserves the symmetry
    // and gives the observed shift pattern (4 eigenvalues +δ, 1 −2δ where
    // δ ≈ E_corr/2).
    //
    // The exact answer falls out of element-wise comparison. To get
    // M_mine directly, we call the sigma function. Since runEOMCCSD does
    // not export the matrix, we replicate the construction here for the
    // 5×5 case by extracting from eom.energies + amplitudes:
    //   M_mine = Σ_k λ_k · v_k · v_k^T   (only valid for symmetric M_mine)
    // For H₂ STO-3G with degenerate triplets, this should be reasonably
    // accurate even if degenerate eigenvectors aren't strictly orthogonal.
    const M_mine_recon = new Float64Array(N * N);
    for (let k = 0; k < N; k++) {
      const lam = eom.energies[k]!;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const idx = i * N + j;
          M_mine_recon[idx] = M_mine_recon[idx]! +
            lam * eom.amplitudes[k * N + i]! * eom.amplitudes[k * N + j]!;
        }
      }
    }
    console.log(`[bf-eom-h2] M_mine reconstructed (5×5, may differ from true M_mine):`);
    for (let i = 0; i < N; i++) {
      const row = Array.from({ length: N }, (_, j) =>
        M_mine_recon[i * N + j]!.toExponential(3).padStart(11),
      ).join(" ");
      console.log(`  row ${i}      ${row}`);
    }

    // Compute M_exact − M_mine_recon. Note: this depends on the eigenvector
    // basis being the SAME for both; for degenerate eigenvalues the basis
    // isn't unique so this difference may have rotation artifacts.
    console.log(`[bf-eom-h2] Diff (M_exact − M_mine_recon), shows in eigenvalue space:`);
    for (let i = 0; i < N; i++) {
      const row = Array.from({ length: N }, (_, j) =>
        (M_exact[i * N + j]! - M_mine_recon[i * N + j]!).toExponential(3).padStart(11),
      ).join(" ");
      console.log(`  row ${i}      ${row}`);
    }

    // Print eigenvalue mismatch summary.
    console.log(`[bf-eom-h2] Eigenvalue gaps (σ − exact, mHa):`);
    for (let k = 0; k < N; k++) {
      const gap = (sortedMine[k]! - sortedExact[k]!) * 1000;
      console.log(`  λ_${k}: ${gap.toFixed(2)} mHa`);
    }

    // ── Direct construction of M_mine by replicating σ matrix logic. ──
    // Below mirrors runEOMCCSD's matrix construction so we can compare
    // element-wise against M_exact.
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
    const W_mnij = makeW_mnij(ccsd.T1, tau_full, eri_SO, NOCC_full, NVIRT_full, NSO_full);
    const W_abef = makeW_abef(ccsd.T1, tau_full, eri_SO, NOCC_full, NVIRT_full, NSO_full);
    const W_mbej = makeW_mbej(ccsd.T1, ccsd.T2, eri_SO, NOCC_full, NVIRT_full, NSO_full);

    console.log(`[bf-eom-h2] Probed σ-equation intermediates (T1=0):`);
    console.log(`  F_ae[0,0] = ${F_ae[0]!.toExponential(3)}`);
    console.log(`  F_ae[1,1] = ${F_ae[3]!.toExponential(3)}`);
    console.log(`  F_mi[0,0] = ${F_mi[0]!.toExponential(3)}`);
    console.log(`  F_mi[1,1] = ${F_mi[3]!.toExponential(3)}`);
    console.log(`  W_mbej[0,0_v,0_v,0] = ${W_mbej[0]!.toExponential(3)}`);
    console.log(`  W_mbej[0,1_v,1_v,0] = ${W_mbej[((0 * 2 + 1) * 2 + 1) * 2 + 0]!.toExponential(3)}`);

    // tK = E_corr (CCSD ground-state amplitude relation).
    // For H₂ STO-3G the only T2 amplitude is t = T2[1,0,1,0].
    const t = ccsd.T2[((1 * 2 + 0) * 2 + 1) * 2 + 0]!;
    const K_01 = eri_MO[((0 * n + 1) * n + 0) * n + 1]!;
    const tK = t * K_01;
    console.log(`[bf-eom-h2] t = ${t.toFixed(6)}, K_01 = ${K_01.toFixed(6)}, tK = ${tK.toExponential(3)}, E_corr = ${ccsd.correlationEnergy.toExponential(3)}`);

    // Build M_mine explicitly using sigma-equation logic.
    const buildMMine = (): Float64Array => {
      const M = new Float64Array(dim * dim);
      const VO = NOCC_full;
      const V = (P: number, Q: number, R: number, S: number): number =>
        eri_SO[((P * NSO_full + Q) * NSO_full + R) * NSO_full + S]!;
      const sigmaFn = (R_1: Float64Array, R_2: Float64Array): {
        s1: Float64Array; s2: Float64Array;
      } => {
        const s1 = new Float64Array(NOCC_full * NVIRT_full);
        const s2 = new Float64Array(NOCC_full * NOCC_full * NVIRT_full * NVIRT_full);
        for (let i = 0; i < NOCC_full; i++) {
          for (let a = 0; a < NVIRT_full; a++) {
            let s = 0;
            s += (eps_full[a + VO]! - eps_full[i]!) * R_1[i * NVIRT_full + a]!;
            for (let e = 0; e < NVIRT_full; e++) s += F_ae[a * NVIRT_full + e]! * R_1[i * NVIRT_full + e]!;
            for (let m = 0; m < NOCC_full; m++) s -= F_mi[m * NOCC_full + i]! * R_1[m * NVIRT_full + a]!;
            for (let m = 0; m < NOCC_full; m++) {
              for (let e = 0; e < NVIRT_full; e++) {
                s += F_me[m * NVIRT_full + e]! * R_2[((i * NOCC_full + m) * NVIRT_full + a) * NVIRT_full + e]!;
              }
            }
            for (let m = 0; m < NOCC_full; m++) {
              for (let e = 0; e < NVIRT_full; e++) {
                s += W_mbej[((m * NVIRT_full + a) * NVIRT_full + e) * NOCC_full + i]! * R_1[m * NVIRT_full + e]!;
              }
            }
            for (let m = 0; m < NOCC_full; m++) {
              for (let nn = 0; nn < NOCC_full; nn++) {
                for (let e = 0; e < NVIRT_full; e++) {
                  s -= 0.5 * V(m, nn, i, e + VO) * R_2[((m * NOCC_full + nn) * NVIRT_full + a) * NVIRT_full + e]!;
                }
              }
            }
            for (let m = 0; m < NOCC_full; m++) {
              for (let e = 0; e < NVIRT_full; e++) {
                for (let f = 0; f < NVIRT_full; f++) {
                  s += 0.5 * V(m, a + VO, e + VO, f + VO) * R_2[((i * NOCC_full + m) * NVIRT_full + e) * NVIRT_full + f]!;
                }
              }
            }
            s1[i * NVIRT_full + a] = s;
          }
        }
        // σ_2 (compact reproduction — same formula as eom-ccsd.ts):
        for (let i = 0; i < NOCC_full; i++) {
          for (let j = 0; j < NOCC_full; j++) {
            for (let a = 0; a < NVIRT_full; a++) {
              for (let b = 0; b < NVIRT_full; b++) {
                let z = 0;
                z += (eps_full[a + VO]! + eps_full[b + VO]! - eps_full[i]! - eps_full[j]!) *
                     R_2[((i * NOCC_full + j) * NVIRT_full + a) * NVIRT_full + b]!;
                for (let e = 0; e < NVIRT_full; e++) {
                  z += F_ae[a * NVIRT_full + e]! * R_2[((i * NOCC_full + j) * NVIRT_full + e) * NVIRT_full + b]!;
                  z -= F_ae[b * NVIRT_full + e]! * R_2[((i * NOCC_full + j) * NVIRT_full + e) * NVIRT_full + a]!;
                }
                for (let m = 0; m < NOCC_full; m++) {
                  z -= F_mi[m * NOCC_full + i]! * R_2[((m * NOCC_full + j) * NVIRT_full + a) * NVIRT_full + b]!;
                  z += F_mi[m * NOCC_full + j]! * R_2[((m * NOCC_full + i) * NVIRT_full + a) * NVIRT_full + b]!;
                }
                for (let m = 0; m < NOCC_full; m++) {
                  for (let nn = 0; nn < NOCC_full; nn++) {
                    z += 0.5 * W_mnij[((m * NOCC_full + nn) * NOCC_full + i) * NOCC_full + j]! *
                         R_2[((m * NOCC_full + nn) * NVIRT_full + a) * NVIRT_full + b]!;
                  }
                }
                for (let e = 0; e < NVIRT_full; e++) {
                  for (let f = 0; f < NVIRT_full; f++) {
                    z += 0.5 * W_abef[((a * NVIRT_full + b) * NVIRT_full + e) * NVIRT_full + f]! *
                         R_2[((i * NOCC_full + j) * NVIRT_full + e) * NVIRT_full + f]!;
                  }
                }
                for (let m = 0; m < NOCC_full; m++) {
                  for (let e = 0; e < NVIRT_full; e++) {
                    z += W_mbej[((m * NVIRT_full + b) * NVIRT_full + e) * NOCC_full + j]! *
                         R_2[((i * NOCC_full + m) * NVIRT_full + a) * NVIRT_full + e]!;
                    z -= W_mbej[((m * NVIRT_full + a) * NVIRT_full + e) * NOCC_full + j]! *
                         R_2[((i * NOCC_full + m) * NVIRT_full + b) * NVIRT_full + e]!;
                    z -= W_mbej[((m * NVIRT_full + b) * NVIRT_full + e) * NOCC_full + i]! *
                         R_2[((j * NOCC_full + m) * NVIRT_full + a) * NVIRT_full + e]!;
                    z += W_mbej[((m * NVIRT_full + a) * NVIRT_full + e) * NOCC_full + i]! *
                         R_2[((j * NOCC_full + m) * NVIRT_full + b) * NVIRT_full + e]!;
                  }
                }
                for (let e = 0; e < NVIRT_full; e++) {
                  let Wabej_j = V(a + VO, b + VO, e + VO, j);
                  let Wabej_i = V(a + VO, b + VO, e + VO, i);
                  for (let mm = 0; mm < NOCC_full; mm++) {
                    for (let nn = 0; nn < NOCC_full; nn++) {
                      const t2 = ccsd.T2[((mm * NOCC_full + nn) * NVIRT_full + a) * NVIRT_full + b]!;
                      Wabej_j += 0.5 * t2 * eri_SO[((mm * NSO_full + nn) * NSO_full + (e + VO)) * NSO_full + j]!;
                      Wabej_i += 0.5 * t2 * eri_SO[((mm * NSO_full + nn) * NSO_full + (e + VO)) * NSO_full + i]!;
                    }
                  }
                  z += Wabej_j * R_1[i * NVIRT_full + e]!;
                  z -= Wabej_i * R_1[j * NVIRT_full + e]!;
                }
                for (let m = 0; m < NOCC_full; m++) {
                  let Wmbij = V(m, b + VO, i, j);
                  let Wmaij = V(m, a + VO, i, j);
                  for (let e = 0; e < NVIRT_full; e++) {
                    for (let f = 0; f < NVIRT_full; f++) {
                      const t2 = ccsd.T2[((i * NOCC_full + j) * NVIRT_full + e) * NVIRT_full + f]!;
                      Wmbij += 0.5 * t2 * eri_SO[((m * NSO_full + (b + VO)) * NSO_full + (e + VO)) * NSO_full + (f + VO)]!;
                      Wmaij += 0.5 * t2 * eri_SO[((m * NSO_full + (a + VO)) * NSO_full + (e + VO)) * NSO_full + (f + VO)]!;
                    }
                  }
                  z -= Wmbij * R_1[m * NVIRT_full + a]!;
                  z += Wmaij * R_1[m * NVIRT_full + b]!;
                }
                s2[((i * NOCC_full + j) * NVIRT_full + a) * NVIRT_full + b] = z;
              }
            }
          }
        }
        return { s1, s2 };
      };

      // Build M column-by-column.
      const R_1 = new Float64Array(NOCC_full * NVIRT_full);
      const R_2 = new Float64Array(NOCC_full * NOCC_full * NVIRT_full * NVIRT_full);
      const nS = NOCC_full * NVIRT_full;
      // Map our basisLabels packing to runEOMCCSD's column ordering.
      // Singles: index i*NVIRT + a. Our basis: same.
      // Doubles packed (i>j): pair (1, 0). 1 packed double for H₂.
      for (let col = 0; col < dim; col++) {
        R_1.fill(0); R_2.fill(0);
        if (col < nS) {
          R_1[col] = 1;
        } else {
          // packed double: (i=1, j=0, a=1, b=0)
          R_2[((1 * NOCC_full + 0) * NVIRT_full + 1) * NVIRT_full + 0] = 1;
          R_2[((0 * NOCC_full + 1) * NVIRT_full + 1) * NVIRT_full + 0] = -1;
          R_2[((1 * NOCC_full + 0) * NVIRT_full + 0) * NVIRT_full + 1] = -1;
          R_2[((0 * NOCC_full + 1) * NVIRT_full + 0) * NVIRT_full + 1] = 1;
        }
        const { s1, s2 } = sigmaFn(R_1, R_2);
        for (let row = 0; row < nS; row++) M[row * dim + col] = s1[row]!;
        // doubles row (packed (1, 0, 1, 0)):
        M[nS * dim + col] = s2[((1 * NOCC_full + 0) * NVIRT_full + 1) * NVIRT_full + 0]!;
      }
      return M;
    };

    const M_mine = buildMMine();
    console.log(`[bf-eom-h2] M_mine (5×5, in basis [R₁_00, R₁_01, R₁_10, R₁_11, R₂_1010]):`);
    for (let i = 0; i < dim; i++) {
      const row = Array.from({ length: dim }, (_, j) =>
        M_mine[i * dim + j]!.toExponential(3).padStart(11),
      ).join(" ");
      console.log(`  ${basisLabels[i]!.padEnd(10)} ${row}`);
    }

    // Diff matrix: where M_mine − M_exact ≠ 0.
    console.log(`[bf-eom-h2] M_mine − M_exact:`);
    for (let i = 0; i < dim; i++) {
      const row = Array.from({ length: dim }, (_, j) => {
        const d = M_mine[i * dim + j]! - M_exact[i * dim + j]!;
        return Math.abs(d) > 1e-9 ? d.toExponential(3).padStart(11) : "    .      ";
      }).join(" ");
      console.log(`  ${basisLabels[i]!.padEnd(10)} ${row}`);
    }
  });
});
