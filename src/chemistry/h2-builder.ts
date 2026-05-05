// ─────────────────────────────────────────────────────────────
// h2-builder.ts — build the H₂ qubit Hamiltonian as a 16×16
// dense Hermitian matrix, parameterized by bond length R.
//
// For VQE we don't actually need the Pauli decomposition — the
// expectation ⟨ψ|H|ψ⟩ for a 4-qubit system is just a 16-vector
// inner product against H·ψ. So we skip the Jordan–Wigner step
// and feed the dense matrix directly to a custom expectation.
//
// The pipeline:
//   1. STO-3G H 1s contracted-Gaussian integrals at the two
//      atom centers (integrals.ts).
//   2. Symmetry MOs: σ_g = (φ_A + φ_B)/√(2(1+S)), σ_u likewise.
//   3. Transform 1- and 2-electron integrals to MO basis.
//   4. Build the 16×16 second-quantized Hamiltonian:
//        H = Σ_pq h_pq^MO  Σ_σ a†_pσ a_qσ
//          + ½ Σ_pqrs (pq|rs)^MO  Σ_στ a†_pσ a†_rτ a_sτ a_qσ
//          + V_nn · I
//      via direct enumeration over 16 occupation-number basis
//      states with proper Jordan–Wigner sign bookkeeping.
//
// Cross-check: at R = 0.7414 Å the Hamiltonian built here must
// agree with the hardcoded OpenFermion Pauli-sum Hamiltonian's
// dense matrix to within a few mHa (small basis-rounding gap).
// ─────────────────────────────────────────────────────────────

import { S_AB, T_AB, V_AB, ERI } from "./integrals.js";

const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

type Center = readonly [number, number, number];

export interface H2Integrals {
  readonly R_Bohr: number;
  /** One-electron integrals in MO basis: [σ_g, σ_u] × [σ_g, σ_u]. */
  readonly h_MO: readonly [readonly [number, number], readonly [number, number]];
  /** Two-electron integrals (chemist notation), 2×2×2×2 in MO basis. */
  readonly v_MO: readonly (readonly (readonly (readonly number[])[])[])[];
  /** Nuclear repulsion energy 1/R (atomic units). */
  readonly Vnn: number;
  /** Overlap ⟨φ_A|φ_B⟩ — useful diagnostic, used in MO normalization. */
  readonly S_AB: number;
}

/** Compute all atomic + MO integrals for H₂ at bond length R (in Å). */
export function computeH2Integrals(R_angstrom: number): H2Integrals {
  const R = R_angstrom * ANGSTROM_TO_BOHR;
  const A: Center = [0, 0, 0];
  const B: Center = [0, 0, R];

  // ── AO integrals (2×2 symmetric matrices) ────────────────
  // S_AA = S_BB = 1 by normalization; only the cross-overlap matters.
  const Sab = S_AB(A, B);

  const Taa = T_AB(A, A);
  const Tab = T_AB(A, B);
  const Tbb = T_AB(B, B);

  // Nuclear attraction terms: V[mu, nu, on-atom-X] for X ∈ {A, B}.
  // We need each since both nuclei attract every electron.
  const VA_aa = V_AB(A, A, 1, A);
  const VA_ab = V_AB(A, B, 1, A);
  const VA_bb = V_AB(B, B, 1, A);
  const VB_aa = V_AB(A, A, 1, B);
  const VB_ab = V_AB(A, B, 1, B);
  const VB_bb = V_AB(B, B, 1, B);

  // h^AO_{mu,nu} = T + V_A + V_B
  const h_AO = [
    [Taa + VA_aa + VB_aa, Tab + VA_ab + VB_ab],
    [Tab + VA_ab + VB_ab, Tbb + VA_bb + VB_bb],
  ];

  // ── ERIs over AO 1s functions on A and B ────────────────
  // 5 distinct (μν|λσ) up to chemist 8-fold symmetry on H₂:
  //   (AA|AA), (BB|BB), (AA|BB), (AB|AB), (AA|AB)
  const eri_AAAA = ERI(A, A, A, A);
  const eri_BBBB = ERI(B, B, B, B);
  const eri_AABB = ERI(A, A, B, B);   // = (BB|AA)
  const eri_ABAB = ERI(A, B, A, B);   // = (BA|BA) — exchange-like
  const eri_AAAB = ERI(A, A, A, B);   // mixed
  // By symmetry of H₂ (homonuclear), AA|AA == BB|BB, AA|AB == BB|AB.
  // We compute both for verification.

  // ── Symmetry-adapted MOs ──────────────────────────────
  // σ_g = (φ_A + φ_B) / N_g,  N_g = √(2(1 + S_ab))
  // σ_u = (φ_A − φ_B) / N_u,  N_u = √(2(1 − S_ab))
  const Ng = Math.sqrt(2 * (1 + Sab));
  const Nu = Math.sqrt(2 * (1 - Sab));

  // MO coefficient matrix C: C[μ, p] where μ ∈ {A, B}, p ∈ {g, u}.
  const C_MO: [readonly [number, number], readonly [number, number]] = [
    [ 1 / Ng,  1 / Nu],   // φ_A row
    [ 1 / Ng, -1 / Nu],   // φ_B row
  ];

  // ── Transform h^AO to h^MO ────────────────────────────
  // h^MO_{p,q} = Σ_{μν} C[μ,p] · h^AO_{μν} · C[ν,q]
  const h_MO_arr: number[][] = [[0, 0], [0, 0]];
  for (let p = 0; p < 2; p++) {
    for (let q = 0; q < 2; q++) {
      let s = 0;
      for (let mu = 0; mu < 2; mu++) {
        for (let nu = 0; nu < 2; nu++) {
          s += C_MO[mu]![p]! * h_AO[mu]![nu]! * C_MO[nu]![q]!;
        }
      }
      h_MO_arr[p]![q] = s;
    }
  }

  // ── Transform two-electron AO ERIs to MO ──────────────
  // (pq|rs)^MO = Σ_{μνλσ} C[μ,p] C[ν,q] C[λ,r] C[σ,s] (μν|λσ)^AO
  function eri_AO(mu: number, nu: number, la: number, si: number): number {
    // mu, nu, la, si ∈ {0=A, 1=B}. Combine the 5 unique values via 8-fold symmetry:
    //   (μν|λσ) is invariant under: μ↔ν, λ↔σ, (μν)↔(λσ).
    // Canonicalize the 4-tuple to a sorted form for lookup.
    const a = Math.min(mu, nu), b = Math.max(mu, nu);
    const c = Math.min(la, si), d = Math.max(la, si);
    const lhs = a * 2 + b;  // {0..3}
    const rhs = c * 2 + d;
    const lo = Math.min(lhs, rhs), hi = Math.max(lhs, rhs);
    // Encode as "lo,hi" with lo ≤ hi and a ≤ b within each pair.
    const key = lo * 4 + hi;
    // Decode: which integral?
    // (AA|AA) -> lo=0, hi=0  → key 0
    // (AA|AB) -> lo=0, hi=1  → key 1
    // (AA|BB) -> lo=0, hi=3  → key 3
    // (AB|AB) -> lo=1, hi=1  → key 5
    // (AB|BB) -> lo=1, hi=3  → key 7
    // (BB|BB) -> lo=3, hi=3  → key 15
    switch (key) {
      case 0:  return eri_AAAA;
      case 1:  return eri_AAAB;
      case 3:  return eri_AABB;
      case 5:  return eri_ABAB;
      case 7:  return eri_AAAB;   // by H₂ homonuclear symmetry
      case 15: return eri_BBBB;
      default: return 0;
    }
  }

  const v_MO: number[][][][] = [
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
  ];
  for (let p = 0; p < 2; p++) {
    for (let q = 0; q < 2; q++) {
      for (let r = 0; r < 2; r++) {
        for (let s = 0; s < 2; s++) {
          let acc = 0;
          for (let mu = 0; mu < 2; mu++) {
            const cmp = C_MO[mu]![p]!;
            for (let nu = 0; nu < 2; nu++) {
              const cnq = C_MO[nu]![q]!;
              for (let la = 0; la < 2; la++) {
                const clr = C_MO[la]![r]!;
                for (let si = 0; si < 2; si++) {
                  const css = C_MO[si]![s]!;
                  acc += cmp * cnq * clr * css * eri_AO(mu, nu, la, si);
                }
              }
            }
          }
          v_MO[p]![q]![r]![s] = acc;
        }
      }
    }
  }

  return {
    R_Bohr: R,
    h_MO: [[h_MO_arr[0]![0]!, h_MO_arr[0]![1]!], [h_MO_arr[1]![0]!, h_MO_arr[1]![1]!]],
    v_MO,
    Vnn: 1 / R,
    S_AB: Sab,
  };
}

// ── Fermionic operator → dense matrix (16×16) ────────────────
//
// Spin-orbital ordering: 0=σ_g↑, 1=σ_g↓, 2=σ_u↑, 3=σ_u↓.
// Basis state |n_0 n_1 n_2 n_3⟩ stored at index n_0 + 2 n_1 + 4 n_2 + 8 n_3.
//
// JW sign for an operator at orbital q: (-1)^(# occupied orbitals < q
// in the current state). Standard left-to-right JW string.

const N = 16;

/** Apply a (creation, qubit) op to a state. Returns null if forbidden. */
function applyOp(state: number, qubit: number, creation: boolean): { newState: number; sign: number } | null {
  const occupied = (state >>> qubit) & 1;
  if (creation && occupied) return null;       // can't create where occupied
  if (!creation && !occupied) return null;     // can't annihilate vacuum
  // JW string sign: (-1)^(parity of bits at positions < qubit).
  let sign = 1;
  let mask = 1;
  for (let k = 0; k < qubit; k++) {
    if ((state & mask) !== 0) sign = -sign;
    mask <<= 1;
  }
  const newState = state ^ (1 << qubit);
  return { newState, sign };
}

/** Add coeff · a†_p a_q to dense H. */
function addOneBody(H: Float64Array, coeff: number, p: number, q: number): void {
  if (Math.abs(coeff) < 1e-15) return;
  for (let s = 0; s < N; s++) {
    const r1 = applyOp(s, q, false);
    if (!r1) continue;
    const r2 = applyOp(r1.newState, p, true);
    if (!r2) continue;
    H[r2.newState * N + s]! += coeff * r1.sign * r2.sign;
  }
}

/** Add coeff · a†_p a†_r a_s a_q to dense H. (Chemist (pq|rs) ordering). */
function addTwoBody(H: Float64Array, coeff: number, p: number, q: number, r: number, s: number): void {
  if (Math.abs(coeff) < 1e-15) return;
  for (let st = 0; st < N; st++) {
    const r1 = applyOp(st, q, false);
    if (!r1) continue;
    const r2 = applyOp(r1.newState, s, false);
    if (!r2) continue;
    const r3 = applyOp(r2.newState, r, true);
    if (!r3) continue;
    const r4 = applyOp(r3.newState, p, true);
    if (!r4) continue;
    const sign = r1.sign * r2.sign * r3.sign * r4.sign;
    H[r4.newState * N + st]! += coeff * sign;
  }
}

/** Build 16×16 dense Hermitian H₂ Hamiltonian at bond length R (Å). */
export function buildH2Dense(R_angstrom: number): { H: Float64Array; integrals: H2Integrals } {
  const integrals = computeH2Integrals(R_angstrom);
  const { h_MO, v_MO, Vnn } = integrals;
  const H = new Float64Array(N * N);

  // Identity term: V_nn on the diagonal.
  for (let i = 0; i < N; i++) H[i * N + i] = Vnn;

  // One-body: Σ_{pq, σ} h_pq^MO a†_{2p+σ} a_{2q+σ}.
  for (let p = 0; p < 2; p++) {
    for (let q = 0; q < 2; q++) {
      const hpq = h_MO[p]![q]!;
      for (let sigma = 0; sigma < 2; sigma++) {
        addOneBody(H, hpq, 2 * p + sigma, 2 * q + sigma);
      }
    }
  }

  // Two-body: ½ Σ_{pqrs, στ} (pq|rs)^MO a†_{2p+σ} a†_{2r+τ} a_{2s+τ} a_{2q+σ}.
  for (let p = 0; p < 2; p++) {
    for (let q = 0; q < 2; q++) {
      for (let r = 0; r < 2; r++) {
        for (let s = 0; s < 2; s++) {
          const pqrs = v_MO[p]![q]![r]![s]!;
          if (Math.abs(pqrs) < 1e-15) continue;
          for (let sigma = 0; sigma < 2; sigma++) {
            for (let tau = 0; tau < 2; tau++) {
              addTwoBody(H, 0.5 * pqrs, 2 * p + sigma, 2 * q + sigma, 2 * r + tau, 2 * s + tau);
            }
          }
        }
      }
    }
  }

  return { H, integrals };
}

/**
 * ⟨ψ|H|ψ⟩ for a dense real-symmetric H acting on a length-2·dim
 * interleaved [re, im] Float64 statevector.
 *
 * Dimension is inferred from `psi.length / 2`. H must be (dim × dim)
 * row-major real Float64. Originally hard-coded to dim=16 (H₂);
 * generalized in Phase C so LiH (dim=64) and other multi-orbital
 * molecules can share the same energy oracle.
 */
export function expectationDense(psi: Float64Array, H: Float64Array): number {
  const dim = psi.length >>> 1;
  let acc = 0;
  for (let i = 0; i < dim; i++) {
    let hpsiRe = 0, hpsiIm = 0;
    for (let j = 0; j < dim; j++) {
      const h = H[i * dim + j]!;
      hpsiRe += h * psi[j * 2]!;
      hpsiIm += h * psi[j * 2 + 1]!;
    }
    acc += psi[i * 2]! * hpsiRe + psi[i * 2 + 1]! * hpsiIm;
  }
  return acc;
}
