// ─────────────────────────────────────────────────────────────
// hn-builder.ts — generalize the H₂ molecular-integral pipeline
// to a linear chain of n equally-spaced hydrogen atoms.
//
//   H_n chain in STO-3G:
//     • n atomic 1s functions, one per H, on the z-axis
//     • n × n overlap matrix S (non-orthogonal)
//     • n × n one-electron core h^AO = T + Σ_C V_C
//     • n^4 two-electron AO integrals (μν|λσ)
//     • Löwdin orthogonalization X = S^{-1/2} → orthonormal basis
//     • Spin-orbital ordering: q = 2·spatial + spin (spin ∈ {0,1})
//     • Jordan-Wigner → 2^(2n)-dim Hermitian Hamiltonian
//
// Output: a dense 2^(2n) × 2^(2n) matrix and the integral block
// for inspection / downstream MPS use.
//
// Scaling: n=2 → 16-dim (H₂, our existing system); n=3 → 64;
// n=4 → 256; n=5 → 1024; n=6 → 4096. Beyond n=6 our Jacobi
// eigensolver is too slow — switch to imag-time MPS for those.
// ─────────────────────────────────────────────────────────────

import { S_AB, T_AB, V_AB, ERI } from "./integrals.js";
import { eigsymmetric } from "../manybody/dense-eig.js";

const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

type Center = readonly [number, number, number];

export interface HnIntegrals {
  /** Number of atoms = number of spatial orbitals. */
  n: number;
  /** Inter-atom spacing (Bohr). */
  R_Bohr: number;
  /** Atom centers (Bohr). */
  centers: ReadonlyArray<Center>;
  /** AO overlap matrix S, n×n, row-major. */
  S_AO: Float64Array;
  /** AO one-electron core h = T + Σ_C V_C, n×n. */
  h_AO: Float64Array;
  /** AO two-electron tensor (μν|λσ), shape n^4 flat row-major (μν λσ). */
  eri_AO: Float64Array;
  /** Orthogonalizing transform X = S^{-1/2}, n×n column-major. */
  X: Float64Array;
  /** Orthogonal one-electron h^OAO = X^T h^AO X, n×n. */
  h_OAO: Float64Array;
  /** Orthogonal two-electron, shape n^4. */
  eri_OAO: Float64Array;
  /** Nuclear repulsion energy (Hartree). */
  Vnn: number;
}

/** Compute all integrals for an n-H chain at uniform spacing R (Å). */
export function computeHnIntegrals(n: number, R_angstrom: number): HnIntegrals {
  if (n < 2) throw new Error(`hn: n must be ≥ 2`);
  const R = R_angstrom * ANGSTROM_TO_BOHR;
  const centers: Center[] = [];
  for (let i = 0; i < n; i++) {
    centers.push([0, 0, (i - (n - 1) / 2) * R]);
  }

  // AO matrices.
  const S_AO = new Float64Array(n * n);
  const h_AO = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu; nu < n; nu++) {
      const sij = S_AB(centers[mu]!, centers[nu]!);
      const tij = T_AB(centers[mu]!, centers[nu]!);
      let vij = 0;
      for (let c = 0; c < n; c++) vij += V_AB(centers[mu]!, centers[nu]!, 1, centers[c]!);
      S_AO[mu * n + nu] = sij;
      S_AO[nu * n + mu] = sij;
      h_AO[mu * n + nu] = tij + vij;
      h_AO[nu * n + mu] = tij + vij;
    }
  }

  // ERI tensor: (μν|λσ). 8-fold symmetry — fill all but compute unique only.
  const eri_AO = new Float64Array(n * n * n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          // Compute on demand; ERI is reasonably symmetric so we just call.
          const v = ERI(centers[mu]!, centers[nu]!, centers[la]!, centers[si]!);
          eri_AO[mu * n * n * n + nu * n * n + la * n + si] = v;
        }
      }
    }
  }

  // Löwdin orthogonalization: X = U Σ^{-1/2} U^T where S = U Σ U^T.
  const eig = eigsymmetric(S_AO, n);
  const X = new Float64Array(n * n);
  // X[r, c] = Σ_k U[r, k] · σ_k^{-1/2} · U[c, k]; vectors[k*N+r] = U[r,k].
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        s += eig.vectors[k * n + r]! * Math.pow(eig.values[k]!, -0.5) * eig.vectors[k * n + c]!;
      }
      X[r * n + c] = s;
    }
  }

  // Transform h_AO and eri_AO to orthogonalized basis: h^OAO = X^T h^AO X.
  const h_OAO = transform2(h_AO, X, n);
  const eri_OAO = transform4(eri_AO, X, n);

  // Nuclear-nuclear repulsion: 1/R for every i<j pair.
  let Vnn = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = centers[i]![0] - centers[j]![0];
      const dy = centers[i]![1] - centers[j]![1];
      const dz = centers[i]![2] - centers[j]![2];
      Vnn += 1 / Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
  }

  return {
    n,
    R_Bohr: R,
    centers,
    S_AO,
    h_AO,
    eri_AO,
    X,
    h_OAO,
    eri_OAO,
    Vnn,
  };
}

/** Two-index AO→OAO transform: h^OAO[p,q] = Σ_{μν} X[μ,p] h^AO[μ,ν] X[ν,q]. */
function transform2(M_AO: Float64Array, X: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      let s = 0;
      for (let mu = 0; mu < n; mu++) {
        for (let nu = 0; nu < n; nu++) {
          s += X[mu * n + p]! * M_AO[mu * n + nu]! * X[nu * n + q]!;
        }
      }
      out[p * n + q] = s;
    }
  }
  return out;
}

/** Four-index AO→OAO transform of (μν|λσ) → (pq|rs). Uses 4 successive 2-index passes. */
function transform4(eri_AO: Float64Array, X: Float64Array, n: number): Float64Array {
  // Pass 1: (μν|λσ) → (pν|λσ): contract first index.
  let buf1 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let nu = 0; nu < n; nu++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let mu = 0; mu < n; mu++) {
            s += X[mu * n + p]! * eri_AO[mu * n * n * n + nu * n * n + la * n + si]!;
          }
          buf1[p * n * n * n + nu * n * n + la * n + si] = s;
        }
      }
    }
  }
  // Pass 2: (pν|λσ) → (pq|λσ).
  let buf2 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let nu = 0; nu < n; nu++) {
            s += X[nu * n + q]! * buf1[p * n * n * n + nu * n * n + la * n + si]!;
          }
          buf2[p * n * n * n + q * n * n + la * n + si] = s;
        }
      }
    }
  }
  // Pass 3: (pq|λσ) → (pq|rσ).
  buf1 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let la = 0; la < n; la++) {
            s += X[la * n + r]! * buf2[p * n * n * n + q * n * n + la * n + si]!;
          }
          buf1[p * n * n * n + q * n * n + r * n + si] = s;
        }
      }
    }
  }
  // Pass 4: (pq|rσ) → (pq|rs).
  buf2 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let s = 0; s < n; s++) {
          let acc = 0;
          for (let si = 0; si < n; si++) {
            acc += X[si * n + s]! * buf1[p * n * n * n + q * n * n + r * n + si]!;
          }
          buf2[p * n * n * n + q * n * n + r * n + s] = acc;
        }
      }
    }
  }
  return buf2;
}

// ─── Build the dense 2^(2n) × 2^(2n) Hermitian for the H_n chain ────
//
// Spin-orbital ordering: index q = 2·spatial + spin (spin ∈ {0=α, 1=β}).
// Basis state |s_0 s_1 … s_{2n-1}⟩ at index Σ_q s_q · 2^q.

function applyOp(state: number, qubit: number, creation: boolean): { newState: number; sign: number } | null {
  const occupied = (state >>> qubit) & 1;
  if (creation && occupied) return null;
  if (!creation && !occupied) return null;
  let sign = 1;
  let mask = 1;
  for (let k = 0; k < qubit; k++) {
    if ((state & mask) !== 0) sign = -sign;
    mask <<= 1;
  }
  return { newState: state ^ (1 << qubit), sign };
}

function addOneBody(H: Float64Array, dim: number, coeff: number, p: number, q: number): void {
  if (Math.abs(coeff) < 1e-15) return;
  for (let s = 0; s < dim; s++) {
    const r1 = applyOp(s, q, false);
    if (!r1) continue;
    const r2 = applyOp(r1.newState, p, true);
    if (!r2) continue;
    H[r2.newState * dim + s]! += coeff * r1.sign * r2.sign;
  }
}

function addTwoBody(H: Float64Array, dim: number, coeff: number, p: number, q: number, r: number, s: number): void {
  if (Math.abs(coeff) < 1e-15) return;
  for (let st = 0; st < dim; st++) {
    const r1 = applyOp(st, q, false);
    if (!r1) continue;
    const r2 = applyOp(r1.newState, s, false);
    if (!r2) continue;
    const r3 = applyOp(r2.newState, r, true);
    if (!r3) continue;
    const r4 = applyOp(r3.newState, p, true);
    if (!r4) continue;
    H[r4.newState * dim + st]! += coeff * r1.sign * r2.sign * r3.sign * r4.sign;
  }
}

/** Build the full 2^(2n) × 2^(2n) dense Hamiltonian for an H_n chain at bond R (Å). */
export function buildHnDense(n: number, R_angstrom: number): {
  H: Float64Array;
  integrals: HnIntegrals;
  nQubits: number;
} {
  const integrals = computeHnIntegrals(n, R_angstrom);
  const nQubits = 2 * n;
  const dim = 1 << nQubits;
  if (dim > 8192) {
    throw new Error(`buildHnDense: 2^(2n)=${dim} too large for dense storage. n=${n} → ${nQubits} qubits.`);
  }
  const H = new Float64Array(dim * dim);

  // V_nn on the diagonal.
  for (let i = 0; i < dim; i++) H[i * dim + i] = integrals.Vnn;

  // One-body Σ_{pq, σ} h_pq^OAO a†_{2p+σ} a_{2q+σ}.
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      const hpq = integrals.h_OAO[p * n + q]!;
      for (let sigma = 0; sigma < 2; sigma++) {
        addOneBody(H, dim, hpq, 2 * p + sigma, 2 * q + sigma);
      }
    }
  }

  // Two-body ½ Σ_{pqrs, στ} (pq|rs)^OAO a†_{2p+σ} a†_{2r+τ} a_{2s+τ} a_{2q+σ}.
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let s = 0; s < n; s++) {
          const pqrs = integrals.eri_OAO[p * n * n * n + q * n * n + r * n + s]!;
          if (Math.abs(pqrs) < 1e-15) continue;
          for (let sigma = 0; sigma < 2; sigma++) {
            for (let tau = 0; tau < 2; tau++) {
              addTwoBody(H, dim, 0.5 * pqrs, 2 * p + sigma, 2 * q + sigma, 2 * r + tau, 2 * s + tau);
            }
          }
        }
      }
    }
  }

  return { H, integrals, nQubits };
}
