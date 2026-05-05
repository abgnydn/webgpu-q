// ─────────────────────────────────────────────────────────────
// lih-builder.ts — LiH ground-state Hamiltonian, STO-3G s-only.
//
// Basis (3 spatial orbitals, 6 spin-orbitals):
//   • Li 1s   (Z = 3 core, 3 primitive Gaussians)
//   • Li 2s   (s-component of the L-shell)
//   • H 1s    (3 primitive Gaussians)
//
// Phase-C scope deliberately excludes the Li 2p sub-shell. With
// 2p the basis would be 6 spatial / 12 spin-orbitals = 4096-dim
// Hilbert space — manageable but slow for the dense Jacobi
// solver; and STO-3G p-shell ERIs need angular-momentum-aware
// Obara–Saika integrals not yet implemented in integrals.ts.
//
// Pipeline:
//   1. Build 3×3 AO matrices S, h^AO = T + V_Li + V_H.
//   2. Build 3⁴ AO ERI tensor (μν|λσ).
//   3. Löwdin orthogonalize: X = S^{-1/2}, transform h and ERI.
//   4. Spin-orbital ordering q = 2·spatial + spin.
//      Spatial 0 = Li 1s, 1 = Li 2s, 2 = H 1s.
//      HF reference: Li 1s² 2s¹ + H 1s¹ → spin-orbitals
//      {0, 1, 2, 4} occupied (4 electrons).
//   5. JW-build the 64×64 dense Hamiltonian.
//
// Cross-checks (lih.test.ts):
//   • Hermiticity (real symmetric).
//   • Block-diagonal in particle number — eigenstates conserve N_e.
//   • Singlet ground state (S_z = 0 sector dominates lowest band).
//   • Dissociation: at large R, E_GS(R) approaches E(Li atom in
//     Li-only basis) + E(H atom in H-only basis).
// ─────────────────────────────────────────────────────────────

import {
  STO3G_LI_1S, STO3G_LI_2S, STO3G_H_1S,
  S_shells, T_shells, V_shells, ERI_shells,
  makeShell, type Shell,
} from "./integrals.js";
import { eigsymmetric } from "../manybody/dense-eig.js";

const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

type Center = readonly [number, number, number];

export interface LiHIntegrals {
  /** Bond length (Bohr). */
  readonly R_Bohr: number;
  /** Atom centers — [0]=Li, [1]=H. */
  readonly centers: readonly [Center, Center];
  /** Atomic shells in spatial-orbital order (Li 1s, Li 2s, H 1s). */
  readonly shells: readonly [Shell, Shell, Shell];
  /** AO overlap S, 3×3 row-major. */
  readonly S_AO: Float64Array;
  /** AO core h = T + V_Li + V_H, 3×3 row-major. */
  readonly h_AO: Float64Array;
  /** AO ERI (μν|λσ), 3⁴ row-major as [μν λσ]. */
  readonly eri_AO: Float64Array;
  /** Löwdin transform X = S^{-1/2}, 3×3 row-major. */
  readonly X: Float64Array;
  /** Orthogonalized one-electron h^OAO, 3×3. */
  readonly h_OAO: Float64Array;
  /** Orthogonalized two-electron tensor, 3⁴. */
  readonly eri_OAO: Float64Array;
  /** Nuclear-nuclear repulsion 3·1/R (Z_Li · Z_H = 3). */
  readonly Vnn: number;
}

export interface LiHHamiltonian {
  readonly H: Float64Array;          // 64×64 dense, real-symmetric
  readonly nQubits: 6;
  readonly integrals: LiHIntegrals;
  /** Spin-orbital indices occupied in the Hartree–Fock reference
   *  state |Li 1s² 2s¹ + H 1s¹⟩, ready for runVQE_HEA_Dense. */
  readonly hfOccupied: readonly number[];
}

const N_ATOMIC = 3;       // 3 spatial orbitals
const N_QUBITS = 6;       // 6 spin-orbitals
const DIM = 1 << N_QUBITS; // 64

/** Compute all integrals for LiH at bond length R (Å). */
export function computeLiHIntegrals(R_angstrom: number): LiHIntegrals {
  const R = R_angstrom * ANGSTROM_TO_BOHR;
  const Li_pos: Center = [0, 0, 0];
  const H_pos: Center = [0, 0, R];

  const liShell1s = makeShell(STO3G_LI_1S, Li_pos, "Li:1s");
  const liShell2s = makeShell(STO3G_LI_2S, Li_pos, "Li:2s");
  const hShell    = makeShell(STO3G_H_1S,  H_pos,  "H:1s");
  const shells = [liShell1s, liShell2s, hShell] as const;

  // ── AO overlap and one-electron core ────────────────────
  const S_AO = new Float64Array(N_ATOMIC * N_ATOMIC);
  const h_AO = new Float64Array(N_ATOMIC * N_ATOMIC);
  for (let mu = 0; mu < N_ATOMIC; mu++) {
    for (let nu = mu; nu < N_ATOMIC; nu++) {
      const sij = S_shells(shells[mu]!, shells[nu]!);
      const tij = T_shells(shells[mu]!, shells[nu]!);
      // V_Li (Z = 3) + V_H (Z = 1).
      const vLi = V_shells(shells[mu]!, shells[nu]!, 3, Li_pos);
      const vH  = V_shells(shells[mu]!, shells[nu]!, 1, H_pos);
      S_AO[mu * N_ATOMIC + nu] = sij;
      S_AO[nu * N_ATOMIC + mu] = sij;
      const h = tij + vLi + vH;
      h_AO[mu * N_ATOMIC + nu] = h;
      h_AO[nu * N_ATOMIC + mu] = h;
    }
  }

  // ── AO two-electron tensor ──────────────────────────────
  const eri_AO = new Float64Array(N_ATOMIC * N_ATOMIC * N_ATOMIC * N_ATOMIC);
  for (let mu = 0; mu < N_ATOMIC; mu++) {
    for (let nu = 0; nu < N_ATOMIC; nu++) {
      for (let la = 0; la < N_ATOMIC; la++) {
        for (let si = 0; si < N_ATOMIC; si++) {
          const v = ERI_shells(shells[mu]!, shells[nu]!, shells[la]!, shells[si]!);
          eri_AO[idx4(mu, nu, la, si)] = v;
        }
      }
    }
  }

  // ── Löwdin orthogonalization X = S^{-1/2} ──────────────
  // S = U Σ U^T → X = U Σ^{-1/2} U^T.  eigsymmetric returns
  // eigenvectors stored column-major: vectors[k*N+r] = U[r,k].
  const eig = eigsymmetric(S_AO, N_ATOMIC);
  const X = new Float64Array(N_ATOMIC * N_ATOMIC);
  for (let r = 0; r < N_ATOMIC; r++) {
    for (let c = 0; c < N_ATOMIC; c++) {
      let s = 0;
      for (let k = 0; k < N_ATOMIC; k++) {
        const lam = eig.values[k]!;
        if (lam <= 0) {
          throw new Error(`computeLiHIntegrals: AO overlap eigenvalue ${lam} is non-positive`);
        }
        s += eig.vectors[k * N_ATOMIC + r]! * Math.pow(lam, -0.5) * eig.vectors[k * N_ATOMIC + c]!;
      }
      X[r * N_ATOMIC + c] = s;
    }
  }

  // ── Transform h and ERI to orthogonal basis ─────────────
  const h_OAO = transform2(h_AO, X, N_ATOMIC);
  const eri_OAO = transform4(eri_AO, X, N_ATOMIC);

  // ── V_nn = Z_Li · Z_H / R = 3 / R ──────────────────────
  const Vnn = 3 / R;

  return {
    R_Bohr: R,
    centers: [Li_pos, H_pos] as const,
    shells: shells as readonly [Shell, Shell, Shell],
    S_AO, h_AO, eri_AO, X, h_OAO, eri_OAO, Vnn,
  };
}

/**
 * Build the 64×64 dense LiH Hamiltonian at bond length R (Å).
 * The HF reference (Li 1s² 2s¹, H 1s¹) lives at spin-orbital
 * indices {0, 1, 2, 4} — matched to the q = 2·spatial + spin
 * convention used by runVQE_HEA_Dense.
 */
export function buildLiHDense(R_angstrom: number): LiHHamiltonian {
  const integrals = computeLiHIntegrals(R_angstrom);
  const H = new Float64Array(DIM * DIM);

  // V_nn on the diagonal.
  for (let i = 0; i < DIM; i++) H[i * DIM + i] = integrals.Vnn;

  // One-body Σ_{pq, σ} h_pq^OAO a†_{2p+σ} a_{2q+σ}.
  for (let p = 0; p < N_ATOMIC; p++) {
    for (let q = 0; q < N_ATOMIC; q++) {
      const hpq = integrals.h_OAO[p * N_ATOMIC + q]!;
      for (let sigma = 0; sigma < 2; sigma++) {
        addOneBody(H, hpq, 2 * p + sigma, 2 * q + sigma);
      }
    }
  }

  // Two-body ½ Σ_{pqrs, στ} (pq|rs)^OAO a†_{2p+σ} a†_{2r+τ} a_{2s+τ} a_{2q+σ}.
  for (let p = 0; p < N_ATOMIC; p++) {
    for (let q = 0; q < N_ATOMIC; q++) {
      for (let r = 0; r < N_ATOMIC; r++) {
        for (let s = 0; s < N_ATOMIC; s++) {
          const pqrs = integrals.eri_OAO[idx4(p, q, r, s)]!;
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

  // HF reference: Li 1s² (spin-orbitals 0, 1) + Li 2s¹ (sp-orbital 2)
  // + H 1s¹ (sp-orbital 4) = singlet with 4 electrons in the Li 1s/2s
  // and H 1s shells. Spin-orbital 5 (H 1s β) is empty so the 2s¹ + 1s¹
  // pair is high-spin α; the singlet/triplet sorting is left to the
  // optimizer. Common active-space convention.
  // Note: a strict closed-shell HF for LiH would put both Li 2s and
  // H 1s electrons paired in the bonding σ MO — we keep the AO picture
  // here so the VQE oracle has clean orbital labels.
  const hfOccupied = [0, 1, 2, 4] as const;

  return { H, nQubits: N_QUBITS, integrals, hfOccupied: [...hfOccupied] };
}

/**
 * Lowest eigenstate restricted to a single particle-number sector.
 * The full second-quantized H is block-diagonal in N̂ (since [H, N̂] = 0)
 * and our 6-spin-orbital basis spans every N from 0 to 6. To get the
 * physical neutral-LiH ground state we project H to its N=4 block and
 * diagonalize that — much smaller (C(6,4)=15) and unambiguously
 * "the LiH state we mean".
 *
 * Returns { energy, psi } with psi expanded back to the full 64-dim
 * statevector (zero amplitude outside the sector) for downstream use
 * by the VQE oracle / observable expectation code.
 */
export function lowestInParticleSector(
  H: Float64Array,
  nQubits: number,
  particleNumber: number,
): { energy: number; psi: Float64Array } {
  const dim = 1 << nQubits;
  // Enumerate basis states in the target sector.
  const sectorIdx: number[] = [];
  for (let i = 0; i < dim; i++) {
    let n = 0;
    for (let q = 0; q < nQubits; q++) if ((i >>> q) & 1) n++;
    if (n === particleNumber) sectorIdx.push(i);
  }
  const k = sectorIdx.length;
  if (k === 0) {
    throw new Error(`lowestInParticleSector: no basis states with N=${particleNumber} in ${nQubits}-qubit space`);
  }
  // Project H to the sector: H_sec[a, b] = H[sectorIdx[a], sectorIdx[b]].
  const Hsec = new Float64Array(k * k);
  for (let a = 0; a < k; a++) {
    const ia = sectorIdx[a]!;
    for (let b = 0; b < k; b++) {
      Hsec[a * k + b] = H[ia * dim + sectorIdx[b]!]!;
    }
  }
  const eig = eigsymmetric(Hsec, k);
  // Embed the lowest eigenvector back into the full Hilbert space.
  // dense-eig returns column-major eigenvectors: vectors[col*N + row].
  // Column 0 (the smallest-eigenvalue eigenvector) lives at indices
  // vectors[0..k-1].
  const psi = new Float64Array(dim);
  for (let a = 0; a < k; a++) {
    psi[sectorIdx[a]!] = eig.vectors[a]!;
  }
  return { energy: eig.values[0]!, psi };
}

// ── Local helpers (mirror h2-builder / hn-builder) ───────────

function idx4(mu: number, nu: number, la: number, si: number): number {
  return ((mu * N_ATOMIC + nu) * N_ATOMIC + la) * N_ATOMIC + si;
}

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

function transform4(eri_AO: Float64Array, X: Float64Array, n: number): Float64Array {
  // Four sequential 2-index contractions, O(n⁵) each.
  let buf1 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let nu = 0; nu < n; nu++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let mu = 0; mu < n; mu++) {
            s += X[mu * n + p]! * eri_AO[((mu * n + nu) * n + la) * n + si]!;
          }
          buf1[((p * n + nu) * n + la) * n + si] = s;
        }
      }
    }
  }
  let buf2 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let nu = 0; nu < n; nu++) {
            s += X[nu * n + q]! * buf1[((p * n + nu) * n + la) * n + si]!;
          }
          buf2[((p * n + q) * n + la) * n + si] = s;
        }
      }
    }
  }
  buf1 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let la = 0; la < n; la++) {
            s += X[la * n + r]! * buf2[((p * n + q) * n + la) * n + si]!;
          }
          buf1[((p * n + q) * n + r) * n + si] = s;
        }
      }
    }
  }
  buf2 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let s = 0; s < n; s++) {
          let acc = 0;
          for (let si = 0; si < n; si++) {
            acc += X[si * n + s]! * buf1[((p * n + q) * n + r) * n + si]!;
          }
          buf2[((p * n + q) * n + r) * n + s] = acc;
        }
      }
    }
  }
  return buf2;
}

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

function addOneBody(H: Float64Array, coeff: number, p: number, q: number): void {
  if (Math.abs(coeff) < 1e-15) return;
  for (let s = 0; s < DIM; s++) {
    const r1 = applyOp(s, q, false);
    if (!r1) continue;
    const r2 = applyOp(r1.newState, p, true);
    if (!r2) continue;
    H[r2.newState * DIM + s]! += coeff * r1.sign * r2.sign;
  }
}

function addTwoBody(H: Float64Array, coeff: number, p: number, q: number, r: number, s: number): void {
  if (Math.abs(coeff) < 1e-15) return;
  for (let st = 0; st < DIM; st++) {
    const r1 = applyOp(st, q, false);
    if (!r1) continue;
    const r2 = applyOp(r1.newState, s, false);
    if (!r2) continue;
    const r3 = applyOp(r2.newState, r, true);
    if (!r3) continue;
    const r4 = applyOp(r3.newState, p, true);
    if (!r4) continue;
    H[r4.newState * DIM + st]! += coeff * r1.sign * r2.sign * r3.sign * r4.sign;
  }
}
