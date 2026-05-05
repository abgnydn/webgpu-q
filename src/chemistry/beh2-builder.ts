// ─────────────────────────────────────────────────────────────
// beh2-builder.ts — BeH₂ ground-state Hamiltonian, STO-3G s-only.
//
// Linear H–Be–H with Be at the origin and the two H atoms at
// (±R, 0, 0) for symmetric stretch. 4 atomic shells / 8 spin-
// orbitals / 256-dim Hilbert space; 6 electrons in the neutral
// molecule. Minimal s-only basis:
//   • Be 1s   (Z = 4 core, 3 primitives)
//   • Be 2s   (s component of the L-shell)
//   • H 1s × 2 (one per hydrogen)
//
// Phase-C v2 scope deliberately excludes the Be 2p sub-shell.
// Including 2p would lift the basis to 7 spatial / 14 spin-
// orbitals / 16384-dim Hilbert space (sector-restricted FCI to
// N=6 still tractable at C(14, 6) = 3003 dim) but requires
// Cartesian-Gaussian p-shell integrals (Obara-Saika recurrence)
// not yet implemented in integrals.ts. That's Phase C v3 work.
//
// Pipeline mirrors lih-builder.ts:
//   1. Build 4×4 AO matrices S, h^AO = T + V_Be + V_H1 + V_H2.
//   2. Build 4⁴ AO ERI tensor (μν|λσ).
//   3. Löwdin orthogonalize: X = S^{-1/2}, transform h and ERI.
//   4. Spin-orbital ordering q = 2·spatial + spin.
//      Spatial 0 = Be 1s, 1 = Be 2s, 2 = H_left 1s, 3 = H_right 1s.
//      HF reference (singlet, 6 e⁻): doubly occupy the 3 lowest
//      spatial orbitals → spin-orbitals {0,1,2,3,4,5}. The 4th
//      spatial (H_right 1s in this AO labelling) is empty in
//      the HF determinant; the optimizer then mixes it in.
//   5. JW-build the 256×256 dense Hamiltonian.
//
// Cross-checks (beh2.test.ts):
//   • Hermiticity (real symmetric).
//   • [H, N̂] = 0.
//   • Symmetric-stretch dissociation: minimum near R = 1.34 Å
//     (experimental Be-H bond length).
//   • Ground state lives in the N=6 sector, energy < HF reference.
// ─────────────────────────────────────────────────────────────

import {
  STO3G_BE_1S, STO3G_BE_2S, STO3G_H_1S,
  S_shells, T_shells, V_shells, ERI_shells,
  makeShell, type Shell,
} from "./integrals.js";
import { eigsymmetric } from "../manybody/dense-eig.js";

const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

type Center = readonly [number, number, number];

export interface BeH2Integrals {
  /** Be–H bond length (Bohr). The two H atoms are at ±R from Be. */
  readonly R_Bohr: number;
  /** Atom centers — [0]=Be, [1]=H_left, [2]=H_right. */
  readonly centers: readonly [Center, Center, Center];
  /** Atomic shells in spatial-orbital order. */
  readonly shells: readonly [Shell, Shell, Shell, Shell];
  /** AO overlap S, 4×4 row-major. */
  readonly S_AO: Float64Array;
  /** AO core h = T + Σ_C V_C, 4×4 row-major. */
  readonly h_AO: Float64Array;
  /** AO ERI (μν|λσ), 4⁴ row-major as [μν λσ]. */
  readonly eri_AO: Float64Array;
  /** Löwdin transform X = S^{-1/2}, 4×4. */
  readonly X: Float64Array;
  /** Orthogonalized one-electron h^OAO, 4×4. */
  readonly h_OAO: Float64Array;
  /** Orthogonalized two-electron tensor, 4⁴. */
  readonly eri_OAO: Float64Array;
  /** Nuclear repulsion: Z_Be · Z_H / R_BeH × 2 + Z_H² / (2R_BeH).
   *  = 4·1/R + 4·1/R + 1/(2R) = 8/R + 1/(2R) = 17/(2R). */
  readonly Vnn: number;
}

export interface BeH2Hamiltonian {
  readonly H: Float64Array;          // 256×256 dense, real-symmetric
  readonly nQubits: 8;
  readonly integrals: BeH2Integrals;
  /** Spin-orbital indices occupied in the Hartree-Fock reference
   *  (closed-shell singlet, 6 electrons in 3 lowest spatial orbitals). */
  readonly hfOccupied: readonly number[];
}

const N_ATOMIC = 4;        // 4 spatial orbitals
const N_QUBITS = 8;        // 8 spin-orbitals
const DIM = 1 << N_QUBITS; // 256

/** Compute all integrals for BeH₂ at Be–H bond length R (Å). */
export function computeBeH2Integrals(R_angstrom: number): BeH2Integrals {
  const R = R_angstrom * ANGSTROM_TO_BOHR;
  const Be_pos: Center = [0, 0, 0];
  const HL_pos: Center = [0, 0, -R];
  const HR_pos: Center = [0, 0, +R];

  const beShell1s = makeShell(STO3G_BE_1S, Be_pos, "Be:1s");
  const beShell2s = makeShell(STO3G_BE_2S, Be_pos, "Be:2s");
  const hLShell   = makeShell(STO3G_H_1S, HL_pos, "H_L:1s");
  const hRShell   = makeShell(STO3G_H_1S, HR_pos, "H_R:1s");
  const shells = [beShell1s, beShell2s, hLShell, hRShell] as const;
  const nuclei = [
    { Z: 4, pos: Be_pos },
    { Z: 1, pos: HL_pos },
    { Z: 1, pos: HR_pos },
  ];

  // ── AO overlap and one-electron core ────────────────────
  const S_AO = new Float64Array(N_ATOMIC * N_ATOMIC);
  const h_AO = new Float64Array(N_ATOMIC * N_ATOMIC);
  for (let mu = 0; mu < N_ATOMIC; mu++) {
    for (let nu = mu; nu < N_ATOMIC; nu++) {
      const sij = S_shells(shells[mu]!, shells[nu]!);
      const tij = T_shells(shells[mu]!, shells[nu]!);
      let vSum = 0;
      for (const { Z, pos } of nuclei) vSum += V_shells(shells[mu]!, shells[nu]!, Z, pos);
      S_AO[mu * N_ATOMIC + nu] = sij;
      S_AO[nu * N_ATOMIC + mu] = sij;
      const h = tij + vSum;
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

  // ── Löwdin orthogonalization ───────────────────────────
  const eig = eigsymmetric(S_AO, N_ATOMIC);
  const X = new Float64Array(N_ATOMIC * N_ATOMIC);
  for (let r = 0; r < N_ATOMIC; r++) {
    for (let c = 0; c < N_ATOMIC; c++) {
      let s = 0;
      for (let k = 0; k < N_ATOMIC; k++) {
        const lam = eig.values[k]!;
        if (lam <= 0) {
          throw new Error(`computeBeH2Integrals: AO overlap eigenvalue ${lam} ≤ 0 — basis degenerate`);
        }
        s += eig.vectors[k * N_ATOMIC + r]! * Math.pow(lam, -0.5) * eig.vectors[k * N_ATOMIC + c]!;
      }
      X[r * N_ATOMIC + c] = s;
    }
  }

  // ── Transform h, ERI to orthogonal basis ────────────────
  const h_OAO = transform2(h_AO, X, N_ATOMIC);
  const eri_OAO = transform4(eri_AO, X, N_ATOMIC);

  // ── V_nn: Be↔H_L (4·1/R) + Be↔H_R (4·1/R) + H_L↔H_R (1/(2R))
  const Vnn = (4 / R) + (4 / R) + (1 / (2 * R));

  return {
    R_Bohr: R,
    centers: [Be_pos, HL_pos, HR_pos] as const,
    shells: shells as readonly [Shell, Shell, Shell, Shell],
    S_AO, h_AO, eri_AO, X, h_OAO, eri_OAO, Vnn,
  };
}

/**
 * Build the 256×256 dense BeH₂ Hamiltonian at bond length R (Å).
 *
 * HF reference: closed-shell singlet, 6 electrons in the 3 lowest
 * AO-orthogonalized spatial orbitals (Be 1s, Be 2s, H_left 1s after
 * Löwdin). One spatial orbital (H_right 1s in the AO labelling) is
 * empty in the HF determinant — the optimizer mixes it in via the
 * symmetric MO combination during VQE.
 */
export function buildBeH2Dense(R_angstrom: number): BeH2Hamiltonian {
  const integrals = computeBeH2Integrals(R_angstrom);
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

  // 6-electron closed-shell HF: spatial 0, 1, 2 doubly occupied.
  // Spin-orbital indices: 0=Be1s↑, 1=Be1s↓, 2=Be2s↑, 3=Be2s↓, 4=H_L↑, 5=H_L↓.
  // Spin-orbitals 6, 7 (H_R 1s ↑, ↓) are empty in this AO determinant;
  // the σ_g/σ_u MO combinations mix them in via the optimizer.
  const hfOccupied = [0, 1, 2, 3, 4, 5] as const;

  return { H, nQubits: N_QUBITS, integrals, hfOccupied: [...hfOccupied] };
}

// ── Local helpers ─────────────────────────────────────────────

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
  // O(n⁵) per pass, 4 passes — fine for n = 4 (1024 ops per pass).
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
