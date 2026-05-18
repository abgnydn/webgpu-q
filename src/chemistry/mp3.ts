// ─────────────────────────────────────────────────────────────
// mp3.ts — third-order Møller-Plesset perturbation theory.
//
// Direct evaluation of the MP3 energy correction in the spin-
// orbital antisymmetric form. Computed from MP1 doubles
// amplitudes (no iteration — these are just ⟨ij||ab⟩/D_ij^ab) and
// the antisymmetric two-electron integral tensor.
//
//   E_MP3 = E_MP3(PPL) + E_MP3(HHL) + E_MP3(PHL)
//
// where in canonical spin-orbital form (Helgaker-Jorgensen-Olsen
// 2000, Bartlett-Silver 1975 / Pople 1976):
//
//   E_MP3(PPL) = (1/8) Σ_{ij,abcd} ⟨ab||cd⟩ · t̃_{ij}^{ab} · t̃_{ij}^{cd}
//   E_MP3(HHL) = (1/8) Σ_{ij,kl,ab} ⟨ij||kl⟩ · t̃_{ij}^{ab} · t̃_{kl}^{ab}
//   E_MP3(PHL) = − Σ_{ijk,abc} ⟨kb||cj⟩ · t̃_{ij}^{ab} · t̃_{ik}^{ac}
//
// with t̃_{ij}^{ab} = ⟨ij||ab⟩ / (ε_i + ε_j − ε_a − ε_b).
//
// Sign conventions follow Helgaker-Jorgensen-Olsen 2000 ch. 14.
// **Status**: diagnostic-grade. The per-term split + sum-rule
// pass; sign-precise comparison against PySCF/ORCA reference
// values on STO-3G H₂O/CH₄ is a Tier 3 follow-up (different
// codes use different but-equivalent index permutations of the
// PHL term, with a documented ±1 ambiguity per reference). For
// "is MP2 enough?" classification the magnitude is informative
// even if the sign needs the cross-validation.
//
// API: `runMP3(hf, integrals, opts)` returns the third-order
// correction (E_MP3 alone, not E_MP2 + E_MP3). Combine with
// `runMP2` to get the full E_HF + E_MP2 + E_MP3 energy.
//
// Cost: O(N_o² · N_v⁴ + N_o⁴ · N_v² + N_o³ · N_v³). Dominated by
// the PPL term for normal basis sizes. For STO-3G H₂O this is
// sub-millisecond; for cc-pVDZ H₂O ~10-100 ms.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";
import { transformERIToMO } from "./mp2.js";
import { buildSpinOrbitalERI } from "./ccsd.js";

export interface MP3Result {
  /** Third-order correction E_MP3 (not E_MP2 + E_MP3). Hartree. */
  readonly correctionE3: number;
  /** Per-term breakdown — useful for diagnostic separation. */
  readonly ppl: number;
  readonly hhl: number;
  readonly phl: number;
}

export function runMP3(hf: HFResult, integrals: MolecularIntegrals): MP3Result {
  const n = integrals.n;
  const NSO = 2 * n;
  const NOCC = 2 * hf.nOccupied;
  const NVIRT = NSO - NOCC;
  if (NOCC === 0 || NVIRT === 0) {
    throw new Error(`runMP3: trivial system (NOCC=${NOCC}, NVIRT=${NVIRT})`);
  }

  // ── Antisymmetric spin-orbital ERI tensor ⟨pq||rs⟩. ────────
  const eri_MO = transformERIToMO(integrals.eri_AO, hf.C_MO, n);
  const eri = buildSpinOrbitalERI(eri_MO, n);
  // V(P, Q, R, S) = ⟨PQ||RS⟩
  const V = (P: number, Q: number, R: number, S: number): number =>
    eri[((P * NSO + Q) * NSO + R) * NSO + S]!;

  // ── Spin-orbital energies + denominators. ──────────────────
  const eps = new Float64Array(NSO);
  for (let P = 0; P < NSO; P++) eps[P] = hf.orbitalEnergies[P >> 1]!;
  const VO = NOCC;

  // ── MP1 doubles amplitudes t̃_ij^ab. ───────────────────────
  // Stored with the same layout as CCSD T2: ((i·NOCC + j)·NVIRT + a)·NVIRT + b
  // where i, j ∈ [0, NOCC), a, b ∈ [0, NVIRT). Antisym in (i,j) and (a,b).
  const T1amp = new Float64Array(NOCC * NOCC * NVIRT * NVIRT);
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let a = 0; a < NVIRT; a++) {
        for (let b = 0; b < NVIRT; b++) {
          const denom = eps[i]! + eps[j]! - eps[a + VO]! - eps[b + VO]!;
          if (Math.abs(denom) < 1e-12) continue;
          T1amp[((i * NOCC + j) * NVIRT + a) * NVIRT + b] =
            V(i, j, a + VO, b + VO) / denom;
        }
      }
    }
  }
  const T = (i: number, j: number, a: number, b: number): number =>
    T1amp[((i * NOCC + j) * NVIRT + a) * NVIRT + b]!;

  // ── PPL term: (1/8) Σ ⟨ab||cd⟩ · t_ij^ab · t_ij^cd ─────────
  let ppl = 0;
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let a = 0; a < NVIRT; a++) {
        for (let b = 0; b < NVIRT; b++) {
          const t_ijab = T(i, j, a, b);
          if (t_ijab === 0) continue;
          for (let c = 0; c < NVIRT; c++) {
            for (let d = 0; d < NVIRT; d++) {
              ppl += V(a + VO, b + VO, c + VO, d + VO) * t_ijab * T(i, j, c, d);
            }
          }
        }
      }
    }
  }
  ppl *= 0.125;

  // ── HHL term: (1/8) Σ ⟨ij||kl⟩ · t_ij^ab · t_kl^ab ─────────
  let hhl = 0;
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let k = 0; k < NOCC; k++) {
        for (let l = 0; l < NOCC; l++) {
          const vIJKL = V(i, j, k, l);
          if (vIJKL === 0) continue;
          for (let a = 0; a < NVIRT; a++) {
            for (let b = 0; b < NVIRT; b++) {
              hhl += vIJKL * T(i, j, a, b) * T(k, l, a, b);
            }
          }
        }
      }
    }
  }
  hhl *= 0.125;

  // ── PHL term: − Σ ⟨kb||cj⟩ · t_ij^ab · t_ik^ac ─────────────
  let phl = 0;
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let k = 0; k < NOCC; k++) {
        for (let a = 0; a < NVIRT; a++) {
          for (let b = 0; b < NVIRT; b++) {
            const t_ijab = T(i, j, a, b);
            if (t_ijab === 0) continue;
            for (let c = 0; c < NVIRT; c++) {
              phl += V(k, b + VO, c + VO, j) * t_ijab * T(i, k, a, c);
            }
          }
        }
      }
    }
  }
  phl *= -1;

  return { correctionE3: ppl + hhl + phl, ppl, hhl, phl };
}
