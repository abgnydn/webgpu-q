// ─────────────────────────────────────────────────────────────
// mp2.ts — closed-shell Møller-Plesset second-order perturbation
// theory (MP2). Phase E stage 1: the simplest post-HF correlation
// method that still captures most of the dynamic correlation
// missing from a single-determinant HF wavefunction.
//
// Algorithm:
//   1. Take the HF MO coefficients C and orbital energies ε.
//   2. Transform the AO ERI tensor (μν|λσ) to MO basis (pq|rs)
//      via four sequential 2-index contractions — O(n^5) per pass.
//   3. The closed-shell MP2 correlation energy is
//        E_MP2 = Σ_{ij occ} Σ_{ab virt}
//                  (ia|jb) · (2 (ia|jb) − (ib|ja))
//                  / (ε_i + ε_j − ε_a − ε_b)
//      where the orbital indices i, j run over the doubly occupied
//      MOs and a, b run over the virtuals.
//
// This is the simplest meaningful correlation correction and
// scales O(n^5) — vs FCI's O(2^N). It's the workhorse for any
// molecule where FCI is infeasible (cc-pVDZ on water is 24
// spatial orbitals → FCI is hopeless but MP2 finishes in
// milliseconds).
//
// Limitations:
//   • Only captures dynamic correlation, not static (multi-
//     reference) correlation. Won't fix bond breaking, biradicals,
//     transition metals, etc.
//   • Not variational — E_MP2 can in principle go below E_FCI
//     for pathological cases. (For our closed-shell molecules
//     near equilibrium this never happens.)
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";

export interface MP2Result {
  /** MP2 correlation energy (Hartree). E_MP2_corr is negative. */
  readonly correlationEnergy: number;
  /** Total energy = E_HF + E_MP2_corr. */
  readonly totalEnergy: number;
  /** MO-basis ERI tensor (pq|rs), n^4 row-major as ((p·n+q)·n+r)·n+s. */
  readonly eri_MO: Float64Array;
  /** Number of doubly occupied orbitals (= nElectrons/2). */
  readonly nOccupied: number;
}

/**
 * Closed-shell MP2 on top of an existing HF result.
 * `hf.C_MO` columns are the HF MOs (sorted by energy ascending);
 * the lowest `hf.nOccupied` are doubly occupied, the rest virtual.
 */
export function runMP2(hf: HFResult, integrals: MolecularIntegrals): MP2Result {
  const { C_MO, orbitalEnergies, nOccupied } = hf;
  const { eri_AO, n } = integrals;
  const eri_MO = transformERIToMO(eri_AO, C_MO, n);

  // E_MP2 = Σ_{ij occ} Σ_{ab virt}
  //           (ia|jb) · (2 (ia|jb) − (ib|ja)) / (ε_i + ε_j − ε_a − ε_b)
  let E_corr = 0;
  for (let i = 0; i < nOccupied; i++) {
    for (let j = 0; j < nOccupied; j++) {
      for (let a = nOccupied; a < n; a++) {
        for (let b = nOccupied; b < n; b++) {
          const iajb = eri_MO[((i * n + a) * n + j) * n + b]!;
          const ibja = eri_MO[((i * n + b) * n + j) * n + a]!;
          const denom = orbitalEnergies[i]! + orbitalEnergies[j]! - orbitalEnergies[a]! - orbitalEnergies[b]!;
          if (Math.abs(denom) < 1e-12) {
            // Near-degeneracy — skip to avoid 1/0. Real chemistry codes
            // would punt to a multi-reference method here.
            continue;
          }
          E_corr += iajb * (2 * iajb - ibja) / denom;
        }
      }
    }
  }

  return {
    correlationEnergy: E_corr,
    totalEnergy: hf.energy + E_corr,
    eri_MO,
    nOccupied,
  };
}

/**
 * 4-index AO → MO transform via four sequential 2-index passes.
 * (μν|λσ) → (pν|λσ) → (pq|λσ) → (pq|rσ) → (pq|rs).
 * O(n^5) per pass; total O(n^5) — same scaling as MP2 itself.
 *
 * Index convention: (pq|rs) = ∫∫ φ_p(1) φ_q(1) (1/r12) φ_r(2) φ_s(2)
 * stored as `eri_MO[((p·n+q)·n+r)·n+s]`. Matches the AO convention.
 */
export function transformERIToMO(eri_AO: Float64Array, C: Float64Array, n: number): Float64Array {
  // Pass 1: (μν|λσ) → (pν|λσ)
  const t1 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let nu = 0; nu < n; nu++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let mu = 0; mu < n; mu++) {
            s += C[mu * n + p]! * eri_AO[((mu * n + nu) * n + la) * n + si]!;
          }
          t1[((p * n + nu) * n + la) * n + si] = s;
        }
      }
    }
  }
  // Pass 2: (pν|λσ) → (pq|λσ)
  const t2 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let nu = 0; nu < n; nu++) {
            s += C[nu * n + q]! * t1[((p * n + nu) * n + la) * n + si]!;
          }
          t2[((p * n + q) * n + la) * n + si] = s;
        }
      }
    }
  }
  // Pass 3: (pq|λσ) → (pq|rσ)
  const t3 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let la = 0; la < n; la++) {
            s += C[la * n + r]! * t2[((p * n + q) * n + la) * n + si]!;
          }
          t3[((p * n + q) * n + r) * n + si] = s;
        }
      }
    }
  }
  // Pass 4: (pq|rσ) → (pq|rs)
  const out = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let s = 0; s < n; s++) {
          let acc = 0;
          for (let si = 0; si < n; si++) {
            acc += C[si * n + s]! * t3[((p * n + q) * n + r) * n + si]!;
          }
          out[((p * n + q) * n + r) * n + s] = acc;
        }
      }
    }
  }
  return out;
}
