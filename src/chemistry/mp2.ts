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
import { choleskyDecomposeERI, type DFResult } from "./df.js";

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

export interface MP2Opts {
  /**
   * Number of frozen-core spatial orbitals — the lowest `nFrozenCore`
   * occupied MOs are excluded from the correlation sum. Default 0.
   * For first-row chemistry the convention is the 1s of each heavy
   * atom (see `defaultFrozenCore` in atoms.ts).
   */
  readonly nFrozenCore?: number;
  /**
   * Use density fitting for the MP2 (ia|jb) integrals. `true` →
   * Cholesky-DF with τ=1e-10 (machine precision); number → τ;
   * DFResult → reuse a caller-provided B-tensor. The MP2 sum
   * reformulates as Σ_P over an auxiliary index, with memory
   * dropping from O(n⁴) to O(n_occ·n_virt·n_aux). Default false
   * (direct 4-index path).
   */
  readonly useDF?: boolean | number | DFResult;
}

/**
 * Closed-shell MP2 on top of an existing HF result.
 * `hf.C_MO` columns are the HF MOs (sorted by energy ascending);
 * the lowest `hf.nOccupied` are doubly occupied, the rest virtual.
 */
export function runMP2(
  hf: HFResult,
  integrals: MolecularIntegrals,
  opts: MP2Opts = {},
): MP2Result {
  const { C_MO, orbitalEnergies, nOccupied } = hf;
  const { eri_AO, n } = integrals;
  const nFrozen = opts.nFrozenCore ?? 0;
  if (nFrozen < 0 || nFrozen > nOccupied) {
    throw new Error(`runMP2: nFrozenCore=${nFrozen} out of range [0, ${nOccupied}]`);
  }

  // ── DF path: reformulate (ia|jb) via B-tensor contractions. ──
  if (opts.useDF !== undefined && opts.useDF !== false) {
    let df: DFResult;
    if (typeof opts.useDF === "object") {
      df = opts.useDF;
    } else {
      const tau = typeof opts.useDF === "number" ? opts.useDF : 1e-10;
      df = choleskyDecomposeERI(eri_AO, n, tau);
    }
    const E_corr = mp2EnergyDF(C_MO, orbitalEnergies, nOccupied, n, nFrozen, df);
    // Still return the MO-basis ERI for downstream consumers (CCSD).
    // Future refactor: pass the DF B-tensor through to CCSD too.
    const eri_MO_for_downstream = transformERIToMO(eri_AO, C_MO, n);
    return {
      correlationEnergy: E_corr,
      totalEnergy: hf.energy + E_corr,
      eri_MO: eri_MO_for_downstream,
      nOccupied,
    };
  }

  // ── Direct 4-index path. ─────────────────────────────────────
  const eri_MO = transformERIToMO(eri_AO, C_MO, n);

  // E_MP2 = Σ_{ij active-occ} Σ_{ab virt}
  //           (ia|jb) · (2 (ia|jb) − (ib|ja)) / (ε_i + ε_j − ε_a − ε_b)
  let E_corr = 0;
  for (let i = nFrozen; i < nOccupied; i++) {
    for (let j = nFrozen; j < nOccupied; j++) {
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
 * MP2 correlation energy via density fitting.
 *
 * Pre-transforms the AO B-tensor B[μν, P] to the (occ × virt) MO
 * block: B_ov[i, a, P] = Σ_μν C_μi · C_νa · B[μν, P]. Two passes
 * O(n²·n_aux·n_occ + n_occ·n²·n_virt·n_aux) — smaller than the
 * 4-index n⁵ MO transform for typical n_aux ~ 3n. Then the MP2 sum
 * reconstructs (ia|jb) = Σ_P B_ov[i,a,P]·B_ov[j,b,P] on the fly.
 *
 * Memory: O(n_occ·n_virt·n_aux) instead of O(n⁴). For H₂O cc-pVDZ
 * (n=24, nocc=5, nvirt=19, n_aux ~ 100) this is 5·19·100 ≈ 10k
 * elements vs n⁴ = 330k elements — 33× reduction.
 */
function mp2EnergyDF(
  C_MO: Float64Array,
  orbitalEnergies: Float64Array,
  nOccupied: number,
  n: number,
  nFrozen: number,
  df: DFResult,
): number {
  const nVirt = n - nOccupied;
  const nAux = df.nAux;

  // Pass 1: tmp[i, ν, P] = Σ_μ C_μi · B[μν, P]
  const tmp = new Float64Array(nOccupied * n * nAux);
  for (let i = 0; i < nOccupied; i++) {
    for (let nu = 0; nu < n; nu++) {
      for (let P = 0; P < nAux; P++) {
        let s = 0;
        for (let mu = 0; mu < n; mu++) {
          s += C_MO[mu * n + i]! * df.B[(mu * n + nu) * nAux + P]!;
        }
        tmp[(i * n + nu) * nAux + P] = s;
      }
    }
  }
  // Pass 2: B_ov[i, a, P] = Σ_ν C_νa · tmp[i, ν, P]
  const B_ov = new Float64Array(nOccupied * nVirt * nAux);
  for (let i = 0; i < nOccupied; i++) {
    for (let a = 0; a < nVirt; a++) {
      const aMO = nOccupied + a;
      for (let P = 0; P < nAux; P++) {
        let s = 0;
        for (let nu = 0; nu < n; nu++) {
          s += C_MO[nu * n + aMO]! * tmp[(i * n + nu) * nAux + P]!;
        }
        B_ov[(i * nVirt + a) * nAux + P] = s;
      }
    }
  }
  // MP2 energy: E_corr = Σ_ij Σ_ab (ia|jb) · (2(ia|jb) − (ib|ja)) / D
  // with (ia|jb) = Σ_P B_ov[i,a,P]·B_ov[j,b,P].
  let E_corr = 0;
  for (let i = nFrozen; i < nOccupied; i++) {
    for (let j = nFrozen; j < nOccupied; j++) {
      for (let a = 0; a < nVirt; a++) {
        const aMO = nOccupied + a;
        for (let b = 0; b < nVirt; b++) {
          const bMO = nOccupied + b;
          let iajb = 0;
          let ibja = 0;
          const iaOffset = (i * nVirt + a) * nAux;
          const jbOffset = (j * nVirt + b) * nAux;
          const ibOffset = (i * nVirt + b) * nAux;
          const jaOffset = (j * nVirt + a) * nAux;
          for (let P = 0; P < nAux; P++) {
            iajb += B_ov[iaOffset + P]! * B_ov[jbOffset + P]!;
            ibja += B_ov[ibOffset + P]! * B_ov[jaOffset + P]!;
          }
          const denom =
            orbitalEnergies[i]! + orbitalEnergies[j]! -
            orbitalEnergies[aMO]! - orbitalEnergies[bMO]!;
          if (Math.abs(denom) < 1e-12) continue;
          E_corr += iajb * (2 * iajb - ibja) / denom;
        }
      }
    }
  }
  return E_corr;
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
