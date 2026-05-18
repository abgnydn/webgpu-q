// ─────────────────────────────────────────────────────────────
// ump2.ts — open-shell MP2 on a UHF reference.
//
// In the spin-orbital antisymmetric form:
//
//   E_UMP2 = ¼ Σ_{ij,ab} |⟨ij||ab⟩|² / D_ij^ab
//          = ¼ Σ_{ij,ab} ⟨ij||ab⟩² / (ε_i + ε_j − ε_a − ε_b)
//
// with sums over all occupied / virtual spin-orbital indices.
// The factor ¼ comes from the antisymmetric (i, j) and (a, b)
// pair restrictions being absorbed into a full unrestricted sum.
//
// For closed-shell UHF (n_α = n_β, α and β orbitals identical),
// this reduces exactly to the RHF MP2 expression. For genuine
// open-shell (n_α ≠ n_β), gives the standard "unrestricted MP2"
// correlation energy — useful for radicals at a much lower cost
// than UCCSD.
//
// Reuses UCCSD's 3-block ERI transform + antisymmetric SO ERI
// construction (exported as `transformERIBlock` from uccsd.ts).
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { UHFResult } from "./uhf-scf.js";
import { transformERIBlock } from "./uccsd.js";

export interface UMP2Result {
  /** UMP2 correlation energy (Hartree). Negative for closed-shell
   *  and most open-shell ground states at equilibrium. */
  readonly correlationEnergy: number;
  /** Total UMP2 energy = uhf.energy + correlationEnergy. */
  readonly totalEnergy: number;
}

export function runUMP2(
  uhf: UHFResult,
  integrals: MolecularIntegrals,
): UMP2Result {
  const n = integrals.n;
  const NSO = 2 * n;
  const NOCC = uhf.nAlpha + uhf.nBeta;
  const NVIRT = NSO - NOCC;
  if (NOCC === 0 || NVIRT === 0) {
    return { correlationEnergy: 0, totalEnergy: uhf.energy };
  }

  const { nAlpha, nBeta } = uhf;
  const nVirtA = n - nAlpha;
  const occBoundary = NOCC;
  const virtAEnd = occBoundary + nVirtA;
  const spinOf = (P: number): 0 | 1 => {
    if (P < nAlpha) return 0;
    if (P < occBoundary) return 1;
    if (P < virtAEnd) return 0;
    return 1;
  };
  const spatialOf = (P: number): number => {
    if (P < nAlpha) return P;
    if (P < occBoundary) return P - nAlpha;
    if (P < virtAEnd) return P - nBeta;
    return P - n;
  };

  // Build the 3 spatial ERI blocks used by UCCSD (αα|αα), (αα|ββ), (ββ|ββ).
  const eri_aa_aa = transformERIBlock(integrals.eri_AO, uhf.C_alpha, uhf.C_alpha, n);
  const eri_aa_bb = transformERIBlock(integrals.eri_AO, uhf.C_alpha, uhf.C_beta,  n);
  const eri_bb_bb = transformERIBlock(integrals.eri_AO, uhf.C_beta,  uhf.C_beta,  n);
  const eri_bb_aa = (p: number, r: number, q: number, s: number): number =>
    eri_aa_bb[((q * n + s) * n + p) * n + r]!;

  // Spin-orbital antisymmetric ERI accessor ⟨PQ||RS⟩ = ⟨PQ|RS⟩ − ⟨PQ|SR⟩.
  // Inlined (no full NSO⁴ tensor — MP2 only needs ⟨ij||ab⟩ for occ/virt blocks).
  const eriDirect = (P: number, Q: number, R: number, S: number): number => {
    const sp = spinOf(P), sq = spinOf(Q), sr = spinOf(R), ss = spinOf(S);
    if (sp !== sr || sq !== ss) return 0;
    const p = spatialOf(P), q = spatialOf(Q), r = spatialOf(R), s = spatialOf(S);
    if (sp === 0 && sq === 0) return eri_aa_aa[((p * n + r) * n + q) * n + s]!;
    if (sp === 0 && sq === 1) return eri_aa_bb[((p * n + r) * n + q) * n + s]!;
    if (sp === 1 && sq === 0) return eri_bb_aa(p, r, q, s);
    return eri_bb_bb[((p * n + r) * n + q) * n + s]!;
  };
  const eriAntisym = (P: number, Q: number, R: number, S: number): number =>
    eriDirect(P, Q, R, S) - eriDirect(P, Q, S, R);

  // Spin-orbital orbital energies.
  const eps = new Float64Array(NSO);
  for (let P = 0; P < NSO; P++) {
    const p = spatialOf(P);
    eps[P] = spinOf(P) === 0
      ? uhf.orbitalEnergiesAlpha[p]!
      : uhf.orbitalEnergiesBeta[p]!;
  }

  // E_UMP2 = ¼ Σ_ij,ab |⟨ij||ab⟩|² / (ε_i + ε_j − ε_a − ε_b).
  let E_corr = 0;
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let a = 0; a < NVIRT; a++) {
        const A = a + NOCC;
        for (let b = 0; b < NVIRT; b++) {
          const B = b + NOCC;
          const v = eriAntisym(i, j, A, B);
          if (v === 0) continue;
          const denom = eps[i]! + eps[j]! - eps[A]! - eps[B]!;
          if (Math.abs(denom) < 1e-12) continue;
          E_corr += 0.25 * v * v / denom;
        }
      }
    }
  }
  return { correlationEnergy: E_corr, totalEnergy: uhf.energy + E_corr };
}
