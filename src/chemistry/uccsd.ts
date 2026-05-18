// ─────────────────────────────────────────────────────────────
// uccsd.ts — Unrestricted CCSD on top of a UHF reference.
// Tier 2 stage 25: opens correlated treatment of open-shell
// systems (radicals, cations, anions, doublets, triplets).
//
// Architecture: UHF provides spin-resolved α and β MO coefficients
// + orbital energies. We:
//   1. Transform the AO ERI to three spatial-MO blocks:
//        (αα|αα), (αα|ββ), (ββ|ββ).
//        — (ββ|αα) follows from (αα|ββ) by swap symmetry.
//   2. Build the spin-orbital antisymmetric ERI tensor ⟨PQ||RS⟩
//      using these blocks + the spin selection rule
//        ⟨PQ|RS⟩ ≠ 0  ⇔  σ_P = σ_R  AND  σ_Q = σ_S.
//   3. Build spin-resolved orbital energies ε_P
//      (α-eps for α SOs, β-eps for β SOs).
//   4. Call the shared CCSD iteration core (Stanton-Bartlett
//      residual equations — identical for RHF and UHF since
//      canonical UHF orbitals make f_PQ block-diagonal in spin
//      AND diagonal within each spin block).
//
// Spin-orbital index convention (FOR UHF — DIFFERENT from runCCSD):
//   SO P encodes (spatial p_σ, σ) in a flat ordering where
//   occupied SOs come first, then virtuals:
//     P ∈ [0, nα)           → α-occupied, spatial = P
//     P ∈ [nα, nα + nβ)     → β-occupied, spatial = P − nα
//     P ∈ [nα + nβ, n + nβ) → α-virtual,  spatial = P − nβ
//     P ∈ [n + nβ, 2n)      → β-virtual,  spatial = P − n
//   So NOCC = nα + nβ and NVIRT = 2n − NOCC. For closed-shell
//   inputs (nα = nβ), this matches an RHF-style "all-α first,
//   all-β second" ordering, NOT the runCCSD interleaved
//   "P = 2p + σ" convention. The two conventions give the same
//   CCSD energy by spin-orbital basis independence.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { UHFResult } from "./uhf-scf.js";
import { type CCSDOpts, type CCSDResult, ccsdIterate } from "./ccsd.js";

export interface UCCSDResult extends CCSDResult {
  readonly nAlpha: number;
  readonly nBeta: number;
}

/**
 * Run unrestricted CCSD on top of a UHF reference. For closed-shell
 * inputs (nα = nβ) the result agrees with the closed-shell `runCCSD`
 * to numerical precision (different SO ordering, same eigenvalue).
 *
 * For open-shell radicals (nα = nβ + 1, doublets), CCSD correlates
 * the open-shell HF determinant. Spin contamination from UHF
 * propagates into the CCSD wavefunction; for clean doublets (e.g.,
 * Li atom, H₂⁺) it should remain small.
 */
export function runUCCSD(
  uhf: UHFResult,
  integrals: MolecularIntegrals,
  opts: CCSDOpts = {},
): UCCSDResult {
  const n = integrals.n;
  const NSO = 2 * n;
  const NOCC = uhf.nAlpha + uhf.nBeta;
  const NVIRT = NSO - NOCC;
  if (NOCC === 0 || NVIRT === 0) {
    throw new Error(`runUCCSD: trivial system (NOCC=${NOCC}, NVIRT=${NVIRT})`);
  }
  const nFrozenCore = opts.nFrozenCore ?? 0;
  const nFrozenSO = 2 * nFrozenCore;
  if (nFrozenSO < 0 || nFrozenSO >= NOCC) {
    throw new Error(
      `runUCCSD: nFrozenCore=${opts.nFrozenCore} leaves no active occupied orbitals ` +
      `(NOCC=${NOCC} spin-orbitals)`,
    );
  }

  // ── Spin assignment + spatial-index mapping for SO P. ─────
  // Occupied: α at SOs [0..nα), β at SOs [nα..NOCC).
  // Virtual:  α at SOs [NOCC..NOCC+(n−nα)), β at the tail.
  const { nAlpha, nBeta } = uhf;
  const nVirtA = n - nAlpha;
  const occBoundary = NOCC; // = nα + nβ
  const virtAEnd = occBoundary + nVirtA; // = n + nβ
  const spinOf = (P: number): 0 | 1 => {
    if (P < nAlpha) return 0;
    if (P < occBoundary) return 1;
    if (P < virtAEnd) return 0;
    return 1;
  };
  const spatialOf = (P: number): number => {
    if (P < nAlpha) return P;                       // α-occ
    if (P < occBoundary) return P - nAlpha;         // β-occ
    if (P < virtAEnd) return P - nBeta;             // α-virt (offset by nAlpha; mapping P−nβ).
    return P - n;                                   // β-virt
  };

  // ── Transform AO ERI to three spatial MO blocks. ─────────
  const eri_aa_aa = transformERIBlock(integrals.eri_AO, uhf.C_alpha, uhf.C_alpha, n);
  const eri_aa_bb = transformERIBlock(integrals.eri_AO, uhf.C_alpha, uhf.C_beta,  n);
  const eri_bb_bb = transformERIBlock(integrals.eri_AO, uhf.C_beta,  uhf.C_beta,  n);

  // (ββ|αα) is the position-swap of (αα|ββ):
  //   (β_p β_r | α_q α_s)_chem = (α_q α_s | β_p β_r)_chem
  // → eri_bb_aa[p,r,q,s] = eri_aa_bb[q,s,p,r]
  const eri_bb_aa = (p: number, r: number, q: number, s: number): number =>
    eri_aa_bb[((q * n + s) * n + p) * n + r]!;

  // ── Build spin-orbital antisymmetric ERI tensor. ─────────
  // ⟨PQ||RS⟩ = ⟨PQ|RS⟩ − ⟨PQ|SR⟩, each piece subject to
  // σ_P=σ_R and σ_Q=σ_S spin selection.
  const eri = new Float64Array(NSO * NSO * NSO * NSO);
  for (let P = 0; P < NSO; P++) {
    const p = spatialOf(P), sp = spinOf(P);
    for (let Q = 0; Q < NSO; Q++) {
      const q = spatialOf(Q), sq = spinOf(Q);
      for (let R = 0; R < NSO; R++) {
        const r = spatialOf(R), sr = spinOf(R);
        for (let S = 0; S < NSO; S++) {
          const s = spatialOf(S), ss = spinOf(S);
          // ⟨PQ|RS⟩ requires σ_P=σ_R and σ_Q=σ_S; spatial uses
          // (pr|qs) with α-MO or β-MO blocks per spin.
          let direct = 0;
          if (sp === sr && sq === ss) {
            if (sp === 0 && sq === 0) direct = eri_aa_aa[((p * n + r) * n + q) * n + s]!;
            else if (sp === 0 && sq === 1) direct = eri_aa_bb[((p * n + r) * n + q) * n + s]!;
            else if (sp === 1 && sq === 0) direct = eri_bb_aa(p, r, q, s);
            else direct = eri_bb_bb[((p * n + r) * n + q) * n + s]!;
          }
          // ⟨PQ|SR⟩ requires σ_P=σ_S and σ_Q=σ_R; spatial uses
          // (ps|qr) with the same per-spin block selection.
          let exch = 0;
          if (sp === ss && sq === sr) {
            if (sp === 0 && sq === 0) exch = eri_aa_aa[((p * n + s) * n + q) * n + r]!;
            else if (sp === 0 && sq === 1) exch = eri_aa_bb[((p * n + s) * n + q) * n + r]!;
            else if (sp === 1 && sq === 0) exch = eri_bb_aa(p, s, q, r);
            else exch = eri_bb_bb[((p * n + s) * n + q) * n + r]!;
          }
          eri[((P * NSO + Q) * NSO + R) * NSO + S] = direct - exch;
        }
      }
    }
  }

  // ── Spin-orbital orbital energies. ───────────────────────
  const eps = new Float64Array(NSO);
  for (let P = 0; P < NSO; P++) {
    const p = spatialOf(P);
    eps[P] = spinOf(P) === 0
      ? uhf.orbitalEnergiesAlpha[p]!
      : uhf.orbitalEnergiesBeta[p]!;
  }

  // ── Run the shared CCSD iteration core. ──────────────────
  // ── Build the frozen occupied-SO set for UCCSD's "all-α-occ
  //    first, then all-β-occ" ordering. Freezing the lowest
  //    `nFrozenCore` spatials means freezing α-SOs [0, nFrozenCore)
  //    AND β-SOs [nAlpha, nAlpha + nFrozenCore). This is NOT a
  //    contiguous SO range (audit 2026-05; see
  //    `tests/chemistry/frozen-core-audit.test.ts`).
  const frozenOccSet = new Set<number>();
  for (let s = 0; s < nFrozenCore; s++) {
    frozenOccSet.add(s);              // α-spatial-s
    frozenOccSet.add(nAlpha + s);     // β-spatial-s
  }

  const core = ccsdIterate(eri, eps, NOCC, NVIRT, NSO, frozenOccSet, opts);
  return {
    correlationEnergy: core.correlationEnergy,
    totalEnergy: uhf.energy + core.correlationEnergy,
    T1: core.T1,
    T2: core.T2,
    history: core.history,
    iter: core.iter,
    converged: core.converged,
    nAlpha,
    nBeta,
  };
}

/**
 * 4-step O(n^5) AO→MO transform of the chemist-notation ERI
 * (μν|λσ) with potentially different C matrices on positions 1
 * (μ, ν) and 2 (λ, σ). Output is row-major (p_1 q_1 | r_2 s_2)
 * indexed as ((p·n + q)·n + r)·n + s.
 *
 * For closed-shell-like blocks (C1 = C2), this reduces to the
 * standard same-basis transform.
 */
function transformERIBlock(
  eri_AO: Float64Array,
  C1: Float64Array,
  C2: Float64Array,
  n: number,
): Float64Array {
  // Step 1: μ → p (using C1).  t1[p, ν, λ, σ] = Σ_μ C1[μ,p] · eri_AO[μ,ν,λ,σ]
  const t1 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let nu = 0; nu < n; nu++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let mu = 0; mu < n; mu++)
            s += C1[mu * n + p]! * eri_AO[((mu * n + nu) * n + la) * n + si]!;
          t1[((p * n + nu) * n + la) * n + si] = s;
        }
      }
    }
  }
  // Step 2: ν → q (C1).
  const t2 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let nu = 0; nu < n; nu++)
            s += C1[nu * n + q]! * t1[((p * n + nu) * n + la) * n + si]!;
          t2[((p * n + q) * n + la) * n + si] = s;
        }
      }
    }
  }
  // Step 3: λ → r (C2).
  const t3 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let la = 0; la < n; la++)
            s += C2[la * n + r]! * t2[((p * n + q) * n + la) * n + si]!;
          t3[((p * n + q) * n + r) * n + si] = s;
        }
      }
    }
  }
  // Step 4: σ → s (C2).
  const out = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let ss = 0; ss < n; ss++) {
          let s = 0;
          for (let si = 0; si < n; si++)
            s += C2[si * n + ss]! * t3[((p * n + q) * n + r) * n + si]!;
          out[((p * n + q) * n + r) * n + ss] = s;
        }
      }
    }
  }
  return out;
}
