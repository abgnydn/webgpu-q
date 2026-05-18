// ─────────────────────────────────────────────────────────────
// uccsd-t.ts — open-shell perturbative triples on a UHF + UCCSD
// reference. Builds the spin-orbital antisymmetric ERI tensor in
// UCCSD's "all-α-occ first, then all-β-occ" ordering, constructs
// spin-resolved orbital energies, and calls the existing
// `ccsdtFromSO` core (same (T) algorithm as the closed-shell path).
//
// For closed-shell UHF input (n_α = n_β, α/β orbitals identical),
// the result matches `runCCSDT` on the corresponding RHF reference
// to numerical precision — the basis-independent (T) eigenvalue.
//
// Frozen-core: wired through the non-contiguous `frozenOccSO`
// parameter on `ccsdtFromSO` (refactored 2026-05). Freezing the
// lowest `nFrozenCore` spatial cores in UCCSD's all-α-first SO
// ordering means skipping α-SOs [0, nFrozenCore) AND β-SOs
// [nAlpha, nAlpha + nFrozenCore). Same nFrozenCore should typically
// be passed to both `runUCCSD` and `runUCCSDT`; T1/T2 are zero at
// the matching SO positions either way.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { UHFResult } from "./uhf-scf.js";
import type { UCCSDResult } from "./uccsd.js";
import { transformERIBlock } from "./uccsd.js";
import { ccsdtFromSO, type CCSDTResult, type CCSDTOpts } from "./ccsd-t.js";

export function runUCCSDT(
  uccsd: UCCSDResult,
  uhf: UHFResult,
  integrals: MolecularIntegrals,
  opts: CCSDTOpts = {},
): CCSDTResult {
  const tStart = performance.now();
  const n = integrals.n;
  const NSO = 2 * n;
  const NOCC = uhf.nAlpha + uhf.nBeta;
  const NVIRT = NSO - NOCC;
  const nFrozenCore = opts.nFrozenCore ?? 0;
  // After freezing both spins, active occupied SOs = NOCC − 2·nFrozenCore.
  if (NOCC - 2 * nFrozenCore < 3 || NVIRT < 3) {
    return { tripleCorrection: 0, totalEnergy: uccsd.totalEnergy, seconds: 0 };
  }

  // ── Spin/spatial mapping (UCCSD's all-α-first SO ordering). ──
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

  // ── Spatial-block ERIs. ────────────────────────────────────
  const eri_aa_aa = transformERIBlock(integrals.eri_AO, uhf.C_alpha, uhf.C_alpha, n);
  const eri_aa_bb = transformERIBlock(integrals.eri_AO, uhf.C_alpha, uhf.C_beta,  n);
  const eri_bb_bb = transformERIBlock(integrals.eri_AO, uhf.C_beta,  uhf.C_beta,  n);
  const eri_bb_aa = (p: number, r: number, q: number, s: number): number =>
    eri_aa_bb[((q * n + s) * n + p) * n + r]!;

  // ── Full antisymmetric SO ERI tensor ⟨PQ||RS⟩. ─────────────
  const eri = new Float64Array(NSO * NSO * NSO * NSO);
  for (let P = 0; P < NSO; P++) {
    const p = spatialOf(P), sp = spinOf(P);
    for (let Q = 0; Q < NSO; Q++) {
      const q = spatialOf(Q), sq = spinOf(Q);
      for (let R = 0; R < NSO; R++) {
        const r = spatialOf(R), sr = spinOf(R);
        for (let S = 0; S < NSO; S++) {
          const s = spatialOf(S), ss = spinOf(S);
          let direct = 0;
          if (sp === sr && sq === ss) {
            if (sp === 0 && sq === 0) direct = eri_aa_aa[((p * n + r) * n + q) * n + s]!;
            else if (sp === 0 && sq === 1) direct = eri_aa_bb[((p * n + r) * n + q) * n + s]!;
            else if (sp === 1 && sq === 0) direct = eri_bb_aa(p, r, q, s);
            else direct = eri_bb_bb[((p * n + r) * n + q) * n + s]!;
          }
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

  // ── Per-SO orbital energies. ───────────────────────────────
  const eps = new Float64Array(NSO);
  for (let P = 0; P < NSO; P++) {
    const p = spatialOf(P);
    eps[P] = spinOf(P) === 0
      ? uhf.orbitalEnergiesAlpha[p]!
      : uhf.orbitalEnergiesBeta[p]!;
  }

  // ── Build the non-contiguous frozen occupied-SO set for UCCSD's
  // "all-α-occ first, then all-β-occ" ordering. Freezing the lowest
  // `nFrozenCore` spatials → freeze α-SOs [0, nFrozenCore) AND
  // β-SOs [nAlpha, nAlpha + nFrozenCore).
  const frozenOccSO = new Set<number>();
  for (let s = 0; s < nFrozenCore; s++) {
    frozenOccSO.add(s);
    frozenOccSO.add(uhf.nAlpha + s);
  }
  const E_T = ccsdtFromSO(eri, eps, uccsd.T1, uccsd.T2, NOCC, NVIRT, frozenOccSO);

  return {
    tripleCorrection: E_T,
    totalEnergy: uccsd.totalEnergy + E_T,
    seconds: (performance.now() - tStart) / 1000,
  };
}
