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
import type { DFResult } from "./df.js";

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

/**
 * DF-UMP2 correlation energy from a precomputed DF B-tensor — open-shell MP2
 * without the 4-index ERI, so it works for large radicals where transformERIBlock
 * can't allocate. Spin-resolved:
 *   E = E_αα + E_ββ + E_αβ
 *   E_σσ = ¼ Σ_{ij∈σ, ab∈σ} [(ia|jb)−(ib|ja)]² / (ε_i+ε_j−ε_a−ε_b)
 *   E_αβ = Σ_{i∈α,j∈β, a∈α,b∈β} (ia|jb)² / (ε_i^α+ε_j^β−ε_a^α−ε_b^β)
 * with (ia|jb)_σσ' = Σ_P B_ov^σ[i,a,P]·B_ov^σ'[j,b,P], B_ov^σ[i,a,P] = Σ_μν
 * C^σ_μi C^σ_νa B[μν,P]. Reduces exactly to RHF-MP2 when α≡β (verified
 * algebraically). nFrozen freezes the lowest occ in each spin.
 */
export function mp2EnergyUDF(
  C_alpha: Float64Array, C_beta: Float64Array,
  epsA: Float64Array, epsB: Float64Array,
  nAlpha: number, nBeta: number, n: number, nFrozen: number, df: DFResult,
): number {
  const nAux = df.nAux;
  // B_ov^σ[i,a,P] = Σ_μν C_μi C_νa B[μν,P]  (two 2-index passes)
  const buildBov = (C: Float64Array, nOcc: number): Float64Array => {
    const nVirt = n - nOcc;
    const tmp = new Float64Array(nOcc * n * nAux); // tmp[i,ν,P] = Σ_μ C_μi B[μν,P]
    for (let i = 0; i < nOcc; i++)
      for (let nu = 0; nu < n; nu++)
        for (let P = 0; P < nAux; P++) {
          let s = 0;
          for (let mu = 0; mu < n; mu++) s += C[mu * n + i]! * df.B[(mu * n + nu) * nAux + P]!;
          tmp[(i * n + nu) * nAux + P] = s;
        }
    const Bov = new Float64Array(nOcc * nVirt * nAux);
    for (let i = 0; i < nOcc; i++)
      for (let a = 0; a < nVirt; a++) {
        const aMO = nOcc + a;
        for (let P = 0; P < nAux; P++) {
          let s = 0;
          for (let nu = 0; nu < n; nu++) s += C[nu * n + aMO]! * tmp[(i * n + nu) * nAux + P]!;
          Bov[(i * nVirt + a) * nAux + P] = s;
        }
      }
    return Bov;
  };

  const nVirtA = n - nAlpha, nVirtB = n - nBeta;
  const BovA = buildBov(C_alpha, nAlpha);
  const BovB = buildBov(C_beta, nBeta);
  const ddot = (B1: Float64Array, o1: number, B2: Float64Array, o2: number): number => {
    let s = 0; for (let P = 0; P < nAux; P++) s += B1[o1 + P]! * B2[o2 + P]!; return s;
  };

  // Same-spin: ¼ Σ_ijab [(ia|jb)−(ib|ja)]² / D
  const sameSpin = (Bov: Float64Array, nOcc: number, nVirt: number, eps: Float64Array): number => {
    let e = 0;
    for (let i = nFrozen; i < nOcc; i++)
      for (let j = nFrozen; j < nOcc; j++)
        for (let a = 0; a < nVirt; a++)
          for (let b = 0; b < nVirt; b++) {
            const iajb = ddot(Bov, (i * nVirt + a) * nAux, Bov, (j * nVirt + b) * nAux);
            const ibja = ddot(Bov, (i * nVirt + b) * nAux, Bov, (j * nVirt + a) * nAux);
            const D = eps[i]! + eps[j]! - eps[nOcc + a]! - eps[nOcc + b]!;
            if (Math.abs(D) < 1e-12) continue;
            const num = iajb - ibja;
            e += 0.25 * num * num / D;
          }
    return e;
  };

  let E = sameSpin(BovA, nAlpha, nVirtA, epsA) + sameSpin(BovB, nBeta, nVirtB, epsB);

  // Opposite-spin: Σ (ia|jb)² / D, counted once (i∈α,j∈β,a∈α,b∈β).
  for (let i = nFrozen; i < nAlpha; i++)
    for (let j = nFrozen; j < nBeta; j++)
      for (let a = 0; a < nVirtA; a++)
        for (let b = 0; b < nVirtB; b++) {
          const iajb = ddot(BovA, (i * nVirtA + a) * nAux, BovB, (j * nVirtB + b) * nAux);
          const D = epsA[i]! + epsB[j]! - epsA[nAlpha + a]! - epsB[nBeta + b]!;
          if (Math.abs(D) < 1e-12) continue;
          E += iajb * iajb / D;
        }
  return E;
}
