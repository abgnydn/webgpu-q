// ───────────────────────────────────────────────────────────────────────
// Ported from PySCF (https://github.com/pyscf/pyscf), Apache 2.0 license.
//
// Source: pyscf/cc/eom_rccsd.py
//   - eeccsd_matvec_singlet (lines ~1190–1287)
//   - eeccsd_matvec_triplet (lines ~1287–1380)
//   - _make_tau helper
// Upstream commit: pyscf master @ 2026-05-13 fetch
//
// Original authors: PySCF developers (see PySCF/AUTHORS).
//
// Adaptations for webgpu-q:
//   - RHF singlet/triplet spatial-orbital matvec (PySCF) → spin-orbital
//     antisym (R₁ + antisym R₂) basis used by runEOMCCSD. Singlet roots
//     come from σ_S; triplet roots from σ_T; eigenvalues are then merged
//     into the M_S=0 spectrum to preserve the current runEOMCCSD API.
//   - NumPy einsum → explicit nested TypeScript loops with typed-array
//     index arithmetic. Some loop blocking from PySCF (memory) is
//     dropped since browser memory ceilings are looser than HPC nodes
//     within the system sizes we target.
//   - Intermediates F_vv / F_oo / F_ov / W_oOoO / W_oVoO / W_vOvV /
//     W_oVVo / W_oVvO / W_oOoV ← built by `makeIMDs` (this file), mirror
//     of PySCF `pyscf/cc/eom_rccsd_init_amps.py` make_imds.
//
// Status: SCAFFOLDED. Translation is in progress; do not import from
// this file in production yet. The hand-derived `runEOMCCSD` in
// `eom-ccsd.ts` is still the production entry point.
//
// Verifier:
//   tests/chemistry/eom-ccsd-bruteforce-lih.test.ts
// After the port lands, M_mine − M_exact in that test's full 14×14
// diff matrix should collapse to ≤ 10⁻⁴ Ha across every entry.
//
// See:
//   LICENSE-PYSCF (root) for the Apache 2.0 notice.
//   MIGRATION.md for the project-wide port policy.
//   LIMITATIONS.md for the bug this port closes (E35 singlet gap).
// ───────────────────────────────────────────────────────────────────────

import type { HFResult } from "./hf-scf.js";
import type { CCSDResult } from "./ccsd.js";
import type { MolecularIntegrals } from "./cg-molecular.js";

/** Spatial-orbital intermediates for RHF EOM-CCSD, matching PySCF's
 *  `imds` attributes. All tensors are stored row-major with TypeScript's
 *  flat-array convention. Indices: o = occupied spatial, v = virtual
 *  spatial. */
export interface RHFEOMIntermediates {
  /** F_vv[a, b] (n_vir × n_vir). */
  readonly Fvv: Float64Array;
  /** F_oo[i, j] (n_occ × n_occ). */
  readonly Foo: Float64Array;
  /** F_ov[i, a] (n_occ × n_vir). */
  readonly Fov: Float64Array;
  /** W_oOoO[m, n, i, j] (n_occ⁴) — spin-adapted MNIJ. */
  readonly woOoO: Float64Array;
  /** W_oVoO[m, b, i, j] (n_occ · n_vir · n_occ²). */
  readonly woVoO: Float64Array;
  /** W_vOvV[a, j, b, e] (n_vir · n_occ · n_vir²). */
  readonly wvOvV: Float64Array;
  /** W_oVVo[m, b, e, j] (n_occ · n_vir² · n_occ) — "exchange" component. */
  readonly woVVo: Float64Array;
  /** W_oVvO[m, b, e, j] (n_occ · n_vir² · n_occ) — "direct" component. */
  readonly woVvO: Float64Array;
  /** W_oOoV[m, n, i, e] (n_occ³ · n_vir). */
  readonly woOoV: Float64Array;
  /** Cached t1 and t2 from CCSD. */
  readonly t1: Float64Array;
  /** Cached t2 from CCSD (spatial RHF). */
  readonly t2: Float64Array;
  /** Spatial ERI ⟨ov|ov⟩ block (n_occ · n_vir · n_occ · n_vir). */
  readonly eris_ovov: Float64Array;
  /** Spatial ERI ⟨ov|vv⟩ block (n_occ · n_vir³). */
  readonly eris_ovvv: Float64Array;
  readonly nocc: number;
  readonly nvir: number;
}

/** _make_tau helper from PySCF eom_rccsd.py.
 *  tau[i,j,a,b] = 0.5·fac·(t1[i,a]·r1[j,b] + t1[j,b]·r1[i,a]) + t2[i,j,a,b]
 *
 *  For singlet matvec PySCF calls this with `fac=2`, so the prefactor
 *  on the (t1·r1) symmetrized term is exactly 1 (fac·.5 = 1).
 */
export function makeTauPySCF(
  r2: Float64Array,
  r1: Float64Array,
  t1: Float64Array,
  t2: Float64Array,
  nocc: number,
  nvir: number,
  fac: number,
): Float64Array {
  void r2; // r2 is unused in PySCF _make_tau (signature mirror only)
  const tau = new Float64Array(nocc * nocc * nvir * nvir);
  const half = 0.5 * fac;
  for (let i = 0; i < nocc; i++) {
    for (let j = 0; j < nocc; j++) {
      for (let a = 0; a < nvir; a++) {
        for (let b = 0; b < nvir; b++) {
          const tia = t1[i * nvir + a]!;
          const t1jb = t1[j * nvir + b]!;
          const r1jb = r1[j * nvir + b]!;
          const r1ia = r1[i * nvir + a]!;
          const t2v = t2[((i * nocc + j) * nvir + a) * nvir + b]!;
          // symmetrize: T[i,j,a,b] + T[j,i,b,a] under sigma(t·r):
          //   t1[i,a]·r1[j,b]  +  t1[j,b]·r1[i,a]
          const symT = tia * r1jb + t1jb * r1ia;
          tau[((i * nocc + j) * nvir + a) * nvir + b] = half * symT + t2v;
        }
      }
    }
  }
  return tau;
}

// TODO(port-1): Build RHFEOMIntermediates from (ccsd, integrals, hf).
//   - Mirror PySCF's pyscf/cc/eom_rccsd_init_amps.py `make_imds` function.
//   - F_vv, F_oo, F_ov come straight from RHF CCSD intermediates.
//   - W tensors need careful translation from PySCF's eom_rccsd_init_amps
//     since they include T1 + T2 dressings.
//   - Verifier: each intermediate tensor should agree with PySCF's output
//     on LiH STO-3G to 10⁻⁹ Ha element-wise. (Add a TS test that runs
//     PySCF via scripts/dump-pyscf-imds.py and diffs JSON-serialized
//     tensors.)
export function makeIMDsPorted(
  ccsd: CCSDResult,
  integrals: MolecularIntegrals,
  hf: HFResult,
): RHFEOMIntermediates {
  // STUB. To be filled in next session.
  void ccsd; void integrals; void hf;
  throw new Error("makeIMDsPorted: not yet implemented — see MIGRATION.md");
}

// TODO(port-2): eeccsd_matvec_singlet — direct translation of PySCF
//   lines 1190–1287.
//
// Pseudocode mapping (PySCF → ours):
//   Hr1  = lib.einsum('ae,ie->ia', imds.Fvv, r1)
//     → for i,a: Hr1[i,a] = sum_e Fvv[a,e] * r1[i,e]
//
//   Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
//     → for i,a: Hr1[i,a] -= sum_m Foo[m,i] * r1[m,a]
//
//   Hr1 += np.einsum('me,imae->ia', imds.Fov, r2) * 2
//     → for i,a: Hr1[i,a] += 2 * sum_m,e Fov[m,e] * r2[i,m,a,e]
//
//   Hr1 -= np.einsum('me,imea->ia', imds.Fov, r2)
//     → for i,a: Hr1[i,a] -= sum_m,e Fov[m,e] * r2[i,m,e,a]
//
//   ... [continue line by line through PySCF singlet matvec]
//
// The full translation is ~150 lines of explicit nested loops.
// Each einsum translates 1:1.
export function eeccsdMatvecSingletPorted(
  r1: Float64Array,
  r2: Float64Array,
  imds: RHFEOMIntermediates,
): { Hr1: Float64Array; Hr2: Float64Array } {
  // STUB. To be filled in next session.
  void r1; void r2; void imds;
  throw new Error("eeccsdMatvecSingletPorted: not yet implemented — see MIGRATION.md");
}

// TODO(port-3): eeccsd_matvec_triplet — same structure as singlet but
//   with different spin coefficients. PySCF lines 1287–1380.
export function eeccsdMatvecTripletPorted(
  r1: Float64Array,
  r2: Float64Array,
  imds: RHFEOMIntermediates,
): { Hr1: Float64Array; Hr2: Float64Array } {
  // STUB.
  void r1; void r2; void imds;
  throw new Error("eeccsdMatvecTripletPorted: not yet implemented — see MIGRATION.md");
}

// TODO(port-4): public wrapper that mirrors runEOMCCSD's API but uses
//   the ported singlet+triplet matvecs internally. After verification
//   against the LiH brute-force diagnostic, this REPLACES runEOMCCSD
//   in `eom-ccsd.ts`. The hand-derived sigma() function is removed.
//
// Until then, this file is dormant — production still uses
// `eom-ccsd.ts`.
