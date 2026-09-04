import type { CGShell } from "./integrals-cg.js";
import type { DFResult } from "./df.js";
import { eigsymmetric } from "../manybody/dense-eig.js";
import { loadWasm, packShells } from "./df-aux.js";
import { DF_CHOLESKY_TOL } from "./numerical-tolerances.js";

/** Pivoted incomplete Cholesky of a PSD n × n matrix M, returning
 *  L such that M ≈ L · L^T with rank ≤ n. L has shape n × r where
 *  r is the discovered rank. Pivot indices are returned to enable
 *  back-substitution against L^T.
 *
 *  Drops dimensions where the remaining diagonal residual falls
 *  below `threshold`. For a well-conditioned PSD matrix this is
 *  full rank; for the rank-deficient case (e.g., auto-aux M with
 *  cross-atom redundancy at benzene scale) it naturally truncates
 *  to the effective rank without producing spurious modes the way
 *  eigendecomp + regularization does.
 *
 *  Cost: O(n²·r). Memory: O(n·r). */
export function pivotedCholesky(
  M: Float64Array,
  n: number,
  threshold: number,
): { L: Float64Array; pivots: number[] } {
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = M[i * n + i]!;
  const L_cols: Float64Array[] = [];
  const pivots: number[] = [];

  for (let step = 0; step < n; step++) {
    let piv = -1;
    let pmax = 0;
    for (let i = 0; i < n; i++) {
      if (d[i]! > pmax) { pmax = d[i]!; piv = i; }
    }
    if (piv < 0 || pmax < threshold) break;
    const sqrtPiv = Math.sqrt(pmax);
    const newCol = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let v = M[i * n + piv]!;
      for (let j = 0; j < L_cols.length; j++) {
        v -= L_cols[j]![i]! * L_cols[j]![piv]!;
      }
      newCol[i] = v / sqrtPiv;
    }
    L_cols.push(newCol);
    pivots.push(piv);
    for (let i = 0; i < n; i++) {
      const v = newCol[i]!;
      d[i] = Math.max(0, d[i]! - v * v);
    }
    d[piv] = 0;
  }

  // Pack columns into row-major L[i, k] = L_cols[k][i].
  const r = L_cols.length;
  const L = new Float64Array(n * r);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < r; k++) {
      L[i * r + k] = L_cols[k]![i]!;
    }
  }
  return { L, pivots };
}

/** TS form_b_from_cholesky — kept as the active path.
 *
 *  An honest negative documented 2026-05-28: porting this to Rust+WASM
 *  was SLOWER than TS (91 s vs 40 s on naphthalene cc-pVDZ at n=190,
 *  n_aux=1080). Two reasons:
 *   1. wasm-bindgen `&[f64]` copies the V tensor (~312 MB on
 *      naphthalene) into WASM linear memory every call.
 *   2. The piv_k-indirect access into V and L is cache-unfriendly;
 *      WASM lost the same way TS did, plus paid the copy cost.
 *
 *  Output B is contiguous (μν, k), so V8 JIT can prefetch the writes
 *  better than expected. Inner k loop has serial dependency on
 *  B[μν, j<k], so SIMD doesn't help the back-substitution.
 *
 *  Cost: O(n_orb² · r²). Memory: O(n_orb² · r). */
export function formBFromCholesky(
  V: Float64Array,
  L: Float64Array,
  pivots: number[],
  nOrb: number,
  nAux: number,
): { B: Float64Array; r: number } {
  const r = pivots.length;
  const B = new Float64Array(nOrb * nOrb * r);
  for (let mu = 0; mu < nOrb; mu++) {
    for (let nu = 0; nu < nOrb; nu++) {
      const vBase = (mu * nOrb + nu) * nAux;
      const bBase = (mu * nOrb + nu) * r;
      for (let k = 0; k < r; k++) {
        const pivK = pivots[k]!;
        let s = V[vBase + pivK]!;
        for (let j = 0; j < k; j++) {
          s -= L[pivK * r + j]! * B[bBase + j]!;
        }
        const diag = L[pivK * r + k]!;
        B[bBase + k] = s / diag;
      }
    }
  }
  return { B, r };
}

/**
 * Build the density-fitting B-tensor from explicit 3-index and
 * 2-index integrals. Aux basis defaults to the orbital basis (Phase 1
 * PoC); for production a proper cc-pVDZ-jkfit basis would be supplied.
 *
 * Algorithm:
 *   V[μν, P] = (μν|P)                    via eri_3idx_build
 *   M[P, Q]  = (P|Q)                     via eri_2idx_build
 *   M = U · Λ · U^T                      eigendecomp (Jacobi)
 *   M^(-1/2) = U · Λ^(-1/2) · U^T        ignoring λ_i < ε (regularize)
 *   B[μν, P] = Σ_Q V[μν, Q] · M^(-1/2)[Q, P]
 *
 * Returns the standard DFResult shape used by `buildJK_DF`.
 */
export async function buildAuxBasisDF(
  orbitalShells: readonly CGShell[],
  auxShells?: readonly CGShell[],
  metricRegularization = DF_CHOLESKY_TOL,
): Promise<DFResult> {
  const mod = await loadWasm();
  const orb = packShells(orbitalShells);
  const aux = auxShells ? packShells(auxShells) : orb;
  const n = orbitalShells.length;
  const nAux = auxShells ? auxShells.length : n;

  // 3-index V[μν, P] and 2-index M[P, Q].
  const V = mod.eri_3idx_build(
    n, nAux,
    orb.nPrims, orb.primOff, orb.alpha, orb.c, orb.center, orb.angular,
    aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );
  const M = mod.eri_2idx_build(
    nAux,
    aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );

  // Eigendecomp of M. (Jacobi returns ascending eigenvalues + column-major V_eig.)
  const eig = eigsymmetric(M, nAux);
  // M^(-1/2) = U · diag(1/√λ) · U^T  (where U columns are eigenvectors).
  // Regularize: zero out eigenvalues below `metricRegularization`.
  const invSqrtLam = new Float64Array(nAux);
  let nKept = 0;
  for (let i = 0; i < nAux; i++) {
    const lam = eig.values[i]!;
    if (lam > metricRegularization) {
      invSqrtLam[i] = 1.0 / Math.sqrt(lam);
      nKept++;
    } else {
      invSqrtLam[i] = 0.0;
    }
  }

  // Form M^(-1/2) (we don't need it explicitly — fold into B directly).
  // B[μν, P] = Σ_Q V[μν, Q] · (U · Λ^(-1/2) · U^T)[Q, P]
  //          = Σ_Q Σ_i V[μν, Q] · U[Q, i] · λ_i^(-1/2) · U[P, i]
  // Pre-compute T[μν, i] = Σ_Q V[μν, Q] · U[Q, i] · λ_i^(-1/2)
  // Then B[μν, P] = Σ_i T[μν, i] · U[P, i]
  //
  // Layout: eig.vectors is column-major, so U[Q, i] = vectors[i*nAux + Q].
  // The full B = V · U · diag(λ⁻¹⸍²) · Uᵀ matmul runs in Rust+WASM
  // with f64x2 SIMD on the inner P loop — ~5× over the TS version.
  const B = mod.form_b_tensor(n, nAux, V, eig.vectors, invSqrtLam);

  // nAux (NOT nKept) is the correct report: form_b_tensor emits B at the full
  // nAux stride and regularizes by zeroing λ⁻¹⸍², so dropped modes are present
  // as exact-zero columns. Reporting nKept here told buildJK_DF to stride by a
  // narrower width than B actually has, misaligning every read: on H₂O/STO-3G a
  // single dropped mode (reg 1e-4, 65 of 66 kept) took max|ΔJ| from 3.2e-4 Ha to
  // 5.0 Ha. Contrast buildAuxBasisDFStreaming, which physically compacts B to
  // nKept and therefore correctly reports nKept.
  return {
    B,
    nAux,
    nKeptModes: nKept,
    threshold: metricRegularization,
    n,
  };
}
