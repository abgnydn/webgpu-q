// ─────────────────────────────────────────────────────────────
// df.ts — Density fitting (RI) for the 2-electron repulsion tensor.
// Tier 2 stage 26.
//
// Replaces the rank-4 ERI tensor (μν|λσ) with a rank-3 "B-tensor":
//
//   (μν|λσ) ≈ Σ_P B_{μν,P} · B_{λσ,P}
//
// where P runs over an auxiliary index of size M_aux ≤ n². Two
// approaches in the literature:
//
//   A) Aux basis (Weigend JKFIT, def2-SVP-JKFIT, etc.): build
//      (μν|P) and (P|Q) integrals over a pre-tabulated aux basis,
//      form B = (μν|P) · (P|Q)^(-1/2). Independent of the 4-index
//      ERI — actually avoids computing it. Standard in production.
//   B) Cholesky decomposition (CD-DF): treat the ERI tensor as a
//      symmetric positive-semidefinite (n², n²) matrix and apply
//      pivoted incomplete Cholesky factorization with threshold τ.
//      The resulting L-factor IS the B-tensor; the aux index is
//      "discovered" from the data, not chosen a priori. No external
//      basis tables needed.
//
// This module implements (B). The advantage: self-contained, with
// the truncation threshold giving direct control over accuracy.
// The disadvantage: requires the full ERI tensor as input — so it
// doesn't reduce the cost of the integral build itself. Downstream
// J + K and post-HF correlations DO benefit from the compressed
// representation. Approach (A) for true integral-build speedup is a
// follow-up (needs aux basis data tables).
//
// Accuracy: |ERI − B·B^T|_max ≤ τ by construction (the residual
// diagonal is bounded by τ at termination, and the residual is
// positive semi-definite). For HF energies, τ = 1e-6 typically gives
// sub-µHa absolute errors; τ = 1e-9 is machine-precision.
//
// Cost: pivoted Cholesky on the (n², n²) ERI matrix is
// O(n⁴ · M_aux) (M_aux columns × O(n²) per column update). For
// M_aux ≪ n², this is cheap relative to the n⁴ tensor build.
// ─────────────────────────────────────────────────────────────

export interface DFResult {
  /** Density-fitting B-tensor, row-major B[(μ·n + ν)·M_aux + P]. */
  readonly B: Float64Array;
  /** B's auxiliary STRIDE M_aux — every consumer indexes `B[i·nAux + P]`, so this
   *  MUST equal `B.length / n²`, never a smaller "kept modes" count. Builders that
   *  regularize by zeroing λ⁻¹⸍² in place keep the full stride (the dropped columns
   *  are exactly zero and contribute nothing); builders that physically compact B
   *  report the compacted width. Reporting kept-modes here while B stays full-width
   *  silently misaligns every J/K read — measured 5.0 Ha error on H₂O/STO-3G from a
   *  SINGLE dropped mode. Use {@link nKeptModes} for the diagnostic count. */
  readonly nAux: number;
  /** Diagnostic: aux modes surviving the metric threshold. Equals `nAux` unless the
   *  metric was rank-deficient. Never use as a stride. */
  readonly nKeptModes?: number;
  /** Cholesky truncation threshold τ used to derive the tensor. */
  readonly threshold: number;
  /** Original AO basis dimension n. */
  readonly n: number;
}

/**
 * Pivoted incomplete Cholesky decomposition of the AO ERI tensor.
 *
 * View the rank-4 tensor (μν|λσ) as a symmetric PSD matrix on the
 * product space (μν), (λσ): M[μν, λσ] = (μν|λσ). Decompose
 * M ≈ L · L^T using pivoted Cholesky with the largest diagonal
 * residual as the pivot at each step. The factor L has shape
 * (n², M_aux) and IS the density-fitting B-tensor.
 *
 * The PSD property of the ERI tensor follows from its Coulomb-metric
 * Gram structure: (μν|λσ) = ∫∫ φ_μ(1)φ_ν(1) · r₁₂⁻¹ · φ_λ(2)φ_σ(2),
 * so M = ⟨g_μν | r⁻¹ | g_λσ⟩ where g_μν = φ_μ · φ_ν. Pivoted
 * Cholesky on a PSD matrix is unconditionally stable; the diagonal
 * residual is monotonically non-increasing.
 *
 * Default threshold τ = 1e-8 — sub-µHa HF accuracy in practice.
 */
export function choleskyDecomposeERI(
  eri_AO: Float64Array,
  n: number,
  threshold = 1e-8,
): DFResult {
  const N = n * n;
  // Diagonal of M: d[μν] = (μν|μν) initially.
  const d = new Float64Array(N);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      const i = mu * n + nu;
      d[i] = eri_AO[((mu * n + nu) * n + mu) * n + nu]!;
    }
  }
  // L stored column-by-column as Float64Array of length N. Each new
  // Cholesky column is appended to L_cols.
  const L_cols: Float64Array[] = [];
  // Standard PSD: d[i] ≥ 0 within numerical tolerance. Clamp small
  // negatives to 0 to avoid sqrt(-eps).
  const MAX_AUX = N;
  for (let step = 0; step < MAX_AUX; step++) {
    // Pivot: index with largest residual diagonal.
    let piv = -1;
    let pmax = 0;
    for (let i = 0; i < N; i++) {
      if (d[i]! > pmax) {
        pmax = d[i]!;
        piv = i;
      }
    }
    if (piv < 0 || pmax < threshold) break;
    const sqrtPiv = Math.sqrt(pmax);
    const piv_mu = Math.floor(piv / n);
    const piv_nu = piv - piv_mu * n;
    const newCol = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const mu = Math.floor(i / n);
      const nu = i - mu * n;
      // M[i, piv] = (μν | piv_μ piv_ν) from the original ERI tensor.
      let v = eri_AO[((mu * n + nu) * n + piv_mu) * n + piv_nu]!;
      // Subtract contributions from previously-computed columns:
      // v -= Σ_j L_cols[j][i] · L_cols[j][piv].
      for (let j = 0; j < L_cols.length; j++) {
        v -= L_cols[j]![i]! * L_cols[j]![piv]!;
      }
      newCol[i] = v / sqrtPiv;
    }
    L_cols.push(newCol);
    // Update residual diagonal: d[i] -= L_cols[step][i]^2.
    for (let i = 0; i < N; i++) {
      const v = newCol[i]!;
      d[i] = Math.max(0, d[i]! - v * v);
    }
    d[piv] = 0;
  }

  // Pack the M_aux Cholesky columns into a row-major B[(μν)·M_aux + P] tensor.
  const M_aux = L_cols.length;
  const B = new Float64Array(N * M_aux);
  for (let i = 0; i < N; i++) {
    for (let P = 0; P < M_aux; P++) {
      B[i * M_aux + P] = L_cols[P]![i]!;
    }
  }
  return { B, nAux: M_aux, threshold, n };
}

/**
 * Reconstruct the full rank-4 ERI tensor from the B-tensor.
 * For testing and threshold verification: |ERI_exact − ERI_DF|_max
 * is bounded by the Cholesky truncation threshold.
 */
export function reconstructERI(df: DFResult): Float64Array {
  const { B, nAux, n } = df;
  const N = n * n;
  const eri = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let P = 0; P < nAux; P++) {
        s += B[i * nAux + P]! * B[j * nAux + P]!;
      }
      eri[i * N + j] = s;
    }
  }
  // Reshape: eri is laid out as eri[(μν)·N + (λσ)] = eri_AO[((μ·n+ν)·n+λ)·n+σ].
  return eri;
}

/**
 * Build Coulomb (J) and exchange (K) Fock contributions from a
 * density matrix using the DF B-tensor.
 *
 * J[μν] = Σ_λσ (μν|λσ) · D[λσ]
 *       = Σ_P B_{μν,P} · γ_P,   γ_P = Σ_λσ B_{λσ,P} · D[λσ]
 *
 * K[μν] = Σ_λσ (μλ|νσ) · D[λσ]
 *       = Σ_P Σ_σ X_{P,μ,σ} · B_{νσ,P},   X_{P,μ,σ} = Σ_λ B_{μλ,P} · D[λσ]
 *
 * Both contractions are matrix-matrix and BLAS-friendly. For
 * M_aux ~ 3n, K dominates at O(n³·M_aux) ~ O(n⁴) — similar
 * scaling to the 4-index direct path but with a smaller constant
 * factor at larger system sizes.
 *
 * The Fock from these is F = h + J − ½ K (closed-shell) or
 * F_σ = h + J − K_σ (per-spin, open-shell). This routine returns
 * J and K (without ½ on K) — the caller composes the Fock matrix.
 */
// Lazy-cached WASM module for the JK build. Set by `preloadWasmJK()`.
// When present, `buildJK_DF` uses the native f64 kernel; otherwise
// falls back to the TS implementation.
let wasmJKModule: {
  build_jk_df: (n: number, nAux: number, b: Float64Array, d: Float64Array) => Float64Array;
} | null = null;

export async function preloadWasmJK(): Promise<void> {
  if (wasmJKModule) return;
  const mod = await import(
    /* @vite-ignore */
    "../../wasm-eri/pkg/wasm_eri.js" as string,
  ) as {
    default(): Promise<unknown>;
    build_jk_df: (n: number, nAux: number, b: Float64Array, d: Float64Array) => Float64Array;
  };
  await mod.default();
  wasmJKModule = mod;
}

export function buildJK_DF(
  df: DFResult,
  D: Float64Array,
): { J: Float64Array; K: Float64Array } {
  const { B, nAux, n } = df;
  const N = n * n;
  // WASM fast path when preloaded — packs J + K into one buffer.
  if (wasmJKModule) {
    const out = wasmJKModule.build_jk_df(n, nAux, B, D);
    const J = new Float64Array(N);
    const K = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      J[i] = out[i]!;
      K[i] = out[N + i]!;
    }
    return { J, K };
  }
  const J = new Float64Array(N);
  const K = new Float64Array(N);

  // γ[P] = Σ_λσ B[λσ, P] · D[λσ]
  const gamma = new Float64Array(nAux);
  for (let P = 0; P < nAux; P++) {
    let s = 0;
    for (let i = 0; i < N; i++) {
      s += B[i * nAux + P]! * D[i]!;
    }
    gamma[P] = s;
  }
  // J[μν] = Σ_P B[μν, P] · γ[P]
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let P = 0; P < nAux; P++) {
      s += B[i * nAux + P]! * gamma[P]!;
    }
    J[i] = s;
  }

  // X[P, μ, σ] = Σ_λ B[μλ, P] · D[λ, σ]
  // Layout X as X[P * n² + μ * n + σ].
  const X = new Float64Array(nAux * N);
  for (let P = 0; P < nAux; P++) {
    for (let mu = 0; mu < n; mu++) {
      for (let si = 0; si < n; si++) {
        let s = 0;
        for (let la = 0; la < n; la++) {
          s += B[(mu * n + la) * nAux + P]! * D[la * n + si]!;
        }
        X[(P * n + mu) * n + si] = s;
      }
    }
  }
  // K[μν] = Σ_P Σ_σ X[P, μ, σ] · B[νσ, P]
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let P = 0; P < nAux; P++) {
        for (let si = 0; si < n; si++) {
          s += X[(P * n + mu) * n + si]! * B[(nu * n + si) * nAux + P]!;
        }
      }
      K[mu * n + nu] = s;
    }
  }
  return { J, K };
}
