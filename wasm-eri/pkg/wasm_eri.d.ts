/* tslint:disable */
/* eslint-disable */

/**
 * Compute the gamma vector γ[P] = Σ_λσ B[λσ, P] · D[λσ] — shared
 * across workers in the parallel build_jk_df path.
 */
export function build_gamma_df(n: number, n_aux: number, b: Float64Array, d: Float64Array): Float64Array;

export function build_jk_df(n: number, n_aux: number, b: Float64Array, d: Float64Array): Float64Array;

/**
 * DF Fock build: given B-tensor B[μν, P] and density D[μν], compute
 *   J[μν] = Σ_P B[μν, P] · γ[P],  γ[P] = Σ_λσ B[λσ, P] · D[λσ]
 *   K[μν] = Σ_P Σ_σ X[P, μ, σ] · B[νσ, P],
 *           X[P, μ, σ] = Σ_λ B[μλ, P] · D[λ, σ]
 *
 * Returns packed [J(0..N), K(0..N)] of length 2N where N = n².
 * Layout: out[0..N] = J, out[N..2N] = K.
 *
 * Hot path: the K inner pass is O(n²·n_aux·n) = ~1 B FLOPs for
 * benzene cc-pVDZ at n_aux≈660. SIMD on the innermost contiguous
 * loops; the (la, si) and (P, si) inner pairs are both length-n
 * f64 reductions perfect for f64x2 wasm-simd128.
 * Worker-parallel slice of build_jk_df: computes J[μ, :] and K[μ, :]
 * for μ ∈ `mus`. Returns packed [J_slice; K_slice] of length 2·|mus|·n.
 * Each worker handles a disjoint μ-row range; combine on main thread.
 *
 * The full J and K depend on the same γ[P] and X[P, μ', σ] precomputes,
 * but each row μ only needs:
 *   J[μ, ν] = Σ_P B[μν, P] · γ[P]                    — local to row μ
 *   K[μ, ν] = Σ_P Σ_σ X[P, μ, σ] · B[νσ, P]           — needs X[*, μ, *]
 *
 * γ is shared (computed once per call, small: n_aux entries).
 * X[*, μ, *] for μ ∈ mus is the per-worker partition (n_aux · n entries
 * per μ). Build that per-worker locally.
 */
export function build_jk_df_slice(mus: Uint32Array, n: number, n_aux: number, b: Float64Array, d: Float64Array, gamma: Float64Array): Float64Array;

/**
 * Build the 2-index AO ERI metric M[P, Q] = (P|Q) on the auxiliary
 * basis. Symmetric n_aux × n_aux matrix.
 *
 * This is the Coulomb metric for density fitting: B = V · M^(-1/2)
 * where V is the 3-index tensor from `eri_3idx_build`.
 */
export function eri_2idx_build(n_aux: number, n_prims_per_aux: Uint32Array, prim_offsets_aux: Uint32Array, alpha_aux: Float64Array, c_aux: Float64Array, center_aux: Float64Array, angular_aux: Int32Array): Float64Array;

/**
 * Build the 3-index AO ERI tensor V[μν, P] = (μν|P) for aux-basis
 * density fitting. Returns a flat `n² · n_aux` Float64Array with
 * layout V[(μ · n + ν) · n_aux + P].
 *
 * For prototype, the aux basis is just another shell list; production
 * would pass a separate jkfit/auxiliary set. Currently the orbital
 * and aux shell representations are identical (CGShell-shaped).
 *
 * Cost: n_bra_pairs · n_aux · prim_eri_3idx (much cheaper than
 * 4-index since the inner loop is 3+3 deep instead of 6 deep).
 */
export function eri_3idx_build(n_orbital: number, n_aux: number, n_prims_per_orb: Uint32Array, prim_offsets_orb: Uint32Array, alpha_orb: Float64Array, c_orb: Float64Array, center_orb: Float64Array, angular_orb: Int32Array, n_prims_per_aux: Uint32Array, prim_offsets_aux: Uint32Array, alpha_aux: Float64Array, c_aux: Float64Array, center_aux: Float64Array, angular_aux: Int32Array): Float64Array;

/**
 * Worker-parallel 3-index ERI build. Computes (μν|P) for μ ∈ `mus`
 * only and returns a packed flat array of length 4·K:
 *   [μ, ν, P, value, μ, ν, P, value, …]
 * Indices stored as f64 (n ≤ 2^53 fits exactly). Caller scatters the
 * values into the n²·n_aux V-tensor at both (μν, P) and (νμ, P)
 * positions.
 *
 * Cost per worker: O(|mus| · n · n_aux · prim_eri_3idx). For benzene
 * cc-pVDZ with n=120, n_aux≈400, |mus|=15: ~720K 3-index ERIs ≈ ~3 s
 * on M2 Pro. Parallel across 8 workers: <1 s wall.
 */
export function eri_3idx_build_slice(mus: Uint32Array, n_orbital: number, n_aux: number, n_prims_per_orb: Uint32Array, prim_offsets_orb: Uint32Array, alpha_orb: Float64Array, c_orb: Float64Array, center_orb: Float64Array, angular_orb: Int32Array, n_prims_per_aux: Uint32Array, prim_offsets_aux: Uint32Array, alpha_aux: Float64Array, c_aux: Float64Array, center_aux: Float64Array, angular_aux: Int32Array): Float64Array;

export function eri_build(n_shells: number, n_prims_per_shell: Uint32Array, prim_offsets: Uint32Array, alpha_flat: Float64Array, c_flat: Float64Array, center_flat: Float64Array, angular_flat: Int32Array, schwarz_tol: number): Float64Array;

/**
 * Compute the canonical ERIs (μν|λσ) for μ ∈ mus only.
 *
 * Returns a packed flat array: [μ, ν, λ, σ, v, μ, ν, λ, σ, v, ...] of
 * length 5K where K is the number of unique non-screened ERIs in this
 * slice. Indices are stored as f64 (n ≤ 2^53 fits exactly).
 *
 * Q-table is precomputed by the caller (cheap, n² ERIs) so multiple
 * workers can share it via postMessage clone. Schwarz screening is
 * applied identically to the full-build path.
 *
 * Worker-side: the caller is responsible for writing the 8 symmetric
 * positions for each (μ, ν, λ, σ, v) into the shared output buffer.
 */
export function eri_build_slice(mus: Uint32Array, n_shells: number, n_prims_per_shell: Uint32Array, prim_offsets: Uint32Array, alpha_flat: Float64Array, c_flat: Float64Array, center_flat: Float64Array, angular_flat: Int32Array, q_table: Float64Array, schwarz_tol: number): Float64Array;

/**
 * Compute the Fock matrix G slice G[μ, ν] for μ ∈ `mus` (a subset of
 * rows), from the AO ERI tensor and the density matrix D.
 *
 *   G[μ, ν] = Σ_{λ, σ} D[λ, σ] · ( (μν|λσ) − ½ (μλ|νσ) )
 *
 * Inputs:
 *   - `mus`: which global μ indices this worker owns (length K).
 *   - `n`: AO basis size.
 *   - `eri_slab`: a flat `K · n³` chunk of the ERI tensor laid out as
 *     `eri_slab[k * n³ + a * n² + b * n + c] = eri[mus[k], a, b, c]`,
 *     i.e. row-major over (k = local μ index, a, b, c). Caller is
 *     responsible for gathering this slab from the full n⁴ ERI before
 *     the per-iteration SCF loop and reusing it across iterations.
 *   - `d`: the full n × n density matrix, row-major.
 *
 * Output: `K · n` Fock entries, `g_slice[k * n + nu] = G[mus[k], nu]`.
 * The caller scatters these back into the full G via the same `mus`
 * indices.
 *
 * Why slab-not-tensor: WASM linear memory is separate from the JS
 * SAB, and copying the full n⁴ ERI (1.65 GB on benzene cc-pVDZ) into
 * WASM would dominate the kernel. The slab is 8 × smaller per
 * worker on N=8 and changes never during SCF — copy once, reuse.
 */
export function fock_build_slice(mus: Uint32Array, n: number, eri_slab: Float64Array, d: Float64Array): Float64Array;

/**
 * Single-μ variant of fock_build_slice. Computes G[μ, :] for one μ.
 *
 *   G[μ, ν] = Σ_{λ, σ} D[λ, σ] · ( (μν|λσ) − ½ (μλ|νσ) )
 *
 * `eri_mu_row` is the n³ slab eri[μ, :, :, :], laid out row-major as
 * eri_mu_row[a · n² + b · n + c] = eri[μ, a, b, c].
 *
 * Used by the per-μ WASM JK kernel: the worker copies only this μ's
 * n³ slab into WASM linear memory per call (rather than caching the
 * full per-worker slab of |mus|·n³ entries, which doubles browser
 * memory pressure on benzene cc-pVDZ). The copy amortizes against
 * the ~10ms WASM compute per μ at n=120. Inner σ-loop uses the
 * 2-lane f64 SIMD `jk_dot` helper above.
 */
export function fock_one_mu_row(n: number, eri_mu_row: Float64Array, d: Float64Array): Float64Array;

/**
 * Form the density-fitting B tensor: B[μν, P] = Σ_Q V[μν, Q] · M⁻¹⸍²[Q, P]
 * where M⁻¹⸍² = U · diag(λ⁻¹⸍²) · Uᵀ from the eigendecomposition of M.
 *
 * Inputs:
 *   - `v`: 3-index tensor, shape (n·n, n_aux), row-major over (μν, Q)
 *   - `u`: eigenvector matrix from `eigsymmetric`, column-major:
 *          u[i·n_aux + Q] = U[Q, i] (Q-th component of i-th eigenvector)
 *   - `inv_sqrt_lam`: 1/√λ_i for each eigenvalue (0 for regularized-out modes)
 *   - n, n_aux: dimensions
 *
 * Returns flat B of length n·n·n_aux, row-major over (μν, P).
 *
 * Hot path: O(n² · n_aux²). For benzene cc-pVDZ (n=120, n_aux≈400)
 * that's ~4.6 B FLOPs — was ~5 s in TypeScript even with cache-
 * friendly loop reorder. WASM with f64x2 SIMD on the inner P loop
 * (length n_aux=400, perfect for vectorization) drops it to ~1 s.
 *
 * Note: a Cholesky-back-sub variant (form_b_from_cholesky) was tried
 * in Rust+WASM and lost to TS (91 s vs 40 s on naphthalene).
 * wasm-bindgen copies the 312 MB V tensor per call, and the
 * pivot-indirect access into V/L is cache-unfriendly. See
 * `src/chemistry/df-aux.ts` for the active TS path.
 */
export function form_b_tensor(n: number, n_aux: number, v: Float64Array, u: Float64Array, inv_sqrt_lam: Float64Array): Float64Array;

/**
 * Worker-parallel slice of form_b_tensor: computes B for μ rows
 * in `mus`. Returns flat (|mus| · n · n_aux) packed contiguously
 * by (k, ν, P) where k is the local μ index.
 *
 * Caller scatters the returned slice into a shared full-B SAB at
 * the correct μ-offsets. Each worker owns disjoint μ rows so no
 * atomics needed.
 */
export function form_b_tensor_slice(mus: Uint32Array, n: number, n_aux: number, v: Float64Array, u: Float64Array, inv_sqrt_lam: Float64Array): Float64Array;

/**
 * Compute just the Schwarz Q table (diagonal-pair ERIs sqrt-abs).
 * Cheap, but JS-side construction is also slow on TS — expose this for
 * workers that want to skip the postMessage clone.
 */
export function schwarz_q_table(n_shells: number, n_prims_per_shell: Uint32Array, prim_offsets: Uint32Array, alpha_flat: Float64Array, c_flat: Float64Array, center_flat: Float64Array, angular_flat: Int32Array): Float64Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly build_gamma_df: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly build_jk_df: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly build_jk_df_slice: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly eri_2idx_build: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly eri_3idx_build: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: number, x: number, y: number, z: number) => [number, number];
    readonly eri_3idx_build_slice: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: number, x: number, y: number, z: number, a1: number, b1: number) => [number, number];
    readonly eri_build: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number];
    readonly eri_build_slice: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number) => [number, number];
    readonly fock_build_slice: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly fock_one_mu_row: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly form_b_tensor: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly form_b_tensor_slice: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly schwarz_q_table: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
