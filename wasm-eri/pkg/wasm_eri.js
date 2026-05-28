/* @ts-self-types="./wasm_eri.d.ts" */

/**
 * Build the 2-index AO ERI metric M[P, Q] = (P|Q) on the auxiliary
 * basis. Symmetric n_aux × n_aux matrix.
 *
 * This is the Coulomb metric for density fitting: B = V · M^(-1/2)
 * where V is the 3-index tensor from `eri_3idx_build`.
 * @param {number} n_aux
 * @param {Uint32Array} n_prims_per_aux
 * @param {Uint32Array} prim_offsets_aux
 * @param {Float64Array} alpha_aux
 * @param {Float64Array} c_aux
 * @param {Float64Array} center_aux
 * @param {Int32Array} angular_aux
 * @returns {Float64Array}
 */
export function eri_2idx_build(n_aux, n_prims_per_aux, prim_offsets_aux, alpha_aux, c_aux, center_aux, angular_aux) {
    const ptr0 = passArray32ToWasm0(n_prims_per_aux, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(prim_offsets_aux, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(alpha_aux, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(c_aux, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArrayF64ToWasm0(center_aux, wasm.__wbindgen_malloc);
    const len4 = WASM_VECTOR_LEN;
    const ptr5 = passArray32ToWasm0(angular_aux, wasm.__wbindgen_malloc);
    const len5 = WASM_VECTOR_LEN;
    const ret = wasm.eri_2idx_build(n_aux, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5);
    var v7 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v7;
}

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
 * @param {number} n_orbital
 * @param {number} n_aux
 * @param {Uint32Array} n_prims_per_orb
 * @param {Uint32Array} prim_offsets_orb
 * @param {Float64Array} alpha_orb
 * @param {Float64Array} c_orb
 * @param {Float64Array} center_orb
 * @param {Int32Array} angular_orb
 * @param {Uint32Array} n_prims_per_aux
 * @param {Uint32Array} prim_offsets_aux
 * @param {Float64Array} alpha_aux
 * @param {Float64Array} c_aux
 * @param {Float64Array} center_aux
 * @param {Int32Array} angular_aux
 * @returns {Float64Array}
 */
export function eri_3idx_build(n_orbital, n_aux, n_prims_per_orb, prim_offsets_orb, alpha_orb, c_orb, center_orb, angular_orb, n_prims_per_aux, prim_offsets_aux, alpha_aux, c_aux, center_aux, angular_aux) {
    const ptr0 = passArray32ToWasm0(n_prims_per_orb, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(prim_offsets_orb, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(alpha_orb, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(c_orb, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArrayF64ToWasm0(center_orb, wasm.__wbindgen_malloc);
    const len4 = WASM_VECTOR_LEN;
    const ptr5 = passArray32ToWasm0(angular_orb, wasm.__wbindgen_malloc);
    const len5 = WASM_VECTOR_LEN;
    const ptr6 = passArray32ToWasm0(n_prims_per_aux, wasm.__wbindgen_malloc);
    const len6 = WASM_VECTOR_LEN;
    const ptr7 = passArray32ToWasm0(prim_offsets_aux, wasm.__wbindgen_malloc);
    const len7 = WASM_VECTOR_LEN;
    const ptr8 = passArrayF64ToWasm0(alpha_aux, wasm.__wbindgen_malloc);
    const len8 = WASM_VECTOR_LEN;
    const ptr9 = passArrayF64ToWasm0(c_aux, wasm.__wbindgen_malloc);
    const len9 = WASM_VECTOR_LEN;
    const ptr10 = passArrayF64ToWasm0(center_aux, wasm.__wbindgen_malloc);
    const len10 = WASM_VECTOR_LEN;
    const ptr11 = passArray32ToWasm0(angular_aux, wasm.__wbindgen_malloc);
    const len11 = WASM_VECTOR_LEN;
    const ret = wasm.eri_3idx_build(n_orbital, n_aux, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, ptr7, len7, ptr8, len8, ptr9, len9, ptr10, len10, ptr11, len11);
    var v13 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v13;
}

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
 * @param {Uint32Array} mus
 * @param {number} n_orbital
 * @param {number} n_aux
 * @param {Uint32Array} n_prims_per_orb
 * @param {Uint32Array} prim_offsets_orb
 * @param {Float64Array} alpha_orb
 * @param {Float64Array} c_orb
 * @param {Float64Array} center_orb
 * @param {Int32Array} angular_orb
 * @param {Uint32Array} n_prims_per_aux
 * @param {Uint32Array} prim_offsets_aux
 * @param {Float64Array} alpha_aux
 * @param {Float64Array} c_aux
 * @param {Float64Array} center_aux
 * @param {Int32Array} angular_aux
 * @returns {Float64Array}
 */
export function eri_3idx_build_slice(mus, n_orbital, n_aux, n_prims_per_orb, prim_offsets_orb, alpha_orb, c_orb, center_orb, angular_orb, n_prims_per_aux, prim_offsets_aux, alpha_aux, c_aux, center_aux, angular_aux) {
    const ptr0 = passArray32ToWasm0(mus, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(n_prims_per_orb, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray32ToWasm0(prim_offsets_orb, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(alpha_orb, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArrayF64ToWasm0(c_orb, wasm.__wbindgen_malloc);
    const len4 = WASM_VECTOR_LEN;
    const ptr5 = passArrayF64ToWasm0(center_orb, wasm.__wbindgen_malloc);
    const len5 = WASM_VECTOR_LEN;
    const ptr6 = passArray32ToWasm0(angular_orb, wasm.__wbindgen_malloc);
    const len6 = WASM_VECTOR_LEN;
    const ptr7 = passArray32ToWasm0(n_prims_per_aux, wasm.__wbindgen_malloc);
    const len7 = WASM_VECTOR_LEN;
    const ptr8 = passArray32ToWasm0(prim_offsets_aux, wasm.__wbindgen_malloc);
    const len8 = WASM_VECTOR_LEN;
    const ptr9 = passArrayF64ToWasm0(alpha_aux, wasm.__wbindgen_malloc);
    const len9 = WASM_VECTOR_LEN;
    const ptr10 = passArrayF64ToWasm0(c_aux, wasm.__wbindgen_malloc);
    const len10 = WASM_VECTOR_LEN;
    const ptr11 = passArrayF64ToWasm0(center_aux, wasm.__wbindgen_malloc);
    const len11 = WASM_VECTOR_LEN;
    const ptr12 = passArray32ToWasm0(angular_aux, wasm.__wbindgen_malloc);
    const len12 = WASM_VECTOR_LEN;
    const ret = wasm.eri_3idx_build_slice(ptr0, len0, n_orbital, n_aux, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, ptr7, len7, ptr8, len8, ptr9, len9, ptr10, len10, ptr11, len11, ptr12, len12);
    var v14 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v14;
}

/**
 * @param {number} n_shells
 * @param {Uint32Array} n_prims_per_shell
 * @param {Uint32Array} prim_offsets
 * @param {Float64Array} alpha_flat
 * @param {Float64Array} c_flat
 * @param {Float64Array} center_flat
 * @param {Int32Array} angular_flat
 * @param {number} schwarz_tol
 * @returns {Float64Array}
 */
export function eri_build(n_shells, n_prims_per_shell, prim_offsets, alpha_flat, c_flat, center_flat, angular_flat, schwarz_tol) {
    const ptr0 = passArray32ToWasm0(n_prims_per_shell, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(prim_offsets, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(alpha_flat, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(c_flat, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArrayF64ToWasm0(center_flat, wasm.__wbindgen_malloc);
    const len4 = WASM_VECTOR_LEN;
    const ptr5 = passArray32ToWasm0(angular_flat, wasm.__wbindgen_malloc);
    const len5 = WASM_VECTOR_LEN;
    const ret = wasm.eri_build(n_shells, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, schwarz_tol);
    var v7 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v7;
}

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
 * @param {Uint32Array} mus
 * @param {number} n_shells
 * @param {Uint32Array} n_prims_per_shell
 * @param {Uint32Array} prim_offsets
 * @param {Float64Array} alpha_flat
 * @param {Float64Array} c_flat
 * @param {Float64Array} center_flat
 * @param {Int32Array} angular_flat
 * @param {Float64Array} q_table
 * @param {number} schwarz_tol
 * @returns {Float64Array}
 */
export function eri_build_slice(mus, n_shells, n_prims_per_shell, prim_offsets, alpha_flat, c_flat, center_flat, angular_flat, q_table, schwarz_tol) {
    const ptr0 = passArray32ToWasm0(mus, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(n_prims_per_shell, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray32ToWasm0(prim_offsets, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(alpha_flat, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArrayF64ToWasm0(c_flat, wasm.__wbindgen_malloc);
    const len4 = WASM_VECTOR_LEN;
    const ptr5 = passArrayF64ToWasm0(center_flat, wasm.__wbindgen_malloc);
    const len5 = WASM_VECTOR_LEN;
    const ptr6 = passArray32ToWasm0(angular_flat, wasm.__wbindgen_malloc);
    const len6 = WASM_VECTOR_LEN;
    const ptr7 = passArrayF64ToWasm0(q_table, wasm.__wbindgen_malloc);
    const len7 = WASM_VECTOR_LEN;
    const ret = wasm.eri_build_slice(ptr0, len0, n_shells, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, ptr7, len7, schwarz_tol);
    var v9 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v9;
}

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
 * @param {Uint32Array} mus
 * @param {number} n
 * @param {Float64Array} eri_slab
 * @param {Float64Array} d
 * @returns {Float64Array}
 */
export function fock_build_slice(mus, n, eri_slab, d) {
    const ptr0 = passArray32ToWasm0(mus, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(eri_slab, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(d, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.fock_build_slice(ptr0, len0, n, ptr1, len1, ptr2, len2);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
}

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
 * @param {number} n
 * @param {Float64Array} eri_mu_row
 * @param {Float64Array} d
 * @returns {Float64Array}
 */
export function fock_one_mu_row(n, eri_mu_row, d) {
    const ptr0 = passArrayF64ToWasm0(eri_mu_row, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(d, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.fock_one_mu_row(n, ptr0, len0, ptr1, len1);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

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
 * @param {number} n
 * @param {number} n_aux
 * @param {Float64Array} v
 * @param {Float64Array} u
 * @param {Float64Array} inv_sqrt_lam
 * @returns {Float64Array}
 */
export function form_b_tensor(n, n_aux, v, u, inv_sqrt_lam) {
    const ptr0 = passArrayF64ToWasm0(v, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(u, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(inv_sqrt_lam, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.form_b_tensor(n, n_aux, ptr0, len0, ptr1, len1, ptr2, len2);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * Compute just the Schwarz Q table (diagonal-pair ERIs sqrt-abs).
 * Cheap, but JS-side construction is also slow on TS — expose this for
 * workers that want to skip the postMessage clone.
 * @param {number} n_shells
 * @param {Uint32Array} n_prims_per_shell
 * @param {Uint32Array} prim_offsets
 * @param {Float64Array} alpha_flat
 * @param {Float64Array} c_flat
 * @param {Float64Array} center_flat
 * @param {Int32Array} angular_flat
 * @returns {Float64Array}
 */
export function schwarz_q_table(n_shells, n_prims_per_shell, prim_offsets, alpha_flat, c_flat, center_flat, angular_flat) {
    const ptr0 = passArray32ToWasm0(n_prims_per_shell, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(prim_offsets, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(alpha_flat, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(c_flat, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArrayF64ToWasm0(center_flat, wasm.__wbindgen_malloc);
    const len4 = WASM_VECTOR_LEN;
    const ptr5 = passArray32ToWasm0(angular_flat, wasm.__wbindgen_malloc);
    const len5 = WASM_VECTOR_LEN;
    const ret = wasm.schwarz_q_table(n_shells, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5);
    var v7 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v7;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./wasm_eri_bg.js": import0,
    };
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('wasm_eri_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
