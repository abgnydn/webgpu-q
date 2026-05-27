/* tslint:disable */
/* eslint-disable */

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
 * Compute just the Schwarz Q table (diagonal-pair ERIs sqrt-abs).
 * Cheap, but JS-side construction is also slow on TS — expose this for
 * workers that want to skip the postMessage clone.
 */
export function schwarz_q_table(n_shells: number, n_prims_per_shell: Uint32Array, prim_offsets: Uint32Array, alpha_flat: Float64Array, c_flat: Float64Array, center_flat: Float64Array, angular_flat: Int32Array): Float64Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly eri_build: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number];
    readonly eri_build_slice: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number) => [number, number];
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
