/* tslint:disable */
/* eslint-disable */

/**
 * Allocate `len` f64 slots in wasm memory; returns the byte pointer.
 */
export function alloc_f64(len: number): number;

/**
 * C = alpha * A * B + beta * C. Returns the resulting C (length m*n).
 */
export function dgemm(m: number, n: number, k: number, a: Float64Array, b: Float64Array, c: Float64Array, alpha: number, beta: number): Float64Array;

/**
 * C = alpha * A^T * B + beta * C. A is k x m. Returns C (length m*n).
 */
export function dgemm_at(m: number, n: number, k: number, a: Float64Array, b: Float64Array, c: Float64Array, alpha: number, beta: number): Float64Array;

/**
 * Raw C = alpha*A^T*B + beta*C (A is k x m) on wasm-memory pointers.
 */
export function dgemm_at_raw(m: number, n: number, k: number, a: number, b: number, c: number, alpha: number, beta: number): void;

/**
 * Raw C = alpha*A*B + beta*C on wasm-memory pointers (element offsets).
 */
export function dgemm_raw(m: number, n: number, k: number, a: number, b: number, c: number, alpha: number, beta: number): void;

/**
 * Free a buffer previously returned by `alloc_f64`.
 */
export function free_f64(ptr: number, len: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly alloc_f64: (a: number) => number;
    readonly dgemm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly dgemm_at: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly dgemm_at_raw: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly dgemm_raw: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly free_f64: (a: number, b: number) => void;
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
