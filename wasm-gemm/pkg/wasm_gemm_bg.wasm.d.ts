/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const alloc_f64: (a: number) => number;
export const dgemm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
export const dgemm_at: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
export const dgemm_at_raw: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
export const dgemm_raw: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
export const free_f64: (a: number, b: number) => void;
export const __wbindgen_externrefs: WebAssembly.Table;
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __wbindgen_free: (a: number, b: number, c: number) => void;
export const __wbindgen_start: () => void;
