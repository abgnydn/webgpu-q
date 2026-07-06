// Bridge from JS/TS to the Rust+WASM cache-blocked SIMD dgemm kernel.
//
// The reusable fast-CPU double-precision matrix-multiply that will later
// replace the naive TS triple/quad loops dominating MP2/CCSD contractions
// (those are BLAS-bound). Built to wasm32 with `-C target-feature=+simd128`
// (f64x2 v128 lanes), register + L1 cache-blocked. See `wasm-gemm/`.
//
// NOTE: this loader is intentionally NOT wired into any chemistry path yet.
// It exists so the kernel is *integrable* (typed, lazy-loaded, same import
// shape as `wasm-eri.ts`) and so `tsc --noEmit` exercises the pkg types.
//
// Loaded lazily so the ~20 KB WASM only downloads when a caller opts in.

interface WasmGemmModule {
  default(input?: unknown): Promise<unknown>;
  /** C = alpha*A*B + beta*C, row-major f64. A: m×k, B: k×n, C: m×n. */
  dgemm(
    m: number,
    n: number,
    k: number,
    a: Float64Array,
    b: Float64Array,
    c: Float64Array,
    alpha: number,
    beta: number,
  ): Float64Array;
  /** C = alpha*Aᵀ*B + beta*C. A is k×m (row-major). */
  dgemm_at(
    m: number,
    n: number,
    k: number,
    a: Float64Array,
    b: Float64Array,
    c: Float64Array,
    alpha: number,
    beta: number,
  ): Float64Array;
}

let wasmModule: WasmGemmModule | null = null;
let initPromise: Promise<WasmGemmModule> | null = null;

async function loadWasm(): Promise<WasmGemmModule> {
  if (wasmModule) return wasmModule;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    // wasm-pack --target web output lives at <repo>/wasm-gemm/pkg/. Vite
    // resolves this relative import + the wasm_gemm.js's internal
    // `new URL('wasm_gemm_bg.wasm', import.meta.url)` pattern natively.
    const mod = (await import(
      /* @vite-ignore */
      "../../wasm-gemm/pkg/wasm_gemm.js" as string
    )) as WasmGemmModule;
    await mod.default();
    wasmModule = mod;
    return mod;
  })();
  return initPromise;
}

/**
 * C = alpha·A·B + beta·C, row-major f64. A is m×k, B is k×n, C is m×n.
 * `c` may be a zero array when beta = 0. Returns a fresh Float64Array (m·n).
 * Lazy-loads the WASM module on first call; subsequent calls reuse it.
 */
export async function dgemmWasm(
  m: number,
  n: number,
  k: number,
  a: Float64Array,
  b: Float64Array,
  c: Float64Array,
  alpha = 1,
  beta = 0,
): Promise<Float64Array> {
  const mod = await loadWasm();
  return mod.dgemm(m, n, k, a, b, c, alpha, beta);
}

/** C = alpha·Aᵀ·B + beta·C. A is k×m (row-major). Returns C (m·n). */
export async function dgemmATWasm(
  m: number,
  n: number,
  k: number,
  a: Float64Array,
  b: Float64Array,
  c: Float64Array,
  alpha = 1,
  beta = 0,
): Promise<Float64Array> {
  const mod = await loadWasm();
  return mod.dgemm_at(m, n, k, a, b, c, alpha, beta);
}
