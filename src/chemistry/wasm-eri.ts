// Bridge from JS to the Rust+WASM ERI kernel.
//
// Builds the same dense AO ERI tensor as ERI_cg-based serial code,
// but executes the inner kernel in native-compiled wasm32 (3-5×
// per-thread speedup target over JIT'd TypeScript).
//
// Loaded lazily so the WASM bundle (~80 KB) only downloads when
// callers explicitly opt in via `buildERIWasm()`.

import type { CGShell } from "./integrals-cg.js";

interface WasmEriModule {
  default(): Promise<unknown>;
  eri_build(
    nShells: number,
    nPrimsPerShell: Uint32Array,
    primOffsets: Uint32Array,
    alphaFlat: Float64Array,
    cFlat: Float64Array,
    centerFlat: Float64Array,
    angularFlat: Int32Array,
    schwarzTol: number,
  ): Float64Array;
  schwarz_q_table(
    nShells: number,
    nPrimsPerShell: Uint32Array,
    primOffsets: Uint32Array,
    alphaFlat: Float64Array,
    cFlat: Float64Array,
    centerFlat: Float64Array,
    angularFlat: Int32Array,
  ): Float64Array;
}

let wasmModule: WasmEriModule | null = null;
let initPromise: Promise<WasmEriModule> | null = null;

async function loadWasm(): Promise<WasmEriModule> {
  if (wasmModule) return wasmModule;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    // wasm-pack output lives at <repo>/wasm-eri/pkg/. Vite resolves
    // this relative import + the wasm_eri.js's internal
    // `new URL('wasm_eri_bg.wasm', import.meta.url)` pattern natively.
    const mod = await import(
      /* @vite-ignore */
      "../../wasm-eri/pkg/wasm_eri.js" as string,
    ) as WasmEriModule;
    await mod.default();
    wasmModule = mod;
    return mod;
  })();
  return initPromise;
}

/** Pack a shell list into the flat-array layout the Rust kernel expects. */
function packShells(shells: readonly CGShell[]): {
  nPrimsPerShell: Uint32Array;
  primOffsets: Uint32Array;
  alphaFlat: Float64Array;
  cFlat: Float64Array;
  centerFlat: Float64Array;
  angularFlat: Int32Array;
} {
  const n = shells.length;
  const nPrimsPerShell = new Uint32Array(n);
  const primOffsets = new Uint32Array(n);
  let totalPrims = 0;
  for (let i = 0; i < n; i++) {
    nPrimsPerShell[i] = shells[i]!.alpha.length;
    primOffsets[i] = totalPrims;
    totalPrims += shells[i]!.alpha.length;
  }
  const alphaFlat = new Float64Array(totalPrims);
  const cFlat = new Float64Array(totalPrims);
  for (let i = 0; i < n; i++) {
    const offset = primOffsets[i]!;
    const sh = shells[i]!;
    for (let p = 0; p < sh.alpha.length; p++) {
      alphaFlat[offset + p] = sh.alpha[p]!;
      cFlat[offset + p] = sh.c[p]!;
    }
  }
  const centerFlat = new Float64Array(n * 3);
  const angularFlat = new Int32Array(n * 3);
  for (let i = 0; i < n; i++) {
    centerFlat[i * 3]     = shells[i]!.center[0];
    centerFlat[i * 3 + 1] = shells[i]!.center[1];
    centerFlat[i * 3 + 2] = shells[i]!.center[2];
    angularFlat[i * 3]     = shells[i]!.angular[0];
    angularFlat[i * 3 + 1] = shells[i]!.angular[1];
    angularFlat[i * 3 + 2] = shells[i]!.angular[2];
  }
  return { nPrimsPerShell, primOffsets, alphaFlat, cFlat, centerFlat, angularFlat };
}

/**
 * Build the dense AO ERI tensor via the Rust+WASM kernel.
 * Returns a fresh Float64Array of length n⁴.
 *
 * Lazy-loads the WASM module on first call; subsequent calls reuse it.
 */
export async function buildERIWasm(
  shells: readonly CGShell[],
  schwarzTol = 1e-10,
): Promise<Float64Array> {
  const mod = await loadWasm();
  const n = shells.length;
  const packed = packShells(shells);
  return mod.eri_build(
    n,
    packed.nPrimsPerShell,
    packed.primOffsets,
    packed.alphaFlat,
    packed.cFlat,
    packed.centerFlat,
    packed.angularFlat,
    schwarzTol,
  );
}

/**
 * Compute the Schwarz Q-table (sqrt-abs of (μν|μν) for all canonical
 * (μ, ν) pairs) via the Rust+WASM kernel. Used by parallel ERI paths
 * to avoid the ~175 ms main-thread TS serialization on benzene.
 */
export async function schwarzQTableWasm(
  shells: readonly CGShell[],
): Promise<Float64Array> {
  const mod = await loadWasm();
  const n = shells.length;
  const packed = packShells(shells);
  return mod.schwarz_q_table(
    n,
    packed.nPrimsPerShell,
    packed.primOffsets,
    packed.alphaFlat,
    packed.cFlat,
    packed.centerFlat,
    packed.angularFlat,
  );
}
