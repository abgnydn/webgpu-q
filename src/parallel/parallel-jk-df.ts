// Parallel DF Fock build (J + K) using shared worker pool.
//
// Each worker computes J[μ,:] and K[μ,:] for μ ∈ its assigned μ list,
// and writes into the shared JK SAB (μ ranges are disjoint, no
// atomics needed). The γ vector is precomputed on main thread once
// (small: n_aux entries) and cloned to workers via postMessage.
//
// Requires SharedArrayBuffer (COOP/COEP isolation). Falls back to
// synchronous build_jk_df when SAB unavailable.

import type { DFResult } from "../chemistry/df.js";
import { sabAvailable, toSAB } from "./worker-pool.js";
import { getSharedWorkerPool } from "./worker-pool-shared.js";

interface WasmJKModule {
  default(): Promise<unknown>;
  build_jk_df: (n: number, nAux: number, b: Float64Array, d: Float64Array) => Float64Array;
  build_gamma_df: (n: number, nAux: number, b: Float64Array, d: Float64Array) => Float64Array;
}

let wasmMod: WasmJKModule | null = null;
async function loadWasm(): Promise<WasmJKModule> {
  if (wasmMod) return wasmMod;
  const mod = await import(
    /* @vite-ignore */
    "../../wasm-eri/pkg/wasm_eri.js" as string,
  ) as WasmJKModule;
  await mod.default();
  wasmMod = mod;
  return mod;
}

// Cache: ERI-B's SAB once per (DFResult identity) so workers don't
// re-copy B every iter.
let cachedDFIdentity: Float64Array | null = null;
let cachedBSAB: SharedArrayBuffer | null = null;

function getBSAB(df: DFResult): SharedArrayBuffer {
  if (cachedDFIdentity === df.B && cachedBSAB) return cachedBSAB;
  cachedBSAB = toSAB(df.B);
  cachedDFIdentity = df.B;
  return cachedBSAB;
}

/**
 * Parallel build_jk_df: returns J, K for the full ERI in O(n²·n_aux²)
 * work split across N workers. Each worker handles a subset of μ rows.
 *
 * @param df  Density-fitting result with B and nAux.
 * @param D   Density matrix, n × n row-major.
 * @param poolSize  Number of workers; 0 → hardwareConcurrency-1.
 */
export async function buildJK_DF_Parallel(
  df: DFResult,
  D: Float64Array,
  poolSize = 0,
): Promise<{ J: Float64Array; K: Float64Array }> {
  if (!sabAvailable()) {
    // Fallback to sync WASM path (still fast).
    const mod = await loadWasm();
    const out = mod.build_jk_df(df.n, df.nAux, df.B, D);
    const N = df.n * df.n;
    const J = new Float64Array(N);
    const K = new Float64Array(N);
    for (let i = 0; i < N; i++) { J[i] = out[i]!; K[i] = out[N + i]!; }
    return { J, K };
  }
  const mod = await loadWasm();
  const N = poolSize > 0 ? poolSize : (navigator.hardwareConcurrency ?? 4) - 1;
  const { n, nAux } = df;

  // γ on main thread (shared across workers).
  const gamma = mod.build_gamma_df(n, nAux, df.B, D);

  // Shared B SAB (cached across HF iters by DF identity).
  const bSAB = getBSAB(df);
  const dSAB = toSAB(D);
  const jkSAB = new SharedArrayBuffer(2 * n * n * 8);

  // Round-robin μ partition.
  const muAssignments: number[][] = Array.from({ length: N }, () => []);
  for (let mu = 0; mu < n; mu++) muAssignments[mu % N]!.push(mu);

  const workers = getSharedWorkerPool("wasm", N);
  await Promise.all(workers.map((w, i) => new Promise<void>((resolve, reject) => {
    const onMessage = (ev: MessageEvent): void => {
      w.removeEventListener("message", onMessage);
      if (ev.data?.ok) resolve();
      else reject(new Error(ev.data?.error ?? "worker failed"));
    };
    w.addEventListener("message", onMessage);
    w.postMessage({
      kind: "buildJK-df-slice",
      mus: muAssignments[i]!,
      muStart: 0, muEnd: n,
      n, nAux,
      B: bSAB,
      D: dSAB,
      gamma,
      JK: jkSAB,
    });
  })));

  const view = new Float64Array(jkSAB);
  const NN = n * n;
  const J = new Float64Array(NN);
  const K = new Float64Array(NN);
  J.set(view.subarray(0, NN));
  K.set(view.subarray(NN, 2 * NN));
  return { J, K };
}
