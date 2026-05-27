// WASM × Workers JK build. Per HF SCF iter, each worker processes a
// subset of μ rows. The ERI tensor is SAB-shared (zero copy across the
// worker boundary); each worker copies one μ's n³ slab into WASM linear
// memory per call (via wasm-bindgen `&[f64]`) and runs `fock_one_mu_row`.
//
// Why per-μ copy and not per-worker slab cache: caching all of a
// worker's μ slabs in WASM memory doubles browser memory pressure
// (full ERI = ~1.65 GB on benzene cc-pVDZ; caching adds another
// 1.65 GB across 8 workers). Per-μ copy is ~7 MB at n=120, well
// amortized by the ~10 ms compute that follows.

import { sabAvailable, toSAB } from "./worker-pool.js";
import { getSharedWorkerPool } from "./worker-pool-shared.js";

// Per-molecule cache: ERI SAB + μ assignments. Workers are pulled from
// the shared (kind, size) pool — same instances as buildERIWasmParallel.
type CachedState = {
  size: number;
  muAssignments: number[][];
  eriSAB: SharedArrayBuffer;
  n: number;
  eriIdentity: Float64Array | null;
};

let cached: CachedState | null = null;

function getState(size: number, n: number, eri: Float64Array): CachedState {
  if (
    cached &&
    cached.size === size &&
    cached.n === n &&
    cached.eriIdentity === eri
  ) {
    return cached;
  }
  // Pool size or molecule changed — rebuild μ assignments + SAB.
  const muAssignments: number[][] = Array.from({ length: size }, () => []);
  for (let mu = 0; mu < n; mu++) {
    muAssignments[mu % size]!.push(mu);
  }
  const eriSAB = toSAB(eri);
  cached = { size, muAssignments, eriSAB, n, eriIdentity: eri };
  return cached;
}

/** Compute the Fock G matrix in parallel via WASM-per-μ workers.
 *  Returns a fresh n × n Float64Array (not SAB-backed). */
export async function buildGWasmParallel(
  D: Float64Array,
  eri_AO: Float64Array,
  n: number,
  poolSize = 0,
): Promise<Float64Array> {
  if (!sabAvailable()) {
    throw new Error("buildGWasmParallel: SharedArrayBuffer unavailable");
  }
  const N = poolSize > 0 ? poolSize : (navigator.hardwareConcurrency ?? 4) - 1;
  const state = getState(N, n, eri_AO);
  const workers = getSharedWorkerPool("wasm", N);

  // D changes every iter → fresh SAB.
  const dSAB = toSAB(D);
  const gSAB = new SharedArrayBuffer(n * n * 8);

  await Promise.all(workers.map((w, i) => new Promise<void>((resolve, reject) => {
    const onMessage = (ev: MessageEvent): void => {
      w.removeEventListener("message", onMessage);
      if (ev.data?.ok) resolve();
      else reject(new Error(ev.data?.error ?? "worker failed"));
    };
    w.addEventListener("message", onMessage);
    w.postMessage({
      kind: "buildG-wasm-mu-slice",
      mus: state.muAssignments[i]!,
      muStart: 0, muEnd: n,
      n,
      eri: state.eriSAB,
      D: dSAB,
      G: gSAB,
    });
  })));

  const out = new Float64Array(n * n);
  out.set(new Float64Array(gSAB));
  return out;
}

/** Reset the cached (ERI, μ assignments) state. Does NOT terminate
 *  workers — those live in the shared pool and may be in use by other
 *  kernels. Call `disposeAllSharedPools()` to actually free them. */
export function disposeParallelBuildGWasm(): void {
  cached = null;
}
