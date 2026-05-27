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

// Round-robin μ pool reused across HF iterations.
type CachedPool = {
  workers: Worker[];
  size: number;
  muAssignments: number[][];
  eriSAB: SharedArrayBuffer;
  n: number;
  /** Identity of the ERI tensor — different ERI ⇒ rebuild SAB. */
  eriIdentity: Float64Array | null;
};

let cached: CachedPool | null = null;

function getPool(size: number, n: number, eri: Float64Array): CachedPool {
  if (
    cached &&
    cached.size === size &&
    cached.n === n &&
    cached.eriIdentity === eri
  ) {
    return cached;
  }
  // Pool size or molecule changed — rebuild.
  if (cached) {
    for (const w of cached.workers) w.terminate();
  }
  const workers: Worker[] = [];
  for (let i = 0; i < size; i++) {
    workers.push(new Worker(
      new URL("./kernels-worker.ts", import.meta.url),
      { type: "module" },
    ));
  }
  const muAssignments: number[][] = Array.from({ length: size }, () => []);
  for (let mu = 0; mu < n; mu++) {
    muAssignments[mu % size]!.push(mu);
  }
  const eriSAB = toSAB(eri);
  cached = {
    workers, size, muAssignments, eriSAB, n,
    eriIdentity: eri,
  };
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
  const pool = getPool(N, n, eri_AO);

  // D changes every iter → fresh SAB.
  const dSAB = toSAB(D);
  const gSAB = new SharedArrayBuffer(n * n * 8);

  await Promise.all(pool.workers.map((w, i) => new Promise<void>((resolve, reject) => {
    const onMessage = (ev: MessageEvent): void => {
      w.removeEventListener("message", onMessage);
      if (ev.data?.ok) resolve();
      else reject(new Error(ev.data?.error ?? "worker failed"));
    };
    w.addEventListener("message", onMessage);
    w.postMessage({
      kind: "buildG-wasm-mu-slice",
      mus: pool.muAssignments[i]!,
      muStart: 0, muEnd: n,
      n,
      eri: pool.eriSAB,
      D: dSAB,
      G: gSAB,
    });
  })));

  const out = new Float64Array(n * n);
  out.set(new Float64Array(gSAB));
  return out;
}

/** Tear down cached worker pool. */
export function disposeParallelBuildGWasm(): void {
  if (cached) {
    for (const w of cached.workers) w.terminate();
    cached = null;
  }
}
