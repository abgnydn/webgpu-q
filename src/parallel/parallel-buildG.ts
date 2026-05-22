// Parallel HF buildG via Web Workers + SharedArrayBuffer.
//
// API mirrors the single-threaded `buildG` in hf-scf.ts so SCF can call
// either by feature flag. The pool is module-level so multiple SCF
// iterations within one process reuse the same workers — spawning per
// iteration would dominate for small systems.

import { createWorkerPool, type WorkerPool, toSAB, sabAvailable } from "./worker-pool.js";

let cachedPool: WorkerPool | null = null;
let cachedPoolSize: number | null = null;
let cachedERI: { src: Float64Array; sab: SharedArrayBuffer } | null = null;

function getPool(size: number): WorkerPool {
  if (cachedPool && cachedPoolSize === size) return cachedPool;
  if (cachedPool) cachedPool.dispose();
  cachedPool = createWorkerPool(size);
  cachedPoolSize = size;
  return cachedPool;
}

/** Compute G in parallel. Returns a fresh Float64Array (not SAB-backed). */
export async function buildGParallel(
  D: Float64Array,
  eri_AO: Float64Array,
  n: number,
  poolSize = 0,
): Promise<Float64Array> {
  if (!sabAvailable()) {
    throw new Error("buildGParallel: SharedArrayBuffer unavailable (need COOP/COEP isolation)");
  }
  const pool = getPool(poolSize > 0 ? poolSize : (navigator.hardwareConcurrency ?? 4) - 1);

  // ERI is constant across SCF iterations — cache its SAB.
  let eriSAB: SharedArrayBuffer;
  if (cachedERI && cachedERI.src === eri_AO) {
    eriSAB = cachedERI.sab;
  } else {
    eriSAB = toSAB(eri_AO);
    cachedERI = { src: eri_AO, sab: eriSAB };
  }

  // D changes every iteration → fresh SAB each call.
  const dSAB = toSAB(D);
  // G output SAB; zero-initialized.
  const gSAB = new SharedArrayBuffer(n * n * 8);

  await pool.runChunked(
    { kind: "buildG-row-slice", n, eri: eriSAB, D: dSAB, G: gSAB },
    { start: 0, end: n },
  );

  // Return a regular Float64Array so the rest of the SCF (which uses
  // ArrayBuffer-backed arrays everywhere) doesn't need to special-case
  // SAB-backed inputs.
  const result = new Float64Array(n * n);
  result.set(new Float64Array(gSAB));
  return result;
}

/** Release worker resources. Call when SCF is done. */
export function disposeParallelBuildG(): void {
  if (cachedPool) {
    cachedPool.dispose();
    cachedPool = null;
    cachedPoolSize = null;
  }
  cachedERI = null;
}
