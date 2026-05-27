// Shared worker pool registry. Multiple parallel kernels (ERI build,
// JK build, etc.) reuse the same physical Worker instances — the
// kernels-worker.ts dispatcher routes on task `kind`. Avoids
// double-spawning + double-WASM-init when one HF call exercises both
// the ERI build and the JK build paths.

type PoolKind = "ts" | "wasm";
const poolCache: Map<string, Worker[]> = new Map();

function poolKey(kind: PoolKind, n: number): string { return `${kind}:${n}`; }

/** Get or spawn a worker pool of the given size, keyed by (kind, size).
 *  Pools are persistent for the page lifetime unless explicitly disposed. */
export function getSharedWorkerPool(kind: PoolKind, n: number): Worker[] {
  const key = poolKey(kind, n);
  const existing = poolCache.get(key);
  if (existing) return existing;
  const workers: Worker[] = [];
  for (let i = 0; i < n; i++) {
    workers.push(new Worker(
      new URL("./kernels-worker.ts", import.meta.url),
      { type: "module" },
    ));
  }
  poolCache.set(key, workers);
  return workers;
}

/** Tear down all cached worker pools. */
export function disposeAllSharedPools(): void {
  for (const workers of poolCache.values()) {
    for (const w of workers) w.terminate();
  }
  poolCache.clear();
}
