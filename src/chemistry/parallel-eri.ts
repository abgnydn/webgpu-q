// Parallel ERI build via Web Workers + SharedArrayBuffer.
//
// The dense n⁴ ERI tensor is embarrassingly parallel over the outer
// μ-row: canonical encoding (μ·n+ν ≤ λ·n+σ) guarantees each unique
// integral is owned by exactly one worker, so we can partition by μ
// and have workers write the 8 symmetric positions of their slice
// without any atomic operations or merging.
//
// Schwarz Q table is precomputed serially on the main thread (cheap,
// n² ERIs at "self" position) and shared as a read-only SAB.

import { ERI_cg, type CGShell } from "./integrals-cg.js";
import { sabAvailable, toSAB } from "../parallel/worker-pool.js";
import { getSharedWorkerPool, disposeAllSharedPools } from "../parallel/worker-pool-shared.js";
import { schwarzQTableWasm } from "./wasm-eri.js";

// Pool spawning is now shared across all parallel chemistry kernels
// (ERI build, JK build, …). See ../parallel/worker-pool-shared.ts.
// Re-exported under the old name for callers that imported it.
export const disposeAllPools = disposeAllSharedPools;

/**
 * Compute the dense AO ERI tensor in parallel across N workers.
 * Returns a fresh Float64Array of length n⁴ (not SAB-backed).
 *
 * - `shells`: the full shell list (n total).
 * - `schwarzTol`: drop pairs with Cauchy-Schwarz upper bound below this.
 * - `poolSize`: how many workers to spawn. 0 → use hardwareConcurrency-1.
 *
 * Honest behavior: when SAB isn't available (no COOP/COEP isolation)
 * this throws — callers should check `sabAvailable()` first and fall
 * back to the sync path.
 */
export async function buildERIParallel(
  shells: readonly CGShell[],
  n: number,
  schwarzTol = 1e-10,
  poolSize = 0,
): Promise<Float64Array> {
  if (!sabAvailable()) {
    throw new Error("buildERIParallel: SharedArrayBuffer unavailable (need COOP/COEP isolation)");
  }
  const N = poolSize > 0 ? poolSize : (navigator.hardwareConcurrency ?? 4) - 1;

  // ── Schwarz Q table (serial, cheap). ──
  const Q = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu; nu < n; nu++) {
      const v = ERI_cg(shells[mu]!, shells[nu]!, shells[mu]!, shells[nu]!);
      const q = Math.sqrt(Math.abs(v));
      Q[mu * n + nu] = q;
      Q[nu * n + mu] = q;
    }
  }
  const qSAB = toSAB(Q);
  const eriSAB = new SharedArrayBuffer(n * n * n * n * 8);

  const serializedShells = shells.map((s) => ({
    center: s.center,
    alpha: s.alpha,
    c: s.c,
    angular: s.angular,
  }));

  // ── Cyclic μ distribution. Work per μ row ∝ (n−μ) under canonical
  // encoding, so contiguous chunks give worker 0 ~1.8× the work of
  // worker N-1. Round-robin assignment {0, N, 2N, …} for worker 0,
  // {1, N+1, 2N+1, …} for worker 1, etc. evens it out.
  const muAssignments: number[][] = Array.from({ length: N }, () => []);
  for (let mu = 0; mu < n; mu++) {
    muAssignments[mu % N]!.push(mu);
  }

  // ── Persistent worker pool (TS kernel). ──
  const workers = getSharedWorkerPool("ts", N);
  await Promise.all(workers.map((w, i) => new Promise<void>((resolve, reject) => {
    const onMessage = (ev: MessageEvent): void => {
      w.removeEventListener("message", onMessage);
      if (ev.data?.ok) resolve();
      else reject(new Error(ev.data?.error ?? "worker failed"));
    };
    w.addEventListener("message", onMessage);
    w.postMessage({
      kind: "eri-row-slice",
      mus: muAssignments[i]!,
      muStart: 0, muEnd: n,
      n,
      shells: serializedShells,
      eri: eriSAB,
      qTable: qSAB,
      schwarzTol,
    });
  })));

  const out = new Float64Array(n * n * n * n);
  out.set(new Float64Array(eriSAB));
  return out;
}

/** Pack a shell list into the flat-array layout the Rust kernel expects.
 *  Same shape as src/chemistry/wasm-eri.ts → packShells(), duplicated to
 *  keep parallel-eri.ts independent. */
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
 * WASM × Workers compound. Each worker loads its own wasm-eri instance
 * and computes its μ-slice in native code, then writes the 8 symmetric
 * positions to the shared n⁴ ERI SAB. Multiplies the WASM single-thread
 * win (~4.7× on benzene) with the Worker parallelism (~2-3× on browsers
 * with ≥8 logical cores).
 *
 * Schwarz Q-table is built once on the main thread (n² ERIs, cheap) and
 * shared as a read-only SAB.
 */
export async function buildERIWasmParallel(
  shells: readonly CGShell[],
  n: number,
  schwarzTol = 1e-10,
  poolSize = 0,
): Promise<Float64Array> {
  if (!sabAvailable()) {
    throw new Error("buildERIWasmParallel: SharedArrayBuffer unavailable (need COOP/COEP isolation)");
  }
  const N = poolSize > 0 ? poolSize : (navigator.hardwareConcurrency ?? 4) - 1;

  // ── Schwarz Q-table (WASM single-thread, main thread). 4.7× faster
  // than the TS path on benzene (~50 ms vs ~225 ms). Cheap enough at
  // n² that parallelizing the table itself isn't worth the message
  // ping-pong. ──
  const Q = await schwarzQTableWasm(shells);
  const qSAB = toSAB(Q);
  const eriSAB = new SharedArrayBuffer(n * n * n * n * 8);

  // Flatten shells once for all workers.
  const packed = packShells(shells);

  // Round-robin μ distribution (same logic as TS-only parallel-eri).
  const muAssignments: number[][] = Array.from({ length: N }, () => []);
  for (let mu = 0; mu < n; mu++) {
    muAssignments[mu % N]!.push(mu);
  }

  // ── Persistent worker pool (WASM kernel). ──
  const workers = getSharedWorkerPool("wasm", N);
  await Promise.all(workers.map((w, i) => new Promise<void>((resolve, reject) => {
    const onMessage = (ev: MessageEvent): void => {
      w.removeEventListener("message", onMessage);
      if (ev.data?.ok) resolve();
      else reject(new Error(ev.data?.error ?? "worker failed"));
    };
    w.addEventListener("message", onMessage);
    w.postMessage({
      kind: "eri-wasm-slice",
      mus: muAssignments[i]!,
      muStart: 0, muEnd: n,
      n,
      nPrimsPerShell: packed.nPrimsPerShell,
      primOffsets: packed.primOffsets,
      alphaFlat: packed.alphaFlat,
      cFlat: packed.cFlat,
      centerFlat: packed.centerFlat,
      angularFlat: packed.angularFlat,
      eri: eriSAB,
      qTable: qSAB,
      schwarzTol,
    });
  })));

  const out = new Float64Array(n * n * n * n);
  out.set(new Float64Array(eriSAB));
  return out;
}
