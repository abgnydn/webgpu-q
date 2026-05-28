// Parallel formBFromCholesky — pivoted-Cholesky back-substitution
// split across workers. Outer (μ,ν) loop is fully parallel; inner k
// loop has the serial back-sub dependency on B[μν, j<k] but that's
// per-pair-local.
//
// Memory: V can be huge (312 MB on naphthalene cc-pVDZ). We share it
// via SharedArrayBuffer so workers see it zero-copy. L (~9 MB) too.
// Pivots is tiny — Uint32Array clone. B output is SAB with disjoint
// μ-row writes per worker.
//
// Falls back to single-thread TS when SAB unavailable.

import { sabAvailable, toSAB } from "./worker-pool.js";
import { getSharedWorkerPool } from "./worker-pool-shared.js";

/**
 * Parallel back-substitution computing B[μν, k] = (L_sub^{-1} · V[μν, pivots])_k.
 *
 * @param V          n²·nAux Float64Array, contiguous over P last.
 * @param L          nAux·r lower-triangular factor from pivotedCholesky.
 * @param pivots     length-r selected aux indices.
 * @param n          orbital count.
 * @param nAux       full aux count (= V's third stride).
 * @param poolSize   workers; 0 → hardwareConcurrency-1.
 */
export async function formBFromCholeskyParallel(
  V: Float64Array,
  L: Float64Array,
  pivots: ReadonlyArray<number>,
  n: number,
  nAux: number,
  poolSize = 0,
): Promise<Float64Array> {
  const r = pivots.length;
  if (!sabAvailable()) {
    // Synchronous TS fallback (same algorithm).
    const B = new Float64Array(n * n * r);
    for (let mu = 0; mu < n; mu++) {
      for (let nu = 0; nu < n; nu++) {
        const vBase = (mu * n + nu) * nAux;
        const bBase = (mu * n + nu) * r;
        for (let k = 0; k < r; k++) {
          const pivK = pivots[k]!;
          let s = V[vBase + pivK]!;
          const lRow = pivK * r;
          for (let j = 0; j < k; j++) {
            s -= L[lRow + j]! * B[bBase + j]!;
          }
          B[bBase + k] = s / L[lRow + k]!;
        }
      }
    }
    return B;
  }

  const N = poolSize > 0 ? poolSize : Math.max(1, (navigator.hardwareConcurrency ?? 4) - 1);
  const vSAB = toSAB(V);
  const lSAB = toSAB(L);
  const bSAB = new SharedArrayBuffer(n * n * r * 8);

  // Round-robin μ partition. Per-μ work is constant in μ (always
  // n·r²/2 FLOPs), so contiguous chunks would balance fine too; the
  // round-robin keeps consistency with parallel-jk-df.ts.
  const muAssignments: number[][] = Array.from({ length: N }, () => []);
  for (let mu = 0; mu < n; mu++) muAssignments[mu % N]!.push(mu);

  const pivotsU32 = new Uint32Array(pivots);

  const workers = getSharedWorkerPool("wasm", N);
  await Promise.all(workers.map((w, i) => new Promise<void>((resolve, reject) => {
    const onMessage = (ev: MessageEvent): void => {
      w.removeEventListener("message", onMessage);
      if (ev.data?.ok) resolve();
      else reject(new Error(ev.data?.error ?? "worker failed"));
    };
    w.addEventListener("message", onMessage);
    w.postMessage({
      kind: "form-b-cholesky-slice",
      mus: muAssignments[i]!,
      muStart: 0, muEnd: n,
      n, nAux, r,
      V: vSAB,
      L: lSAB,
      pivots: pivotsU32,
      B: bSAB,
    });
  })));

  // Return an SAB-backed view directly: parallel-jk-df.ts caches
  // df.B's underlying SAB by identity. If df.B were a regular
  // Float64Array, the first SCF iter would copy the 303 MB B-tensor
  // into a fresh SAB (~150 ms one-time, but mostly we want zero-copy
  // throughout SCF). Returning the SAB-backed view here also keeps
  // the worker-shared buffer alive without an extra reference.
  return new Float64Array(bSAB);
}
