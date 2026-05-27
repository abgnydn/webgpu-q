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

  // ── Spawn workers + dispatch. ──
  const workers: Worker[] = [];
  for (let i = 0; i < N; i++) {
    workers.push(new Worker(new URL("../parallel/kernels-worker.ts", import.meta.url), { type: "module" }));
  }
  try {
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
        muStart: 0, muEnd: n, // unused by the eri kernel but in the type
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
  } finally {
    for (const w of workers) w.terminate();
  }
}
