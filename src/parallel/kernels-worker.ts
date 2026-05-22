// Web Worker entrypoint for parallel chemistry kernels.
// Each message is a WorkerTask; we dispatch on `kind` and write results
// into the shared output SAB.

import type { WorkerTask } from "./worker-pool.js";

self.addEventListener("message", (ev: MessageEvent<WorkerTask>) => {
  try {
    const task = ev.data;
    switch (task.kind) {
      case "buildG-row-slice":
        buildGRowSlice(task.muStart, task.muEnd, task.n,
          new Float64Array(task.eri),
          new Float64Array(task.D),
          new Float64Array(task.G));
        break;
      default:
        throw new Error(`unknown kernel kind: ${(task as { kind: string }).kind}`);
    }
    (self as unknown as Worker).postMessage({ ok: true });
  } catch (e) {
    (self as unknown as Worker).postMessage({
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    });
  }
});

/**
 * Compute G[μ, ν] = Σ_{λ, σ} D[λ, σ] · ( (μν|λσ) − ½ (μλ|νσ) )
 * for μ ∈ [muStart, muEnd) and write into the shared G buffer.
 * Other μ ranges are left untouched (assumed handled by sibling workers).
 */
function buildGRowSlice(
  muStart: number, muEnd: number, n: number,
  eri: Float64Array, D: Float64Array, G: Float64Array,
): void {
  for (let mu = muStart; mu < muEnd; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          const Dls = D[la * n + si]!;
          if (Dls === 0) continue;
          const J = eri[((mu * n + nu) * n + la) * n + si]!;
          const K = eri[((mu * n + la) * n + nu) * n + si]!;
          s += Dls * (J - 0.5 * K);
        }
      }
      G[mu * n + nu] = s;
    }
  }
}
