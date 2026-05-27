// Web Worker entrypoint for parallel chemistry kernels.
// Each message is a WorkerTask; we dispatch on `kind` and write results
// into the shared output SAB.

import type { WorkerTask } from "./worker-pool.js";
import { ERI_cg, type CGShell } from "../chemistry/integrals-cg.js";

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
      case "eri-row-slice":
        eriRowSlice(task.mus, task.n,
          task.shells as readonly CGShell[],
          new Float64Array(task.eri),
          new Float64Array(task.qTable),
          task.schwarzTol);
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
 * Compute the canonical ERIs (μν|λσ) for μ ∈ [muStart, muEnd) and
 * write all 8 symmetric positions into the shared eri buffer.
 *
 * Canonical encoding (μ·n+ν ≤ λ·n+σ, ν ≥ μ, σ ≥ λ) guarantees each
 * unique integral is owned by exactly one worker — the one whose μ
 * slice contains the "small-first" μ. So no two workers ever write
 * to the same address; no atomics needed.
 */
function eriRowSlice(
  mus: ReadonlyArray<number>, n: number,
  shells: readonly CGShell[], eri: Float64Array, Q: Float64Array,
  schwarzTol: number,
): void {
  for (const mu of mus) {
    for (let nu = mu; nu < n; nu++) {
      const qMuNu = Q[mu * n + nu]!;
      const pairMuNu = mu * n + nu;
      for (let la = 0; la < n; la++) {
        for (let si = la; si < n; si++) {
          if (pairMuNu > la * n + si) continue;
          if (qMuNu < schwarzTol || qMuNu * Q[la * n + si]! < schwarzTol) continue;
          const v = ERI_cg(shells[mu]!, shells[nu]!, shells[la]!, shells[si]!);
          eri[((mu * n + nu) * n + la) * n + si] = v;
          eri[((nu * n + mu) * n + la) * n + si] = v;
          eri[((mu * n + nu) * n + si) * n + la] = v;
          eri[((nu * n + mu) * n + si) * n + la] = v;
          eri[((la * n + si) * n + mu) * n + nu] = v;
          eri[((si * n + la) * n + mu) * n + nu] = v;
          eri[((la * n + si) * n + nu) * n + mu] = v;
          eri[((si * n + la) * n + nu) * n + mu] = v;
        }
      }
    }
  }
}

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
