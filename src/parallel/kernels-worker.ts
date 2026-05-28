// Web Worker entrypoint for parallel chemistry kernels.
// Each message is a WorkerTask; we dispatch on `kind` and write results
// into the shared output SAB.

import type { WorkerTask, ERIWasmSliceTask, BuildGWasmMuSliceTask, ERI3idxWasmSliceTask, BuildJKDFSliceTask, FormBTensorSliceTask, FormBCholeskySliceTask } from "./worker-pool.js";
import { ERI_cg, type CGShell } from "../chemistry/integrals-cg.js";

interface WasmEriModule {
  default(): Promise<unknown>;
  eri_build_slice(
    mus: Uint32Array,
    nShells: number,
    nPrimsPerShell: Uint32Array,
    primOffsets: Uint32Array,
    alphaFlat: Float64Array,
    cFlat: Float64Array,
    centerFlat: Float64Array,
    angularFlat: Int32Array,
    qTable: Float64Array,
    schwarzTol: number,
  ): Float64Array;
  fock_one_mu_row(
    n: number,
    eriMuRow: Float64Array,
    d: Float64Array,
  ): Float64Array;
  eri_3idx_build_slice(
    mus: Uint32Array,
    nOrbital: number, nAux: number,
    nPrimsOrb: Uint32Array, primOffOrb: Uint32Array,
    alphaOrb: Float64Array, cOrb: Float64Array,
    centerOrb: Float64Array, angularOrb: Int32Array,
    nPrimsAux: Uint32Array, primOffAux: Uint32Array,
    alphaAux: Float64Array, cAux: Float64Array,
    centerAux: Float64Array, angularAux: Int32Array,
  ): Float64Array;
  build_jk_df_slice(
    mus: Uint32Array,
    n: number, nAux: number,
    b: Float64Array, d: Float64Array,
    gamma: Float64Array,
  ): Float64Array;
  form_b_tensor_slice(
    mus: Uint32Array,
    n: number, nAux: number,
    v: Float64Array, u: Float64Array,
    invSqrtLam: Float64Array,
  ): Float64Array;
}

// Per-worker WASM module instance. Lazy-loaded on first eri-wasm-slice
// task; reused across subsequent calls.
let workerWasm: WasmEriModule | null = null;
let workerWasmInit: Promise<WasmEriModule> | null = null;
async function loadWorkerWasm(): Promise<WasmEriModule> {
  if (workerWasm) return workerWasm;
  if (workerWasmInit) return workerWasmInit;
  workerWasmInit = (async () => {
    const mod = await import(
      /* @vite-ignore */
      "../../wasm-eri/pkg/wasm_eri.js" as string,
    ) as WasmEriModule;
    await mod.default();
    workerWasm = mod;
    return mod;
  })();
  return workerWasmInit;
}

self.addEventListener("message", (ev: MessageEvent<WorkerTask>) => {
  const task = ev.data;
  const handle = async (): Promise<void> => {
    switch (task.kind) {
      case "buildG-row-slice":
        buildGRowSlice(task.muStart, task.muEnd, task.n,
          new Float64Array(task.eri),
          new Float64Array(task.D),
          new Float64Array(task.G));
        return;
      case "eri-row-slice":
        eriRowSlice(task.mus, task.n,
          task.shells as readonly CGShell[],
          new Float64Array(task.eri),
          new Float64Array(task.qTable),
          task.schwarzTol);
        return;
      case "eri-wasm-slice":
        await eriWasmSlice(task);
        return;
      case "buildG-wasm-mu-slice":
        await buildGWasmMuSlice(task);
        return;
      case "eri-3idx-wasm-slice":
        await eri3idxWasmSlice(task);
        return;
      case "buildJK-df-slice":
        await buildJKDFSlice(task);
        return;
      case "form-b-tensor-slice":
        await formBTensorSlice(task);
        return;
      case "form-b-cholesky-slice":
        formBCholeskySlice(task);
        return;
      default:
        throw new Error(`unknown kernel kind: ${(task as { kind: string }).kind}`);
    }
  };
  handle().then(
    () => (self as unknown as Worker).postMessage({ ok: true }),
    (e: unknown) => (self as unknown as Worker).postMessage({
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    }),
  );
});

/** Pure-TS back-substitution on a μ slice. V8 JITs the hot loop into
 *  reasonably tight code; the win over single-thread is pure
 *  parallelism (N_workers× on what's a compute-bound kernel at ~1
 *  GFLOP/s per thread). Cache locality of L's piv_k-indirect access is
 *  the same as TS path; we don't try to fix it here. */
function formBCholeskySlice(task: FormBCholeskySliceTask): void {
  const { mus, n, nAux, r, pivots } = task;
  const V = new Float64Array(task.V);
  const L = new Float64Array(task.L);
  const B = new Float64Array(task.B);
  for (const mu of mus) {
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
}

async function formBTensorSlice(task: FormBTensorSliceTask): Promise<void> {
  const mod = await loadWorkerWasm();
  const mus = new Uint32Array(task.mus);
  const out = mod.form_b_tensor_slice(
    mus, task.n, task.nAux, task.V, task.U, task.invSqrtLam,
  );
  // out shape: (|mus| · n · nAux). Scatter into shared B at μ-rows.
  const B = new Float64Array(task.B);
  const n = task.n;
  const nAux = task.nAux;
  const kMus = task.mus.length;
  for (let k = 0; k < kMus; k++) {
    const mu = task.mus[k]!;
    for (let nu = 0; nu < n; nu++) {
      const srcBase = (k * n + nu) * nAux;
      const dstBase = (mu * n + nu) * nAux;
      for (let P = 0; P < nAux; P++) {
        B[dstBase + P] = out[srcBase + P]!;
      }
    }
  }
}

async function buildJKDFSlice(task: BuildJKDFSliceTask): Promise<void> {
  const mod = await loadWorkerWasm();
  const mus = new Uint32Array(task.mus);
  const b = new Float64Array(task.B);
  const d = new Float64Array(task.D);
  const out = mod.build_jk_df_slice(mus, task.n, task.nAux, b, d, task.gamma);
  // out = [J_slice (k_mus·n); K_slice (k_mus·n)]
  const JK = new Float64Array(task.JK);
  const kMus = task.mus.length;
  const n = task.n;
  for (let k = 0; k < kMus; k++) {
    const mu = task.mus[k]!;
    for (let nu = 0; nu < n; nu++) {
      JK[mu * n + nu] = out[k * n + nu]!;            // J
      JK[n * n + mu * n + nu] = out[kMus * n + k * n + nu]!;  // K
    }
  }
}

async function eri3idxWasmSlice(task: ERI3idxWasmSliceTask): Promise<void> {
  const mod = await loadWorkerWasm();
  const mus = new Uint32Array(task.mus);
  const packed = mod.eri_3idx_build_slice(
    mus,
    task.nOrbital, task.nAux,
    task.nPrimsOrb, task.primOffsetsOrb,
    task.alphaOrb, task.cOrb, task.centerOrb, task.angularOrb,
    task.nPrimsAux, task.primOffsetsAux,
    task.alphaAux, task.cAux, task.centerAux, task.angularAux,
  );
  // packed = [μ, ν, P, value, …] of length 4·K. Scatter both (μν, P)
  // and (νμ, P) symmetric positions into the shared V SAB.
  const V = new Float64Array(task.v);
  const n = task.nOrbital;
  const nx = task.nAux;
  const K = packed.length / 4;
  for (let k = 0; k < K; k++) {
    const base = k * 4;
    const mu = packed[base]! | 0;
    const nu = packed[base + 1]! | 0;
    const P = packed[base + 2]! | 0;
    const v = packed[base + 3]!;
    V[(mu * n + nu) * nx + P] = v;
    if (mu !== nu) V[(nu * n + mu) * nx + P] = v;
  }
}

async function buildGWasmMuSlice(task: BuildGWasmMuSliceTask): Promise<void> {
  const mod = await loadWorkerWasm();
  const n = task.n;
  const n3 = n * n * n;
  const eri = new Float64Array(task.eri);
  const d = new Float64Array(task.D);
  const G = new Float64Array(task.G);
  for (const mu of task.mus) {
    const eriRow = eri.subarray(mu * n3, (mu + 1) * n3);
    const gRow = mod.fock_one_mu_row(n, eriRow, d);
    const gBase = mu * n;
    for (let nu = 0; nu < n; nu++) {
      G[gBase + nu] = gRow[nu]!;
    }
  }
}

async function eriWasmSlice(task: ERIWasmSliceTask): Promise<void> {
  const mod = await loadWorkerWasm();
  const eri = new Float64Array(task.eri);
  const qTable = new Float64Array(task.qTable);
  const mus = new Uint32Array(task.mus);
  const packed = mod.eri_build_slice(
    mus,
    task.n,
    task.nPrimsPerShell,
    task.primOffsets,
    task.alphaFlat,
    task.cFlat,
    task.centerFlat,
    task.angularFlat,
    qTable,
    task.schwarzTol,
  );
  // packed = [mu, nu, la, si, v, mu, nu, la, si, v, ...]
  // Write 8 symmetric positions for each canonical ERI.
  const n = task.n;
  const n2 = n * n;
  const n3 = n2 * n;
  const K = packed.length / 5;
  for (let k = 0; k < K; k++) {
    const base = k * 5;
    const mu = packed[base]! | 0;
    const nu = packed[base + 1]! | 0;
    const la = packed[base + 2]! | 0;
    const si = packed[base + 3]! | 0;
    const v = packed[base + 4]!;
    eri[mu * n3 + nu * n2 + la * n + si] = v;
    eri[nu * n3 + mu * n2 + la * n + si] = v;
    eri[mu * n3 + nu * n2 + si * n + la] = v;
    eri[nu * n3 + mu * n2 + si * n + la] = v;
    eri[la * n3 + si * n2 + mu * n + nu] = v;
    eri[si * n3 + la * n2 + mu * n + nu] = v;
    eri[la * n3 + si * n2 + nu * n + mu] = v;
    eri[si * n3 + la * n2 + nu * n + mu] = v;
  }
}

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
