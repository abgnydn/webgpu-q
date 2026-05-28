// Minimal Web Worker pool for parallelizing CPU-bound chemistry kernels.
//
// Workers are spawned lazily on first use and reused across calls (SCF
// iterates many times — spawning per iteration would dominate the cost
// for small molecules). Pool is sized by `navigator.hardwareConcurrency`
// clamped to [2, 16] but caller can override.
//
// Communication model:
//   - Shared inputs (ERI tensor, density matrix) live in
//     SharedArrayBuffers so workers see them without copying.
//   - Outputs (Fock chunks) write into a shared output SAB partitioned
//     by row range so workers never collide.
//   - Each work item is a small JSON message with the row range +
//     dimensions; workers know which kernel to run via `kind`.

export type KernelKind =
  | "buildG-row-slice"
  | "buildG-wasm-mu-slice"
  | "buildJK-df-slice"
  | "eri-row-slice"
  | "eri-wasm-slice"
  | "eri-3idx-wasm-slice";

export interface BuildGRowSliceTask {
  readonly kind: "buildG-row-slice";
  readonly muStart: number;
  readonly muEnd: number;
  readonly n: number;
  readonly eri: SharedArrayBuffer;
  readonly D: SharedArrayBuffer;
  readonly G: SharedArrayBuffer;
}

/** ERI build kernel — each worker computes the (μν|λσ) integrals for
 *  μ in `mus` and writes the 8-fold symmetric positions to the shared
 *  eri SAB. Canonical encoding (μ·n+ν ≤ λ·n+σ) ensures each unique
 *  ERI is owned by exactly one worker → no write races. Using an
 *  explicit μ list (instead of [muStart, muEnd)) lets the dispatcher
 *  distribute rows cyclically — work per μ row decreases monotonically
 *  with μ (canonical encoding), so contiguous chunks are unbalanced. */
export interface ERIRowSliceTask {
  readonly kind: "eri-row-slice";
  /** Indices of μ rows this worker is responsible for. */
  readonly mus: ReadonlyArray<number>;
  /** Carry muStart/muEnd as 0,n for compatibility with the generic
   *  runChunked signature — workers iterate via `mus` instead. */
  readonly muStart: number;
  readonly muEnd: number;
  readonly n: number;
  /** All shells serialized as plain JSON-clonable data. */
  readonly shells: ReadonlyArray<{
    readonly center: readonly [number, number, number];
    readonly alpha: readonly number[];
    readonly c: readonly number[];
    readonly angular: readonly [number, number, number];
  }>;
  /** Output ERI tensor (n⁴ Float64). Worker writes only its slice. */
  readonly eri: SharedArrayBuffer;
  /** Precomputed Schwarz Q table (n² Float64). Read-only. */
  readonly qTable: SharedArrayBuffer;
  /** Schwarz tolerance. */
  readonly schwarzTol: number;
}

/** ERI build kernel — WASM variant. The worker computes the (μν|λσ)
 *  integrals for μ ∈ `mus` via the native-compiled Rust kernel, then
 *  writes the 8-fold symmetric positions into the shared eri SAB.
 *  Shells arrive pre-flattened so the worker can hand them straight to
 *  WASM without per-call serialization. Q-table is precomputed by the
 *  main thread and shared read-only. */
export interface ERIWasmSliceTask {
  readonly kind: "eri-wasm-slice";
  readonly mus: ReadonlyArray<number>;
  /** Unused but kept for runChunked typing. */
  readonly muStart: number;
  readonly muEnd: number;
  readonly n: number;
  /** Flat shell representation (matches the WASM API). */
  readonly nPrimsPerShell: Uint32Array;
  readonly primOffsets: Uint32Array;
  readonly alphaFlat: Float64Array;
  readonly cFlat: Float64Array;
  readonly centerFlat: Float64Array;
  readonly angularFlat: Int32Array;
  /** Output ERI tensor (n⁴ Float64). Worker writes only its slice. */
  readonly eri: SharedArrayBuffer;
  /** Precomputed Schwarz Q table (n² Float64). Read-only. */
  readonly qTable: SharedArrayBuffer;
  readonly schwarzTol: number;
}

/** Fock JK build kernel — WASM per-μ variant. Workers process a list of
 *  μ rows. For each μ they copy that μ's n³ ERI slab into WASM linear
 *  memory and call `fock_one_mu_row` for the G[μ, :] row. This avoids
 *  the doubled-memory cost of caching all the worker's μ slabs in
 *  WASM (which would be |mus|·n³ × 8 B per worker — 207 MB at benzene
 *  cc-pVDZ × 8 workers ≈ 1.65 GB, same as the SAB). The per-μ copy is
 *  ~7 MB/call at n=120; well-amortized by the ~10 ms compute. */
export interface BuildGWasmMuSliceTask {
  readonly kind: "buildG-wasm-mu-slice";
  readonly mus: ReadonlyArray<number>;
  readonly muStart: number;
  readonly muEnd: number;
  readonly n: number;
  readonly eri: SharedArrayBuffer;
  readonly D: SharedArrayBuffer;
  readonly G: SharedArrayBuffer;
}

/** 3-index ERI build slice — aux-basis density fitting. Each worker
 *  computes (μν|P) for μ ∈ `mus` and writes into the shared V tensor
 *  SAB at both (μν, P) and (νμ, P) symmetric positions. Returns no
 *  result; the worker scatters into V directly. */
export interface ERI3idxWasmSliceTask {
  readonly kind: "eri-3idx-wasm-slice";
  readonly mus: ReadonlyArray<number>;
  readonly muStart: number;
  readonly muEnd: number;
  readonly nOrbital: number;
  readonly nAux: number;
  readonly nPrimsOrb: Uint32Array;
  readonly primOffsetsOrb: Uint32Array;
  readonly alphaOrb: Float64Array;
  readonly cOrb: Float64Array;
  readonly centerOrb: Float64Array;
  readonly angularOrb: Int32Array;
  readonly nPrimsAux: Uint32Array;
  readonly primOffsetsAux: Uint32Array;
  readonly alphaAux: Float64Array;
  readonly cAux: Float64Array;
  readonly centerAux: Float64Array;
  readonly angularAux: Int32Array;
  /** Output V tensor: shape n_orbital × n_orbital × n_aux, f64. */
  readonly v: SharedArrayBuffer;
}

/** Parallel buildJK_DF: each worker computes (J, K) for μ ∈ its mus list.
 *  The shared γ vector is computed once (small, n_aux entries) and
 *  cloned to each worker via postMessage. B and D are SAB-shared.
 *  Worker writes J slice + K slice into the shared J, K SAB.
 */
export interface BuildJKDFSliceTask {
  readonly kind: "buildJK-df-slice";
  readonly mus: ReadonlyArray<number>;
  readonly muStart: number;
  readonly muEnd: number;
  readonly n: number;
  readonly nAux: number;
  readonly B: SharedArrayBuffer;
  readonly D: SharedArrayBuffer;
  readonly gamma: Float64Array;
  /** Output: 2·n² Float64 — first half J, second half K. */
  readonly JK: SharedArrayBuffer;
}

export type WorkerTask =
  | BuildGRowSliceTask
  | BuildGWasmMuSliceTask
  | BuildJKDFSliceTask
  | ERIRowSliceTask
  | ERIWasmSliceTask
  | ERI3idxWasmSliceTask;

export interface WorkerPool {
  readonly size: number;
  /** Run a task on the worker pool, partitioned across all workers. */
  runChunked<T extends WorkerTask>(
    template: Omit<T, "muStart" | "muEnd">,
    range: { start: number; end: number },
  ): Promise<void>;
  /** Tear down. */
  dispose(): void;
}

export function createWorkerPool(size?: number): WorkerPool {
  const N = clamp(
    size ?? Math.max(2, Math.floor((globalThis.navigator?.hardwareConcurrency ?? 4) - 1)),
    1,
    16,
  );

  const workers: Worker[] = [];
  for (let i = 0; i < N; i++) {
    workers.push(new Worker(new URL("./kernels-worker.ts", import.meta.url), { type: "module" }));
  }

  return {
    size: N,
    runChunked: (template, range) => {
      const total = range.end - range.start;
      if (total <= 0) return Promise.resolve();
      const baseChunk = Math.max(1, Math.floor(total / N));
      const remainder = total - baseChunk * N;
      const promises: Promise<void>[] = [];
      let cursor = range.start;
      for (let i = 0; i < N; i++) {
        const extra = i < remainder ? 1 : 0;
        const chunkEnd = Math.min(range.end, cursor + baseChunk + extra);
        if (chunkEnd > cursor) {
          const task = { ...template, muStart: cursor, muEnd: chunkEnd } as unknown as WorkerTask;
          promises.push(callWorker(workers[i]!, task));
        }
        cursor = chunkEnd;
      }
      return Promise.all(promises).then(() => undefined);
    },
    dispose: () => {
      for (const w of workers) w.terminate();
      workers.length = 0;
    },
  };
}

function callWorker(worker: Worker, task: WorkerTask): Promise<void> {
  return new Promise((resolve, reject) => {
    const onMessage = (ev: MessageEvent): void => {
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      if (ev.data?.ok) resolve();
      else reject(new Error(ev.data?.error ?? "worker failed"));
    };
    const onError = (ev: ErrorEvent): void => {
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      reject(new Error(ev.message));
    };
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
    worker.postMessage(task);
  });
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/** Check whether SharedArrayBuffer is available (requires COOP/COEP). */
export function sabAvailable(): boolean {
  return typeof SharedArrayBuffer !== "undefined" &&
         typeof crossOriginIsolated !== "undefined" &&
         crossOriginIsolated;
}

/** Copy a Float64Array's bytes into a fresh SharedArrayBuffer (or return
 *  the source SAB if the input already lives in one). */
export function toSAB(arr: Float64Array): SharedArrayBuffer {
  if (arr.buffer instanceof SharedArrayBuffer) {
    return arr.buffer;
  }
  const sab = new SharedArrayBuffer(arr.byteLength);
  new Float64Array(sab).set(arr);
  return sab;
}
