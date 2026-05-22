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

export type KernelKind = "buildG-row-slice";

export interface BuildGRowSliceTask {
  readonly kind: "buildG-row-slice";
  readonly muStart: number;
  readonly muEnd: number;
  readonly n: number;
  readonly eri: SharedArrayBuffer;
  readonly D: SharedArrayBuffer;
  readonly G: SharedArrayBuffer;
}

export type WorkerTask = BuildGRowSliceTask;

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
          const task = { ...template, muStart: cursor, muEnd: chunkEnd } as WorkerTask;
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
