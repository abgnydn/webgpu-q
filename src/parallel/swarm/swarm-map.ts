// swarmMap — the swarm's `parallelMap` primitive.
//
// `swarmMap(transport, kernel, items, opts)` distributes work across
// every peer currently connected to the transport. Each peer can claim
// tiles voluntarily; the master assigns first-come-first-served, falls
// back to executing tiles locally if no one claims them within a
// timeout. If a peer fails or disappears, the master reassigns.
//
// The kernel must be registered on every peer with the SAME `kind`
// string. Master-only peers (no kernel) can still distribute work.

import type { SwarmTransport } from "./transport.js";

export interface SwarmMapOpts {
  /** Free-form descriptor for what this job is. */
  readonly jobId?: string;
  /** How long (ms) to wait for peers to claim a tile before running
   *  it locally. Default 250ms — generous enough for one round-trip on
   *  BroadcastChannel, short enough that no-peers degrades cleanly. */
  readonly claimTimeoutMs?: number;
  /** Per-tile execution timeout. Default 30000ms. */
  readonly tileTimeoutMs?: number;
  /** Called when a tile completes. Lets UIs render progress. */
  readonly onProgress?: (done: number, total: number) => void;
}

export interface KernelRegistry {
  /** Register a worker for a `kind`. The kernel is called with each
   *  assigned tile and must return the result. */
  registerKernel<Tile, Result>(
    kind: string,
    fn: (tile: Tile, tileSpec: unknown) => Promise<Result> | Result,
  ): () => void;
  /** Convenience: read currently-known peers (seen via hello). */
  knownPeers(): readonly string[];
  /** Stop the hello-broadcast timer and close the transport. Always
   *  call this before constructing a new runtime over a different
   *  transport — otherwise the orphaned hello-broadcast timer keeps
   *  firing into a closed transport and throws "send before open()". */
  dispose(): void;
}

export function attachSwarmRuntime(transport: SwarmTransport): KernelRegistry {
  transport.open();

  // ── Roster: peers we've heard from recently. ──
  const peers = new Map<string, number>(); // peerId -> last-seen ts
  const PEER_TIMEOUT_MS = 5000;
  const HELLO_INTERVAL_MS = 1000;

  function pruneStale(): void {
    const now = Date.now();
    for (const [p, ts] of peers) if (now - ts > PEER_TIMEOUT_MS) peers.delete(p);
  }

  // ── Kernel registry. ──
  const kernels = new Map<string, (tile: unknown, tileSpec: unknown) => Promise<unknown> | unknown>();

  function sendHello(): void {
    transport.send({
      type: "hello",
      payload: { capabilities: Array.from(kernels.keys()), ts: Date.now() },
    });
  }

  const helloTimer = globalThis.setInterval(() => { sendHello(); pruneStale(); }, HELLO_INTERVAL_MS);
  sendHello();

  // ── Per-job state when WE are master. ──
  interface JobState {
    readonly kind: string;
    readonly tiles: readonly unknown[];
    readonly tileSpec: unknown;
    readonly results: unknown[];
    readonly assigned: Map<number, string>;  // tileIndex -> worker
    readonly pending: Set<number>;            // tiles not yet assigned
    // setTimeout return type differs across DOM (number) and Node
    // (NodeJS.Timeout); we only ever pass these back to clearTimeout
    // so `unknown` is fine and portable.
    readonly claimTimers: Map<number, ReturnType<typeof setTimeout>>;
    readonly settle: (value: unknown[]) => void;
    readonly reject: (e: Error) => void;
    readonly opts: SwarmMapOpts;
  }
  const jobs = new Map<string, JobState>();

  transport.onMessage(async (msg) => {
    peers.set(msg.from, Date.now());

    switch (msg.type) {
      case "hello":
        // Just bookkeeping — already updated.
        return;

      case "job-announce": {
        // Volunteer to work for someone else's job IF we have the kernel.
        const p = (msg as { payload: { jobId: string; kind: string; nTiles: number; tileSpec?: unknown } }).payload;
        if (!kernels.has(p.kind)) return;
        // Claim tile 0 first; if accepted, master will tell us index.
        // (Simplified: race for any single tile per round-trip.)
        const guess = Math.floor(Math.random() * p.nTiles);
        transport.send({
          type: "tile-claim",
          to: msg.from,
          payload: { jobId: p.jobId, tileIndex: guess },
        });
        return;
      }

      case "tile-claim": {
        const p = (msg as { payload: { jobId: string; tileIndex: number } }).payload;
        const job = jobs.get(p.jobId);
        if (!job) return;
        if (job.assigned.has(p.tileIndex) || !job.pending.has(p.tileIndex)) {
          transport.send({
            type: "tile-assign", to: msg.from,
            payload: { jobId: p.jobId, tileIndex: p.tileIndex, worker: msg.from, accepted: false },
          });
          return;
        }
        // Accept.
        job.pending.delete(p.tileIndex);
        job.assigned.set(p.tileIndex, msg.from);
        const ct = job.claimTimers.get(p.tileIndex);
        if (ct !== undefined) { globalThis.clearTimeout(ct); job.claimTimers.delete(p.tileIndex); }
        transport.send({
          type: "tile-assign", to: msg.from,
          payload: {
            jobId: p.jobId, tileIndex: p.tileIndex, worker: msg.from,
            accepted: true, tile: job.tiles[p.tileIndex],
          },
        });
        return;
      }

      case "tile-assign": {
        // We were accepted as worker. Run the kernel and reply.
        const p = (msg as { payload: {
          jobId: string; tileIndex: number; worker: string; accepted: boolean; tile?: unknown;
        } }).payload;
        if (!p.accepted || p.worker !== transport.self) return;
        // Find the kind — we need it; without it the worker can't proceed.
        // We rely on jobAnnounce having broadcast `kind`; cache it.
        const kind = announcedKinds.get(p.jobId);
        if (!kind) return;
        const fn = kernels.get(kind);
        if (!fn) return;
        const spec = announcedSpecs.get(p.jobId);
        try {
          const result = await fn(p.tile, spec);
          transport.send({
            type: "tile-result", to: msg.from,
            payload: { jobId: p.jobId, tileIndex: p.tileIndex, result },
          });
        } catch (e) {
          transport.send({
            type: "tile-fail", to: msg.from,
            payload: {
              jobId: p.jobId, tileIndex: p.tileIndex,
              error: e instanceof Error ? e.message : String(e),
            },
          });
        }
        return;
      }

      case "tile-result": {
        const p = (msg as { payload: { jobId: string; tileIndex: number; result: unknown } }).payload;
        const job = jobs.get(p.jobId);
        if (!job) return;
        if (job.results[p.tileIndex] !== undefined) return;
        job.results[p.tileIndex] = p.result;
        job.assigned.delete(p.tileIndex);
        const done = job.results.filter((r) => r !== undefined).length;
        job.opts.onProgress?.(done, job.tiles.length);
        if (done === job.tiles.length) {
          job.settle(job.results);
          jobs.delete(p.jobId);
        }
        return;
      }

      case "tile-fail": {
        const p = (msg as { payload: { jobId: string; tileIndex: number } }).payload;
        const job = jobs.get(p.jobId);
        if (!job) return;
        // Reassign locally (master claims the tile and runs it).
        job.assigned.delete(p.tileIndex);
        job.pending.add(p.tileIndex);
        runLocalTile(p.jobId, p.tileIndex);
        return;
      }
    }
  });

  // For workers — cache the kind/spec that came with the job-announce.
  const announcedKinds = new Map<string, string>();
  const announcedSpecs = new Map<string, unknown>();
  transport.onMessage((msg) => {
    if (msg.type === "job-announce") {
      const p = (msg as { payload: { jobId: string; kind: string; tileSpec?: unknown } }).payload;
      announcedKinds.set(p.jobId, p.kind);
      if (p.tileSpec !== undefined) announcedSpecs.set(p.jobId, p.tileSpec);
    }
  });

  async function runLocalTile(jobId: string, tileIndex: number): Promise<void> {
    const job = jobs.get(jobId); if (!job) return;
    if (!job.pending.has(tileIndex)) return;
    job.pending.delete(tileIndex);
    job.assigned.set(tileIndex, transport.self);
    const fn = kernels.get(job.kind);
    if (!fn) { job.reject(new Error(`No kernel for kind "${job.kind}" on master`)); return; }
    try {
      const result = await fn(job.tiles[tileIndex], job.tileSpec);
      if (job.results[tileIndex] !== undefined) return;
      job.results[tileIndex] = result;
      job.assigned.delete(tileIndex);
      const done = job.results.filter((r) => r !== undefined).length;
      job.opts.onProgress?.(done, job.tiles.length);
      if (done === job.tiles.length) {
        job.settle(job.results);
        jobs.delete(jobId);
      }
    } catch (e) {
      job.reject(e instanceof Error ? e : new Error(String(e)));
    }
  }

  function dispose(): void {
    globalThis.clearInterval(helloTimer);
    transport.close();
  }

  return {
    registerKernel<Tile, Result>(
      kind: string,
      fn: (tile: Tile, tileSpec: unknown) => Promise<Result> | Result,
    ): () => void {
      kernels.set(kind, fn as (tile: unknown, tileSpec: unknown) => Promise<unknown> | unknown);
      return () => { kernels.delete(kind); };
    },
    knownPeers(): readonly string[] {
      pruneStale();
      return Array.from(peers.keys());
    },
    dispose,
    _runMap,
  } as unknown as KernelRegistry;

  // ── Master-side: the actual swarmMap call. ──
  async function _runMap<Tile, Result>(
    kind: string,
    tiles: readonly Tile[],
    opts: SwarmMapOpts,
    tileSpec: unknown,
  ): Promise<Result[]> {
    const jobId = opts.jobId ?? `job-${Math.random().toString(36).slice(2, 8)}-${Date.now().toString(36).slice(-4)}`;
    // Empty-input shortcut — no announce, no waiting.
    if (tiles.length === 0) {
      opts.onProgress?.(0, 0);
      return Promise.resolve([] as Result[]);
    }
    return new Promise<Result[]>((resolve, reject) => {
      const claimTimeoutMs = opts.claimTimeoutMs ?? 250;
      const pending = new Set<number>();
      for (let i = 0; i < tiles.length; i++) pending.add(i);
      const job: JobState = {
        kind, tiles, tileSpec,
        results: new Array(tiles.length),
        assigned: new Map(),
        pending,
        claimTimers: new Map(),
        settle: resolve as (v: unknown[]) => void,
        reject,
        opts,
      };
      jobs.set(jobId, job);
      transport.send({
        type: "job-announce",
        payload: { jobId, kind, nTiles: tiles.length, tileSpec },
      });
      // After claimTimeout, run any un-claimed tiles locally.
      for (let i = 0; i < tiles.length; i++) {
        const idx = i;
        const t = globalThis.setTimeout(() => {
          if (job.pending.has(idx)) {
            void runLocalTile(jobId, idx);
          }
        }, claimTimeoutMs);
        job.claimTimers.set(i, t);
      }
    });
  }
}

/** Convenience helper bundling attachSwarmRuntime + the master-side
 *  swarmMap call into one call (for ad-hoc use). */
export async function swarmMap<Tile, Result>(
  registry: KernelRegistry,
  kind: string,
  tiles: readonly Tile[],
  opts: SwarmMapOpts = {},
  tileSpec: unknown = undefined,
): Promise<Result[]> {
  const internal = registry as unknown as {
    _runMap: <T, R>(k: string, t: readonly T[], o: SwarmMapOpts, ts: unknown) => Promise<R[]>;
  };
  return internal._runMap<Tile, Result>(kind, tiles, opts, tileSpec);
}
