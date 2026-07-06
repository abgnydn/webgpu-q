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

import type { SwarmTransport, PeerId } from "./transport.js";

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
  /** Cost-aware scheduling (LPT — longest-processing-time-first). When given, the
   *  master hands tiles out heaviest-first instead of FIFO, so a heavy tile starts
   *  at t=0 and overlaps the light ones rather than tailing past everyone when
   *  pulled late. Pure ordering hint — does NOT change results (independent tiles).
   *  Cost only needs to be monotonic in real runtime (e.g. basis-function count). */
  readonly costFn?: (tile: unknown, index: number) => number;
}

/** Compute-only makespan of the greedy-pull scheduler: k workers, tiles consumed
 *  in `costsInOrder`, each worker taking the next tile when it goes free. Returns
 *  the finish time of the slowest worker. FIFO = original order; LPT = costs sorted
 *  descending. Models the parallel CEILING (ignores comms/scheduler overhead), so
 *  it's the throughput analogue of the (T) S/C decomposition — runnable without
 *  real tabs. speedup = Σcost / makespan, efficiency = speedup / k. */
export function simulateGreedyMakespan(costsInOrder: readonly number[], k: number): number {
  const nWorkers = Math.max(1, k);
  const freeAt = new Array<number>(nWorkers).fill(0);
  for (const c of costsInOrder) {
    let m = 0;
    for (let w = 1; w < nWorkers; w++) if (freeAt[w]! < freeAt[m]!) m = w;
    freeAt[m] = freeAt[m]! + c;
  }
  return Math.max(...freeAt);
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
  // Scheduling model = greedy pull queue (dynamic load balancing). The master
  // owns `pending` (a FIFO of tile indices); every peer, the master included,
  // pulls ONE tile at a time and asks for another only after finishing. A peer
  // can therefore hold at most one tile in flight, so a slow tile parks only its
  // own puller while everyone else keeps draining the queue — which balances
  // uneven tile costs automatically (the old single-claim-per-worker protocol
  // could not: workers grabbed one tile then idled, leaving the master to run
  // the rest via a timeout fallback → master-heavy, no real parallelism).
  interface JobState {
    readonly kind: string;
    readonly tiles: readonly unknown[];
    readonly tileSpec: unknown;
    readonly results: unknown[];
    readonly assigned: Map<number, string>;  // tileIndex -> worker currently running it
    readonly assignedAt: Map<number, number>; // tileIndex -> ts we handed it to a remote worker
    readonly pending: number[];               // FIFO queue of unassigned tile indices
    readonly settle: (value: unknown[]) => void;
    readonly reject: (e: Error) => void;
    readonly opts: SwarmMapOpts;
    /** Flips true once any remote worker pulls a tile — tells the master pump to
     *  keep yielding (workers-first) instead of draining the queue locally. */
    remoteActive: boolean;
  }
  const jobs = new Map<string, JobState>();

  // Worker-side cache of the kind/spec each job-announce carried.
  const announcedKinds = new Map<string, string>();
  const announcedSpecs = new Map<string, unknown>();

  const delay = (ms: number): Promise<void> => new Promise((r) => globalThis.setTimeout(r, ms));

  // Master: hand the next queued tile to `worker`, or signal the queue is drained.
  function serveRequest(jobId: string, worker: PeerId): void {
    const job = jobs.get(jobId);
    if (!job) return;
    job.remoteActive = true;       // a live remote worker is pulling — keep yielding to it
    const idx = job.pending.shift();
    if (idx === undefined) {
      transport.send({ type: "tile-assign", to: worker, payload: { jobId, worker, accepted: false, done: true } });
      return;
    }
    job.assigned.set(idx, worker);
    job.assignedAt.set(idx, Date.now());
    transport.send({ type: "tile-assign", to: worker, payload: { jobId, tileIndex: idx, worker, accepted: true, tile: job.tiles[idx] } });
  }

  // Master: record a finished tile; settle the whole job when the last one lands.
  function recordResult(jobId: string, tileIndex: number, result: unknown): void {
    const job = jobs.get(jobId);
    if (!job) return;
    if (job.results[tileIndex] !== undefined) return;  // duplicate / already settled
    job.results[tileIndex] = result;
    job.assigned.delete(tileIndex);
    job.assignedAt.delete(tileIndex);
    const done = job.results.filter((r) => r !== undefined).length;
    job.opts.onProgress?.(done, job.tiles.length);
    if (done === job.tiles.length) { job.settle(job.results); jobs.delete(jobId); }
  }

  transport.onMessage(async (msg) => {
    peers.set(msg.from, Date.now());

    switch (msg.type) {
      case "hello":
        return;

      case "job-announce": {
        // Worker side: cache the job, then start pulling tiles if we can run them.
        const p = (msg as { payload: { jobId: string; kind: string; tileSpec?: unknown } }).payload;
        announcedKinds.set(p.jobId, p.kind);
        if (p.tileSpec !== undefined) announcedSpecs.set(p.jobId, p.tileSpec);
        if (!kernels.has(p.kind)) return;
        transport.send({ type: "tile-request", to: msg.from, payload: { jobId: p.jobId } });
        return;
      }

      case "tile-request": {
        // Master side: give the asker the next tile (the greedy-pull handout).
        const p = (msg as { payload: { jobId: string } }).payload;
        serveRequest(p.jobId, msg.from);
        return;
      }

      case "tile-assign": {
        // Worker side: we were handed a tile (or told the queue is drained).
        const p = (msg as { payload: {
          jobId: string; tileIndex?: number; worker: string; accepted: boolean; done?: boolean; tile?: unknown;
        } }).payload;
        if (p.worker !== transport.self) return;            // addressed to another peer
        if (!p.accepted || p.done || p.tileIndex === undefined) return; // no more work → stop pulling
        const kind = announcedKinds.get(p.jobId);
        const fn = kind ? kernels.get(kind) : undefined;
        if (!fn) return;
        const spec = announcedSpecs.get(p.jobId);
        try {
          const result = await fn(p.tile, spec);
          transport.send({ type: "tile-result", to: msg.from, payload: { jobId: p.jobId, tileIndex: p.tileIndex, result } });
        } catch (e) {
          transport.send({ type: "tile-fail", to: msg.from, payload: { jobId: p.jobId, tileIndex: p.tileIndex, error: e instanceof Error ? e.message : String(e) } });
        }
        // Immediately pull the next tile — keep working until the queue is empty.
        transport.send({ type: "tile-request", to: msg.from, payload: { jobId: p.jobId } });
        return;
      }

      case "tile-result": {
        const p = (msg as { payload: { jobId: string; tileIndex: number; result: unknown } }).payload;
        recordResult(p.jobId, p.tileIndex, p.result);
        return;
      }

      case "tile-fail": {
        // A worker's kernel threw on this tile. Do NOT requeue it to the shared
        // pool: a worker whose kernel always fails would instantly re-pull and
        // re-fail the same tile in a tight loop, starving everyone else. Instead
        // the master runs it itself (its kernel presumably works); if the master
        // also can't, the job rejects rather than hanging.
        const p = (msg as { payload: { jobId: string; tileIndex: number } }).payload;
        void runLocally(p.jobId, p.tileIndex);
        return;
      }
    }
  });

  // Master runs one tile directly (recovery path for a failed remote tile, or
  // any tile when there are no workers). A failure of the master's OWN kernel is
  // unrecoverable → reject the job instead of looping forever.
  async function runLocally(jobId: string, idx: number): Promise<void> {
    const job = jobs.get(jobId);
    if (!job || job.results[idx] !== undefined) return;
    const fn = kernels.get(job.kind);
    if (!fn) {
      job.reject(new Error(`tile ${idx} failed and master has no "${job.kind}" kernel to recover`));
      jobs.delete(jobId);
      return;
    }
    job.assigned.set(idx, transport.self);
    try {
      recordResult(jobId, idx, await fn(job.tiles[idx], job.tileSpec));
    } catch (e) {
      job.assigned.delete(idx);
      job.reject(e instanceof Error ? e : new Error(String(e)));
      jobs.delete(jobId);
    }
  }

  // The master is also a worker: it pulls one tile at a time from the same queue
  // and runs it locally, interleaved (through the kernel's await points) with
  // serving remote pull requests. Holding at most one tile keeps it from hogging
  // the queue — remote peers drain the rest while the master grinds its single
  // tile. With no remote workers this cleanly degrades to sequential local work.
  async function masterPump(jobId: string): Promise<void> {
    const job = jobs.get(jobId);
    if (!job) return;
    const fn = kernels.get(job.kind);
    const pollMs = job.opts.claimTimeoutMs ?? 250;
    // One-time head start: let the just-sent announce reach remote workers and
    // them claim before the master grabs anything. (Can't gate on the peer roster
    // — swarmMap often runs before the first hello lands, so the roster is empty.)
    await delay(pollMs);
    while (jobs.has(jobId)) {
      // Watchdog: reclaim any tile whose assigned remote worker has gone silent
      // past tileTimeoutMs (tab closed / peer vanished with NO tile-fail — the
      // tile-fail path only fires on a kernel throw, not on a dead peer). Without
      // this the master spins here forever: `pending` is empty but `assigned`
      // still holds the lost tile, so the job never settles. Requeueing is safe —
      // recordResult()'s duplicate guard makes a late reply from a merely-slow
      // (still-alive) worker a no-op (first result wins), so a false reclaim only
      // ever costs redundant compute, never a wrong or double-counted result.
      const tileTimeout = job.opts.tileTimeoutMs ?? 30_000;
      const now = Date.now();
      for (const [idx, ts] of job.assignedAt) {
        if (now - ts <= tileTimeout) continue;
        job.assignedAt.delete(idx);
        job.assigned.delete(idx);
        if (job.results[idx] === undefined && !job.pending.includes(idx)) {
          job.pending.unshift(idx);   // front of queue → retried promptly
        }
      }
      // While remote workers are actively pulling, yield each round so they get
      // priority — the master takes only what they leave. Local kernel compute
      // blocks the event loop, so without this the master would drain everything
      // via microtasks before a remote macrotask is processed. If no worker ever
      // pulled (true solo), drain locally at full speed.
      if (job.remoteActive) await delay(pollMs);
      const idx = fn ? job.pending.shift() : undefined;
      if (idx === undefined) {
        // Queue empty for us. Done? exit. Otherwise wait for tiles still in flight
        // on remote workers. With no kernel and nothing outstanding, leave it all
        // to remote workers.
        if (job.results.filter((r) => r !== undefined).length === job.tiles.length) return;
        if (!fn && job.assigned.size === 0 && job.pending.length === 0) return;
        await delay(pollMs);
        continue;
      }
      await runLocally(jobId, idx);   // runs + records, or rejects on master-kernel failure
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
      const pending: number[] = [];
      for (let i = 0; i < tiles.length; i++) pending.push(i);
      // Cost-aware (LPT): hand heaviest tiles out first. shift() pulls from the
      // front, so sort the queue descending by cost. No-op without costFn (FIFO).
      if (opts.costFn) {
        const cost = tiles.map((t, i) => opts.costFn!(t, i));
        pending.sort((a, b) => cost[b]! - cost[a]!);
      }
      const job: JobState = {
        kind, tiles, tileSpec,
        results: new Array(tiles.length),
        assigned: new Map(),
        assignedAt: new Map(),
        pending,
        settle: resolve as (v: unknown[]) => void,
        reject,
        opts,
        remoteActive: false,
      };
      jobs.set(jobId, job);
      transport.send({ type: "job-announce", payload: { jobId, kind, nTiles: tiles.length, tileSpec } });
      void masterPump(jobId);   // master joins as a puller too
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
