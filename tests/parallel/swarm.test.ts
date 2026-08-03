// Single-process swarm round-trip with an in-memory transport.
// Verifies the swarmMap claim/assign/result protocol routes correctly
// between a "master" peer and a "worker" peer in the same process.

import { afterEach, describe, expect, test } from "vitest";
import type {
  SwarmTransport, SwarmMessage, SwarmListener, PeerId,
} from "../../src/parallel/swarm/transport.js";
import { attachSwarmRuntime, swarmMap } from "../../src/parallel/swarm/swarm-map.js";

// KernelRegistry.dispose() stops a 1 s hello-broadcast interval and closes the
// transport; its own docstring says "always call this". No test did, so every
// attachSwarmRuntime here leaked a live timer for the whole run — which is also
// why vitest.config.ts's "no stray async handlers" justification for
// dangerouslyIgnoreUnhandledErrors was not actually true. Track and dispose.
const liveRuntimes: { dispose(): void }[] = [];
function attachTracked(transport: Parameters<typeof attachSwarmRuntime>[0]) {
  const reg = attachSwarmRuntime(transport);
  liveRuntimes.push(reg);
  return reg;
}
afterEach(() => {
  while (liveRuntimes.length > 0) {
    try { liveRuntimes.pop()!.dispose(); } catch { /* already closed */ }
  }
});

class FakeBus {
  private readonly tabs: { id: PeerId; listeners: Set<SwarmListener> }[] = [];
  newTransport(id: PeerId): SwarmTransport {
    const tab = { id, listeners: new Set<SwarmListener>() };
    this.tabs.push(tab);
    const bus = this;
    return {
      self: id,
      open() {},
      close() {},
      send(msg) {
        const full: SwarmMessage = { ...msg, from: id };
        // Deliver async (microtask) to mimic real transport.
        queueMicrotask(() => {
          for (const other of bus.tabs) {
            if (other.id === id) continue;
            if (full.to && full.to !== other.id) continue;
            for (const l of other.listeners) l(full);
          }
        });
      },
      onMessage(listener) {
        tab.listeners.add(listener);
        return () => tab.listeners.delete(listener);
      },
    };
  }
}

describe("Swarm — in-process protocol round-trip", () => {
  test("swarmMap distributes work across 2 peers and reassembles results", async () => {
    const bus = new FakeBus();
    const master = bus.newTransport("master");
    const worker = bus.newTransport("worker");

    // Stub window.setInterval / setTimeout to no-op for hello timer
    // and tile claim timeout — we don't need them for this test.
    // (jsdom env provides these by default; just rely on it.)

    const mReg = attachTracked(master);
    const wReg = attachTracked(worker);

    mReg.registerKernel<number, number>("square", (t) => t * t);
    wReg.registerKernel<number, number>("square", (t) => t * t);

    // Let hello round-trip so both peers see each other.
    await new Promise((r) => setTimeout(r, 50));

    const inputs = [1, 2, 3, 4, 5, 6, 7, 8];
    const results = await swarmMap<number, number>(mReg, "square", inputs, {
      claimTimeoutMs: 30,
    });
    expect(results).toEqual([1, 4, 9, 16, 25, 36, 49, 64]);
  }, 5000);

  test("swarmMap with no peers runs all tiles locally on master", async () => {
    const bus = new FakeBus();
    const master = bus.newTransport("solo");
    const mReg = attachTracked(master);
    mReg.registerKernel<number, number>("cube", (t) => t * t * t);

    const results = await swarmMap<number, number>(mReg, "cube", [2, 3, 4], {
      claimTimeoutMs: 10,
    });
    expect(results).toEqual([8, 27, 64]);
  }, 5000);

  test("swarmMap with 3 peers spreads tiles among all of them", async () => {
    const bus = new FakeBus();
    const m = bus.newTransport("m");
    const w1 = bus.newTransport("w1");
    const w2 = bus.newTransport("w2");
    const mReg = attachTracked(m);
    const w1Reg = attachTracked(w1);
    const w2Reg = attachTracked(w2);
    const workers = { m: 0, w1: 0, w2: 0 };
    mReg.registerKernel<number, number>("id", (t) => { workers.m++; return t; });
    w1Reg.registerKernel<number, number>("id", (t) => { workers.w1++; return t; });
    w2Reg.registerKernel<number, number>("id", (t) => { workers.w2++; return t; });
    await new Promise((r) => setTimeout(r, 60));

    const N = 20;
    const tiles = Array.from({ length: N }, (_, i) => i);
    const results = await swarmMap<number, number>(mReg, "id", tiles, {
      claimTimeoutMs: 30,
    });
    expect(results).toEqual(tiles);
    expect(workers.m + workers.w1 + workers.w2).toBe(N);
    // Hard guarantee is sum = N; each worker getting > 0 isn't
    // guaranteed because the master ALSO runs local fallbacks if a
    // peer is slow to claim. We just sanity-check that work spread.
    const involved = [workers.m, workers.w1, workers.w2].filter((x) => x > 0).length;
    expect(involved).toBeGreaterThanOrEqual(2);
  }, 5000);

  test("swarmMap reassigns to master when worker reports tile-fail", async () => {
    const bus = new FakeBus();
    const m = bus.newTransport("m");
    const flaky = bus.newTransport("flaky");
    const mReg = attachTracked(m);
    const fReg = attachTracked(flaky);

    // Master kernel: succeeds.
    mReg.registerKernel<number, number>("k", (t) => t + 100);
    // Worker kernel: always throws — master should catch tile-fail
    // and reassign to itself.
    fReg.registerKernel<number, number>("k", () => { throw new Error("flaky"); });

    await new Promise((r) => setTimeout(r, 60));

    const results = await swarmMap<number, number>(mReg, "k", [1, 2, 3], {
      claimTimeoutMs: 30,
    });
    expect(results).toEqual([101, 102, 103]);
  }, 5000);

  test("swarmMap with 0 tiles resolves immediately to []", async () => {
    const bus = new FakeBus();
    const m = bus.newTransport("solo");
    const mReg = attachTracked(m);
    mReg.registerKernel<number, number>("noop", (t) => t);
    const results = await swarmMap<number, number>(mReg, "noop", [], { claimTimeoutMs: 10 });
    expect(results).toEqual([]);
  });

  test("swarmMap ignores peers that don't have the matching kernel", async () => {
    const bus = new FakeBus();
    const m = bus.newTransport("m");
    const useless = bus.newTransport("useless");
    const mReg = attachTracked(m);
    const uReg = attachTracked(useless);
    mReg.registerKernel<number, number>("k", (t) => t * 10);
    uReg.registerKernel<number, number>("OTHER", (t) => t * -1);
    await new Promise((r) => setTimeout(r, 60));

    const results = await swarmMap<number, number>(mReg, "k", [1, 2, 3], {
      claimTimeoutMs: 25,
    });
    expect(results).toEqual([10, 20, 30]);
  }, 5000);

  test("tileSpec is delivered to the kernel as the second arg", async () => {
    const bus = new FakeBus();
    const m = bus.newTransport("m");
    const w = bus.newTransport("w");
    const mReg = attachTracked(m);
    const wReg = attachTracked(w);
    const seenSpecs: unknown[] = [];
    mReg.registerKernel<number, number>("withSpec", (t, spec) => {
      seenSpecs.push(spec);
      return t + (spec as { offset: number }).offset;
    });
    wReg.registerKernel<number, number>("withSpec", (t, spec) => {
      seenSpecs.push(spec);
      return t + (spec as { offset: number }).offset;
    });
    await new Promise((r) => setTimeout(r, 60));

    const results = await swarmMap<number, number>(mReg, "withSpec", [1, 2, 3], {
      claimTimeoutMs: 30,
    }, { offset: 1000 });
    expect(results).toEqual([1001, 1002, 1003]);
    for (const s of seenSpecs) {
      expect(s).toEqual({ offset: 1000 });
    }
  }, 5000);

  test("two concurrent swarmMap jobs both complete with correct results", async () => {
    const bus = new FakeBus();
    const m = bus.newTransport("m");
    const w = bus.newTransport("w");
    const mReg = attachTracked(m);
    const wReg = attachTracked(w);
    mReg.registerKernel<number, number>("doubler", (t) => t * 2);
    wReg.registerKernel<number, number>("doubler", (t) => t * 2);
    mReg.registerKernel<number, string>("stringer", (t) => `n${t}`);
    wReg.registerKernel<number, string>("stringer", (t) => `n${t}`);
    await new Promise((r) => setTimeout(r, 60));

    const [a, b] = await Promise.all([
      swarmMap<number, number>(mReg, "doubler", [1, 2, 3, 4], { claimTimeoutMs: 30 }),
      swarmMap<number, string>(mReg, "stringer", [10, 20, 30], { claimTimeoutMs: 30 }),
    ]);
    expect(a).toEqual([2, 4, 6, 8]);
    expect(b).toEqual(["n10", "n20", "n30"]);
  }, 5000);

  test("knownPeers() reflects who has announced themselves", async () => {
    const bus = new FakeBus();
    const m = bus.newTransport("m");
    const a = bus.newTransport("a");
    const b = bus.newTransport("b");
    const mReg = attachTracked(m);
    attachTracked(a);
    attachTracked(b);
    // Initially knownPeers() may be empty — wait one hello tick.
    await new Promise((r) => setTimeout(r, 80));
    const peers = mReg.knownPeers();
    expect(peers).toContain("a");
    expect(peers).toContain("b");
    expect(peers).not.toContain("m"); // self isn't in the roster
  }, 5000);

  test("onProgress fires with monotone (done, total) counts", async () => {
    const bus = new FakeBus();
    const m = bus.newTransport("m");
    const w = bus.newTransport("w");
    const mReg = attachTracked(m);
    const wReg = attachTracked(w);
    mReg.registerKernel<number, number>("k", (t) => t);
    wReg.registerKernel<number, number>("k", (t) => t);
    await new Promise((r) => setTimeout(r, 60));

    const ticks: Array<[number, number]> = [];
    const results = await swarmMap<number, number>(mReg, "k", [1, 2, 3, 4, 5], {
      claimTimeoutMs: 25,
      onProgress: (done, total) => ticks.push([done, total]),
    });
    expect(results).toEqual([1, 2, 3, 4, 5]);
    expect(ticks.length).toBeGreaterThan(0);
    // Every tick has total = 5; done counts ascend.
    for (const [, total] of ticks) expect(total).toBe(5);
    let prev = 0;
    for (const [done] of ticks) {
      expect(done).toBeGreaterThanOrEqual(prev);
      prev = done;
    }
    expect(ticks[ticks.length - 1]![0]).toBe(5);
  }, 5000);
});

describe("BroadcastChannelTransport — wire-level behavior (FakeBus stand-in)", () => {
  // The BroadcastChannel API isn't available in vitest's node env, so
  // we test the FakeBus's invariants here as a contract for what
  // BroadcastChannelTransport must satisfy (self-echo suppression,
  // targeted routing). The browser implementation is exercised by
  // e2e/swarm.spec.ts.

  test("transport ignores its own echoed messages", async () => {
    const bus = new FakeBus();
    const a = bus.newTransport("a");
    const b = bus.newTransport("b");
    const aSaw: string[] = [];
    a.onMessage((m) => aSaw.push(m.from));
    b.onMessage(() => {});
    a.send({ type: "hello", payload: {} });
    await new Promise((r) => setTimeout(r, 10));
    // 'a' should NOT receive its own broadcast.
    expect(aSaw).not.toContain("a");
  });

  test("targeted send delivers only to the addressee", async () => {
    const bus = new FakeBus();
    const m = bus.newTransport("m");
    const a = bus.newTransport("a");
    const b = bus.newTransport("b");
    const aSaw: string[] = [];
    const bSaw: string[] = [];
    a.onMessage((msg) => aSaw.push(msg.type));
    b.onMessage((msg) => bSaw.push(msg.type));
    m.send({ type: "hello", to: "a", payload: {} });
    await new Promise((r) => setTimeout(r, 10));
    expect(aSaw).toEqual(["hello"]);
    expect(bSaw).toEqual([]);
  });

  test("onMessage unsubscribe stops delivery", async () => {
    const bus = new FakeBus();
    const a = bus.newTransport("a");
    const b = bus.newTransport("b");
    const aSaw: string[] = [];
    const unsubscribe = a.onMessage((msg) => aSaw.push(msg.type));
    b.send({ type: "hello", payload: {} });
    await new Promise((r) => setTimeout(r, 10));
    expect(aSaw).toEqual(["hello"]);
    unsubscribe();
    b.send({ type: "hello", payload: {} });
    await new Promise((r) => setTimeout(r, 10));
    expect(aSaw).toEqual(["hello"]); // unchanged
  });
});
