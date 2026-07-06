// Watchdog regression: a peer that pulls a tile then vanishes (tab closed,
// crash) sends NO tile-fail — the tile-fail recovery path only fires on a
// kernel throw. Before the tileTimeoutMs watchdog, the master's pump spun
// forever (pending empty, the lost tile stuck in `assigned`) and swarmMap
// never resolved. This asserts the master now reclaims the silent tile after
// tileTimeoutMs and completes the job with correct results.

import { describe, expect, test } from "vitest";
import type {
  SwarmTransport, SwarmMessage, SwarmListener, PeerId,
} from "../../src/parallel/swarm/transport.js";
import { attachSwarmRuntime, swarmMap } from "../../src/parallel/swarm/swarm-map.js";

class FakeBus {
  private readonly tabs: { id: PeerId; listeners: Set<SwarmListener>; live: boolean }[] = [];
  newTransport(id: PeerId): SwarmTransport & { goSilent(): void } {
    const tab = { id, listeners: new Set<SwarmListener>(), live: true };
    this.tabs.push(tab);
    const bus = this;
    return {
      self: id,
      open() {},
      close() {},
      goSilent() { tab.live = false; },   // simulate the tab dying: stop emitting
      send(msg) {
        if (!tab.live) return;
        const full: SwarmMessage = { ...msg, from: id };
        queueMicrotask(() => {
          for (const other of bus.tabs) {
            if (other.id === id || !other.live) continue;
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

describe("Swarm — tileTimeoutMs watchdog reclaims a vanished peer's tile", () => {
  test("job completes with correct results after a worker goes silent mid-tile", async () => {
    const bus = new FakeBus();
    const master = bus.newTransport("master");
    const worker = bus.newTransport("worker");

    const mReg = attachSwarmRuntime(master);
    const wReg = attachSwarmRuntime(worker);

    // Master's kernel works normally. The worker's kernel HANGS on its first
    // tile (never resolves) and then the tab "dies" — the classic silent-peer
    // failure: one tile stuck in `assigned` with no tile-result, no tile-fail.
    mReg.registerKernel<number, number>("square", (t) => t * t);
    let workerGotOne = false;
    wReg.registerKernel<number, number>("square", (t) => {
      if (!workerGotOne) {
        workerGotOne = true;
        (worker as unknown as { goSilent(): void }).goSilent();
        return new Promise<number>(() => { /* never resolves — peer is gone */ });
      }
      return t * t;
    });

    await new Promise((r) => setTimeout(r, 50));   // let hello round-trip

    const inputs = [1, 2, 3, 4, 5, 6];
    const results = await swarmMap<number, number>(mReg, "square", inputs, {
      claimTimeoutMs: 20,
      tileTimeoutMs: 150,   // short so the test reclaims quickly
    });

    expect(results).toEqual(inputs.map((x) => x * x));
    expect(workerGotOne).toBe(true);   // the worker really did claim (then lose) a tile

    mReg.dispose();
    wReg.dispose();
  }, 10_000);
});
