// Single-process swarm round-trip with an in-memory transport.
// Verifies the swarmMap claim/assign/result protocol routes correctly
// between a "master" peer and a "worker" peer in the same process.

import { describe, expect, test } from "vitest";
import type {
  SwarmTransport, SwarmMessage, SwarmListener, PeerId,
} from "../../src/parallel/swarm/transport.js";
import { attachSwarmRuntime, swarmMap } from "../../src/parallel/swarm/swarm-map.js";

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

    const mReg = attachSwarmRuntime(master);
    const wReg = attachSwarmRuntime(worker);

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
    const mReg = attachSwarmRuntime(master);
    mReg.registerKernel<number, number>("cube", (t) => t * t * t);

    const results = await swarmMap<number, number>(mReg, "cube", [2, 3, 4], {
      claimTimeoutMs: 10,
    });
    expect(results).toEqual([8, 27, 64]);
  }, 5000);
});
