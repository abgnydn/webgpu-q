// Cost-aware (LPT) swarm scheduling — makespan model + result-order safety.
//
// The throughput axis: N independent molecules, one per tile. swarmMap hands them
// out FIFO, so a heavy molecule pulled late tails past everyone (the documented
// 59%-efficiency ceiling). LPT (longest-processing-time-first) sorts the queue
// heaviest-first so the heavy tile starts at t=0. These tests pin (a) the makespan
// model and (b) that reordering the PULL QUEUE never reorders the RESULTS.

import { afterEach, describe, expect, test } from "vitest";
import type { SwarmTransport, SwarmMessage, SwarmListener, PeerId } from "../../src/parallel/swarm/transport.js";
import { attachSwarmRuntime, swarmMap, simulateGreedyMakespan } from "../../src/parallel/swarm/swarm-map.js";

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
      open() {}, close() {},
      send(msg) {
        const full: SwarmMessage = { ...msg, from: id };
        queueMicrotask(() => {
          for (const other of bus.tabs) {
            if (other.id === id) continue;
            if (full.to && full.to !== other.id) continue;
            for (const l of other.listeners) l(full);
          }
        });
      },
      onMessage(listener) { tab.listeners.add(listener); return () => tab.listeners.delete(listener); },
    };
  }
}

describe("simulateGreedyMakespan — greedy-pull makespan model", () => {
  test("single worker = serial sum", () => {
    expect(simulateGreedyMakespan([3, 1, 4, 1, 5], 1)).toBe(14);
  });
  test("perfectly even split halves the makespan", () => {
    expect(simulateGreedyMakespan([2, 2, 2, 2], 2)).toBe(4);
  });
  test("LPT beats FIFO when a heavy tile is last (the bottleneck case)", () => {
    const costs = [1, 1, 1, 1, 1, 1, 6]; // heavy tile at the END — worst case for FIFO
    const fifo = simulateGreedyMakespan(costs, 3);
    const lpt = simulateGreedyMakespan([...costs].sort((a, b) => b - a), 3);
    expect(fifo).toBe(8); // the 6 lands on an already-busy worker → long tail
    expect(lpt).toBe(6);  // the 6 starts at t=0; the six 1s fill the other two workers
    expect(lpt).toBeLessThan(fifo);
  });
  test("LPT lowers makespan on average + ≤ FIFO in the vast majority (a worst-case/average win, NOT pointwise dominance — Graham's 4/3 bound, not ≤ every instance)", () => {
    let rng = 12345;
    const rand = (): number => (rng = (rng * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const N = 500;
    let fifoTotal = 0, lptTotal = 0, lptAtLeastAsGood = 0;
    for (let trial = 0; trial < N; trial++) {
      const n = 2 + Math.floor(rand() * 12);
      const costs = Array.from({ length: n }, () => 1 + Math.floor(rand() * 20));
      const k = 1 + Math.floor(rand() * 4);
      const fifo = simulateGreedyMakespan(costs, k);
      const lpt = simulateGreedyMakespan([...costs].sort((a, b) => b - a), k);
      fifoTotal += fifo; lptTotal += lpt;
      if (lpt <= fifo) lptAtLeastAsGood++;
    }
    expect(lptTotal).toBeLessThan(fifoTotal);            // aggregate makespan: LPT wins
    expect(lptAtLeastAsGood / N).toBeGreaterThan(0.85);  // ≤ FIFO in the large majority of instances
  });
});

describe("swarmMap costFn (LPT) — reorders the queue, NOT the results", () => {
  test("results stay index-correct under cost-aware scheduling", async () => {
    const bus = new FakeBus();
    const mReg = attachTracked(bus.newTransport("m"));
    const wReg = attachTracked(bus.newTransport("w"));
    mReg.registerKernel<number, number>("sq", (t) => t * t);
    wReg.registerKernel<number, number>("sq", (t) => t * t);
    await new Promise((r) => setTimeout(r, 50));

    const inputs = [1, 2, 3, 4, 5, 6, 7, 8];
    // Rank larger inputs heavier → they're pulled first, but results must still
    // come back in INPUT order (the reduction is keyed by original tile index).
    const results = await swarmMap<number, number>(mReg, "sq", inputs, {
      claimTimeoutMs: 30,
      costFn: (t) => t as number,
    });
    expect(results).toEqual([1, 4, 9, 16, 25, 36, 49, 64]);
  }, 5000);
});
