import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: ["tests/**/*.test.ts"],
    // CI was failing on "[vitest-worker]: Timeout calling onTaskUpdate"
    // even though all 845 tests PASS — exit 1 after green.
    //
    // The label is the clue: the WORKER calls onTaskUpdate ON THE MAIN
    // process and times out waiting for the main process to ACK. So the
    // worker isn't the problem — the main reporter process is starved.
    // On a 4-core CI runner, several heavy chemistry tests (82 s, 79 s,
    // 63 s synchronous blocks: aug-cc-pVDZ HF, DMRG TFIM N=24, CCSD)
    // run concurrently across all cores, then finish and bombard the
    // main process with reports at once — it can't ack within the RPC
    // window and the worker errors out. (Switching threads→forks did
    // NOT help, confirming it's contention, not pool transport.)
    //
    // Fix: run test files serially. With one worker active, the main
    // process always has free cores to ack reports promptly. Slower
    // wall-time (~10 min serial) but well within the CI budget, and it
    // removes the contention entirely. Generous per-test timeouts kept.
    pool: "forks",
    fileParallelism: false,
    testTimeout: 120_000,
    hookTimeout: 120_000,
    teardownTimeout: 60_000,
  },
});
