import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: ["tests/**/*.test.ts"],
    // CI was failing on "Timeout calling onTaskUpdate" even though all
    // 845 tests PASS — it's a teardown-phase IPC flake, not a real
    // failure. Root cause: heavy chemistry tests (aug-cc-pVDZ HF on
    // CH₄ ≈ 38 s, DMRG TFIM N=24 ≈ 20 s) run as one long *synchronous*
    // block, so the worker_threads MessagePort can't service the
    // reporter's RPC ping and vitest kills the worker. testTimeout
    // does NOT fix this — it governs how long a test may run, not the
    // RPC keepalive. The fix is the `forks` pool: child_process IPC
    // tolerates long synchronous blocking that worker_threads does not.
    pool: "forks",
    poolOptions: { forks: { singleFork: false } },
    testTimeout: 120_000,
    hookTimeout: 120_000,
    teardownTimeout: 60_000,
  },
});
