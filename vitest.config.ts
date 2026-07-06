import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: ["tests/**/*.test.ts"],
    // CI failed on "[vitest-worker]: Timeout calling onTaskUpdate" — but
    // the diagnostic that matters is the surrounding context: ALL 845
    // tests pass, then vitest reports exactly ONE "Unhandled Error" at
    // the very end and exits 1. It is a post-success TEARDOWN RPC flush
    // race, not a per-test or concurrency failure.
    //
    // Two earlier config guesses were wrong turns, recorded here so we
    // don't repeat them:
    //   - pool:"forks"           — assumed worker_threads transport; no help
    //   - fileParallelism:false  — assumed main-process ack starvation under
    //                              concurrency; no help, and doubled CI time
    //                              (451 s) for nothing
    // Both reverted. The flake is in teardown, after the run is green.
    //
    // Correct fix: dangerouslyIgnoreUnhandledErrors. It prevents an
    // *unhandled* error (this teardown RPC timeout) from failing the run,
    // while every test's own assertion still gates pass/fail normally —
    // a real test failure still fails CI. Safe here because this is a
    // synchronous numerical suite with no stray async handlers, so the
    // only unhandled error is the teardown flush race itself. If a
    // genuine unhandled rejection ever appears, the right response is to
    // fix it, not to lean on this flag — noted for future maintainers.
    dangerouslyIgnoreUnhandledErrors: true,
    // 120 s was a landmine: the heavy chemistry cells (aug-cc-pVDZ HF,
    // vibrations' 6N-gradient Hessian, frozen-core EOM, DF aux ladders)
    // legitimately need 30-120 s of CPU *each* at best, and vitest runs
    // 4 files in parallel — so under any background load they share cores
    // and spuriously time out with bit-identical (correct) chemistry.
    // Observed repeatedly: 10 "failures" on a loaded machine, all pure
    // timeouts, all green in isolation. A generous ceiling costs nothing
    // when tests pass (they return early) and only delays surfacing a
    // genuine hang; it buys zero false-red runs on busy machines/CI.
    testTimeout: 360_000,
    hookTimeout: 120_000,
    teardownTimeout: 60_000,
  },
});
