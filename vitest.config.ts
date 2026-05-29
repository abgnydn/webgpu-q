import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: ["tests/**/*.test.ts"],
    // The full suite runs ~10 min on CI (Ubuntu 4-core); aug-cc-pVDZ
    // HF on CH₄ takes 38 s, DMRG TFIM N=24 takes 20 s. The default
    // worker IPC keepalive trips "Timeout calling onTaskUpdate" when
    // a single test exceeds ~30 s. Bump generously so a slow but
    // still-running test doesn't kill the worker. Has no effect on
    // local short runs; just stops the CI flake.
    testTimeout: 120_000,
    hookTimeout: 120_000,
    teardownTimeout: 30_000,
  },
});
