import { defineConfig } from "vite";
import { execSync } from "node:child_process";
import { resolve } from "node:path";

function gitSha(): string {
  try {
    return execSync("git rev-parse --short=12 HEAD", { encoding: "utf-8" }).trim();
  } catch {
    return "dev-unknown";
  }
}

export default defineConfig({
  root: ".",
  base: "./",
  define: {
    __GIT_SHA__: JSON.stringify(gitSha()),
  },
  build: {
    outDir: "dist",
    target: "esnext",
    sourcemap: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        demo: resolve(__dirname, "demo.html"),
        experiments: resolve(__dirname, "experiments/index.html"),
        viz: resolve(__dirname, "viz.html"),
        molecule: resolve(__dirname, "molecule.html"),
        swarm: resolve(__dirname, "swarm.html"),
        gpuMps: resolve(__dirname, "experiments/gpu-mps/index.html"),
      },
    },
  },
  worker: {
    // ES format so workers can dynamic-import the wasm-eri pkg
    // (default IIFE doesn't support code-splitting).
    format: "es",
  },
  server: {
    port: 5175,
    host: true,
    // Mirror vercel.json's COOP/COEP headers so SharedArrayBuffer +
    // crossOriginIsolated are available in dev (required for Web Worker
    // parallel HF buildG). Without these, runRHFSCFAsync silently falls
    // back to single-threaded — which silently breaks the bench.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  // WGSL imported via ?raw
  assetsInclude: ["**/*.wgsl"],
});
