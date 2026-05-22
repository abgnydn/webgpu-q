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
  server: {
    port: 5175,
    host: true,
  },
  // WGSL imported via ?raw
  assetsInclude: ["**/*.wgsl"],
});
