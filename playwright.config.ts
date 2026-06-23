import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.WEBGPU_Q_PORT ?? 5175);

// Opt-in lane for validating on a real NVIDIA GPU on Linux (e.g. a cloud T4):
// adds the Vulkan/Dawn flags headless Chrome needs there + a hard non-fallback
// adapter gate (see e2e/lib/run-level.ts). Leave UNSET on macOS — it uses Metal
// natively and these flags would break it. See docs/webgpu-ci-providers.md.
const NVIDIA = !!process.env.WEBGPU_Q_NVIDIA;

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./e2e/.artifacts",
  timeout: 10 * 60 * 1000,
  expect: { timeout: 30 * 1000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["html", { outputFolder: "e2e/.report", open: "never" }]],
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        ...devices["Desktop Chrome"],
        channel: "chromium",
        launchOptions: {
          // WebGPU in headless Chromium needs explicit unblocking. By default we
          // DON'T pin a backend — macOS picks Metal, Linux picks Vulkan, and
          // SwiftShader is the last-resort software path. The NVIDIA-on-Linux
          // validation lane (WEBGPU_Q_NVIDIA=1) is the exception: pin Vulkan, drop
          // Dawn's driver blocklist, and add --disable-vulkan-surface so headless
          // GPU *compute* works (canvas off — fine for the L1/L3/L6 benches).
          // These would break macOS/Metal, so they're opt-in only.
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,WebGPU",
            "--ignore-gpu-blocklist",
            "--no-sandbox",
            ...(NVIDIA
              ? [
                  "--use-angle=vulkan",
                  "--disable-vulkan-surface",
                  "--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist",
                ]
              : []),
          ],
          // Strip Playwright's injected software-fallback args on the NVIDIA lane
          // so the hard adapter gate can't pass on SwiftShader.
          ...(NVIDIA
            ? { ignoreDefaultArgs: ["--use-angle=swiftshader-webgl", "--enable-unsafe-swiftshader"] }
            : {}),
        },
      },
    },
  ],
  webServer: {
    command: `npm run dev -- --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/experiments/`,
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
