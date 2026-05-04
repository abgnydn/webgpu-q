import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.WEBGPU_Q_PORT ?? 5175);

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
          // WebGPU in headless Chromium needs explicit unblocking. On Mac
          // this routes through Metal natively; on Linux through Vulkan;
          // SwiftShader is the last-resort software path. Don't pin a
          // backend here — let the OS pick the best available.
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,WebGPU",
            "--ignore-gpu-blocklist",
            "--no-sandbox",
          ],
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
