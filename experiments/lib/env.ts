// ─────────────────────────────────────────────────────────────
// env.ts — capture the reproducibility block for every artifact.
//
// RESEARCH.md §Reproducibility requires every JSON artifact to
// record git SHA, userAgent, adapter.info, device limits, OS, and
// UTC timestamp. This file assembles that block so experiment
// code can just call `await captureEnv(device)` and forget.
//
// Git SHA is sourced from `import.meta.env.VITE_GIT_SHA` (see
// vite.config.ts `define` block). When running `vite dev` without
// that wired up the SHA falls back to "dev-unknown" — fine for
// local iteration, unacceptable for a publishable result.
// ─────────────────────────────────────────────────────────────

export interface AdapterInfoBlock {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
}

export interface DeviceLimitsBlock {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeWorkgroupsPerDimension: number;
  readonly maxComputeInvocationsPerWorkgroup: number;
  readonly maxComputeWorkgroupStorageSize: number;
}

export interface EnvBlock {
  readonly gitSha: string;
  readonly timestamp: string;         // UTC ISO8601
  readonly userAgent: string;
  readonly platform: string;
  readonly adapter: AdapterInfoBlock;
  readonly limits: DeviceLimitsBlock;
  readonly hardwareConcurrency: number;
  readonly devicePixelRatio: number;
}

// Vite injects this at build time via `define`.
declare const __GIT_SHA__: string | undefined;

function readGitSha(): string {
  try {
    const sha: unknown = typeof __GIT_SHA__ !== "undefined" ? __GIT_SHA__ : undefined;
    if (typeof sha === "string" && sha.length > 0) return sha;
  } catch { /* not wired up in this build */ }
  return "dev-unknown";
}

/** Grab adapter info in a way that works across Chrome / Firefox / Safari. */
async function adapterInfo(adapter: GPUAdapter): Promise<AdapterInfoBlock> {
  // Chrome ≥121 + Safari 17.4: adapter.info is a sync property.
  // Older Chrome: adapter.requestAdapterInfo() returns a promise.
  // Firefox Nightly: may expose neither — fall back to empty strings.
  type MaybeInfo = Partial<AdapterInfoBlock>;
  type Legacy = { requestAdapterInfo?: () => Promise<MaybeInfo> };
  const syncInfo = (adapter as unknown as { info?: MaybeInfo }).info;
  const asyncInfo = syncInfo
    ? undefined
    : await (adapter as unknown as Legacy).requestAdapterInfo?.().catch(() => ({} as MaybeInfo));
  const info: MaybeInfo = syncInfo ?? asyncInfo ?? {};
  return {
    vendor: String(info.vendor ?? ""),
    architecture: String(info.architecture ?? ""),
    device: String(info.device ?? ""),
    description: String(info.description ?? ""),
  };
}

function deviceLimits(device: GPUDevice): DeviceLimitsBlock {
  const l = device.limits;
  return {
    maxBufferSize: l.maxBufferSize,
    maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: l.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize,
  };
}

export async function captureEnv(
  device: GPUDevice,
  adapter: GPUAdapter,
): Promise<EnvBlock> {
  return {
    gitSha: readGitSha(),
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    platform: typeof navigator !== "undefined" && "platform" in navigator
      ? String((navigator as unknown as { platform: string }).platform)
      : "",
    adapter: await adapterInfo(adapter),
    limits: deviceLimits(device),
    hardwareConcurrency: typeof navigator !== "undefined" ? navigator.hardwareConcurrency : 0,
    devicePixelRatio: typeof window !== "undefined" ? window.devicePixelRatio : 1,
  };
}
