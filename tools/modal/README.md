# tools/modal — second-GPU validation (real NVIDIA T4)

One honest non-Apple GPU artifact for ~$0, to retire the n=1 (single M2 Pro)
sample behind every webgpu-q benchmark. Background + provider survey:
[`docs/webgpu-ci-providers.md`](../../docs/webgpu-ci-providers.md).

## Run it

```bash
pip install modal && modal setup            # one-time (free $30/mo credits)

# 1) De-risk first — cheap adapter probe (~30s). Confirms Modal's T4 yields a
#    REAL, non-fallback adapter (not SwiftShader/lavapipe). This is the gate.
modal run tools/modal/webgpu_t4_probe.py

# 2) Once the probe is green, run the actual kernels on the T4:
modal run tools/modal/webgpu_t4_probe.py --kernels                 # L1 + L3
modal run tools/modal/webgpu_t4_probe.py --kernels --levels 1,3,E34   # + (T) CPU/GPU bench (heavy)
```

It drives the **live deployed** `/experiments/` page (override with `--url`), so
no clone/build runs in the container — the WGSL/WASM executes in the headless
Chrome on the T4 wherever the page is served.

## Status / first-run expectations

Written from the verified recipe (jasonmayes / Chrome team + tigerabrodi, Feb
2026) but **not yet run** — there's no GPU in the dev container. The script is
self-diagnosing: every run prints `nvidia-smi` + `vulkaninfo` so the **first run
tells you immediately** whether you're on the NVIDIA ICD or fell back to
lavapipe. The two `# VERIFY:` points to expect to touch once:

1. **NVIDIA Vulkan userspace / ICD** inside Modal's container. The image sets
   `NVIDIA_DRIVER_CAPABILITIES=all`; if `vulkaninfo` still shows only `llvmpipe`,
   install a `libnvidia-gl-<branch>` matching `nvidia-smi`'s driver, or set
   `VK_ICD_FILENAMES`.
2. **Dawn's driver-version blocklist** cutoff (the `570+` number drifts with
   Chrome/Dawn versions).

The adapter **gate is strict on purpose**: if it lands on software it *raises*
rather than emitting fake "GPU" numbers — a rejected run is the gate working.
