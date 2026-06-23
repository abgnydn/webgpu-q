# WebGPU testing on CI / cloud GPU — provider survey (2026-06)

Where can webgpu-q's GPU tests run on a **real, non-Apple GPU** — to retire the
n=1 (single M2 Pro) hardware sample behind every benchmark? This is the survey.

Researched 2026-06-23 via a 5-angle fan-out (hosted CI, self-hosted runners,
on-demand GPU clouds, headless-browser SaaS, native/Dawn recipes). Load-bearing
claims are triangulated across primary sources; the recipe and "wgpu CI =
lavapipe" were read from raw source files. Several vendor pages 403 automated
fetch, so some pricing rests on search-index extracts — **treat every $ as a
snapshot, re-pull before committing.**

---

## Verdict

1. **Every *default* CI runner is software-only** (SwiftShader / lavapipe /
   llvmpipe). GitHub, GitLab, CircleCI standard runners have **no GPU at all**.
   So does every managed cross-browser cloud (BrowserStack, Sauce, LambdaTest,
   Browserbase, Browserless self-serve).
2. **Real-GPU headless WebGPU is solved and *empirically confirmed* — but only
   on DIY cloud GPU** (Modal / Colab / RunPod). The recipe is authored by a
   Google Chrome dev and linked from the official Chromium docs.
3. **Honesty gap:** *no one has published an actual non-fallback-adapter result
   on any managed GPU **CI** runner.* The "hardware" cells for GitHub gpu-t4 /
   CircleCI / GitLab are **inferred from driver presence, not demonstrated.**
4. The whole game is **dodging silent software fallback** and **asserting a hard
   gate**: `adapter.info.isFallbackAdapter === false`.

---

## First decide what you're validating

This fork decides which option you pick.

| You want… | What you need | Cost | Automatable today? |
|---|---|---|---|
| **Correctness portability** — does the WGSL compute the *right answer* on a different impl | A **software** Vulkan lane suffices. The f32 pass-bar (F ≥ 1−1e-5) will likely *pass* on SwiftShader/lavapipe (deterministic CPU). | **$0** | **Yes, now** — Deno+lavapipe or Chrome+SwiftShader on normal CI |
| **Performance reproduction** on a real non-Apple GPU — the actual n=1 *speed* claim | **Real silicon.** Software is useless (slower than the M2; proves nothing about f32 GPU perf). | $0–cents one-shot; ~$0.07/min CI | One-shot trivial; CI lane = real work |

Sharp implication: a **free** SwiftShader/lavapipe lane already buys you "the
kernels are correct on a second WebGPU implementation." Only the **speed**
numbers need real silicon.

---

## Options (software-fallback risk per row)

| Option | Real GPU? | Shape | Automation | Cost | Fallback risk |
|---|---|---|---|---|---|
| **Modal** | ✅ verified (T4) | scriptable Python container | API/script | **~$0** ($30/mo free credits) | low — strip injected SwiftShader arg |
| **Google Colab** | ✅ verified (T4) | notebook | manual cell | **$0** | low — Chrome's own documented path |
| **RunPod** | ✅ verified (A40) | root container | API/SSH | $0.34–0.44/hr, per-sec, $10 float | low |
| **Kaggle** | ⚠️ likely (T4×2) | notebook | manual | $0, 30h/wk | med — *verify the adapter* |
| **GitHub `gpu-t4-4-core`** | ⚠️ HW yes, WebGPU unproven | first-party CI runner | **push-gated** | **$0.07/min**, Team/Enterprise only | med — driver+flags yours to set |
| **Self-hosted runner, AWS g4dn.xlarge T4 spot** | ⚠️ HW yes (proven recipe) | self-hosted CI | **push-gated** | **~$0.16–0.22/hr** + ephemeral (RunsOn/HyperEnv) | med — you own the setup |
| **CircleCI GPU** (A10G/T4) | ⚠️ HW yes | CI | push-gated | Scale plan only (~$2k/mo floor) | med |
| **GitLab GPU SaaS** (T4) | ⚠️ HW yes | CI | push-gated | Premium/Ultimate; cost-factor unverified | med |
| **GitHub macOS (Apple Silicon)** | ⚠️ paravirtual, flaky | CI | push-gated | ~$0.08–0.16/min | **high** — unsupported; browsers blocklist paravirtual |
| BrowserStack / Sauce / LambdaTest / Browserbase / Browserless self-serve | ❌ software | SaaS browser | — | — | **certain SwiftShader** |
| Deno / Dawn-node native | ✅ on a GPU box / ❌ on hosted CI | no-browser native | either | depends on host | hosted CI → lavapipe |
| Kernel.sh, Browserless Enterprise | ⚠️ HW vGPU, WebGPU unconfirmed | managed GPU browser | API | opaque / sales-gated | unknown |

**Excluded:** Replicate (Cog-locked, can't shell a browser), Fly.io GPUs
(discontinued Aug 1), CoreWeave (8-GPU minimum), Paperspace free (M4000 too old
for Vulkan WebGPU), Lambda (no cheap T4 in 2026).

---

## Ranked picks

**(a) Cheapest ONE honest second-GPU artifact → Modal free tier** (or Colab for
absolute $0). Scriptable Python, verified real T4, a ~2-min run sits inside the
$30/mo free credits → effectively **$0**, and unlike Colab it's a *script*, not a
babysat notebook. Also the right place to **de-risk before any CI work**: prove
webgpu-q's kernels report a non-fallback NVIDIA adapter here first. See
[`tools/modal/webgpu_t4_probe.py`](../tools/modal/webgpu_t4_probe.py).

**(b) Best automatable lane → staged.** Prototype on Modal/Colab (a), then
promote to either **GitHub `gpu-t4-4-core` ($0.07/min)** if you're on
Team/Enterprise (simplest, first-party), or a **self-hosted ephemeral runner on
g4dn.xlarge T4 spot (~$0.18/hr)** via RunsOn/HyperEnv (cheapest sustained, any
plan, and the config with the *proven* recipe). **Don't build the CI lane until
the one-shot confirms hardware-real** — the managed-CI "hardware" cells are
unproven for WebGPU specifically.

---

## Known-good recipe + the hard gate

Authored by jasonmayes (Chrome team), linked from official Chromium docs.
Confirmed on T4/V100/A100.

```bash
# Driver layer (Debian/Ubuntu, NVIDIA). The GL/Vulkan *userspace* is load-bearing —
# the compute driver / nvidia-smi alone is NOT enough.
apt-get update && apt-get install -y vulkan-tools libnvidia-gl-525   # match your driver branch
vulkaninfo | grep -i deviceName     # MUST show NVIDIA, NOT llvmpipe / lavapipe
```

```js
// Playwright: kill the injected SwiftShader arg, force Vulkan, then GATE on it.
chromium.launch({
  ignoreDefaultArgs: ['--use-angle=swiftshader-webgl'],   // trap #1: Playwright/Puppeteer inject this
  args: ['--no-sandbox','--headless=new','--use-angle=vulkan','--enable-features=Vulkan',
         '--disable-vulkan-surface',                        // compute works; canvas disabled (fine for our math)
         '--enable-unsafe-webgpu',
         '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist'], // trap #2: Dawn blocklists NVIDIA driver ≥570
});
// HARD GATE — fail the build on software:
if (info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe/i.test(info.vendor + info.architecture))
  throw new Error(`Software adapter: ${JSON.stringify(info)}`);
```

The silent-software traps, in order of how often they bite:

1. **Puppeteer/Playwright inject `--use-angle=swiftshader-webgl`** → forces
   SwiftShader. Strip via `ignoreDefaultArgs`.
2. **Dawn keeps its *own* blocklist**, separate from `--ignore-gpu-blocklist`,
   and rejects NVIDIA driver ≥570 by default. Override with
   `--enable-dawn-features=…disable_adapter_blocklist` (the `570` cutoff is
   version-sensitive — re-verify against your stack).
3. **`libnvidia-gl` userspace** (not just the compute driver) must be installed,
   or `vulkaninfo` shows only `llvmpipe`.
4. **In Docker:** `--gpus all` mounts compute only. Set
   `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` (or `all`), and the
   toolkit frequently fails to create `/etc/vulkan/icd.d/nvidia_icd.json` — copy
   one in / set `VK_ICD_FILENAMES`.
5. **`--disable-vulkan-surface`** is what makes headless GPU work — it disables
   canvas rendering but leaves **compute** untouched (exactly webgpu-q's
   L1/L3/(T) case).

**Verify, don't trust the flags:** assert `adapter.info.isFallbackAdapter ===
false` and `vendor`/`architecture` ∉ {swiftshader, llvmpipe, lavapipe}.
`chrome://gpu` saying "Hardware accelerated" is **necessary but not sufficient** —
WebGPU adapter acquisition can still fail (Chromium #332726571).

---

## What this means for webgpu-q

`playwright.config.ts` today launches with `--enable-unsafe-webgpu
--enable-features=Vulkan,WebGPU --ignore-gpu-blocklist --no-sandbox`. On
Mac/Metal that's fine; **on an NVIDIA box it would likely fall back or null
out** — it's missing `--use-angle=vulkan`, `--disable-vulkan-surface`, the
**Dawn**-blocklist flag (note: `--ignore-gpu-blocklist` is Chrome's blocklist,
*not* Dawn's), the `--use-angle=swiftshader-webgl` strip, and the hard gate.

Half the gate is already built: `experiments/lib/env.ts → captureEnv` records
`adapter.info` (vendor / architecture / description). To stand up a second-GPU
lane you'd: (1) add the flags above, (2) assert non-fallback on `captureEnv`'s
output, (3) point it at a real T4 (start with `tools/modal/webgpu_t4_probe.py`).

---

## Native path (Deno / Dawn-node) — worth it?

**No, not to "make CI easier."** The hard part — a working vendor Vulkan driver +
dodging software fallback — is identical in all three worlds. Hosted CI has no
GPU, so Deno/Dawn-node fall to lavapipe/SwiftShader there exactly as Chrome does;
**nothing gives you a free hardware GPU in hosted CI.** And a native runner tests
a *different* WebGPU implementation (wgpu / standalone Dawn) than users hit
(Chrome's Dawn) — f32 compilation, FP contraction, and subgroup behavior can
differ, so "green on native" wouldn't guarantee the browser artifact. wgpu's own
CI runs on **lavapipe (software)**; hardware testing happens on contributors'
machines.

**Where native *is* worth it:** a fast, hardware-free **correctness/smoke lane** —
run the WGSL compute under Deno (wgpu+lavapipe) or Dawn-node (SwiftShader) on
ordinary CI to catch shader-compile errors and f64-reference diffs that don't
depend on GPU timing. Keep the **performance + f32-fidelity** gates on the
headless-Chrome-on-GPU lane.

---

## Sources

- [jasonmayes/headless-chrome-nvidia-t4-gpu-support](https://github.com/jasonmayes/headless-chrome-nvidia-t4-gpu-support) — the canonical recipe (Chrome team)
- [Chromium: using GPU hardware in headless Chrome](https://chromium.googlesource.com/chromium/src/+/main/docs/gpu/using-gpu-hardware-in-headless-chrome.md)
- [tigerabrodi — WebGPU in headless Chrome on cloud GPUs (Feb 2026)](https://tigerabrodi.blog/how-to-get-webgpu-in-headless-chrome-on-cloud-gpus) — Dawn/Puppeteer traps
- [Chrome for Developers — Web AI testing in Colab](https://developer.chrome.com/docs/web-platform/webgpu/colab-headless)
- [Modal pricing](https://modal.com/pricing) · [RunPod pricing](https://www.runpod.io/pricing) · [Kaggle GPU docs](https://www.kaggle.com/docs/efficient-gpu-usage)
- [GitHub Actions GPU runners GA](https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/) · [RunsOn GPU](https://runs-on.com/runners/gpu/)
- [CircleCI GPU execution env](https://circleci.com/docs/using-gpu/) · [GitLab GPU runners](https://docs.gitlab.com/ci/runners/hosted_runners/gpu_enabled/)
- [wgpu CI uses lavapipe (raw)](https://github.com/gfx-rs/wgpu/blob/trunk/.github/actions/install-mesa/action.yml) · [Deno WebGPU](https://deno.com/blog/v1.39) · [@kmamal/gpu (Dawn-node)](https://github.com/kmamal/gpu)
- [tfjs #7631 — BrowserStack runs WebGPU on SwiftShader](https://github.com/tensorflow/tfjs/issues/7631)
- [MDN: GPUAdapter.isFallbackAdapter](https://developer.mozilla.org/en-US/docs/Web/API/GPUAdapter/isFallbackAdapter)
