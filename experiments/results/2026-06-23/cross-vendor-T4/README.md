# Cross-vendor validation — NVIDIA T4 (Google Colab), 2026-06-23

First non-Apple GPU sample in the project. Retires the n=1 (single Apple M2 Pro)
hardware sample behind every L1/L3 benchmark **for correctness**. Produced by
[`tools/colab/webgpu_t4_probe.ipynb`](../../../../tools/colab/webgpu_t4_probe.ipynb)
on a free Colab T4 via headless Chrome → Vulkan (see
[`docs/webgpu-ci-providers.md`](../../../../docs/webgpu-ci-providers.md)).

## Environment

| field | value |
|---|---|
| adapter | `vendor=nvidia, architecture=turing`, **`isFallbackAdapter=false`** (device/description masked by Chrome) |
| GPU | NVIDIA Tesla **T4** (Turing, GDDR6, 320 GB/s) |
| browser | HeadlessChrome/148, Linux x86_64 |
| flags | `--headless=new --use-angle=vulkan --enable-features=Vulkan --disable-vulkan-surface --enable-unsafe-webgpu --enable-dawn-features=…disable_adapter_blocklist` (Playwright's injected SwiftShader args stripped) |
| device limits | `maxBufferSize=4294967292`, `maxStorageBufferBindingSize=2147483644`, `maxComputeInvocationsPerWorkgroup=256`, `maxComputeWorkgroupStorageSize=49152` |
| hardwareConcurrency | 2 (shared Colab VM) |

## Result

**Correctness reproduces cross-vendor — every fidelity cell passes.**

| level | protocol | status | result |
|---|---|---|---|
| L1 | E1-gate-fidelity | **pass** | 23/23 cells, worst min **F = 0.999999287** (random-6L, N=14) |
| L1 | E2-bandwidth-roofline | noisy | timing noisy (real 320 GB/s peak via `?peak=320`) |
| L1 | E3-scaling-law | noisy | timing noisy |
| L1 | E4-dispatch-overhead | noisy | timing noisy |
| L3 | E8-fusion-correctness | **pass** | 360/360 cells, worst **F = 0.999997549** (N=24, k=32) |
| L3 | E9-dispatch-collapse | noisy | timing noisy |
| L3 | E10-throughput | noisy | timing noisy |
| L3 | E11-brickwall-fusion | noisy | correctness OK (F=1.0000000), timing noisy |
| L3 | E12-tier-c-fusion | noisy | correctness OK (F=0.9999988), timing noisy |
| L3 | E13-tier-d-fusion | noisy | correctness OK (F=0.9999995), timing noisy |

**383 / 383 correctness cells within pass bar** (L1 23 + L3 360).

## Scope — what this does and does NOT establish

- ✅ **Correctness, cross-vendor.** The f32 statevector (L1) and kernel-fusion
  (L3) WGSL produce results matching the f64 CPU reference at F ≥ 1−1e-5 on
  NVIDIA Turing — not just Apple Metal. This is the load-bearing claim and it
  reproduces.
- ❌ **Performance, NOT established here.** Every timing protocol flagged NOISY
  (`std/median > 0.1`) because the Colab T4 is a **shared, throttled** instance
  (`hardwareConcurrency=2`, noisy neighbors). The headline speedups (fusion
  4.22×, etc.) remain Apple-M2-only. A *dedicated* GPU (Modal dedicated / RunPod,
  not shared Colab) is required for trustworthy cross-vendor timing.
- ⚠️ **Indicative-only datapoint:** an earlier, equally-noisy run reported
  Tier-D fusion ≈ 2.91× on the T4 vs 3.78× on the M2 — suggesting the speedup is
  vendor-dependent (unified memory vs GDDR6), but Colab noise makes even that
  unreliable. Do not cite as a measurement.
- ⚠️ Chrome masks `adapter.device`/`description` on the web, so the roofline
  table can't auto-identify the card; the T4's 320 GB/s peak was supplied via
  `?peak=320`.

## Reproduce

Open the notebook in Colab (T4 runtime), Run all:
<https://colab.research.google.com/github/abgnydn/webgpu-q/blob/claude/blissful-davinci-7zm63h/tools/colab/webgpu_t4_probe.ipynb>

Full per-row artifacts (`webgpu-q-T4-runLevel1.json`, `-runLevel3.json`) are
saved + downloaded client-side by the notebook.
