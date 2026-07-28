---
title: webgpu-q — quantum chemistry in a browser tab
emoji: ⚛️
colorFrom: indigo
colorTo: pink
sdk: static
app_file: learn.html
header: mini
pinned: false
short_description: Hartree–Fock to CCSD(T) in your browser. No install.
tags:
  - webgpu
  - webassembly
  - client-side
  - scientific-computing
  - quantum-chemistry
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# webgpu-q

The whole electronic-structure ladder — **Hartree–Fock, UHF, DFT (LDA/GGA/hybrid), MP2, CCSD, CCSD(T), EOM-CCSD** — running entirely in your browser tab. TypeScript plus a Rust/WASM SIMD integrals core. No Python, no BLAS, no CUDA, no server round-trip. The computation happens on your machine, in the tab, and the state is shareable as a URL.

**This Space opens on the interactive lesson.** Drag the water molecule and watch a real SCF converge as the geometry changes — that's live Hartree–Fock in a Web Worker, not a precomputed animation. Every other page is one link away.

## Why this exists

The interesting claim here is a **systems** claim, not a chemistry one. The methods are textbook and validated against PySCF; nothing new is asserted about molecules. What's demonstrated is *delivery*: that a class of computation everyone assumes needs a workstation, a Python environment, and an optimized BLAS fits inside a URL, stays fully inspectable, and reproduces the reference to sub-milliHartree — with that agreement enforced in CI, not claimed in a README.

The second result is the **browser-tab swarm**: the density-fitted Fock build partitioned across same-origin tabs over `BroadcastChannel` + `SharedArrayBuffer`, pushing HF past a single tab's memory wall to C₆₀ on a 16 GB laptop while reproducing the single-machine energy to ~1e-15. Cross-origin isolation is enabled on this Space, so the parallel path is live here.

## What it is not

It is not a PySCF replacement, and the repo says so with measurements. At production basis sets the pure-JS/WASM path is roughly **480× behind optimized BLAS** on the dominant GEMM, and WebGPU has no f64, which caps what the GPU path can do for correlated methods. Those are measured, documented, committed negatives — they're part of the result, not omissions from it. If you need production throughput, use PySCF or gpu4pyscf. Use this when zero-install, inspectability, or shareability is what actually matters.

## Try

- **`learn.html`** — drag water, watch HF converge; four depths from intuition to the numbers
- **`index.html`** — the main engine
- **`swarm.html`** — the multi-tab compute swarm. Open it at the direct host — `https://abgunaydin-webgpu-q.static.hf.space/swarm.html` — and fan out sibling tabs from there; the swarm needs same-origin tabs, not the embedded frame.
- **`screening.html`** — batch molecular screening
- **`experiments/`** — the research dashboard: pre-registrations, results, and the honest negatives

Requires a WebGPU-capable browser for the GPU paths; the CPU/WASM paths run anywhere modern.

## Provenance

This Space is a **pinned static snapshot**, not the canonical deployment.

- Snapshot: `v0.12.0` @ `ec257087725b`
- Canonical site: **https://webgpu-q.vercel.app**
- Source, methods, benchmarks, and pre-registered experiments: see the repository linked from the canonical site

The build stamps its own git SHA into the UI, so what you see running is traceable to the commit above.
