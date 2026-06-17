# Collapsing the dispatch ceiling: kernel fusion for browser-native WebGPU quantum-circuit simulation

**Ahmet Barış Günaydın** · Independent researcher · `github.com/abgnydn/webgpu-q`

> **Canonical manuscript.** The full, current paper is the LaTeX source
> [`main-fusion.tex`](./main-fusion.tex) and its rendered PDF
> [`main-fusion.pdf`](./main-fusion.pdf), with figures `fig-dispatch.pdf` /
> `fig-ladder.pdf` and references [`refs-fusion.bib`](./refs-fusion.bib). This
> file is a readable abstract only — kept as a single source of truth alongside
> the LaTeX.

## Abstract

Sequential GPU workloads in the browser are throttled by a fixed per-dispatch
cost: every WebGPU compute dispatch pays a driver/submission overhead that, for
fine-grained work, dominates the arithmetic. We quantify this for browser
quantum-circuit simulation — a per-gate dispatch cost of α ≈ 5–500 µs on
commodity hardware — and show that **kernel fusion**, executing a chain of *k*
gates inside a single WGSL `main()`, drives the effective per-gate cost toward
α/*k* + C. Implemented as hand-written dense-tile WGSL kernels (2-, 3-, and
4-qubit dense operators), a tier ladder against the simulator's own unfused
multi-dispatch path reaches **3.04×** (brick-wall two-qubit) and **4.22×**
(three-qubit tile, Tier-C), both at fidelity F ≥ 0.99999. Four-qubit-tile
(Tier-D) fusion is an **honest negative** — it tops out at **3.78×** because it
crosses from the dispatch-bound into the compute-bound regime. Fusion is
correctness-preserving: across 360 (N, k, trial) configurations the worst
fidelity loss against the unfused statevector is 1 − F = 2.4×10⁻⁶. All results
run in a browser tab with no installation; every number is backed by a
committed, environment-stamped JSON artifact. We do not benchmark against native
GPU simulators (cuQuantum, Qiskit Aer) — those require hardware unavailable to
us — and position relative to them only through a bandwidth-fraction argument.

---

*See [`main-fusion.pdf`](./main-fusion.pdf) for the complete manuscript
(methods, the dispatch-ceiling measurement, the fusion tier ladder, the
compute-bound crossover, and figures).*
