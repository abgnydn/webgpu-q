# Level 3 — Kernel fusion

## Thesis fragment
> The dispatch ceiling collapses; gates/sec scales with memory
> bandwidth, not the number of driver calls.

## Why this level
E4 establishes a fixed per-gate dispatch cost α ≈ 100–500 μs on commodity
WebGPU. At N = 10, that α dominates — a single-gate-per-dispatch
scheduler spends more time in the driver than in the ALUs. Fusion runs
an entire chain of gates inside one `main()` function in WGSL,
amortizing α over k gates so the effective α drops to α/k.

## Status
Protocol only. Awaits `src/shaders/fused-chain.wgsl` + JIT generator in
`src/fusion/`.

## Baselines

- **This repo's own Level 1 numbers** — E4's α and β are the "before"
  datapoint.
- **cuQuantum custatevec** — published gate throughput on A100 / H100 is
  roughly memory-bandwidth-bound for long contiguous chains. Target
  parity on relative throughput (bw fraction), not absolute FLOPs.
- **Qiskit Aer GPU** — secondary baseline.

## Experiments

### E8 — Fusion correctness
- **Hypothesis:** For every chain of length k ∈ {1, 2, 4, 8, 16, 32} of
  single-qubit gates on the same qubit, the fused kernel produces a
  statevector matching the unfused (k × single-dispatch) path at
  F ≥ 1 − 1e-5.
- **Method:** Random gate chains, 20 trials per k, N ∈ {8, 16, 24}.
  Compare fused output to the Level 1 unfused reference.
- **Pass bar:** F ≥ 1 − 1e-5 on every cell; |norm − 1| < 1e-4.

### E9 — Dispatch-cost collapse
- **Hypothesis:** For chain length k, measured α_effective ≈ α_raw / k to
  within 20%. At k = 32, effective dispatch overhead is below 20 μs.
- **Method:** Re-run E4's measurement but with k-fused chains. Fit α and
  β for each k; plot α(k).
- **Pass bar:** α(32) ≤ α(1) / 16 AND α(32) ≤ 20 μs. If the fusion runtime
  composes the WGSL at request time, include compile cost in a separate
  "cold-start α" column — it must be amortized across a reasonable
  batch (≥ 100 dispatches) to be meaningful.

### E10 — Throughput ceiling on long circuits
- **Hypothesis:** On random brick-wall circuits with depth ≥ 40 layers
  and N = 20, the fused simulator reaches ≥ 2× the unfused throughput
  (gates/sec) at ≥ 60% of adapter peak BW.
- **Method:** Measure gates/sec for fused vs unfused across depth ∈
  {10, 20, 40, 80, 160} and N = 20.
- **Pass bar:** Speedup ≥ 2.0× at depth ≥ 40. Bandwidth ≥ 60% of peak.
- **Secondary:** JIT-compile latency as its own reported metric.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-3/E{8,9,10}-*.json`
