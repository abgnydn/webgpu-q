# Level 1 — Statevector simulator

## Thesis fragment
> A full-featured quantum circuit simulator runs in a browser tab and
> reaches bandwidth-bound performance on commodity hardware.

## Roofline baseline
Every gate on a statevector of 2^N amplitudes reads N·8 bytes (vec2<f32>)
and writes the same back. The theoretical lower bound on gate time is
`2 · 8 · 2^N / peak_bw`. The adapter's advertised memory bandwidth sets
the roofline.

| Device class | Peak BW (GB/s) | N=24 min time (ms) |
|--------------|----------------|---------------------|
| M2 Pro (LPDDR5, 200 GB/s) | 200 | 1.34 |
| M3 Max (LPDDR5, 400 GB/s) | 400 | 0.67 |
| RTX 4090 (DDR6X, 1008 GB/s) | 1008 | 0.27 |

A passing Level 1 run reaches ≥ 40% of the adapter's peak bandwidth at
N ≥ 20. Below 40% we call it dispatch-limited and escalate to Level 3
(fusion).

## Experiments

### E1 — gate fidelity vs CPU reference
- **Hypothesis:** For every circuit in the suite and every N ∈ [4, 16], the
  GPU statevector agrees with a Float64 CPU reference at fidelity
  F ≥ 1 − 1e-5.
- **Method:** For each (circuit, N), run 20 trials. Each trial builds a
  random brick-wall circuit with seed `E1_FIDELITY` and a per-trial sub-seed.
  Compute F = |⟨ψ_cpu|ψ_gpu⟩|², TVD, max|Δp|, ||Δp||₂.
- **Pass bar:** F ≥ 1 − 1e-5 AND |norm_gpu − 1| < 1e-4 on ALL trials.
- **Failure = evidence:** If any trial misses, commit the artifact with
  `status: "fail"` and diagnosis naming the circuit + trial index.
- **Secondary:** Record CPU vs GPU wall-clock speedup for context.

### E2 — bandwidth vs roofline
- **Hypothesis:** Gate throughput is bandwidth-bound at N ≥ 20. The
  measured effective bandwidth is ≥ 40% of the adapter's peak memory BW.
- **Method:** For N ∈ {16, 18, 20, 22, 24}, dispatch 64 single-qubit H gates
  to qubit 0 (the cheapest memory-bandwidth-bound primitive). Retain 20
  trials × 5 warmup. Effective BW = `2 · 8 · 2^N · n_gates / median_time`.
- **Pass bar:** For N = 22 and N = 24, effective BW ≥ 0.40 × peak BW.
- **Peak BW sourcing:** browsers often redact the GPU name (Chrome
  surfaces only "apple metal-3", not the specific M-series chip).
  Lookup order: (1) `opts.peakGbps` programmatic override, (2) URL
  `?peak=<GB/s>` query param, (3) regex table in `E2-bandwidth-roofline.ts`
  against `adapter.description`. If none resolve, peak = 200 GB/s
  placeholder and the run is flagged `"noisy"` — the raw GB/s number is
  still measured and reported, just not gated against a peak bar.
- **Diagnosis surface:** If it fails, report (effective BW, peak BW, ratio,
  median gate time, gates/sec) so we know whether it's dispatch-limited
  or memory-limited.

### E3 — runtime scaling T(N)
- **Hypothesis:** Per-gate time scales linearly with state size in the
  memory-bound regime: T(N) = α + β · 2^N ≈ β · 2^N once β · 2^N ≫ α.
  Fit log–log slope; expect 1.00 ± 0.08 at N ≥ 22.
- **Method:** For N ∈ {12, 16, 18, 20, 22, 24}, run a fixed brick-wall
  circuit with 8 layers, seed `E3_SCALING`. Record median per-gate time.
  Fit slope of log(t) vs log(2^N) over **N ≥ 22**.
- **Pass bar:** Slope ∈ [0.92, 1.08] AND R² ≥ 0.95 (R² waived if the
  regime contains only 2 fit points — two-point fits produce trivial
  R² = 1).
- **Why N ≥ 22:** E4 puts α ≈ 25–100 μs on consumer WebGPU. At N = 22,
  per-gate time reaches ~400 μs → α is only ~6% of the signal. At
  N = 20, α is still ~45% → fitting there mixes regimes. For N < 22 the
  row is recorded for context but excluded from the fit.

### E4 — dispatch overhead
- **Hypothesis:** Per-gate cost decomposes as T(N) = α + β · 2^N. α is
  the fixed submit + sync cost in microseconds; β is the per-amplitude
  cost in nanoseconds. At N = 4, α dominates (16 amplitudes do not move
  the needle); by N = 18 β · 2^N dominates. Expect α ∈ [5, 500] μs.
- **Method:** Dispatch 256 single-qubit H gates at N ∈ {4, 6, 8, 10, 12,
  14, 16, 18}. The low-N half characterises α (flat regime); the high-N
  half characterises β (linear regime). Linear-fit T̄(N) = α + β · 2^N
  via ordinary least squares on the medians.
- **Pass bar:** α ∈ [5, 500] μs AND β > 0 AND R² ≥ 0.85. Run status
  becomes NOISY if > 50% of cells flag std/median > 0.1 (individual noisy
  cells are noted in diagnosis but do not block).
- **Motivation:** This is the floor that Level 3 (kernel fusion) has to
  attack. Publishing it as a baseline makes the fusion improvement real.

## Artifacts

Each experiment writes `experiments/results/<YYYY-MM-DD>/level-1/E<k>-<slug>.json`
via `downloadArtifact()`. The top-level `run-all.ts` orchestrator emits a
summary index with status for each E1–E4 row.

## Running

```
npm run dev          # then open /experiments/index.html
# OR
npm run dev          # then open /experiments/index.html
```

Click "Run E1–E4" in the panel. Each artifact is offered as a JSON download
and mirrored to the console. Commit the JSON to
`experiments/results/<date>/level-1/`.
