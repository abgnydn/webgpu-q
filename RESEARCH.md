# webgpu-q — Research Protocol

## Thesis

> **A full-featured quantum circuit simulator can run in a single browser
> tab on commodity hardware, reach bandwidth-bound performance, scale to
> 70+ qubits via MPS, and distribute across a WebRTC swarm — without
> requiring CUDA, a data-center GPU, or an install.**

Every level below decomposes that thesis into falsifiable experiments.

## Standards (apply to every experiment)

### Reproducibility

- Every experiment records the exact git commit SHA, `navigator.userAgent`,
  `adapter.info` (vendor / architecture / device / description), WebGPU
  device limits, OS, and timestamp (UTC ISO8601) alongside its numbers.
- Every random input is driven by a **named deterministic seed** from
  `experiments/lib/seeds.ts`. No `Math.random()` in an experiment path.
- Every JSON artifact includes `protocol`, `hypothesis`, `seed`, `trials`,
  `warmup`, `pass_bar`, so an outsider can re-run and compare.

### Timing

- All wall-clock measurements use `performance.now()` with **forced GPU
  sync before and after** (a mapped readback of a tiny buffer) — `queue.submit`
  is non-blocking.
- First **W = 5** samples per configuration are discarded (shader compile
  + warm-up).
- Next **T = 20** samples are retained.
- Reported stats: **median, p10, p90, p99, std, IQR**. Never single-shot.
- If `std/median > 0.1` for any cell, the experiment flags it as
  **NOISY** and the author must investigate before publishing.

### Correctness

- Quantum-state comparisons use **fidelity** F = |⟨ψ_ref | ψ_test⟩|², not
  just `max|Δp|`. Probability agreement is necessary but not sufficient —
  phases matter for any downstream controlled gate.
- Pass bar for a bit-for-bit test: **F ≥ 1 − 10⁻⁵** (f32 amplitude drift).
- Secondary metrics: **TVD** (total variation distance), **L1**, **L2**,
  **max|Δp|** all reported.

### Honest negative results

- If an experiment fails its pass bar, the JSON is still committed with
  `"status": "fail"` and a short `"diagnosis"` string. **Failures are
  the evidence.** No rerunning until it passes.

## The six levels

| # | Level | Thesis fragment | Experiments |
|---|-------|-----------------|-------------|
| 1 | Statevector | "a full-featured simulator runs in a browser tab and hits bandwidth-bound perf" | E1–E4 |
| 2 | MPS | "70+ qubits for low-entanglement circuits, bounded by χ" | E5–E7 |
| 3 | Kernel fusion | "dispatch ceiling collapses; gates/sec scales with memory, not driver calls" | E8–E10 |
| 4 | WebRTC swarm | "distributed statevector or MPS bonds across peers with ≤ 30 ms hops" | E11–E13 |
| 5 | Hardware cross-verify | "Heron r2 / Nighthawk agree with our simulator on shots/bitstrings within χ²" | E14–E15 |
| 6 | Chemistry → webgpu-dna | "VQE H₂ ground state agrees with FCI; cross-sections agree with Geant4" | E16–E17 |

Each level has its own `protocol.md` under `experiments/level-N-<slug>/`.

## Status

| Level | Status | Notes |
|-------|--------|-------|
| 1 — Statevector | **In progress.** E1 passing, E2–E4 scaffolded. | Piece one is real: 24 qubits, 51 GB/s, F ≥ 1 − 10⁻⁶ on Bell / GHZ / QFT. |
| 2 — MPS | Protocol only. | Awaits `src/mps.ts`. |
| 3 — Fusion | Protocol only. | Awaits fused-gate-chain shader. |
| 4 — Swarm | Protocol only. | Awaits WebRTC coordination layer. |
| 5 — Hardware | Protocol only. | Awaits IBM Quantum token + qiskit-runtime submission script. |
| 6 — Chemistry | Protocol only. | Cross-link to `webgpu-dna`. |

## References

- Pan & Zhang 2021 (arXiv:2103.03074) — Sycamore tensor-network contraction baseline (512 GPU × 15 h).
- Karamitros 2011 — Geant4-DNA IRT chemistry, cross-link to `webgpu-dna` validation.
- IBM Heron r2 (156q, 2025) — target for E14.
- IBM Nighthawk (120q, Jan 2026) — target for E14.
- cuQuantum, Qiskit Aer — classical simulator baselines.

## Bench logs

All run artifacts live under `experiments/results/<date>/<level>/<expt>.json`.
The top-level `experiments/results/README.md` indexes them chronologically.
