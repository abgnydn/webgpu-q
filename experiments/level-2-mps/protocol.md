# Level 2 — Matrix Product States (MPS)

## Thesis fragment
> The simulator scales to 70+ qubits for low-entanglement circuits, with
> error bounded by the bond dimension χ.

## Why this level
Statevector storage is 2^N × 16 B. At N = 30 that's 16 GB — past the
maxBufferSize limit of most WebGPU stacks and well past a consumer GPU's
VRAM. MPS compresses low-entanglement states to O(N · χ² · d) bytes (d = 2
for qubits). For χ = 32 and N = 70, that's under 1 MB of parameters.
Circuits with low entanglement (shallow random, QAOA at low p, 1D
nearest-neighbor Ising) stay in this regime.

## Status
Protocol only. Awaits `src/mps.ts` implementation (right-canonical MPS,
two-site TEBD updates, SVD truncation at bond χ).

## Baselines

- **Schollwöck (2011) DMRG / MPS review** — chi-vs-error curves for 1D Ising.
- **Vidal (2003) iTEBD** — canonical-form MPS algorithm we follow.
- **Statevector ground truth** — for every test N ≤ 20, compare MPS against
  the GPU statevector from Level 1 and take its fidelity as ground truth.

## Experiments

### E5 — MPS correctness vs statevector (N ≤ 20)
- **Hypothesis:** For every circuit with entanglement S ≤ log₂(χ) and
  every N ∈ [4, 20], the MPS state agrees with the statevector reference
  at fidelity F ≥ 0.999 using χ = 64.
- **Method:** 20 trials per (circuit, N). Circuits = GHZ, QFT, shallow
  random brick-wall (depth ≤ 4), 1D Heisenberg Trotter-step. Compute F
  against `QuantumCircuit` ground truth.
- **Pass bar:** F ≥ 0.999 on all cells at χ = 64. Separately record the
  smallest χ that achieves F ≥ 0.999 per cell — the *cheapest χ* curve.
- **Failure-mode map:** log (entanglement_entropy, χ, F) to distinguish
  MPS bugs from "this circuit genuinely needs χ > 64."

### E6 — Qubit-count ceiling
- **Hypothesis:** The MPS simulator reaches N ≥ 70 qubits in a single
  browser tab on commodity hardware at χ = 32 for shallow random circuits
  (depth ≤ 4), keeping wall-clock per TEBD sweep under 1 s.
- **Method:** Sweep N ∈ {32, 48, 64, 72, 96, 128}. Run one TEBD sweep per
  trial, 20 trials per N. Record median sweep time, memory footprint,
  total FLOPs estimate.
- **Pass bar:** Completes without OOM at N = 72. Sweep time ≤ 1.0 s median.
- **Negative result is evidence:** If N = 72 OOMs or exceeds 1 s per
  sweep, we publish the failure with χ, footprint, and the specific
  limit that fired.

### E7 — Entanglement-aware χ scaling
- **Hypothesis:** For Haar-random brick-wall circuits, required χ for
  F ≥ 0.999 scales as χ ≈ 2^(S(depth)) where S(depth) is the bipartite
  entropy at mid-cut. Confirms the simulator operates near the
  information-theoretic bound.
- **Method:** For depth ∈ {1, 2, 3, 4, 5, 6, 8}, N = 16, run 50 random
  seeds. For each (depth, seed) sweep χ ∈ {2, 4, 8, 16, 32, 64, 128} and
  record smallest χ achieving F ≥ 0.999.
- **Pass bar:** Fit χ_required vs depth on log scale; slope within 20% of
  theoretical prediction (S = depth · log₂(2) in 1D brick-wall).
- **Secondary:** Report ESEE (entanglement-entropy error envelope) vs
  expected.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-2/E{5,6,7}-*.json`
