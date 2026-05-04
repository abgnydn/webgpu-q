# Level 5 — Hardware cross-verification

## Thesis fragment
> Our simulator agrees with IBM Heron r2 / Nighthawk on output
> bitstrings / sampled probabilities within χ² goodness-of-fit.

## Why this level
A simulator that disagrees with hardware on the easy circuits isn't
interesting. The reverse is also true: if hardware noise swamps any
signal, we can't distinguish correct from incorrect sim. The bar is
"consistent within shot noise on low-depth circuits."

## Status
Protocol only. Awaits IBM Quantum API token and a `qiskit-runtime`
submission script.

## Baselines

- **IBM Heron r2 (156 qubits, late 2025)** — the target hardware.
- **IBM Nighthawk (120 qubits, Jan 2026)** — secondary target.
- **Qiskit Aer noiseless simulator** — the noiseless ground truth.
- **Qiskit Aer with Heron device-model noise** — the "what IBM's own
  simulator thinks Heron should produce" curve.

## Experiments

### E14 — Low-depth circuit agreement (GHZ, W, QFT-on-n)
- **Hypothesis:** For each of {GHZ_n, W_n, QFT_n} at n ∈ {5, 8, 12, 16},
  sampled bitstring distributions from Heron and from webgpu-q agree
  within the mid-circuit readout-error ceiling published by IBM's device
  data (typical 1–5% per qubit).
- **Method:** Fix 8192 shots on Heron, 8192 shots on webgpu-q (with
  sampled-amplitude statistical noise added to match shot count).
  Compute χ² statistic on the observed vs expected count
  distribution, restrict to the top-k most-likely bitstrings (k ≥ 32).
- **Pass bar:** χ²/dof ≤ 1.5 on the k-most-likely subset AND TVD between
  distributions ≤ published readout-error envelope.
- **Noise handling:** Ours is noiseless; IBM's is noisy. We interpret
  disagreement through IBM's own noise model (second baseline above) —
  disagreement with Heron raw is expected, disagreement with Heron under
  its device noise model is not.

### E15 — Entanglement witness
- **Hypothesis:** For GHZ_n prepared on hardware, the Mermin entanglement
  witness evaluated from measurement statistics matches the simulator's
  predicted value within one standard deviation of the 8192-shot Monte
  Carlo uncertainty.
- **Method:** Construct the Mermin operator for GHZ_n, measure the
  required Pauli strings on hardware, compute witness from count data.
  Simulator provides the theoretical value with Monte Carlo error bars
  matched to the shot count.
- **Pass bar:** |M_hw − M_sim| < 1.5 σ_sampling.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-5/E{14,15}-*.json`

Artifact must also include the Qiskit job IDs for re-verification.
