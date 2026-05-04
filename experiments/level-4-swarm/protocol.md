# Level 4 — WebRTC swarm (distributed statevector / MPS)

## Thesis fragment
> A distributed statevector or MPS-bond lattice is tiled across peers
> over WebRTC data channels with ≤ 30 ms hops.

## Why this level
A single tab tops out at 24–30 qubits for statevector. A swarm of P
browser tabs can hold P × 2^N_local amplitudes, so 8 peers × N_local = 24
gives a logical N = 27. Gates whose partition touches two halves cost
one WebRTC round-trip per touch; gates inside a peer are free (Level 1
numbers apply).

## Status
Protocol only. Awaits `src/net/webrtc-coordinator.ts` and a signalling
endpoint (mock via localStorage for single-box experiments; STUN/TURN
for cross-device).

## Baselines

- **This repo's Level 1** for in-tab throughput.
- **Round-trip latency floor** — raw WebRTC unordered data-channel RTT
  on loopback (~0.1 ms) and across LAN (~1–3 ms) and over WAN (10–60 ms,
  varies wildly).
- **MPI-style distributed simulators (IBM Qiskit-ddsim, HiQ)** — only as
  an order-of-magnitude sanity check; absolute comparisons are
  meaningless across hardware classes.

## Experiments

### E11 — Swarm correctness (P = 2, 4, 8)
- **Hypothesis:** Distributed statevector across P peers produces final
  probabilities matching the single-tab ground-truth run at
  F ≥ 1 − 1e-4 (relaxed vs Level 1 because cross-peer reductions run in
  f32 over the network).
- **Method:** Fixed random seeded brick-wall circuits, N_logical ∈ {10,
  12, 14}. For P ∈ {2, 4, 8}, run distributed and compare against the
  Level 1 single-tab reference. All peers simulated in separate tabs on
  the same box first, then across two physical boxes via WAN STUN.
- **Pass bar:** F ≥ 1 − 1e-4, |norm − 1| < 1e-3.

### E12 — Hop-count and latency
- **Hypothesis:** Per-gate wall-clock = (local-gate-time) +
  k_cross · (RTT + 2^N_local · 8 B / link-BW), with k_cross ∈ {0, 1, 2}
  depending on where the gate qubits lie. Median RTT on LAN ≤ 30 ms.
- **Method:** 100 paired gates across 2, 4, 8-peer topologies, measured
  per-gate end-to-end including the sync. Plot wall-clock vs k_cross.
- **Pass bar:** median LAN RTT ≤ 30 ms; slope of wall-clock vs k_cross
  matches linear model within 25%.

### E13 — Straggler and fault tolerance
- **Hypothesis:** Losing one peer out of P = 8 mid-simulation triggers a
  recoverable re-shard within 2 s; final fidelity unchanged. Losing half
  the peers is detected and the job aborts with a legible error instead
  of silently corrupting state.
- **Method:** Inject peer drop at 50% of the circuit depth. Measure
  recovery wall-clock. Verify final F vs ground truth.
- **Pass bar:** Recovery < 2 s for 1/P loss. F ≥ 1 − 1e-4 after recovery.
  Quarter-peer loss → aborts cleanly (exit status in the artifact);
  half-peer loss → aborts.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-4/E{11,12,13}-*.json`
