# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/) starting
from `0.1.0`.

## [0.1.0] — 2026-05-04

First public release. The six-level research ladder is shipped through
levels 1, 2, 3, and 6 (chemistry); levels 4 (WebRTC swarm) and 5 (IBM
hardware) remain protocol-only.

### Added

- **Statevector simulator (Level 1)** — WebGPU compute kernels for single-
  and two-qubit gates, dispatch overhead α ≈ 22 μs on Apple Metal-3, scaling
  slope ≈ 1.0 in the bandwidth-bound regime. Experiments E1–E4 cover gate
  fidelity, bandwidth roofline, runtime scaling, and dispatch overhead.
- **MPS simulator (Level 2)** — Float64 TypeScript MPS with Jacobi complex
  SVD and canonical-form sweeps. Experiments E5–E7 cover correctness vs CPU
  statevector (180/180 cells), qubit-count ceiling, and χ-vs-entropy scaling
  (E7 ships as an honest negative — slope 0.45 instead of 1.0 because at
  N=16 entanglement entropy saturates around depth 4).
- **Kernel fusion (Level 3)**:
  - JIT-emitted WGSL chains for k same-qubit single-qubit gates (E8–E10).
    Best 2.5× speedup at N=20 D=160; α_eff(64) ≈ 11 μs.
  - Tier B brick-wall layer fusion via 4×4 dense kernel (E11) — best 2.69×
    speedup at N=20 D=80.
  - **Tier C 3-qubit cascade fusion via 8×8 dense kernel (E12)** — collapses
    3 singles + 2 CNOTs into one dispatch. **Best 4.18× speedup at N=15
    D=80** with worst F = 0.9999988.
  - **Tier D 4-qubit cascade fusion via 16×16 dense kernel (E13)** — 7 ops
    per tile collapsed. Best 3.14× speedup, ships as **honest negative**:
    per-block compute scales 4× per tier while memory traffic only 2×, so
    Tier D crosses into compute-bound territory and Tier C remains the
    bandwidth-bound sweet spot.
- **Quantum chemistry (Level 6)** — STO-3G molecular integrals from scratch
  (Boys F₀, contracted Gaussians, 4-center ERIs), full 16×16 dense H from
  occupation-number basis with Jordan-Wigner sign bookkeeping, VQE on the
  full H₂ dissociation curve. **Hits chemical accuracy (≤ 1.6 mHa) on 50/50
  random-init trials**; FCI matches PySCF literature to 7 decimals. Linear
  H_n chains via Löwdin orthogonalization (`hn-builder.ts`) up to H₄.
- **Many-body extension** — Hamiltonian1D library (Heisenberg, TFIM, XXZ),
  real-symmetric eigendecomposition, matrix exponential, imaginary-time
  ground-state evolution, real-time evolution, monitored / measurement-
  induced trajectories. Validated against ITensor DMRG to ≤ 5 mHa on N=8.
- **DMRG-v0 (`src/manybody/dmrg.ts`)** — exact ground-state via dense
  diagonalization + statevector → MPS conversion with chiMax truncation.
  9 tests including ITensor cross-check at N=8 to f64 precision (1e-7).
- **GPU MPS port** — multi-phase port of the CPU MPS to GPU-resident
  tensors:
  - Phase 1A: complex matrix multiplication on GPU (matches CPU at f32).
  - Phase 1B: Jacobi SVD kernels (small n ≤ 24, medium n ≤ 32).
  - Phase 2: full MPS evolution via GPU SVD (CPU↔GPU per gate).
  - Phase 4a: GPU-resident tensors with single-qubit kernel.
  - Phase 4b: full GPU two-site pipeline (merge → SVD → col-norms →
    extract-U → build-T_{q+1}).
  - Phase 5 v0: rectangular SVD via zero-padding to square (lifts the
    `chiL == chiR` constraint, deeper brick-walls now run end-to-end).
  - Phase 5 fast-path: GPU-side σ sort + single-submit per gate (eliminates
    the per-gate CPU↔GPU sync; ~25% per-gate cost reduction).
  - Phase 5 v1: canonical sweeps on GPU (`applyTwoSiteLeft` + `canonicalize`
    bring the chain to mixed-canonical form; arbitrary gate orderings work,
    not just brick-wall).
  - **Phase 6 v0**: large single-workgroup SVD kernel (n ≤ 48) for adapters
    with ≥ 37 KB workgroup storage.
  - **Phase 6 v1**: storage-mode SVD kernel — A/V live in global memory,
    only the 4 active columns of each (p, q) rotation enter shared memory.
    **n ≤ 64 on every adapter**, lifting χ_max to 32 universally.
- **Hyperscope (`/viz.html`)** — three synchronized 3D panels: H₂ electron
  density, conditional pair density (with draggable cursor showing the
  Fermi/Coulomb hole), and a live MPS bond-network. Models include
  brick-wall random circuits, TFIM ground state with phase-transition
  slider, Heisenberg ground state, TFIM quench (Lieb-Robinson light cone),
  and monitored trajectories (measurement-induced phase transition).
- **Mobile-first responsive layout** — every page (landing, hyperscope,
  experiments dashboard, GPU MPS bench, gate demo) tested at 390×844 with
  Playwright. Tables wrap into horizontal scrollers, hero CTAs stack
  full-width, viz panes stack vertically. Tier-1 and Tier-2 viz extensions
  (per-site Bloch arrows, order-parameter sweep, quench heatmap) all stay
  legible on phones.
- **Live deployment** at https://webgpu-q.vercel.app with COOP/COEP headers
  for SharedArrayBuffer-safe contexts.

### Validated against

- **ITensor DMRG** (Julia, `tools/itensor-reference.jl`) — 19 configurations
  across Heisenberg / TFIM / XXZ; exact-diag matches to 1e-7 on N ≤ 8;
  imag-time MPS matches to ≤ 5 mHa on N=8.
- **PySCF** — H₂ STO-3G FCI energy at R = 0.7414 Å matches to 7 decimals.
- **Bethe ansatz / Pfeuty** — analytical thermodynamic-limit energies as
  trend sanity-checks for Heisenberg and TFIM.

### Test surface

- 160 unit tests (Vitest) across linalg, gates, fidelity, stats, MPS, fusion
  (Tier B / C / D), chemistry (integrals, H₂, H_n), many-body, observables,
  trajectories, ITensor reference, DMRG-v0.
- 11 e2e specs (Playwright on headless WebGPU Chromium): landing smoke,
  viz smoke, level 1 / 2 / 3 / 6 full sweeps with JSON artifact dump,
  GPU MPS Phases 1A / 1B / 2 / 4a / 4b / 5 v1, OG image generator, mobile
  layout (390×844).

[0.1.0]: https://github.com/abgnydn/webgpu-q/releases/tag/v0.1.0
