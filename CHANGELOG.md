# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/) starting
from `0.1.0`.

## [0.2.0] — 2026-05-07

The chemistry track went from "VQE on H₂" to a full-spectrum
computational-chemistry tool: HF / DFT / MP2 / CCSD / CCSD(T) /
CIS / TDA / TDDFT, analytical gradients on every level, geometry
optimization, UV-vis spectra, and ground-state property analysis —
all in a browser tab, all PySCF / libxc cross-checked.

### Added — Tier 1 quick-wins bundle

- **DIIS** SCF accelerator: H₂O cc-pVDZ HF 101 → 14 iter (7.2× faster).
- **Frozen-core** option on MP2 / CCSD / CCSD(T).
- **Spherical-harmonic d-shell** basis (`spherical: true` opt) — kills
  the documented Cartesian-vs-spherical-d 4 mHa slack on cc-pVDZ H₂O.
- **f / g / h orbital integrals** via rewritten `boysAll` with per-n
  Taylor + closed-form recurrence anchors. Max relative error at n = 12
  dropped from 1.5e-2 to 8e-10 — unblocks cc-pVTZ and beyond.
- **aug-cc-pVDZ** diffuse functions (H + O wired). HF/H₂O matches
  PySCF to 50 µHa; 14 mHa lower than cc-pVDZ.
- **Schwarz integral screening** in the AO ERI build (Q[μ,ν] threshold
  1e-10) — 2-5× ERI speedup at cc-pVDZ scale.

### Added — Tier 2: ground-state methods

- **Tier 2 stage 1 — geometry optimization on the HF surface.**
  L-BFGS minimization of E_HF over atomic positions with central-FD
  gradients. Sub-mÅ + sub-degree agreement with PySCF references on
  H₂ / H₂O / BeH₂.
- **Tier 2 stage 2 — closed-shell RKS-DFT (LDA = Slater + VWN5).**
  Becke-partitioned molecular grid (Becke M3 radial × Gauss-Chebyshev ×
  Gauss-Legendre × uniform-φ), default 50r × 12θ × 24φ per atom
  integrates ρ to 10⁻⁵–10⁻⁷ e. Matches PySCF SVWN5 within 5 mHa on the
  STO-3G molecule set.
- **Tier 2 stage 3 — GGA + B3-style hybrid DFT.** B88 GGA exchange,
  density-gradient evaluator on the grid, hybrid Fock build with HF
  exchange mixing. Three new functionals: `bvwn5`, `b3vwn5` plus the
  retained `lda-svwn`.
- **Tier 2 stage 4 — LYP correlation + B3LYP5 hybrid.** Closed-shell
  collapse of the LYP correlation kernel cross-checked against the
  libxc Maple source (`gga_c_lyp.mpl`). Caught and avoided a sign-error
  prone hand-collapse with a 20-test FD self-test as a forward moat.
  Two new functionals: `blyp` and the published `b3lyp5`.
- **Tier 2 stage 5 + 5b — analytical HF gradients (Pulay 1969).**
  Primitive integral derivatives via the bra-side Hellmann-Feynman shift
  `2α·prim(I+1) − I_axis·prim(I−1)`. FD-validated to **1e-5 Ha/Bohr**
  per component, translational invariance to 1e-9. 8-fold canonical
  ERI loop + Schwarz screening: H₂O STO-3G gradient **4500 ms → 440 ms
  (10× faster)**, geom-opt **52 s → 6.6 s (8× faster)**.
- **Tier 2 stage 6 + 6b — analytical RKS-DFT gradients.** Same Pulay
  machinery for the HF-like part (with `kFactor = hfMix`); LDA XC term
  via `∂ρ/∂R` on the existing gradient grid; GGA term via
  `∂γ/∂R = 2·∇ρ·∂(∇ρ)/∂R` with new basis-Hessian evaluator (FD-validated
  to 2e-8). Works for all 5 functionals. H₂O LDA STO-3G geom-opt:
  **55.4 s FD → 7.6 s analytical** (7.3× faster).
- **Tier 2 stage 7 — Lebedev angular quadrature.** Lebedev-Laikov 1999
  tables for orders 50, 110, 302 (Christoph van Wuellen Fortran via
  PySCF), expanded via the canonical `genOh` 6-symmetry-class octahedral
  group. Default order 110 (exact for L ≤ 17). H₂O STO-3G: 43200 → 16500
  grid points; SCF **119 ms → 55 ms (2.2× faster)**; energy difference
  31 µHa (sub-chemical accuracy).

### Added — Tier 2: excited states + properties

- **Tier 2 stage 8 — CIS / TDA excited states.** Closed-shell singlet +
  triplet excitation energies via direct dense diagonalization of the
  CIS A matrix in the (occ → virt) singles manifold. H₂ STO-3G first
  singlet at 0.947 Ha = 25.7 eV — matches the textbook reference.
- **Tier 2 stage 9 + 9b + 9c — TDA-DFT and full TDDFT.** First-derivative
  XC kernel `f_xc` via numerical FD on `evalXC` (LDA + GGA + hybrid).
  Full Casida `(A − B)·(A + B) Z = ω² Z` via `M = (A−B)^(1/2)·(A+B)·
  (A−B)^(1/2)` + `eigsymmetric`. Singlet sector across the full
  functional ladder: HF, LDA, BVWN5, BLYP, B3VWN5, B3LYP5. H₂O STO-3G
  first singlet TDA: 13.20 eV (HF) → 11.31 eV (BLYP) → 11.72 eV
  (B3LYP5); TDDFT uniformly ≤ TDA per state — textbook B-correction.
- **Tier 2 stage 10 — oscillator strengths.** Dipole AO integrals
  (`dipole_cg`) via primitive overlap with shifted angular momentum
  + A_axis·S trick. `f_n = (4/3)·ω_n·Σ_axis|T_axis|²` per root for
  TDA; `f_n = (4/3)·Σ_axis|Σ(S·Z')·μ|²` (ω-cancelled) for TDDFT.
  H₂O STO-3G state-2 carries f ≈ 0 by point-group symmetry; state-5
  carries the dominant intensity ~1.1.
- **Tier 2 stage 11 — ground-state dipole moments.** `dipoleMoment`
  helper with `AU_TO_DEBYE` constant. H₂O HF/STO-3G: 1.726 D vs
  experiment 1.85 D (STO-3G truncation explains most of the gap).
- **Tier 2 stage 12 — Mulliken population analysis.** Per-atom partial
  charges `q_A = Z_A − Σ_{μ on A} (P·S)_μμ`. H₂O STO-3G: O = −0.366 e,
  H = +0.183 e (HF); LDA most-polar, GGA least — well-known
  DFT-vs-HF trend.
- **Tier 2 stage 13 — Wiberg-Mayer bond orders.** Per-pair shared-
  electron counts `B_AB = Σ_{μ ∈ A, ν ∈ B} (P·S)_μν · (P·S)_νμ`.
  H₂: 1.000 (single bond); H₂O: B_OH = 0.954 each, valence on O = 1.91;
  BeH₂: B_BeH = 0.998, valence on Be = 2.00.

### Validated against

- **PySCF** for HF / DFT / MP2 / CCSD / CCSD(T) energies on H₂ / H₂O /
  BeH₂ / CH₄ / STO-3G + cc-pVDZ to ≤ 0.5 mHa (35 µHa with spherical-d).
- **libxc** for LYP closed-shell collapse and the GGA TDDFT XC kernel
  (cross-referenced against `gga_c_lyp.mpl`).
- **18 FD-vs-analytical self-tests** on integral derivatives — overlap,
  kinetic, nuclear, ERI gradients, basis Hessians, LDA/GGA XC kernel.

### Test surface

- 433 unit tests (was 160), ~50 s on M2 Pro.
- 11 e2e Playwright specs (unchanged).
- typecheck strict, lint pre-existing only.

[0.2.0]: https://github.com/abgnydn/webgpu-q/releases/tag/v0.2.0

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
