# webgpu-q

> Quantum simulation in a browser tab.

A WebGPU quantum many-body playground — statevector, matrix product states,
kernel fusion, quantum chemistry, and real-time phase-transition / quench
dynamics — running on a research-grade harness with reproducible JSON
artifacts and external validation against ITensor and PySCF.

**Live:** [webgpu-q.vercel.app](https://webgpu-q.vercel.app) — no install, no
Linux, no Python. Open a tab.

```
160 unit tests · 11 e2e specs · ITensor cross-checked · MIT
```

---

## What's in the tab

| Page | What it shows |
| --- | --- |
| `/` | Landing — overview, ladder, companion projects |
| `/viz.html` | **Hyperscope** — 3 synchronized 3D panels: H₂ electron density, conditional pair density (with a draggable cursor finding the Fermi/Coulomb hole), and a live MPS bond-network with TFIM phase-transition slider, quench light-cone heatmap, and monitored-trajectory mode |
| `/experiments/` | **Research dashboard** — run E1–E16 live (gate fidelity, dispatch roofline, MPS correctness, kernel-fusion benchmarks, VQE on H₂ dissociation). Every run produces a downloadable JSON artifact with environment capture |
| `/experiments/gpu-mps/` | GPU MPS phase-by-phase bench (Phase 1A → 6 v1) |
| `/demo.html` | Original gate-throughput demo (Bell, GHZ, QFT, DJ) |

---

## The research ladder

| # | Level | Status | Headline |
| - | ----- | ------ | -------- |
| 1 | Statevector — WebGPU | ✅ shipped | Apple Metal-3, α ≈ 22 μs dispatch overhead |
| 2 | MPS — TypeScript | ✅ shipped | Validated against ITensor DMRG to ≤ 5 mHa on N=8 chains |
| 3 | Kernel fusion | ✅ shipped | **4.18× speedup** (Tier C cascade fusion, 8×8 dense kernel); Tier D 16×16 plateaus at 3.14× — honest negative |
| 4 | WebRTC swarm | 🚧 protocol-only | Two browsers sharing an MPS bond contraction; deferred |
| 5 | Hardware cross-verify | 🚧 protocol-only | IBM Quantum shot-level agreement; blocked on token |
| 6 | Chemistry — VQE | ✅ shipped | H₂ STO-3G FCI matches PySCF to **7 decimals**; full dissociation curve hits chemical accuracy 50/50 trials |

Plus a many-body extension (Heisenberg / TFIM / XXZ ground states + real-time
evolution + monitored trajectories with measurement-induced phase transition)
and the GPU-resident MPS port (Phase 1A → 6 v1).

---

## Research-grade discipline

Every experiment under `experiments/` enforces:

- **Reproducibility.** No `Math.random()` in any experiment path. Every
  random draw uses a named seed from `experiments/lib/seeds.ts`. Every JSON
  artifact records git SHA, `navigator.userAgent`, `adapter.info`, WebGPU
  limits, ISO8601 timestamp, and echoes back the protocol, hypothesis, pass
  bar, seed, warmup, and trial counts.
- **Timing.** `performance.now()` with a forced GPU sync (mapped readback)
  before AND after — `queue.submit` is non-blocking, so raw timing is
  fiction. Median + p10 / p90 / p99 + std + IQR over 5 warmup + 20 trials.
  Cells with `std/median > 0.1` get marked `"status": "noisy"`.
- **Correctness via fidelity.** `F = |⟨ψ_ref | ψ_test⟩|²`, not max|Δp|. Two
  states can share probabilities and differ in phase — that kills any
  downstream controlled gate. Pass bar for f32 GPU paths: `F ≥ 1 − 1e-5`.
  Pass bar for f64 MPS vs f64 statevector: `F ≥ 0.999`.
- **Honest negative results.** If an experiment fails its pass bar, the JSON
  is committed with `status: "fail"` and a `diagnosis` naming the smoking
  gun. Failures are evidence. Example: E7's χ-vs-depth slope (0.45 instead
  of 1.0) and E13's Tier D plateau (3.14× instead of 5×) both ship with
  the explanation rather than silent reruns.

See [`RESEARCH.md`](./RESEARCH.md) for the master protocol.

---

## Quick start

```bash
git clone https://github.com/abgnydn/webgpu-q
cd webgpu-q
npm install

npm run dev          # http://localhost:5175 — landing
                     # /viz.html   /experiments/   /demo.html

npm run test         # vitest (160 tests, ~3 s)
npm run typecheck    # strict, noUncheckedIndexedAccess
npm run lint         # eslint flat config

npm run build        # → dist/
npm run test:e2e     # playwright headless WebGPU Chromium
                     # ~1.5 min, dumps JSON artifacts + screenshots
```

WebGPU support is required (Chrome 113+, Safari 18+, Firefox Nightly with
flag, Edge 113+). For headless testing, Playwright launches Chromium with
`--enable-unsafe-webgpu --enable-features=Vulkan` (configured in
`playwright.config.ts`).

---

## Architecture

```
src/
  shaders/                   # WGSL kernels
    single-qubit.wgsl
    two-qubit.wgsl              # Controlled-U
    two-qubit-dense.wgsl        # 4×4 brick-wall fusion (Tier B)
    three-qubit-dense.wgsl      # 8×8 cascade fusion (Tier C)
    four-qubit-dense.wgsl       # 16×16 quad fusion (Tier D)
    jacobi-svd-{small,medium,large,storage}.wgsl
                                # Single-workgroup Jacobi SVD,
                                # n ≤ {24,32,48,64}
    mps-two-site-{merge,extract-u,extract-uS,build-tj,build-vh}.wgsl
    column-norms.wgsl
    sigma-sort.wgsl
  quantum.ts                 # GPU statevector (QuantumCircuit)
  cpu-reference.ts           # Float64 CPU reference (CpuCircuit)
  mps.ts                     # CPU MPS with canonical-form sweeps
  linalg.ts                  # Complex Jacobi SVD + matmul
  gates.ts                   # H, X, Y, Z, S, T, Rx/Ry/Rz, P
  circuits.ts                # Bell, GHZ, QFT, Deutsch-Jozsa, brick-wall
  two-qubit-dense.ts         # Matrix4 algebra + brick-wall pair fusion
  three-qubit-dense.ts       # Matrix8 + Tier C cascade fusion
  four-qubit-dense.ts        # Matrix16 + Tier D cascade fusion
  fusion-jit.ts              # JIT-emitted single-qubit chain shaders
  gpu-mps/                   # GPU-resident MPS port (Phase 1A → 6 v1)
    mps-gpu.ts
    jacobi-svd.ts
    buffer-pool.ts
    types.ts
  manybody/                  # Hamiltonian1D library
    hamiltonian.ts              # Heisenberg / TFIM / XXZ + buildDense
    dense-eig.ts                # Real-symmetric Jacobi diag
    expm.ts                     # Matrix exponential
    ground-state.ts             # Imag-time MPS ground state
    real-time.ts                # Trotter time evolution
    trajectory.ts               # Monitored / MIPT trajectories
    observables.ts
    dmrg.ts                     # Direct-diag → MPS conversion
  chemistry/
    integrals.ts                # STO-3G primitives, Boys F₀, ERIs
    h2-builder.ts               # H₂ Hamiltonian from integrals
    hn-builder.ts               # H_n linear chain (Löwdin OAO)
    h2.ts                       # Cross-check via OpenFermion-published Pauli table

tests/                       # Vitest, 160 tests
experiments/                 # Research dashboard, E1–E16 protocols
e2e/                         # Playwright specs
public/                      # Static assets (favicon, og-image)
RESEARCH.md                  # Master protocol document
```

---

## Validation

The repo cross-checks against three external references:

1. **ITensor DMRG** (Julia) — `tools/itensor-reference.jl` regenerates
   `tests/manybody/itensor-reference.json` with energies for 19
   configurations across Heisenberg / TFIM / XXZ. Our exact-diag matches to
   1e-7 on N ≤ 8; imag-time MPS matches to ≤ 5 mHa on N=8.
2. **PySCF** — H₂ STO-3G FCI at R = 0.7414 Å gives −1.13727008 Ha; our
   integral-built dense Hamiltonian agrees to **7 decimals**.
3. **Bethe ansatz / Pfeuty** — analytical limits for Heisenberg /
   TFIM serve as sanity checks on the trend with N.

---

## Companion projects

This is one front of a broader research line on GPU-resident compute in the
browser:

- [kernelfusion.dev](https://kernelfusion.dev) — research umbrella
- [webgpudna.com](https://webgpudna.com) — Geant4-DNA radiolysis chemistry in the browser
- [gpubench.dev](https://gpubench.dev) — WebGPU benchmark harness, 592 devices
- [zerotvm.com](https://zerotvm.com) — Phi-3-mini in 10 hand-written WebGPU kernels
- [neuropulse.live](https://neuropulse.live) — live transformer activations
- [barisgunaydin.com](https://barisgunaydin.com) — index of all the above

---

## License

MIT — see [LICENSE](./LICENSE).

Author: Ahmet Baris Gunaydin · [@abgnydn](https://github.com/abgnydn)
