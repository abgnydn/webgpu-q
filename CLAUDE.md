# webgpu-q — CLAUDE.md

Project-local instructions for Claude. Load this first.

---

## One-paragraph read

WebGPU quantum circuit simulator. Runs in a browser tab. Target: piece one of
a six-level research ladder — statevector → MPS → kernel fusion → WebRTC
swarm → IBM hardware cross-verify → quantum chemistry. Each level is a set
of **research-grade experiments** (not just benchmarks): named seed, warmup,
trials, fidelity pass bar, honest negative results. The master doc is
`RESEARCH.md`. Per-level protocols live under
`experiments/level-N-<slug>/protocol.md`.

**Communication mode: hero.** Terse, bold, first-principles, attempt-first.
Scope-honest. See `~/.claude/skills/hero/SKILL.md`.
**Project skill: `webgpu-q-research`.** See `~/.claude/skills/webgpu-q-research/SKILL.md`.

---

## Roadmap to the frontier

The project is past the launchpad. **All six chemistry-track phases are
shipped** (A through E5: foundation → 1D records → real molecules → HF
SCF → MP2 → cc-pVDZ basis → CCSD → CCSD(T) → cc-pVDZ CCSD(T) on H₂O).
Repo is public + CI-green. Path forward is ranked by *what it costs vs
what it unlocks*, not by ladder position.

### Shipped (recap)

- ✓ **L1 statevector**, **L2 MPS** (incl. GPU MPS Phase 6 v1, χ ≤ 64),
  **L3 kernel fusion** (Tier B/C/D — 4.18× headline), **L6 chemistry**
  (full quantum-chemistry stack)
- ✓ **DMRG** with Lanczos + MPO; ITensor cross-checked at N = 8 to f64
- ✓ **Phase B**: TFIM/Heisenberg N = 128 in browser, validated vs Pfeuty/Bethe
- ✓ **Phase C/D/E1-5**: HF / MP2 / FCI / CCSD / CCSD(T) on
  H₂ → LiH → BeH₂ → H₂O → CH₄ in STO-3G; **cc-pVDZ CCSD(T) on H₂O in 106 s**
- ✓ **Tier 1 bundle**: DIIS, frozen-core, spherical-d, f/g/h, aug-cc-pVDZ,
  Schwarz screening
- ✓ **Tier 2 stages 1–23**: geometry optimization → DFT/LDA → GGA + hybrids
  (BVWN5, BLYP, B3VWN5, B3LYP5) → HF + DFT analytical gradients → Lebedev
  grids → CIS / TDA / TDDFT (full functional ladder) → oscillator strengths
  → dipole moments → Mulliken + Mayer-Wiberg analysis → triplet TDA/TDDFT
  (full ladder via spin-polarized LSDA + B88 + LYP) → vibrational
  frequencies + IR + Raman + thermo → polarizability + hyperpolarizability
  → UHF + ΔSCF ionization potentials + electron affinities → molecular SI
  report page (`/molecule.html`)
- ✓ Reference-grounded validation green in CI (PySCF / FCI / ITensor /
  Pfeuty-Bethe gates); e2e browser benches across all levels

### Next: chemistry-track tier roadmap

Ranked by ROI. One focused session ≈ a few hours.

#### Tier 2 — ALL SHIPPED through stage 38 (2026-05-12)

| feature | status | unlocks |
|---|---:|---|
| **DFT (LDA + B3LYP + Lebedev grids)** | ✓ | **~90% of all real chemistry** |
| **HF analytical gradients + BFGS** | ✓ | **geometry optimization** |
| **WebGPU port of (T) kernel** | ✓ (39× on H₂O cc-pVDZ, single-run) | 10-100× speedup; cc-pVTZ CCSD(T) routine |
| **EOM-CCSD (excited states)** | ✓ (+ eigenvectors, oscillator strengths, spin classifier) | UV-vis, photochemistry |
| **UHF + open-shell CCSD** | ✓ (UHF stage 21, UCCSD stage 25) | radicals, transition metals |
| **Density fitting (RI)** | ✓ correctness + **speedup** (aux-basis 3-index DF now shipped: `buildAuxBasisDFStreaming` WASM, never builds the 4-index ERI; the old CD-DF 11-20× regression is retired) | half memory + faster |
| **WebGPU aux-basis DF integrals** | ✓ (`df-gpu.ts`: s/p/d McMurchie–Davidson 3-index V + 2-index metric in WGSL f32, validated ~1e-4 rel; `buildDFAuto` auto-selects GPU in the d-regime) | GPU integral build, 1.1-1.35× d-regime |
| **Fully-GPU DF-HF SCF** | ✓ (`makeGpuDFJK` + `buildDFAuto`: whole HF loop on GPU from a URL, no 4-index ERI; benzene cc-pVDZ **5-6× faster** whole-loop vs WASM; level-0 aux → ~30 mHa screening; f32 JK floor ~6e-4 is element-precision) | fast browser HF (screening) |
| **Hybrid GPU/WASM DF — EXPERIMENTAL** | ✓ but **demoted (decision 2026-06-10)**: `buildV3idxHybrid` is chemistry-grade (DF-vs-exact ladder H₂O 0.19 / CH₂O 0.53 / C₂H₄ 0.36 mHa, `e2e/df-accuracy-ladder`) but the win is only **~1.3× on the integral BUILD in a medium band** — WebGPU has no f64 so it can't touch the f64-bound J/K. **f64 WASM is the recommended chemistry default**; the GPU hybrid is `fast=true` opt-in, proof-of-mechanism only. | marginal; not the default |
| **`runRHFAuto` / `runRKSAuto` entry points** | ✓ (`rhf-auto.ts`: size-gated exact(small)/f64-DF(large)/hybrid-GPU(fast) with honest provenance, for both HF and **DFT** — `runRKSDFT` gained a `useDF` option; pure functionals ride the cheap DF J, hybrids the DF K; validated H₂O LDA 0.07 mHa / B3LYP5 0.02 mHa vs exact) | one call, right method, attributed, HF+DFT |
| **IP-EOM-CCSD / EA-EOM-CCSD** | ✓ (stages 37–38, beyond original Tier 2 plan) | accurate IPs / EAs |

#### Tier 3 — Substantial (~25 sessions)

CCSDT (full triples), CASSCF (multi-reference), TD-DFT, MP2/CCSD
gradients (Z-vector), PCM solvent, coupled-perturbed HF (NMR /
polarizabilities). (WebGPU integral parallelization: DF 3-index/metric
+ DF-JK now shipped — see Tier 2; full 4-index ERI on GPU still open.)

#### Tier 4 — Genuinely hard (a season each)

CASPT2 / NEVPT2, periodic DFT (k-points), spin-orbit / X2C,
analytical CC gradients, QM/MM.

### Deferred moonshots

- **Phase D (WebRTC swarm)** — distributed 1D chain across browsers.
  ~3-5 sessions. Reuse `webgpu-p2p-evolution`'s relay.
- **E.1 — Verify Sycamore** — 2D PEPS + Sycamore gates + distributed
  contraction. ~3-5 sessions on top of Phase D.
- **E.2 — Fault-tolerant qubit** — stabilizer sim + surface code +
  syndrome decoder + threshold curve. ~4-6 sessions.
- **E.3 — Browser-native lattice QCD** — 4D lattice + Wilson Dirac +
  fused CG solver. ~6-10 sessions.

### Cleanest near-term path

**WebGPU (T) → EOM-CCSD → DFT excited-state properties**.
~5-6 sessions to "real chemistry tool in a browser tab, with speed."
Every step ships a publishable artifact.

Unifying thesis: *"every advanced physics simulation in the world ships
as a URL"*. webgpu-q is the proof point; the chemistry track is its
highest-leverage demonstration.

---

## Current state (one-pager — for per-stage detail: `git log`)

**Headline numbers**:
- L1 statevector: F ≥ 0.999999 vs CPU; 4-experiment ladder (E1–E4) green.
- L2 MPS / DMRG: TFIM & Heisenberg N=128 in browser, χ=32, validated to
  Pfeuty/Bethe limits at 1/N. ITensor cross-checked at N=8 to f64.
- L3 kernel fusion: **4.18× headline** (Tier C, 8×8 cascade); Tier D plateau
  is the documented honest negative.
- L6 chemistry: HF (≤ 50 µHa vs PySCF w/ spherical d) → MP2 → FCI (CH₄ to
  0.76 mHa) → CCSD (≥ 99% capture) → **CCSD(T)** (≤ 0.25 mHa vs FCI).
  cc-pVDZ CCSD(T) on H₂O — CPU 116 s, GPU **13.8× median** (5 warmup +
  20 trials, M2 Pro; p10=28×, p90=10×, std/median 42% noisy).
  Full DFT ladder (LDA/GGA/B3-hybrid) on RHF/UHF/RKS/UKS.
  Full {α, α(ω), α(iω), C₆} response matrix.
  EE/IP/EA-EOM-CCSD with eigenvectors, oscillator strengths, spin classifier.
  **DF engine = f64 WASM (recommended default).** `runRHFAuto` + siblings are
  size-gated: small → exact ERI (f64), large → streaming aux-basis DF (f64 WASM,
  `buildAuxBasisDFStreaming`), with honest method/engine/precision provenance.
  **The GPU paths are EXPERIMENTAL (decision 2026-06-10), not the chemistry
  default.** WebGPU has no f64, so the chemistry-grade hybrid (`buildV3idxHybrid`,
  `fast=true`) can only put f32 on the insensitive s/p/d-aux columns (~8 µHa) and
  buys just **~1.3× on the integral BUILD in a medium band** — it can't touch the
  f64-bound J/K, and loses at PAH scale. The fully-GPU f32-JK path (`makeGpuDFJK`,
  benzene 5-6× whole-loop) is ~30 mHa **screening only**. Both kept as
  proof-of-mechanism ("GPU in the browser") and as the seam where a real win lands
  IF df64 emulation ever makes the GPU JK chemistry-grade. GPU genuinely wins on
  the f32-tolerant tracks (statevector, kernel-fusion 4.18×, (T) 39×) — DF
  chemistry just isn't one (needs f64).

**Live**: https://webgpu-q.vercel.app — landing, `/viz.html` (4D
hyperscope), `/molecule.html` (SI report), `/experiments/` (E1–E33+).
**Standing preference: do NOT auto-deploy** — deploy only when explicitly asked.

**Validation surface** (what's checked, not how many): reference-grounded
gates green in CI — bit-exact / sub-µHa vs PySCF, EOM-CCSD full-tensor
brute-force diffs (14×14 LiH), CCSD(T) sub-mHa vs FCI, ITensor N=8 and
Pfeuty/Bethe 1D limits, swarm partition-sum vs single-slab below 1e-12.
`npx tsc --noEmit` clean, `npm run lint` clean, vitest green, e2e browser
benches (`e2e/`) cover all levels + the swarm/acene series.

**Honest negatives / open work** (each its own session):
- IP-EOM-CCSD: **PySCF-ported (2026-05-22)**. σ_1 + σ_2 follow
  Tu-Wang-Li 2012 Eqs (8)-(9) with PySCF eom_gccsd intermediates. The
  earlier R_2 satellite over-count (~60 eV on H₂) is closed; brute-force
  H₂ diff < 1e-10 Ha element-by-element.
- **Phase D — swarm shipped (2026-05-22), all three steps**:
  - Step 1: `swarmMap(items, fn)` primitive + `BroadcastChannelTransport`
    (same-origin multi-tab, no infra).
  - Step 2: `WebRTCTransport` via PeerJS broker (cross-machine, NAT-
    traversal via Google STUN; symmetric-NAT corporate networks may
    need TURN, documented).
  - Step 3: real-chemistry kernel — `chem-energy` runs a molecule tile,
    swarm distributes H₂ bond-length scans (and any 1D parameter scan)
    across tabs. /swarm.html ships both prime-counting (Demo 1) and
    bond-scan (Demo 2) demos.
  - Step 4 (**swarm × GPU**, 2026-06-09): the kernel now runs `runRHFAuto`
    per tile, so each worker tab auto-picks exact / hybrid-GPU DF and
    reports its own provenance. `e2e/swarm-gpu` distributes an N₂ cc-pVDZ
    batch across 2 tabs, every tile `gpu+wasm`, tracing N₂'s bond curve
    (min r=1.098 Å). GPU-accelerated chemistry-grade single-points split
    across the crowd — the project's two theses in one demo.
  - Step 5 (**distributed DF-MP2 + honest-negative measurement**, 2026-06-15):
    the swarm's first *collaborative single-molecule* reduction — ONE molecule's
    MP2 correlation energy `E_corr = Σ_i Σ_j Σ_ab …` partitioned over the outer
    occupied index `i`, each tab owns an i-slice, master sums the scalar partials
    (`mp2-slice` kernel + `mp2EnergyDF(...,iRange)` + `reduceMP2Slices`). Comm-
    optimal (spec in, one f64 out, one round); `reduceMP2Slices` guards the
    deterministic-reference assumption (throws if per-tab E_HF disagree). Validated:
    partition-sum == single-machine to <1e-12 (`tests/chemistry/mp2-slice`), 2-tab
    e2e to <1e-9 (`e2e/swarm-mp2-distributed`). **HONEST NEGATIVE — distributing
    the contraction barely speeds up one molecule** (`e2e/swarm-mp2-speedup`,
    single-shot M2 Pro): H₂O 0.51×, benzene cc-pVDZ **1.10×** (single 96s / 2-tab
    87s). Breakdown: redundant SCF+DF setup S≈79s (82%, paid in full on every tab,
    on the critical path) vs splittable grind C≈17.5s (18%). speedup=(S+C)/(S+C/k)
    is pinned near 1 while S≫C; C≫S needs n≈600 whose DF tensor (~5 GB) won't fit
    a tab. **The swarm's scaling axis is throughput (N independent molecules via
    chem-energy), NOT single-molecule wall-time.** To speed up one molecule you'd
    have to parallelize the SCF+DF *setup*, not the correlation.
  - Step 6 (**screening + honest multi-tab scaling**, 2026-06-15): the throughput
    axis demonstrated. `chem-energy` now returns the HOMO–LUMO gap (eV) as a
    screening descriptor; `e2e/swarm-screening` ranks a 10-molecule library by
    gap, validated to give the IDENTICAL ranking distributed vs single-tab (that
    spec validates ranking correctness only — its timing is indicative).
    `e2e/swarm-scaling` is the honest measurement (warmed JIT per tab, even
    round-robin split, true parallelism, wall = slowest tab): **1→1.00×, 2→1.73×
    (87% eff), 3→2.02× (67%), 4→2.36× (59%)** on the library, HF/cc-pVDZ, M2 Pro.
    Sub-linear because molecule costs are uneven (H₂ ≪ C₂H₄) so the tab holding
    the heavy molecules caps the win. Further efficiency would need cost-aware
    scheduling (big molecules first/alone) + a larger library.
    (Earlier screening "1.59×" was retracted — it was warmup-inflated + a
    master-heavy auto-distribution; the warmed/balanced 2-tab number is 1.73×.)
  - Step 7 (**greedy-pull scheduler fix**, 2026-06-15): rewrote `swarmMap`'s
    distribution from single-claim-per-worker (master-heavy: a worker grabbed ONE
    tile then idled while the master ran the rest via a timeout fallback) to a
    **greedy pull queue** — every peer, master included, pulls one tile at a time
    and requests another only after finishing, so a slow tile parks only its own
    puller while everyone else keeps draining (also auto-balances uneven tile
    costs). Two subtleties fixed during the rewrite: (a) a failing worker's tile
    is run by the MASTER, not requeued to the shared pool — requeueing let a
    persistently-failing worker re-pull and re-fail the same tile in a tight loop
    (livelock, hung `swarm.test.ts`); (b) the master does a one-time head-start
    yield + keeps yielding while remote workers pull, because local kernel compute
    blocks the single-threaded event loop and would otherwise drain the queue via
    microtasks before any remote macrotask is processed. Result: auto-distributed
    screening went 9/1 → 4/6 (master/other), 1.05× → 1.48×; `tests/parallel/swarm`
    (13 tests) green; distributed-MP2 reduction still bit-exact.
- EE-EOM-CCSD: **PySCF-ported (2026-05-21)**. σ_1 + σ_2 follow
  Wang-Tu-Wang 2014 Eqs (9)-(10) with PySCF eom_gccsd intermediates
  (EOM-Fvv/Foo/Wovvo with full t2 dressing + Wovoo / Wvvvo). EE's empirical
  stage-32c patch removed; brute-force LiH diff < 1e-10 Ha
  element-by-element. H₂ STO-3G now matches FCI to 8+ decimals.
- EA-EOM-CCSD: **PySCF-ported + multi-electron-validated (2026-06-16).** The
  2026-06-16 audit surfaced that `ea-eom-ccsd.ts` carried an empirical
  `+½·E_corr·R₂` σ_2 diagonal patch (stage-32e) curve-fit to the H₂ brute-force
  (the diagnostic-loop-trap anti-pattern) — and used BARE integrals where PySCF
  uses dressed Wvvvo/Wvovv. A NEW multi-electron oracle
  (`tests/chemistry/ea-eom-ccsd-bruteforce-lih.test.ts`, LiH NSO=6, T̂²≠0)
  measured the patch ~1 mHa wrong. σ is now a **direct port of PySCF
  eom_gccsd.eaccsd_matvec** onto the shared dressed intermediates
  (`buildEOMIntermediates`: Fvv/Foo/Fov/Wvvvv/Wovvo/Wvovv/Wvvvo) + the proper
  `−½ Σ⟨kl||cd⟩ r_l^{cd} t_{ki}^{ab}` term, **matching the explicit H̄ projection
  to ~5e-13 Ha on LiH** (machine precision). All three EOM variants (EE/IP/EA)
  are now patch-free PySCF ports with asserting brute-force verifiers.
- ✓ Aux-basis DF (stage 31 proper) — **done**: `buildAuxBasisDFStreaming`
  (WASM) + `df-gpu.ts` (WGSL s/p/d 3-index V + metric). No longer open.
- DF-CCSD via B-tensor through spin-orbital ERI build.
- df64 (double-single) emulation on GPU-JK products to push past the f32
  ~6e-4 element-precision floor (Kahan on the sum was a no-op — see jk-df-gpu.ts).
  Lower priority now: the hybrid path (`buildV3idxHybrid`) already gives
  chemistry-grade GPU-accelerated DF via f64 JK, sidestepping the f32-JK floor.
- f-functions in the WGSL 3-index kernel: would raise the GPU-carried aux
  fraction past the current ~91% (hybrid offloads f-aux to WASM) for a bigger
  GPU win — but no longer needed for *accuracy* (the hybrid is chemistry-grade).
- WASM (or GPU-side) merge kernel to replace the JS block-assembly in
  `buildHybridDFStreaming` — THE lever to extend the GPU hybrid past medium
  molecules. The hybrid currently gates off at n²·nAux ≥ 12 M because the
  per-block f32-low + f64-f-aux merge is a JS triple-loop that loses to WASM-SIMD
  streaming at PAH scale (naphthalene >2× slower; honest negative, 2026-06-09).
  Large molecules use all-WASM streaming DF, which is excellent; the GPU hybrid
  is a medium-molecule optimization (chemistry-grade + 1.31× V-build).
- **Naphthalene/PAH-scale DF-HF is feasibility-demonstrated, NOT precision-
  validated.** The capstone (`e2e/naphthalene-capstone`) asserts only a sane
  energy window (ERI never built); DF-vs-exact chemical-accuracy is validated
  only up to the size ladder where the exact 4-index ERI still fits a tab
  (H₂O→CH₂O→C₂H₄, n≤50, `e2e/df-accuracy-ladder`). To claim "chemistry-grade up
  to naphthalene" needs an external PySCF DF-HF reference for that geometry + a
  sub-mHa assertion (exact ERI is uncomputable in a tab there). Surfaced by the
  scientific-critic pass 2026-06-09. Don't conflate *feasible* with *validated*.
- WGSL (T) kernel optimization to push 39× → 100× (no warmup+trials harness yet).
- UKS-TDDFT response α(ω) — only remaining {ref}×{response} cell.
- Z-vector for MP2 / CCSD analytical gradients.
- NMR shielding via magnetic-perturbation CPHF.
- Becke-partition weight derivatives in DFT gradients (~1e-3 Ha/Bohr residual).
- Spherical-d on TDA-DFT / DFT-gradient grid (refuses with clear error).
- Davidson eigensolver for large-basis CIS / TDDFT.
- Continuum representation for E17 σ_ion convergence.
- Degenerate-eigenvector orthogonalization in `eigGeneralWithVectors`.

**Permanent verifiers**:
- `tests/chemistry/eom-ccsd-bruteforce-lih.test.ts` — full 14×14
  M_mine − M_exact diff after any σ_1/σ_2 change. Binary feedback.
- `tests/chemistry/ip-eom-ccsd-bruteforce.test.ts`
- `tests/chemistry/ea-eom-ccsd-bruteforce.test.ts`

---

## Research-grade discipline (non-negotiable)

From `RESEARCH.md`. Every experiment enforces them.

### Reproducibility

- No `Math.random()` in any experiment path. Every random draw uses a named
  seed from `experiments/lib/seeds.ts` via `mulberry32(seed)`.
- Every JSON artifact records: git SHA (when available), `navigator.userAgent`,
  `adapter.info`, WebGPU limits, UTC ISO8601 timestamp, and echoes back
  `protocol`, `hypothesis`, `passBar`, `seed`, `warmup`, `trials`. See
  `experiments/lib/env.ts → captureEnv(device, adapter)`.
- Artifact shape locked: `{ meta, env, rows, status, diagnosis }`. Don't
  add top-level keys without updating `experiments/lib/runner.ts` and the
  downstream dashboard.

### Timing

- `performance.now()` **with a forced GPU sync before AND after** — a mapped
  readback of a tiny buffer. `queue.submit` alone is non-blocking so raw
  timing is fiction. Harness: `experiments/lib/runner.ts → timedRun`.
- Discard 5 warmup samples. Retain 20 trials. Report median, p10, p90, p99,
  std, IQR — never single-shot.
- If `std/median > 0.1` on any cell, mark the artifact `"status": "noisy"`.

### Correctness

- Use **fidelity** F = |⟨ψ_ref | ψ_test⟩|², not max|Δp|. Two states can share
  a probability distribution and differ in phase — that kills any downstream
  controlled gate. Use `experiments/lib/fidelity.ts → stateMetrics`.
- Pass bar for f32-amplitude GPU paths: `F ≥ 1 − 1e-5`.
- Pass bar for f64 MPS vs f64 statevector: `F ≥ 0.999` (MPS has SVD
  truncation + accumulated Jacobi error, ~9 digits realistic at χ = 64).
- Secondary: TVD, L1, L2, max|Δp|, ‖ψ_ref‖², ‖ψ_test‖² — always reported.

### Honest negative results

- If an experiment fails its pass bar, still commit the JSON with
  `"status": "fail"` and a `"diagnosis"` naming the first failing
  cell and the smoking gun. **Failures are the evidence.** No silent
  rerunning until it passes.
- Example (MPS canonical-form bug, 2026-04-22): brick-wall F = 0.25 at
  depth 2. Diagnosis: "non-monotonic two-site gate order breaks
  mixed-canonical invariant, local Frobenius norm ≠ global norm,
  renormalization distorts." Fix: `_canonicalizeBond(q)` before every
  `applyTwoSite`.

---

## Commands

```bash
npm install
npm run dev          # Vite dev server, http://localhost:5175
                     # experiments live at http://localhost:5175/experiments/
npm run test         # Vitest, ~500 ms (one outlier 5 s for the MPS bug repro)
npm run test:watch   # TDD loop
npm run typecheck    # tsc --noEmit (strict, noUncheckedIndexedAccess on)
npm run lint         # ESLint flat config, src/ tests/ experiments/
npm run build        # → dist/
npm run test:e2e     # Playwright, all 4 levels headless (~1.4 min on M2 Pro).
                     # Saves JSON artifacts to experiments/results/<date>/level-N/.
                     # Each level also reachable via window.__webgpuq.runLevelN()
                     # in devtools at /experiments/.
npm run test:e2e:headed   # Same, with a visible browser window.
```

---

## File layout

```
src/
  shaders/
    single-qubit.wgsl    # 1-q gate kernel, N/2 threads, 2×2 complex matrix via uniform
    two-qubit.wgsl       # controlled-U kernel, N/4 threads
  gates.ts               # H, X, Y, Z, S/Sdg, T/Tdg, Rx/Ry/Rz, P, matrixFloats()
  quantum.ts             # QuantumCircuit (GPU) + initGPU() with requiredLimits
  cpu-reference.ts       # CpuCircuit (Float64 TS reference, ground truth)
  circuits.ts            # bell, ghz, qft, deutschJozsa, randomCircuit builders
  linalg.ts              # ComplexMatrix, Jacobi complex SVD, matmul   — Level 2
  mps.ts                 # MPS class with canonical form + TEBD         — Level 2
  bench.ts               # GPU vs CPU throughput sweep (pre-research harness)
  main.ts                # Legacy browser demo entrypoint
  chemistry/             # Level 6: HF, MP2, CCSD, CCSD(T), DFT, CIS/TDA/TDDFT,
                         # properties, gradients, geom-opt, vibrational analysis

tests/                   # Vitest unit tests (chemistry/, gates, linalg, mps, …)

experiments/
  index.html             # Research dashboard (run buttons, result tables)
  runner.ts              # Dashboard entry point — wires each level's run-all
  lib/
    seeds.ts             # Named deterministic seeds (no Math.random)
    runner.ts            # timedRun harness + Artifact / ArtifactMeta schema
    env.ts               # captureEnv(device, adapter) → EnvBlock
    fidelity.ts          # stateMetrics, FIDELITY_PASS_BAR
    stats.ts             # stats() — median, p10/p90/p99, std, IQR
  level-1-statevector/   # E1–E4 + run-all
  level-2-mps/           # E5–E7, E18, E19 + run-all
  level-3-fusion/        # E8–E13 shipped (Tiers A/B/C/D fusion)
  level-6-chemistry/     # E16, E20–E31 shipped (H₂ → CCSD(T)/cc-pVDZ)
  results/               # JSON artifacts, organized YYYY-MM-DD/level-N/
```

---

## Architecture notes (carry forward)

### Statevector (Level 1)

- Amplitudes stored as `vec2<f32>` interleaved (re, im). Buffer = `2^(N+3)` B.
- Single-qubit gate: `N/2` threads, each processes the pair `(i, j)` where
  bit `q` is 0 and 1. Apply 2×2 complex matrix from uniform buffer.
- Two-qubit (controlled-U): `N/4` threads, index scattered around control
  + target bits, only control=1 is touched.
- `initGPU()` MUST request the adapter's max `maxBufferSize` and
  `maxStorageBufferBindingSize` via `requiredLimits`. Default 128 MiB cap
  silently truncates N ≥ 25 dispatches.
- No atomics needed — gate application is pair-local read / write, zero contention.

### MPS (Level 2)

- Tensor storage: `tensors[i]` is a `ComplexMatrix` of shape
  `(χ_L · 2, χ_R)` — left-grouped. Element `T[l, s, r]` at row `l·2 + s`,
  col `r`. Single-qubit gates apply cleanly this way.
- Statevector convention: qubit 0 is LSB of the index —
  `ψ[s_0 + 2·s_1 + 4·s_2 + …]`. `mps.statevector()` follows this for
  comparison with `CpuCircuit.psi`.
- Two-site gate order within the 4×4: `i = s_lo · 2 + s_hi` — site `q` is
  the MSB within the pair. Controlled-U needs the right ordering;
  see `buildControlledMatrix4(U, controlIsLo)`.
- **Canonical form invariant** (critical). Two-site TEBD needs
  `‖M‖_F² = ‖ψ‖²`, which requires left-canonical on sites `[0..q−1]` and
  right-canonical on `[q+2..N−1]`. `_canonicalizeBond(q)` does the sweep.
  Cost: O(N · χ³) per two-site gate. Trivial at N ≤ 20, χ ≤ 64.
- SVD is one-sided Jacobi on complex matrices: phase-align col q by
  e^(−iφ) so ⟨p, q⟩ is real, then apply the real Jacobi rotation. 60 sweep
  cap, TOL = 1e-14.
- `apply*` returns void (mutates). `statevector()` refuses `N > 24`.
- v1 constraint: `applyTwoSite` / `applyControlled` require `|c − t| = 1`.
  Non-adjacent two-qubit gates need SWAP ladders (not yet implemented).

### Research harness

- `experiments/lib/runner.ts → timedRun(device, fn, cfg)` is the only
  legitimate way to measure wall time on GPU paths. It owns the sync
  fence and the error-scope guards.
- `Artifact<Row>` is the JSON shape. `emitArtifact` logs; `downloadArtifact`
  serves it as a download from a click handler.
- Per-experiment logs use the `[artifact:protocol] status — diagnosis`
  prefix on stdout so CI greps can find pass/fail without parsing JSON.

---

## WebGPU gotchas (carry forward from webgpu-dna)

- `initGPU()` MUST pass `requiredLimits` for `maxStorageBufferBindingSize`
  and `maxBufferSize`. Default 128 MiB cap silently truncates large
  dispatches.
- `atomicAdd` only on `u32`. Not needed in statevector path (no contention).
- No recursive function calls in WGSL. All shaders are single-pass.
- Uniform buffers must be aligned.

---

## Hero-mode conventions for this repo

- Scope-honest. Most research tasks here = hours for a capable agent, not
  weeks. Attempt now; decompose only if truly large.
- Speculation labeled. "This should work" ≠ "tested". Benchmark > belief.
- Raw WGSL > framework. Dispatch ceremony is the enemy.
- Edge hardware underrated. The thesis is "no one has shipped this in a
  browser tab." Don't reinvent it; ship the numbers.

---

## Engineering policy — port, don't re-derive (NEW 2026-05-13)

**Discovered the hard way via E35/E36 EOM-CCSD bug:** webgpu-q's
differentiator is the browser/WebGPU layer. The chemistry methods
themselves are textbook with peer-reviewed reference implementations
(PySCF, libxc, ITensor). Re-deriving them from papers, as we did,
produces bugs that take weeks to find. Going forward:

- **Hand-write only the novel layer**: WGSL shaders, WebGPU dispatch +
  sync, MPS browser memory bookkeeping, kernel fusion, research-grade
  harness.
- **Port from references** with proper Apache 2.0 attribution
  everything else: HF, MP2, CCSD, UCCSD, CCSD(T), EOM-CCSD, DFT
  functionals (libxc), gradients (Pulay), density fitting, integrals
  if vectorizable, basis-set tables (EMSL).

Migration framework in [`MIGRATION.md`](./MIGRATION.md). Per-module
status table (🔴 hand-derived → 🟢 ported), priority order, attribution
recipe. `LICENSE-PYSCF` at root covers ported portions.

**First scheduled port**: `eom-ccsd.ts` σ_2 from PySCF
`pyscf/cc/eom_rccsd.py`. Closes the singlet-sector bug E35 surfaced
on H₂O / NH₃ / CH₄ / BeH₂ / LiH. Verifier is the LiH brute-force
diagnostic (`tests/chemistry/eom-ccsd-bruteforce-lih.test.ts`) — after
the port, M_mine − M_exact should collapse to numerical noise.

## Modern reference standards (audited 2026-05)

What our claims map to in current literature. Run this audit again
before any release or paper draft.

- **Chemical accuracy** = 1 kcal/mol = **1.594 mHa** (Pople pragmatic
  threshold). Our CCSD(T) vs FCI residuals (≤ 0.25 mHa) are sub-chemical;
  our GPU↔CPU |Δ| (≈ 10⁻¹⁰ Ha) is ~6 orders past chemical accuracy and
  characterizes f32 reduction noise, not method error.
- **CCSD(T) is still the gold standard** in 2025/2026 (multiple JCTC
  reviews). MAE ~0.2–0.3 kcal/mol at CBS for noncovalent interactions.
- **AFQMC** (Mahajan et al. JCTC Feb 2025, arXiv:2410.02885) now beats
  CCSD(T) at **O(N⁶)** vs O(N⁷). Tier 4 candidate "beyond CCSD(T)".
- **EOM-CCSD literature accuracy vs FCI** for singlet single-excitations
  is **0.1–0.2 eV (~3.7–7.4 mHa) typical**, 0.3 eV conservative.
  Doubly-excited states: errors up to 1 eV. Our 10⁻⁵ Ha on H₂ STO-3G
  is **algorithmic precision** (T̂² = 0 for 2-electron systems makes
  EOM-CCSD ≡ FCI exactly there) — it validates the implementation, not
  the method on real systems.
- **GMTKN55 best functionals (2024–2025)**: **ωB97M(2)** DH WTMAD2 =
  **2.19 kcal/mol** (best ever), xrevDSD-PBEP86-D4 = 2.23, revDSD-PBEP86-D4
  = 2.33. Best RSH: **ωB97X-V**. Best meta-GGA: **SCAN-D3(BJ)**. We
  benchmark with B3LYP5 / BLYP / LSDA / B88 / LYP — textbook, not
  current SOTA. Modern functionals are in the Tier 3 row.
- **MPS state-of-the-art**: TeNPy / ITensor are the reference libraries.
  Production runs go to **χ = 1000+**. Our χ ≤ 64 is "browser-feasible";
  the comparison Schollwöck 2011 still holds (χ scales with entanglement).
- **WebGPU subgroups**: out of WebGPU 1.0 spec (gpuweb#3950); coming
  later. Would unlock 2× reductions in fusion kernels (shuffle/add).
- **FAIR / Zenodo DOI**: standard for reproducible computational chemistry
  data publishing. We emit JSON artifacts with full env capture but
  don't mint DOIs. Tier 3+ research-publishing improvement.
- **Browser-native quantum chemistry**: as of 2026-05 web search, no
  published WebGPU + HF/DFT/CCSD(T) implementation exists outside this
  repo. Worth a paper if Phase D / hardware verify ever lands.

## Related repos / links

- **Sibling:** `/Users/ahmetbarisgunaydin2/Downloads/webgpu-dna/` —
  Geant4-DNA port. Has its own CLAUDE.md. Level 6 chemistry cross-links here.
- `kernelfusion.dev` — umbrella theory.
- `gpubench.dev` — WebGPU bench harness reuse pattern.
- Pan & Zhang 2021 (arXiv:2103.03074) — Sycamore tensor-network baseline.
- Karamitros 2011 — IRT chemistry, cross-link target.
- IBM Heron r2 (156q, 2025), Nighthawk (120q, Jan 2026) — E14 target.
- Schollwöck 2011 — MPS / DMRG review, χ-vs-error baseline.
- Vidal 2003 — iTEBD algorithm (what `applyTwoSite` implements).
- GMTKN55: Goerigk, Hansen, Bauer et al., PCCP 2017 — main DFT benchmark.
- Mahajan et al. JCTC 2025 — AFQMC beats CCSD(T) at O(N⁶).
- NIST CCCBDB — experimental reference IP, EA, vibrational data.

---

## License

MIT (simulation). Research protocol and experiment artifacts: MIT.
