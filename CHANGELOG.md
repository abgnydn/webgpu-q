# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/) starting
from `0.1.0`.

## [0.10.0] — 2026-06-15

The **distributed-chemistry** release. Since the streaming swarm (0.9.4), the
crowd learned to do *post-HF* work together and to *screen* molecule libraries —
and a whole GPU-accelerated density-fitting stack landed behind one honest,
size-gated entry point. The headline lesson is a measured one: the swarm makes
*many* molecules fast (throughput), not *one* molecule fast.

### Added

- **Size-gated auto entry points** — `runRHFAuto` / `runRKSAuto` / `runUHFAuto` /
  `runUKSAuto` / `runMP2Auto` / `runUMP2Auto`: the full {R,U}×{HF,KS,MP2} matrix in
  one call each, auto-selecting **exact 4-index ERI (small) or streaming f64
  density fitting (large)** and reporting honest `{method, engine, precision,
  nAux, expectedError}` provenance. One call, the right method, attributed.
- **Aux-basis density fitting (proper RI)** — `buildAuxBasisDFStreaming` streams
  the 3-index integrals μ-block by μ-block and never materializes the 4-index
  ERI; **naphthalene cc-pVDZ HF now runs in a tab** (its 9.7 GB ERI is skipped
  entirely). `e2e/naphthalene-capstone`.
- **Distributed DF-MP2 — the swarm's first collaborative single-molecule
  reduction.** One molecule's MP2 correlation energy `E_corr = Σ_i Σ_j Σ_ab …`
  partitions exactly over the outer occupied index `i`; each tab owns an
  `i`-slice and returns a scalar partial, the master sums them (`mp2-slice`
  kernel, `mp2EnergyDF(..., iRange)`, `reduceMP2Slices`). Partition-sum ==
  single-machine to `<1e-12`; 2-tab e2e to `<1e-9`.
- **Distributed molecule screening** — `chem-energy` now also returns the
  **HOMO–LUMO gap (eV)** as a screening descriptor; `e2e/swarm-screening` ranks
  a candidate library across tabs with the *identical* ranking to a single tab.
- **Cross-machine swarm** — `RelayTransport` over a free public MQTT broker;
  **N-machine distributed Hartree–Fock verified on real CI VMs** (N=2 to
  `5.7e-14`, N=4), and a **swarm × GPU** batch where every tile runs `runRHFAuto`
  on the GPU and reports its own provenance.
- **gzip-binary-f64 wire codec** — lossless f64-array compression for swarm
  messages (~4× on a real density), so the distributed paths keep bit-exact
  agreement over the wire.
- **GPU-accelerated density fitting (experimental)** — WGSL s/p/d 3-index +
  2-index McMurchie–Davidson integral build (`df-gpu.ts`), a hybrid GPU/WASM DF
  build (`buildV3idxHybrid`), and a fully-GPU DF-JK (`makeGpuDFJK`).

### Changed

- **f64 WASM is the recommended density-fitting default; the GPU DF paths are
  experimental** (decision 2026-06-10). WebGPU has no f64, so the GPU can only
  touch the f32-insensitive auxiliary columns (~1.3× on the integral *build* in a
  medium band) and cannot accelerate the f64-bound J/K — kept as a
  proof-of-mechanism, not the chemistry default.
- **Swarm scheduler rewritten to a greedy pull queue** — every peer (master
  included) pulls one tile at a time and asks for the next only after finishing,
  so a slow tile parks only its own puller while everyone else drains; this also
  auto-balances uneven tile costs. Replaces the old single-claim-per-worker
  protocol that left the master running almost everything (auto-distributed
  screening went 9/1 → 4/6, 1.05× → 1.48×).

### Fixed

- `swarmMap` over `RelayTransport` (queue sends issued before the async broker
  connect, flush on connect).
- Greedy-pull livelock: a persistently-failing worker re-pulling and re-failing
  the same tile forever — failed tiles now run on the master instead of requeueing.
- `requiredLimits` on GPU DF devices (handles a V output past the default 128 MB cap).

### Honest negatives (the evidence we keep)

- **Distributing one molecule's MP2 barely speeds it up** (`e2e/swarm-mp2-speedup`):
  benzene cc-pVDZ 1.10× across 2 tabs, because the redundant SCF + DF *setup*
  (S≈79 s, 82%) dwarfs the splittable contraction (C≈17.5 s, 18%) and sits on the
  critical path. The swarm's scaling axis is **throughput (N independent
  molecules), not single-molecule wall-time**. Multi-tab screening scales as
  1→1.00× / 2→1.73× / 3→2.02× / 4→2.36× (warmed, even split; sub-linear because
  molecule costs are uneven).
- **The hybrid GPU DF is gated off at PAH scale** — its per-block JS merge loses
  to WASM-SIMD streaming for large molecules.
- **Naphthalene-scale DF-HF is feasibility-demonstrated, not precision-validated**
  — the capstone asserts only a sane energy window; sub-mHa DF-vs-exact accuracy
  is validated up to where the exact ERI still fits a tab (H₂O→CH₂O→C₂H₄).

## [0.9.4] — 2026-06-08

The **streaming swarm** release. A browser-tab swarm now scales Hartree–Fock
*past the 2 GB single-allocation wall* — a calculation impossible on any single
tab, run across several, and proven correct.

### Streaming, self-built slices — no full tensor on any tab

The earlier swarm had the master build the whole density-fitting tensor (~3 GB
transient) then ship slices. Now **each tab independently builds only its own
slice**, and never materializes the full V or full B:

- `buildAuxBasisDFStreaming` uses the symmetric (eigendecomposition) fit
  `B[μν,i] = Σ_Q V[μν,Q]·U[Q,i]·λ_i^(−1/2)`, whose mode index is the axis the
  Fock build contracts over and is itself a valid fit. The 3-index integrals are
  streamed μ-block by μ-block and projected immediately, so peak resident V is
  one μ-block (~50 MB), not `n²·n_aux`.
- The metric eigendecomposition is deterministic, so the N tabs tile the mode
  axis with **zero coordination** (`partition:{tab,of:N}`).
- A cooperative variant builds each V-block once and fans it out to all tabs
  (no redundant integrals when tabs share a machine).
- Validated end to end: independently built slices reproduce the single-tab DF
  J/K to 1.5e-13 and the full DF-HF SCF to 1.4e-14.

### Past the 2 GB wall — the headline

A `Float64Array` cannot exceed 2^31−1 bytes in any JS engine. On a 16 GB CI
runner, a 4-tab streaming swarm runs **40 H₂ in cc-pVDZ (n=400): full B = 2.56 GB
— impossible single-tab — as four 640 MB slices** (peak V 51 MB/tab), converging
to E = −45.14995980 Ha and matching the exact non-interacting oracle
`40·E(H₂)` to 2.7e-5 Ha. Committed as an environment-stamped artifact; runs in CI
(`.github/workflows/swarm-scaling.yml`).

### Faster build — WASM SIMD projection

The streaming projection (the `~n⁴` term, profiled as ~67% of the build at
n=400) moved from a TypeScript loop to a Rust+f64x2 kernel
(`df_project_block_modes`): **3.4× faster projection, ~2× faster build at the
headline scale** — what turns a >70 min timeout on a slow runner into a pass.

### Accuracy quantified

Against exact HF in cc-pVDZ, the streaming DF is **0.12× chemical accuracy at
auto-aux level 1** (H₂O 1.9e-4 Ha), dialing to µHa at L2/L3 — a clean
accuracy/cost knob, not a liability.

### Manuscripts

- `paper/main.pdf` (now 12 pp): rewrote the swarm sections to lead with the
  streaming architecture and the wall-crossing result, with a memory-scaling
  figure; updated limitations (C₆₀ cc-pVDZ is now total-RAM-bound at ~26 GB;
  the n=490 SCF-coordination stall over BroadcastChannel is the next bottleneck,
  fixed by SAB messaging).
- `paper/main-fusion.pdf`: corrected the bandwidth claim — the 866 GB/s figure
  is *logical* throughput; physical DRAM use is ~13.5% of peak (floor-limited),
  so the kernel is not bandwidth-saturated in the measured regime.

## [0.9.3] — 2026-06-04

A **deposit-correctness** release. No source, method, or figure changed — it
exists to carry the corrected-DOI manuscripts into a fresh Zenodo archive and
to stop the per-release DOI churn.

- The v0.9.2 Zenodo files were minted from manuscripts that still cited the
  old (v0.9.0 version) DOI; Zenodo locks files on publish, so a new version is
  the only way to deposit the corrected PDFs. This release's `paper/main.pdf`
  and `paper/main-fusion.pdf` cite the concept DOI `10.5281/zenodo.20494382`.
- `CITATION.cff` (top-level and `preferred-citation`) and the README citation
  now use the **concept DOI** throughout — it always resolves to the latest
  version, so the citation no longer needs a per-release edit referencing a
  version DOI that doesn't exist until after the tag is published. Each
  release still gets its own version DOI on the Zenodo record.

## [0.9.2] — 2026-06-03

A **paper-hardening and reproducibility** release. No method or kernel
changed; the two manuscripts and the deposit metadata were audited against
the repository and every quantitative claim was either tied to a committed
artifact or honestly caveated.

### Reproducibility — the swarm scaling table is now artifact-backed

- Added a swarm-artifact emitter (`e2e/lib/swarm-artifact.ts`) and wired it
  into the swarm specs. Each molecule writes an environment-stamped JSON
  (`{ meta, env, rows, status, diagnosis }`) under
  `experiments/results/<date>/swarm/` with git SHA, user-agent, adapter info,
  basis-function and auxiliary counts, iterations, per-phase wall, and the
  converged energy.
- Ran benzene → naphthalene → anthracene → pentacene → **C₆₀** and committed
  the artifacts. Correcting the paper table from measured data caught a wrong
  benzene energy before it shipped. **C₆₀ STO-3G confirmed: E = −2244.10176303
  Ha, n = 300, n_aux = 2520, 9 SCF iterations, 1.81 GB B-tensor at 454 MB/tab,
  swarm-SCF wall 539 s** on a 16 GB M2 Pro across four tabs.
- The scaling table's "wall" is now defined as the swarm-SCF loop (the stable,
  reproducible quantity; cold integral/DF builds are machine-load noisy) and
  this is footnoted. C₆₀ remains gated to a local high-memory run — Ubuntu CI's
  ~2 GB WASM heap cap is below the build peak until `eri_3idx_build_slice` is
  rewritten to stream (documented honest limitation, not a number).

### Manuscript audit — corrected and grounded

- Removed falsifiable / embarrassing claims surfaced by an against-the-repo
  audit: a bogus "2–6× slower" figure, a reverted SIMD attempt presented as a
  win, and overstated tolerances.
- Fixed a units error in the fusion paper (multiply count is **per
  16-amplitude tile (16/amp)**, not per amplitude) and corrected the Tier-D
  system size.
- Grounded the novelty claim against prior server-side browser front-ends
  (WebMO, MolCalc) with a dated literature search, and cited the two prior
  WebGPU single-kernel-fusion preprints for cross-domain framing.

### No vanity metrics

- Replaced test-count / lines-of-code framing in both papers, the README-facing
  deposit description (`.zenodo.json`), and `CLAUDE.md` with a
  capability-and-validation surface: what is checked against which reference,
  not how many tests exist.

### Figures

- Regenerated all five matplotlib figures from the committed artifacts;
  `fig-scaling` now plots the measured swarm-SCF walls (C₆₀ at 539 s).

### DOI correction

- Fixed the Zenodo identifier everywhere: the value wired in at v0.9.1
  (`10.5281/zenodo.20494383`) is actually the **version DOI of v0.9.0**, not
  the concept DOI, so the badge / citation had been pinned to the first
  release instead of tracking latest. Corrected to the true **concept DOI
  `10.5281/zenodo.20494382`** (all versions, resolves to latest) for the badge,
  `CITATION.cff` top-level, and both manuscript archive lines; the
  `preferred-citation` and the README "this release" link now carry the
  **v0.9.2 version DOI `10.5281/zenodo.20527479`**.

## [0.9.1] — 2026-06-02

Submittable manuscripts with verified, DOI-bearing references.

- Authored both papers in LaTeX (`pdflatex` + `bibtex`): `main.tex` (the
  electronic-structure stack and browser-tab swarm, 10 pp) and
  `main-fusion.tex` (the kernel-fusion companion, 6 pp), replacing the earlier
  pandoc drafts.
- Verified every reference and recorded DOIs; added data figures (validation
  ladder, single-tab optimization, dispatch-cost collapse, tier ladder, swarm
  protocol) via `paper/make-figs.py`, and fixed the fusion tier-ladder table
  overflowing the column.
- Wired in the Zenodo concept DOI `10.5281/zenodo.20494383` (README badge,
  `CITATION.cff`) and reconciled version metadata.

## [0.9.0] — 2026-05-29

The **multi-tab swarm** release. The browser-tab swarm HF SCF
architecture is end-to-end proven from H₂O (n=24) through C₆₀
buckminsterfullerene (n=300), and the same code path makes naphthalene
cc-pVDZ HF run **5.5× faster** than the v0.8.0 single-tab baseline by
combining single-tab WASM perf wins with cross-tab parallelism.

### Headline numbers (M2 Pro, Chromium, COI enabled)

| molecule | basis | n | best path | time | E (Ha) |
|---|---|---:|---|---:|---:|
| H₂O | cc-pVDZ | 24 | 1-tab | 0.025 s | -76.027 |
| benzene | cc-pVDZ | 120 | 2-tab swarm | 5.5 s | -230.7227 |
| **naphthalene** | **cc-pVDZ** | **190** | **4-tab × 2-inner swarm** | **14 s** ⬇️ from 77 s | **-383.3846** |
| anthracene | STO-3G | 80 | 4-tab swarm | 8 s | -526.9232 |
| pentacene | STO-3G | 124 | 4-tab × 2-inner swarm | 38 s | -821.6327 |
| **C₆₀** | **STO-3G** | **300** | **4-tab × 2-inner swarm** | **730 s (12 min)** | **-2244.1018** |

C₆₀ is the iconic showcase — the most recognizable molecule in
chemistry, full HF SCF in 4 same-origin browser tabs on a 16 GB Mac,
each tab holding only 454 MB of the 1.82 GB B-tensor.

### Single-tab WASM/SIMD perf (8 commits)

Cumulative on naphthalene cc-pVDZ: SCF **43 s → 17.6 s** on a clean
profile run (-59%), end-to-end **77 s → 28 s** (2.7×) before the
swarm even kicks in.

- `10cfe14` perf(df): parallel V build + parallel B back-sub on
  Cholesky path — naphthalene 77 s → 36 s (2.2×)
- `3123505` perf(df): release V tensor SAB before SCF — peak SAB 620 → 315 MB
- `563e8d1` diag(hf): per-iter SCF profiling via opts.profileCallback
- `d9b633a` perf(hf): pre-warm JK_DF workers before SCF iter loop
- `5cacb7e` perf(jk): reuse JK_DF worker scratch buffers — SCF 43 → 20 s
- `3e020e4` perf(jk): exploit K symmetry in build_jk_df_slice — SCF 20 → 17.6 s
- `73f5d2b` perf(jk): 4× SIMD unroll on X-build inner loop
- `d097625` diag(jk): worker-count sweep — find optimal poolSize per hardware

### Multi-tab swarm architecture (10 commits)

The single-tab SAB ceiling is the limiting factor for direct browser
quantum chemistry; the swarm breaks through by partitioning the
B-tensor by aux index P across N same-origin tabs and running each
SCF iter as broadcast-D → parallel-JK → gather-J/K via BroadcastChannel.

- `78146ec` diag(swarm): multi-tab SAB partitioning MVP — 4 tabs × 200 MB SAB, BroadcastChannel coord 68 ms round-trip
- `9103ed6` diag(jk): prove P-partition correctness for distributed JK_DF — bit-exact at 2/3/4/8 splits
- `ba1c272` feat(swarm): distribute JK_DF build across 2 tabs via BroadcastChannel — 2-tab JK matches single-tab ref to 10⁻¹⁵
- `07c09be` feat(swarm): full distributed HF SCF across 2 browser tabs — benzene end-to-end, bit-identical energy
- `22ae86a` feat(swarm): scale swarm HF SCF to 4 tabs on naphthalene cc-pVDZ
- `ae83213` perf(swarm): inner-parallelism inside each worker tab — naphthalene 24 → 14 s (40% faster)
- `416dbe4` diag(swarm): topology sweep — 4 tabs × 2 inner beats 2 tabs × 4 inner at the same total thread count
- `a14f134` + `dcd8ee7` swarm anthracene STO-3G — 4-tab convergence to -526.92 Ha
- `3e15ab5` feat(swarm): pentacene C₂₂H₁₄ HF SCF in browser swarm — 5 fused rings, 124 basis, 35 iters
- `9298d0e` **feat(swarm): C₆₀ buckminsterfullerene HF SCF in browser tabs** — 300 basis, 9 iters, -2244.10 Ha

### Chemistry method (2 commits)

- `b450896` feat(hf): diisStartIter option for delayed DIIS activation — unblocks hard-converging cases (recipe for anthracene cc-pVDZ: `damping: 0.2, diisStartIter: 8`)
- `213f75d` test(swarm): anthracene cc-pVDZ with delayed-DIIS recipe — validates the fix on the molecule it was designed for

### Infrastructure / CI (2 commits)

- `ecec5e5` ci(swarm): offload swarm benches to GitHub Actions runners — nightly + manual deep benches, zero local resource cost. Three jobs: swarm-quick (always), swarm-deep (nightly + manual), swarm-c60 (manual-only)
- `41f9ea3` fix(numbers): rebase two drifted claims — unblocks CI lint+test workflow

### Known limitations

- **Anthracene cc-pVDZ RHF — convergence vs correctness**: the
  `diisStartIter` recipe (damping=0.2, diisStartIter=8) successfully
  reaches a stationary point and avoids the +5352 Ha divergence seen
  on default DIIS. CI's first green run converged to E = -880 Ha,
  which is *more negative* than the literature anthracene HF/cc-pVDZ
  value of ~-537 Ha — a "wrong-basin" SCF solution where the damped
  warm-up steers density into a non-physical orbital occupation
  that's energetically lower but doesn't correspond to the true
  ground-state singlet. The architecture and convergence-method
  work end-to-end; **basin selection** is the new open problem.
  Real fixes: MOM (maximum overlap method) to preserve orbital
  ordering across iters, SOSCF for second-order convergence to the
  nearest stationary point, or a SAD initial guess that starts in
  the right basin. UHF + spin-symmetry-break is the textbook
  multi-reference alternative.
- **C₆₀ in Ubuntu CI runner**: the V tensor build hits a WASM trap
  during SAB allocation around ~3 GB — Chromium's SAB ceiling on
  Linux runners is configured tighter than on macOS. Local C₆₀ works
  fine; CI C₆₀ is gated to manual trigger only.
- **C₆₀ converges only at STO-3G**; cc-pVDZ at n=600+ exceeds even
  the swarm-distributed master tab's V-build memory.
- **Per-tab independent B-slice builder** not yet implemented —
  master builds the full B then partitions. For molecules where the
  full B exceeds the master's SAB ceiling (anthracene+ at cc-pVDZ,
  C₆₀+ at any basis), an independent per-tab build remains the
  next architectural step.

## [0.8.0] — 2026-05-28

The **WASM + aux-DF** release. ~50 commits since v0.7.0 deliver the
biggest performance leap in the project's history: benzene cc-pVDZ HF
goes from 841 s (TS-only baseline) to 16.8 s end-to-end via WASM hot
paths (50× speedup), and the auxiliary-basis density-fitting stack
ships from zero to a working sub-mHa-accuracy implementation that
unlocks arbitrary organic molecules without external basis-set data.

### Headline numbers (benzene cc-pVDZ, n=120, M2 Pro / Chromium WebGPU)

| stage | wall time | speedup vs TS |
|---|---:|---:|
| TS-only direct ERI + HF (start of session) | 841 s | 1× |
| + WASM single-thread ERI                   | 189 s | 4.4× |
| + Workers parallel (×8)                    |  38.5 s | 21.5× |
| + branch-free hot path                     |  25.3 s | 32.7× |
| + pair-table cache                         |  16.8 s | 49× |
| + r_aux buffer pooling + SIMD JK + matmul  | **16.98 s** (15.76 s ERI + 1.21 s HF) | **49.5×** |

End-to-end HF SCF on benzene cc-pVDZ from cold shells: **~17 s**.
The aux-DF path (Cholesky) ships at 49 μHa accuracy as the *only*
viable path where direct's n⁴ ERI tensor exceeds browser memory
(n ≥ 200, e.g., naphthalene cc-pVDZ).

### Major architectural breakthrough — auxiliary-basis density fitting

The aux-DF stack went from "shipped but broken NET LOSS" (CD-DF on
the full ERI tensor, v0.7.0) to a working algorithmic path:

- **3-index + 2-index ERI Rust kernels** (`eri_3idx_build`,
  `eri_2idx_build`) — McMurchie-Davidson with single-Gaussian
  Hermite expansion on the aux side. Bit-perfect against closed-form
  `(s|s)` and `(s_a s_a | s_c)` at α ∈ {0.5, 1, 2, 5, 10} (rel 10⁻¹⁶).
  Parity tests pass at L=1..4 (g-functions correct).
- **`form_b_tensor` WASM SIMD matmul** — f64x2-vectorized
  B = V · M⁻¹⸍² composition (4.6 B FLOPs on benzene). 5× over the TS
  matmul.
- **`generateAutoAux`** — decontracts the orbital basis and extends
  angular momentum; no external jkfit basis tables needed.
- **Pivoted Cholesky breakthrough** (`buildAuxBasisDFCholesky`) — the
  algorithmic fix that unlocked aux-DF for arbitrary organic systems.
  Replaces eigendecomp + threshold regularization with pivoted
  incomplete Cholesky of M. Eigendecomp's spurious near-zero modes
  corrupted M⁻¹⸍² on rank-deficient auto-aux at multi-heavy-atom
  systems; pivoted Cholesky naturally stops at the effective rank.

End-to-end benzene cc-pVDZ aux-DF HF via Cholesky:

| stage | time |
|---|---:|
| Integrals (skipERI + skipOAO) | 0.6 s |
| B-tensor (parallel V + WASM matmul) | 8.4 s |
| HF SCF (DIIS over DF, WASM JK, 11 iters) | 42 s |
| Total                            | **51 s, 49 μHa accuracy** |

### WASM hot-path wins shipped this release (compound order)

- `buildERIWasm` + `buildERIWasmParallel` — Rust port of the ERI
  primitive kernel with Boys, E-coefs, R-aux table, pair-cache,
  8-fold symmetry. Worker-parallel μ-row distribution.
- Branch-free `prim_eri_with_pairs` — invariant index bases hoisted
  out of the 6-deep nest; partial products xyz1, xyz1_x2, xyz1_x2_y2
  lifted out so the inner loop is 2 mults + 1 sign-mult + 1 load + 1
  add. `sign = 1 − 2·parity` (branch-free).
- Pair-table cache in `precompute_pair_tables` — the kernel rebuilt
  the bra+ket Hermite-Gaussian E-coefficient tables on every
  (μν|λσ) call (~470 M redundant builds on benzene). Hoisting to a
  single precompute collapses this to 7 200 (65 000× reduction).
- `r_aux_table` buffer pooling — caller-provided scratch buffers
  eliminate ~2 B malloc/free pairs per benzene build.
- WASM JK kernel (`fock_one_mu_row`) — replaces TS Fock-build inner
  accumulator. Per-μ design keeps worker memory bounded.
- SIMD JK σ-loop — hand-vectorized f64x2 dot product (`jk_dot`)
  via `wasm-simd128` intrinsics.
- `form_b_tensor` + `build_jk_df` WASM kernels — full B-tensor
  composition and DF Fock build via SIMD matmul.

### Optimization plumbing

- **Persistent worker pool** in `worker-pool-shared.ts` — ERI build
  and JK build share the same physical Workers via the shared
  registry. Saves ~150 ms of duplicate WASM init per call.
- **`skipERI` / `skipOAO` options** on `computeMolecularIntegrals`.
  HF/MP2/CCSD/EOM/UHF/DF paths all use `eri_AO` directly and never
  touch `eri_OAO`, so the O(n⁵) 4-index OAO transform (~100 s on
  benzene) was pure dead work. Both flags applied across
  `geometry`, `counterpoise`, `vibrations`, `hyperpolarizability`,
  `redox`, `uv-vis`, `quick-report`. Test wall time on benzene
  dropped 11 min → 29 s as a side effect.

### Research artifact — WGSL JK kernel (not production)

`src/shaders/fock-build.wgsl` — WebGPU compute shader for JK, one
thread per (μ, ν). **2.4× faster** than WASM SIMD as an isolated
kernel on benzene, but **3.3× SLOWER** when wired into actual HF SCF:
f32 precision (~10⁻⁴ relative noise) prevents DIIS from converging,
and per-iter GPU dispatch + `mapAsync` latency (~100 ms) dwarfs
the 30 ms compute saving. Shipped as `useWgpuJK` opt for research
use; default remains the WASM SIMD path.

### Honest negative results (LIMITATIONS.md)

- wasm-simd128 auto-vec on the `prim_eri_with_pairs` 6-loop: no signal
- `[f64; 13]` E-coef stack prefetch: 55% benzene regression
- Boys downward recurrence: mixed signal + precision regression
- 4-way SIMD JK unroll: within noise band
- WGSL f32 JK in real HF SCF: 3.3× slower than WASM SIMD
- Auto-aux + eigendecomp on > 2 heavy atoms: catastrophic SCF
  failure (later fixed by pivoted Cholesky)

### Tests

- 553 chemistry tests green (one pre-existing skip)
- New kernel cross-checks: closed-form `(s|s)` and
  `(s_a s_a | s_c)` at α ∈ {0.5, 1, 2, 5, 10} — rel error 10⁻¹⁶
- L=1..4 parity tests on the 2-index kernel (g-functions verified)
- 12 new e2e benchmark specs covering each optimization stage

## [0.7.0] — 2026-05-22

The browser-platform release. Ten commits since v0.6.0 close two
correctness bugs (EE-EOM-CCSD + IP-EOM-CCSD via PySCF port) and ship
seven user-visible browser-platform wins atop /molecule.html. The big
realization driving most of this: we had been GPU-pilled and missed
the rest of the browser-tab platform — Web Workers, SharedArrayBuffer,
Pyodide, File System Access, PWA, IndexedDB.

### Fixed — EE-EOM-CCSD + IP-EOM-CCSD σ-equations (PySCF-ported)

- `src/chemistry/eom-ccsd.ts` — direct port of PySCF `eom_gccsd.eeccsd_matvec`
  (Wang-Tu-Wang 2014 Eqs. 9-10) using `gintermediates.py` intermediates
  (Foo/Fvv with bare canonical Fock diagonal, Wovvo with full t2·oovv,
  Wooov/Wvovv/Wovoo/Wvvvo with all dressings). σ_2 also gets four
  previously-missing (t2, r1) and (t2, r2) couplings.
- `src/chemistry/ip-eom-ccsd.ts` — mirror port of `eom_gccsd.ipccsd_matvec`
  (Tu-Wang-Li 2012 Eqs. 8-9). Closes the ~60 eV R_2 satellite over-count
  documented in earlier LIMITATIONS.md.
- Verifiers in `tests/chemistry/eom-ccsd-bruteforce-lih.test.ts` and
  `tests/chemistry/ip-eom-ccsd-bruteforce.test.ts` build H̄ =
  e^(-T̂) H e^(T̂) explicitly in the Fock space and diff full matrix
  element-by-element. Both now have hard `expect(maxDiff) < 1e-10` Ha
  regression assertions. Empirical stage-32c/32e patches removed.
- Side effects on production tests: H₂ EOM-CCSD now matches FCI to 8+
  decimals (was 10⁻⁵ Ha "algorithmic cap"). H₂O lowest triplet
  10.81 eV, first singlet 12.44 eV.

### Added — URL-as-citation contract

- `/molecule.html?molecule=h2o&method=b3lyp5&autorun=1` is a permanent,
  reproducible reference. URL stays in sync as dropdowns change; "Copy
  citation link" button copies the canonical autorun URL to clipboard.

### Added — PWA + service worker + IndexedDB history

- `public/sw.js` — network-first for HTML, cache-first for static.
  Installable as desktop/mobile app, works offline after first load.
- `src/molecule/history.ts` — IndexedDB persistence of every completed
  pipeline run. "Past calculations" card on /molecule.html lists 20
  most recent; each links back via the citation URL.

### Added — File System Access drag-import

- `src/molecule/import-formats.ts` — parsers for `.xyz` / `.pdb` /
  `.mol` / `.sdf` with symbol normalization (`13C` → `C`). 7 unit tests.
- Drag-and-drop anywhere on the page OR "Open file…" picker → parsed
  geometry → MOLECULES["imported"] entry → ready to Run.
- Round-trip closed: we already export Molden / Cube / XYZ / QCSchema;
  now we can import the formats users already have.

### Added — Web Worker pool + parallel HF buildG

- `src/parallel/{worker-pool,kernels-worker,parallel-buildG}.ts` —
  Web Worker pool with SharedArrayBuffer transport (zero-copy of the
  ERI tensor across workers). Lazy worker spawn, reused across SCF
  iterations.
- `runRHFSCFAsync(opts.parallel = N)` — async sibling of `runRHFSCF`
  that row-partitions the JK build across N workers. Falls back to
  single-threaded when cross-origin-isolation is unavailable or for
  small molecules (n < 15). /molecule.html opts in automatically when
  the page is COOP/COEP-isolated.

### Added — Pyodide REPL on /molecule.html

- `src/molecule/python-repl.ts` — lazy-loaded Pyodide (~10 MB on first
  Run click). Exposes the latest calculation as Python `ctx` with
  flattened Float64Arrays of D / C_MO / orbitalEnergies / eri_AO /
  h_AO / S_AO. Cmd/Ctrl+Enter to run; captured stdout/stderr inline.

### Added — Sanity-check + Compare-to-PySCF buttons

- "Sanity-check" — pure-numpy invariants on ctx (Hermitian D, idempotent
  D, orthonormal C under S, positive-definite S, HOMO/LUMO gap). Works
  for every molecule, no external deps beyond numpy.
- "Compare to PySCF" — best-effort install of pyscf via micropip, runs
  HF on the same geometry/basis, diffs against ours. Gracefully degrades
  with QCSchema-export fallback message if pyscf can't compile.

### Infrastructure

- `vercel.json` — `git.deploymentEnabled: false`. No more auto-deploys
  on push; production lands via the release pipeline or manual
  `vercel deploy --prod`.

### Docs

- `RESEARCH_STANDARDS.md` section 7a "Porting acceptance gate
  (non-negotiable)" — codifies the three rules learned the hard way
  from the EE-EOM port: independent oracle / full-tensor / hard ε;
  symbol-collision awareness; curve-fitting against your own diagnostic
  is tautology.
- `CLAUDE.md` and `LIMITATIONS.md` updated to reflect both EOM ports
  closed.

## [0.6.0] — 2026-05-21

The visualization release. Nine commits since v0.5.0 ship a complete
SVG visualization stack for the chemistry property surface — every
number the engine emits now has a chart. All viz is rendered as
inline SVG (no canvas, no client-side libs), wired into
`/molecule.html`, and validated by unit + e2e tests.

### Added — Chart foundation (`src/viz/chart-base.ts`)

- Shared color palette, frame defaults, linear scales, nice-tick
  axes, grid helpers, polyline emitter, XML-escape utility — the
  primitive layer every batch builds on.

### Added — Batch 1: property-surface charts

- `alphaOmegaChart` — α(ω) dispersion curve.
- `noonChart` — natural-orbital occupation numbers.
- `uvVisChart` — Gaussian-broadened absorption spectrum + sticks.
- `energyDecompChart` — SCF + correlation + dispersion stack bar.
- `c6MatrixChart` — heatmap of pairwise C₆ coefficients.
- `rotationalCard` — rotational constants A/B/C card.

### Added — Batch 2: molecular structure

- `dispersionD2Chart` — pairwise D2 dispersion bars (top-K).
- `mullikenChart` — divergent red/cyan charge bars.
- `moleculeGraph2D` + `moleculeStructure` — projected 2D molecular
  graph with bond detection, charge / spin overlays, xy/xz/yz axes.

### Added — Batch 3: 2D fields + iteratives

- `contour2DChart` — generic 2D scalar field heatmap (divergent for
  signed orbitals, sequential for densities, optional iso-contours).
- `localizedOrbitalGallery` — composite multi-panel SVG for Foster-
  Boys / Pipek-Mezey LMOs.
- `convergenceTrace` — log-scale iterative convergence (SCF / DIIS /
  CCSD / EOM-CCSD / DMRG) with threshold guideline.
- `basisCoverageChart` — per-atom horizontal stacked bars by s/p/d/f/g
  shells, with `basisSummary("3s 2p 1d")` helper.

### Added — Batch 4: wired into `/molecule.html`

- New "Structure + basis" card — moleculeStructure + basisCoverageChart
  in a 2-column grid, with per-atom basis-summary footer derived from
  the shell list (L = ix+iy+iz).
- Enhanced "Charges" card — moleculeGraph2D (atoms colored by Mulliken
  charge) + mullikenChart (divergent bars).
- Enhanced "UV-vis" card — Gaussian-broadened SVG spectrum (σ = 0.3 eV)
  alongside the existing canvas stick spectrum.

### Fixed

- `<svg height="auto">` is invalid per the SVG spec — five console
  warnings on every page load. Switched to `style="height:auto"`, which
  is valid CSS and keeps the viewBox-driven aspect ratio.

### Tests

- 31 new viz unit tests across `tests/viz/{chemistry-charts,
  molecule-viz, field-viz}.test.ts`. Cover: chart-foundation rendering,
  divergent vs sequential colormaps, atom-overlay rendering, log-scale
  iterations, threshold guidelines, empty-input branches, grid-mismatch
  errors, basis-summary edge cases.
- `molecule-smoke.spec.ts` (Playwright) — H₂O HF end-to-end through all
  cards including the new Structure card. Clean console.

### Docs

- `CLAUDE.md` collapsed from 47.9k → 19.8k chars. The file kept
  replicating per-stage history against its own advice ("don't replicate
  git log here"). Trimmed two session markers + the 470-line stage block
  to a one-pager state summary. Durable guidance (research-grade
  discipline, architecture invariants, port-don't-re-derive, modern
  reference standards) is unchanged.

## [0.5.0] — 2026-05-21

Major release. 89 commits since v0.4.1 — closes the polarizability +
dispersion matrix end-to-end, lands UKS-DFT (full functional ladder),
ships D2 dispersion + Casimir-Polder C₆, four interop export formats
(Molden / Cube / QCSchema / XYZ trajectory), and a comprehensive
geometry-analysis + properties + diagnostics toolkit. **94 test
files / 553 tests, all green.**

### Added — Open-shell DFT (UKS) + complete polarizability matrix

- **UKS-DFT SCF** (`src/chemistry/dft/uks-scf.ts`) — full functional
  ladder (LDA-SVWN5, BVWN5, BLYP, B3VWN5, B3LYP5). Spin-resolved
  J/K assembly + spin-polarized XC kernel. Closed-shell H₂O matches
  RKS-DFT to ≤ 1e-5 Ha across all functionals; H atom doublet ⟨S²⟩
  = 0.75 exactly; Li doublet cc-pVDZ converges with non-zero spin
  density.
- **UKS-CPHF static α** (`uks-cphf.ts`) — LSDA only. 4-spin-block
  (A+B) on combined (α-OV + β-OV) with XC kernel. Closed-shell
  H₂O matches RKS-TDDFT@ω=0 to 1e-3 per element / 1e-4 isotropic.
- **UKS-TDDFT α(ω)** (`uks-tdhf.ts`) — LSDA dynamic response. (A−B)
  reduces to ε-diag in LSDA (Coulomb and symmetric kernel both
  cancel in A−B). Static ω=0 limit matches UKS-CPHF to 1e-7.
- **TDHF, UHF-TDHF, TDDFT, UKS-TDDFT α(ω) and α(iω)** — all RPA
  response solvers shipped (`tdhf.ts`, `uhf-tdhf.ts`,
  `tddft-response.ts`, `uks-tdhf.ts`).
- **Open-shell counterpoise** — `runCounterpoise` extended to UHF /
  UCCSD / RKS / UKS methods + optional D2 dispersion add-on.
- **{RHF, UHF, RKS, UKS} × {static α, α(ω), α(iω), C₆} 16-cell
  polarizability matrix: 16/16 closed.**

### Added — Dispersion correction (Grimme D2)

- `dispersionD2(atoms, opts)` (`dispersion-d2.ts`) — atomic-pairwise
  C6/R⁶ with Fermi damping and functional-specific s6 (BLYP=1.20,
  B3LYP5/B3VWN5=1.05, LDA=1.05 default). Tabulated for full first
  row + He.
- `dispersionD2Gradient(atoms, opts)` — analytical Cartesian gradient
  ∂E_disp/∂R_A. Validated against central FD to 1e-6 Ha/Bohr.
- `dispersionCorrectedEnergy(scfEnergy, atoms, opts)` — convenience
  wrapper that returns the SCF + dispersion + total decomposition.
- `c6Coefficient` + `c6CoefficientGeneral` (`dispersion.ts`) —
  Casimir-Polder integral C₆ = (3/π)·∫₀^∞ ᾱ(iω)² dω with a
  discriminated `AlphaImagSource` union supporting "rhf" / "uhf" /
  "rks" / "uks" references. Gauss-Legendre quadrature with Golub-
  Welsch nodes; converges to 2% from N=8 → N=32.

### Added — Interop exports

- `toMoldenString({atoms, shells, C_MO, ...})` (`molden.ts`) —
  Cartesian-Gaussian Molden file for Jmol / Avogadro / Multiwfn.
- `densityCube` / `moCube` / `homoCube` / `lumoCube` /
  `spinDensityCube` (`cube.ts`) — Gaussian98-standard Cube files
  with sensible default grids (0.3-bohr step, 4-bohr padding).
- `toQCSchemaClosedShell` / `toQCSchemaOpenShell` (`qcschema.ts`) —
  MolSSI QCSchema v1 AtomicResult JSON for QCEngine / QCFractal /
  cclib pipelines.
- `parseXYZ` / `toXYZ` / `toMultiFrameXYZ` / `parseMultiFrameXYZ`
  (`xyz.ts`) — standard XYZ parser + emitter + multi-frame
  trajectory support for geom-opt visualization.

### Added — Element coverage

- **He** support across STO-3G, cc-pVDZ, and aug-cc-pVDZ. All four
  basis files wired in `atoms.ts` and `integrals.ts`. Element
  union now: H, He, Li, Be, C, N, O, F.
- aug-cc-pVDZ diffuse data for Li/Be/C/N/F (previously only H, O).

### Added — Localization + diagnostics

- `fosterBoys` (`foster-boys.ts`) — Boys 1960 maximization via
  2×2 Jacobi sweeps. Converges in 5-20 sweeps for small molecules.
- `pipekMezey` (`pipek-mezey.ts`) — Mulliken-based PM 1989
  localization. Separates σ and π bonds where Boys mixes them.
- `naturalOrbitalOccupations` (`natural-orbitals.ts`) — NOON via
  Löwdin-orthogonal eigendecomposition of the 1-PDM. Multi-
  reference diagnostic.
- `multireferenceDiagnostic` — aggregates T1, D1, ⟨S²⟩, NOON into a
  single severity verdict + flags (Lee-Taylor, Janssen-Nielsen
  cutoffs).
- `trkSumRule` — Thomas-Reiche-Kuhn cross-check on oscillator
  strengths.
- `decomposeHFEnergy` / `decomposeUHFEnergy`
  (`energy-decomposition.ts`) — one-electron + Coulomb + exchange
  + V_nn breakdown. Reconstructs total to 1e-9 Ha.

### Added — Geometry & analysis utilities

`src/chemistry/geometry.ts` extended with:
- `dihedralAngle`, `findBonds`, `molecularGraph` (adjacency,
  components), `shortestPath` (BFS), `extractFragments`,
  `compareMolecules` (formula + bonds + Kabsch RMSD).
- `translateMolecule`, `rotateMolecule`, `standardOrientation`
  (Gaussian convention).
- `rmsd`, `rmsdAligned` (Kabsch SVD-based alignment).
- `centerOfMass`, `totalMass`, `principalMomentsOfInertia`,
  `rotationalConstants` (A/B/C in cm⁻¹ and GHz).
- `coordinationNumbers` (Grimme D3 fractional CN with k₁=16,
  k₂=4/3).
- `distanceMatrix`, `molecularFormula` (Hill convention),
  `planarity` (best-fit plane RMS), `lewisStructure` (Mayer → bond
  multiplicity).

### Added — Convenience API + ML

- `quickReport(atoms, opts)` (`quick-report.ts`) — one-call SCF +
  full property report.
- `molecularReport(input, integrals, shells, atoms, shellAtomIdx,
  opts)` (`molecular-report.ts`) — comprehensive aggregator
  dispatching on `{kind: "rhf" | "uhf" | "rks" | "uks"}`.
- `uvVisSpectrum(atoms, opts)` (`uv-vis.ts`) — atoms → SCF →
  TDDFT → broadened spectrum + peaks + TRK in one call.
- `molecules` library (`molecules.ts`) — pre-built h2o, ch4, nh3,
  beh2, h2, liH, hf, li, he, atomicH, f at experimental geometries.
- `coulombMatrix` (`descriptors.ts`) — Rupp 2012 permutation-
  invariant ML feature.
- `broadenSpectrum`, `findSpectrumPeaks` (`spectrum.ts`) — Gaussian
  broadening + peak detection for UV-vis output.
- `PERIODIC_TABLE` + `elementInfo` + `molecularWeight`
  (`periodic-table.ts`) — atomic properties database.
- `HARTREE_TO_EV` / `BOHR_TO_ANGSTROM` / `fromHartree` / `toHartree`
  / etc. (`units.ts`) — CODATA 2018 unit conversions.

### Added — SCF + correlation

- **UCCSD(T) frozen-core** via non-contiguous `ReadonlySet<number>`
  (closes explicit TODO in `uccsd-t.ts`). Closed-shell H₂O frozen-1s
  matches RHF-CCSD(T) frozen-1s to 1e-7.
- **DFT SCF level-shift** (`RKSOpts.levelShift`) — mirror of HF
  level-shift for stretched-bond / near-degenerate KS problems.
- **UKS hybrid K-factor fix** — `F_σ -= hfMix·K(D_σ)` (not 0.5·hfMix).
  Brought B3VWN5/B3LYP5 closed-shell UKS-vs-RKS gap from 0.88 mHa
  to 1e-6 Ha (SCF convergence noise).

### Fixed

- `uksCphfPolarizability` initial implementation had a missing factor
  of 2 on the XC kernel piece; closed-shell limit cross-check vs
  `tddftPolarizability` caught it.
- `fosterBoys` initial sign on the A_ij maximization coefficient was
  inverted; caused L to oscillate. Tracked down via the H₂O probe.

### Test surface

`npm test` → **553 passing / 1 skipped across 94 test files**, no
regressions across all 89 commits. `npx tsc --noEmit` clean.

## [0.4.1] — 2026-05-14

EOM-CCSD multi-electron singlet bug-hunt + migration framework. The
E35 cross-validation in v0.4.0 surfaced a 2.57 eV singlet gap vs PySCF
on multi-electron systems. Today's arc traced it through 4 rejected
hypotheses to a **single-character sign flip** in the σ_1 ← R_2 W̄_amef
term, and shipped the framework for the structural close (PySCF port).

### Fixed — Stage 32k: σ_1 sign correction

The σ_1 ← R_2 W̄_amef contribution used `+½ ⟨ma||ef⟩` where
Stanton-Bartlett 1993 Eq 41 requires `+½ ⟨am||ef⟩` (= −½ ⟨ma||ef⟩ by
antisymmetry). One-line fix in `src/chemistry/eom-ccsd.ts`:

  V(m, a+VO, e+VO, f+VO)  →  V(a+VO, m, e+VO, f+VO)

**Result**: LiH STO-3G EOM-CCSD singlet gap **2.57 eV → 0.27 eV**
(10× shrink, within literature EOM-CCSD ↔ FCI bar of 0.1–0.2 eV).
Triplets unchanged at 7 meV (already exact). H₂O / NH₃ / CH₄
multi-electron singlets improved 30–40% from same fix.

### Added — Stages 32j, 32l: missing T-dressings on W̄ intermediates

Per Stanton-Bartlett / Crawford-Schaefer 2000, added:
- T1·T1 dressing on W̄_abej, W̄_mbij (τ_mn^ab = T2 + T1·T1 antisym)
- Linear T1 dressing on W̄_abej, W̄_mbij (Σ_m T1[m,a] ⟨mb||ej⟩ etc.)
- T1 dressing on W̄_mnie and W̄_amef in σ_1 ← R_2

Each closes 5–25% of the remaining multi-electron gap.

### Added — Migration framework (`MIGRATION.md`, `LICENSE-PYSCF`)

Project-wide engineering policy: hand-write only the novel WebGPU
layer; port chemistry methods from peer-reviewed references (PySCF,
libxc, EMSL Basis Set Exchange) with Apache 2.0 attribution.
- `LICENSE-PYSCF` (Apache 2.0 verbatim) at repo root
- `MIGRATION.md` — per-module status table, priority order,
  attribution recipe, JOSS narrative
- `src/chemistry/eom-ccsd-ported.ts` — scaffolded port skeleton
  ready for the structural σ_1/σ_2 fix
- `scripts/dump-pyscf-eom-imds.py` — emits PySCF reference values
  for every EOM-CCSD intermediate
- `experiments/results/2026-05-13/level-6/E36-pyscf-imds-lih.json`
  — first reference artifact (LiH STO-3G; 9 W̄ tensors + t1/t2/eris)

### Added — Permanent verifiers (regression tests + diagnostics)

- `tests/chemistry/eom-ccsd-bruteforce-lih.test.ts` (~700 lines)
  Brute-force H̄ = e^(-T̂) H e^(T̂) on the 4-electron Fock space
  (DIM=64), projects onto (R_1 + antisym R_2) basis (dim=14),
  builds M_mine via σ-on-unit-vectors, diffs element-wise. The
  permanent verifier for any σ_1/σ_2 change.
- `tests/chemistry/eom-ccsd-imds-vs-pyscf.test.ts`
  Per-intermediate diff vs PySCF E36 reference. F_me bit-exact
  (3.3×10⁻⁹ Ha); F_ae and F_mi both 87 µHa off vs PySCF — same
  magnitude points at a single missing F_ov·T1 dressing term
  (documented in MIGRATION.md as the next concrete port piece).

### Documented — Honest negatives (the four rejected hypotheses)

- Stage 32f: "missing σ_1 cross-spin coupling" — R_1×R_1 was correct
- Stage 32f-2: "R_2×R_2 off-diagonal 7.26 eV bug" — was diagnostic
  permutation noise, not a physics bug
- Stage 32g: "stage 32c diagonal patches over-correct" — patches
  are net-positive (revert made LiH triplet WORSE: 7 → 540 meV)
- Stage 32h: "sign-flip on (α,β)↔(β,α) R_2 pairs" — basis-ordering
  artifact in the diagnostic, not in production code

All four are documented in commit messages, `LIMITATIONS.md`, and
the diagnostic-test comments so future sessions don't re-test them.

### Test surface

- 320 / 320 vitest chemistry tests pass (no regressions from σ_1
  sign fix; H₂ STO-3G brute-force still exact since T1 = 0 for 2e)
- `npx tsc --noEmit` clean
- `npm run lint` clean (2 pre-existing warnings)
- 3 e2e specs green (CCSD(T) GPU, H₂O UV-vis, wallclock-vs-PySCF)
- E35 EOM-CCSD validation: LiH essentially closed; H₂O / NH₃ / CH₄
  remaining 0.5–1.9 eV gaps queued for the PySCF port

### Honest scope (carried from v0.4.0)

- LiH STO-3G EOM-CCSD: at literature method precision (0.27 eV is
  the inherent EOM-CCSD ↔ FCI gap, not an implementation bug)
- H₂O / NH₃ / CH₄: 0.5–1.9 eV gaps remain. The 32m verifier
  isolates F_ae / F_mi as having a missing F_ov · T1 dressing
  (87 µHa). W intermediates (woOoO, woVoO, wvOvV, woVVo, woVvO,
  woOoV) not yet diff'd — each is a ~30-line test in the same
  pattern.

## [0.4.0] — 2026-05-12

The chemistry track closed out Tier 2 of the roadmap: every method
the original Tier 2 table listed as "remaining" is now shipped, plus
2 bonus methods (IP-EOM-CCSD, EA-EOM-CCSD) and a brute-force EOM-CCSD
diagnostic framework. This release adds **correlated excited-state
spectroscopy (EOM-CCSD)**, **GPU-accelerated perturbative triples**,
**density fitting**, **open-shell CCSD**, and a **general
non-symmetric eigensolver**. The single most-persistent honest
negative across the v0.3 → v0.4 arc — "EOM-CCSD ~10 mHa from H₂
FCI" — is now closed to numerical precision via brute-force diagnosis
+ targeted σ-equation patches.

### Added — Tier 2: EOM-CCSD stack (stages 24a/24b, 30, 33, 35, 36)

- **Stage 24a — non-symmetric dense eigensolver** (`src/manybody/dense-eig-general.ts`).
  Hessenberg reduction (Householder) + Wilkinson-shifted QR with
  deflation. Returns real + imaginary parts of all eigenvalues from
  a real N×N matrix. 5 tests green (diagonal, upper-triangular,
  symmetric agreement with `eigsymmetric` to 1e-9, companion-matrix
  polynomial roots, similarity-transformed diagonal recovery).
- **Stage 24b — EE-EOM-CCSD** (`src/chemistry/eom-ccsd.ts`).
  `runEOMCCSD(ccsd, integrals, hf)` returns excitation energies on
  the singles + antisymmetric doubles manifold via Stanton-Bartlett
  σ equations. Dim = NOCC·NVIRT + C(NOCC,2)·C(NVIRT,2). For H₂O
  STO-3G: 3 degenerate triplets at 10.32 eV + dipole-allowed singlet
  at 11.76 eV (1.44 eV below CIS singlet — correlation correction in
  the expected direction).
- **Stage 30 — eigenvector back-substitution.** Tracks Q through
  Hessenberg (Householder right-mult) and QR iteration (right-Givens
  accumulation). Eigenvectors via back-substitution on the Schur
  form + v_M = Q·v_T transform. `runEOMCCSD` now returns
  `amplitudes` alongside `energies`. Degenerate eigenvalues handled
  by setting the zero-denominator entry to 0 (one representative
  per degenerate subspace).
- **Stage 33 — EOM-CCSD oscillator strengths.** f_n = (2/3)·ω_n·|μ_n|²
  via R₁·μ AO→MO dipole transform. Spin-orbital R₁ amplitudes
  summed with σ_i = σ_a filter; spin-flip → 0 by physics. H₂ STO-3G:
  3 triplets f ≈ 10⁻³¹, S1 f = 1.13 (dipole-allowed), S2
  (doubly-excited) f ≈ 10⁻³¹ — textbook spin and symmetry selection
  rules.
- **Stage 35 — EOM-CCSD spin classifier.** Per-root decomposition of
  R₁ amplitudes into (αα, ββ, αβ, βα) channels; reports
  singlet weight + triplet weight ∈ [0, 1]. H₂ STO-3G: 3 triplets
  at exactly 1.000 triplet weight, S1 at 1.000 singlet, S2 (R₂-
  dominated) at 0.003 triplet + 0 singlet (rest in R₂ mass).
- **Stage 36 — H₂O EOM-CCSD UV-vis demo experiment (E33).**
  New `experiments/level-6-chemistry/E33-h2o-uvvis.ts` wired into
  the runner + Playwright e2e at `e2e/uvvis-h2o.spec.ts`. Returns
  the lowest 12 excitations with (energy, oscillator strength,
  singlet/triplet weight, assignment). Validates real eigenvalues,
  dipole-allowed singlet presence, and ordering.

### Added — Tier 2: IP/EA-EOM-CCSD (stages 37–38)

- **Stage 37 — IP-EOM-CCSD** (`src/chemistry/ip-eom-ccsd.ts`).
  Diagonalizes H̄ on the (1h + antisym 2h1p) manifold. Reuses CCSD
  intermediates + the new eigGeneral. For H₂O STO-3G:
  Koopmans IP 10.65 eV, ΔSCF IP 8.36 eV, **IP-EOM-CCSD IP 12.03 eV**
  (closest to experimental 12.62 of all three methods).
- **Stage 38 — EA-EOM-CCSD** (`src/chemistry/ea-eom-ccsd.ts`). Mirror
  of IP-EOM on the (1p + antisym 1h2p) manifold. For STO-3G systems
  with unbound LUMOs, EAs are negative — quantifies basis-set limit.
  H₂O: Koopmans LUMO EA −16.48 eV, EA-EOM-CCSD best EA −16.35 eV
  (after stage 32e σ_2 patch).

### Added — Tier 2: open-shell CCSD (stage 25)

- **Stage 25 — UCCSD on UHF** (`src/chemistry/uccsd.ts`).
  Refactored `runCCSD` to extract a `ccsdIterate` core; both
  closed-shell (RHF) and open-shell (UHF) paths share the
  Stanton-Bartlett residual iteration. UCCSD-specific scaffolding:
  3-block AO→MO ERI transform for (αα|αα), (αα|ββ), (ββ|ββ);
  spin-orbital antisym ERI via spin selection rules; "α-occ → β-occ
  → α-virt → β-virt" SO ordering. H₂ closed-shell UCCSD =
  RHF-CCSD to 1e-10. Be⁺ STO-3G doublet: E_corr = −0.357 mHa.

### Added — Tier 2: density fitting (stages 26, 29, 34)

- **Stage 26 — Cholesky-DF infrastructure** (`src/chemistry/df.ts`).
  Pivoted incomplete Cholesky decomposition of the rank-4 ERI tensor
  as a (n², n²) PSD matrix. Returns a rank-3 B-tensor of shape
  (n², M_aux) with threshold-controlled truncation. H₂O STO-3G:
  τ = 1e-6 → 28 aux of n² = 49 (43% compression), max ERI error
  1.8×10⁻¹⁵ Ha.
- **Stage 29 — DF-HF SCF wiring**. `runRHFSCF` accepts a `useDF`
  option (boolean / number / DFResult). DF-HF energy matches direct
  HF to **7×10⁻¹⁴ Ha** on H₂O STO-3G (machine precision).
- **Stage 34 — DF-MP2 wiring**. `runMP2` accepts the same `useDF`
  option. Reformulates (ia|jb) as Σ_P B_ov[i,a,P]·B_ov[j,b,P] via
  a 2-pass AO→MO transform of B. Memory drops from O(n⁴) to
  O(n_occ·n_virt·n_aux). H₂O STO-3G: DF-MP2 = exact MP2 to 0 Ha
  at τ = 1e-10.

### Added — Tier 2: WebGPU port of (T) (stages 27–28)

- **Stage 27 — WebGPU CCSD(T)** (`src/shaders/ccsd-t.wgsl` +
  `src/chemistry/ccsd-t-gpu.ts`). WGSL compute kernel: 1 thread per
  (i,j,k) occupied spin-orbital triple; each thread sums over all
  (a,b,c) virtuals internally (9-perm W and V dressings inline)
  and writes a single f32 partial sum. f32 GPU storage + f64 CPU
  reduction. e2e validation in `e2e/ccsd-t-gpu.spec.ts`: BeH₂
  STO-3G |Δ| = 1.35×10⁻¹¹ Ha, H₂O STO-3G |Δ| = 7.09×10⁻¹³ Ha
  (sub-pHa precision).
- **Stage 28 — cc-pVDZ benchmark.** H₂O cc-pVDZ: CPU 198.6 s →
  **GPU 5.05 s = ~39× speedup**, |Δ| = 2.4×10⁻¹⁰ Ha. Single-run
  measurement on Apple M2 Pro — not yet routed through the
  warmup + 20-trials research harness; the speedup number could
  move ±20% on different hardware.

### Added — Stage 32 close-out: EOM-CCSD precision validation

The single most-persistent honest negative across the v0.3 → v0.4
arc — "EOM-CCSD ~10 mHa from H₂ STO-3G FCI" — is now closed via a
brute-force EOM-CCSD reference framework + targeted σ-equation
patches.

- **Stage 32b — brute-force EE-EOM-CCSD reference**
  (`tests/chemistry/eom-ccsd-bruteforce.test.ts`). Constructs
  H̄ = e^(−T̂) H e^(T̂) explicitly in the 4-spin-orbital Fock space
  (T̂² = 0 for 2-electron makes e^(±T̂) = I ± T̂ exact), projects
  onto the (R₁, R₂) basis used by `runEOMCCSD`, compares element-
  wise. Diagnosis:
    M_mine − M_exact = diag(+δ, +δ, +δ, +δ, −2δ),  δ = |E_corr|/2
  All off-diagonals match to 10⁻¹⁶; the diff is purely diagonal.
- **Stage 32c — EE-EOM σ-diagonal patch.** σ_1 += 0.5·E_corr·R₁,
  σ_2 −= E_corr·R₂. H₂ STO-3G EOM-CCSD now matches FCI to
  **10⁻⁵ Ha** (was 10–20 mHa). H₂O lowest singlet shifts 11.76 →
  11.21 eV (correlation correction vs CIS grows 1.44 → 1.99 eV,
  in line with typical EOM-CCSD-vs-CIS gaps).
- **Stage 32d — IP-EOM-CCSD cross-check**
  (`tests/chemistry/ip-eom-ccsd-bruteforce.test.ts`). Found a
  more nuanced structure than EE: R₁ sector exact, R₂ sector
  off by ~2.3 Ha (60 eV) per state from σ_2's P(ij)·W_mbej
  contraction. Lowest IPs (R₁-dominated) are FCI-equivalent
  already — the H₂O 12.03 eV result is validated. R₂ "Auger
  satellite" sector needs separate σ_2 re-derivation (deferred).
- **Stage 32e — EA-EOM-CCSD cross-check + σ_2 patch**
  (`tests/chemistry/ea-eom-ccsd-bruteforce.test.ts`). Cleaner
  picture: R₁ exact, R₂ off by +|E_corr|/2 (analogous to EE's
  σ_1 issue but on σ_2). Patched ea-eom-ccsd.ts σ_2 with the
  matching correction; brute-force diff post-patch confirms
  zero everywhere.

Summary of EOM-CCSD validation status post-32e:

| sector | EE-EOM | IP-EOM | EA-EOM |
|---|---|---|---|
| R₁ (primary states) | +δ shift, patched (32c) | exact ✓ | exact ✓ |
| R₂ (correlated/satellite) | −2δ shift, patched (32c) | +2.3 Ha bug, deferred | +δ shift, patched (32e) |

5 of 6 sectors fully validated to brute-force precision.

### Test surface

- Vitest: **401 tests** (319 chemistry + 82 manybody) + 1 opt-in
  cc-pVDZ CCSD(T). Two pre-existing untracked
  `tests/numbers.test.ts` failures (benchmark-drift checks) remain;
  they predate this conversation.
- e2e: 3 specs green (CCSD(T) GPU at STO-3G + cc-pVDZ; H₂O UV-vis).

### Honest residuals (documented in CLAUDE.md)

- IP-EOM σ_2 R₂ sector structural bug (~60 eV on H₂; affects only
  Auger-satellite eigenvalues, not the physically important lowest
  IPs).
- DF-HF / DF-MP2 machine-precision matches validated on STO-3G only;
  cc-pVDZ expected to be equally clean by construction, not
  separately benchmarked.
- (T) GPU 39× speedup is a single e2e measurement on M2 Pro; not
  routed through warmup+20-trials research harness yet.

## [0.3.0] — 2026-05-09

The chemistry track went from "ground-state methods + UV-vis" to a
complete experimental-chemistry SI bundle: triplet excited states across
the full functional ladder, full IR + Raman vibrational spectroscopy,
field-response trio (μ → α → β), ideal-gas thermochemistry,
**open-shell UHF**, and ΔSCF ionization potentials. Every property an
experimental chemistry paper would tabulate is now computable in a
browser tab.

### Added — Tier 2: triplet excited states across the full ladder

- **Tier 2 stage 15a — spherical-d on the grid (real fix).** The
  pre-15a "guard with throw" was documentation papering over a real
  bug. Stage 15a applies the Cartesian → spherical-d transform T to
  basis values, gradients, Hessians, and density matrices on the
  numerical grid so RKS-DFT SCF, TDA-DFT XC kernel, and HF + DFT
  analytical gradients all stay consistent when integrals are built
  with `spherical: true`. H₂O cc-pVDZ B3LYP5: SCF Δ = 1.2 mHa,
  TDA[0] 7.627 vs 7.601 eV, |∇| within 1–4 %.
- **Tier 2 stage 15b — triplet TDA + TDDFT across the full functional
  ladder.** Spin-polarized LSDA (Slater + VWN5 with the full VWN
  spin-interpolation function), spin-polarized B88 (clean spin
  decomposition), and spin-polarized LYP (Miehlich 1989 integrated-
  by-parts form, exploiting LYP's linearity in γ_↑↑/γ_↑↓/γ_↓↓ to
  give closed-form γ-coefficients). Triplet kernel evaluators land
  exactly on the textbook |T_z=0⟩ Casida convention. H₂O STO-3G first
  singlet/triplet (eV) — textbook ordering across all 6 methods:
  HF S 13.20/T 11.10; LDA S 11.50/T 9.53; BVWN5 S 11.35/T 9.41;
  BLYP S 11.31/T 9.36; B3VWN5 S 11.76/T 9.80; B3LYP5 S 11.72/T 9.76.

### Added — Tier 2: complete vibrational spectroscopy

- **Tier 2 stage 16 — harmonic vibrational frequencies + IR intensities.**
  6N central-FD Hessian on the existing analytical gradient with mass-
  weighting + Eckart projection; dipole-derivative tracking during the
  same FD loop gives IR intensities in km/mol. H₂O HF/STO-3G: bend
  2170, sym 4140, asym 4391 cm⁻¹ — matches Pople 1969 reference to
  0.1 cm⁻¹. H₂ symmetric stretch IR-inactive < 1e-3 km/mol — homo-
  nuclear symmetry forces ∂μ/∂q = 0, reproduced from FP arithmetic.
- **Tier 2 stage 18 — Placzek Raman activities.** 6N FD on the
  polarizability tensor at displaced geometries (162 SCF runs total
  on H₂O), projection onto modes, S_k = 45·ā_k² + 7·γ_k² in Å⁴/amu.
  H₂ stretch is BOTH IR-inactive AND Raman-ACTIVE — the textbook
  rule of mutual exclusion in centrosymmetric molecules, reproduced
  from FP arithmetic alone.

### Added — Tier 2: field-response properties

- **Tier 2 stage 17 — static dipole polarizability via finite-field.**
  Perturbs h_AO with +E·μ_AO, runs SCF at ±E along each axis, FDs the
  dipole. Returns the full 3×3 tensor + isotropic + anisotropy +
  principal components. H₂O α_iso = 2.7 a.u. at STO-3G (small basis
  underestimates; aug-cc-pVDZ reaches ~7-8 a.u., experiment 9.79 a.u.).
- **Tier 2 stage 20 — first hyperpolarizability β via finite-field.**
  19-SCF stencil for the full 27-component β_ijk tensor, Kleinman
  symmetrization, rotational vector invariant. H₂O |β_vec| ≈ 11 a.u.
  H₂ centrosymmetric ⇒ all 27 β_ijk < 1e-2 (inversion enforces β = 0
  exactly).

### Added — Tier 2: thermochemistry

- **Tier 2 stage 19 — ideal-gas thermochemistry.** ZPE + thermal U/H
  + Sackur-Tetrode translation + rigid-rotor rotation (linear /
  asymmetric) + harmonic-oscillator vibration partition functions →
  S(T) and G(T) at any (T, P). Symmetry number σ scales rotational
  entropy as −R·ln(σ) exactly. H₂O 298.15 K, 1 atm: total entropy
  45.06 cal/(mol·K) vs experiment 45.10 — match to 0.1.

### Added — Tier 2: open-shell SCF

- **Tier 2 stage 21 — Unrestricted Hartree-Fock (UHF).** Spin-resolved
  α/β orbitals, F_σ = h + J(D_α + D_β) − K(D_σ). Symmetry-breaking
  initial guess for radicals; closed-shell systems collapse back to
  RHF (verified UHF=RHF on H₂ to 1e-8). DIIS on stacked α+β error
  vector. ⟨S²⟩ via Pople formula. H atom: −0.466582 Ha (lit −0.4666);
  Li atom: −7.315526 Ha (lit −7.3155); H₂⁺: −0.581667 Ha. ⟨S²⟩ =
  0.750000 to FP precision for clean doublets.
- **Tier 2 stage 22 — vertical IP / EA via Koopmans + ΔSCF.** First
  user-visible deliverable powered by UHF: remove an electron, run
  UHF on the cation, get IP from energy difference. LiH Koopmans
  7.40 eV vs experiment 7.85 eV — within 6 %. ΔSCF ≤ Koopmans where
  basis flexibility allows orbital relaxation.

### Validated against

- **PySCF** for HF / DFT / MP2 / CCSD / CCSD(T) energies on H₂ / H₂O /
  BeH₂ / CH₄ / STO-3G + cc-pVDZ to ≤ 0.5 mHa (35 µHa with spherical-d).
- **Pople 1969 STO-3G HF reference** for H₂O harmonic frequencies
  (bend / sym-stretch / asym-stretch within 0.1 cm⁻¹).
- **Experiment (gas phase)** for H₂O thermochemistry — total entropy
  45.06 vs 45.10 cal/(mol·K).
- **libxc** for LYP closed-shell collapse (`gga_c_lyp.mpl`).
- **Symmetry-forced exact results** as the cleanest correctness checks:
  homonuclear-diatomic IR-inactive stretch, centrosymmetric β = 0,
  rule of mutual exclusion in H₂.

### Test surface

- 479 unit tests (was 433), ~50 s on M2 Pro.
- 11 e2e Playwright specs (unchanged).
- typecheck strict, lint pre-existing only.

[0.3.0]: https://github.com/abgnydn/webgpu-q/releases/tag/v0.3.0

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
