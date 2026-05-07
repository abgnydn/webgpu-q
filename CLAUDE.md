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

## Roadmap to the frontier (the path you're on)

The project is past the launchpad. **All six chemistry-track phases are
shipped** (A through E5: foundation → 1D records → real molecules → HF
SCF → MP2 → cc-pVDZ basis → CCSD → CCSD(T) → cc-pVDZ CCSD(T) on H₂O).
The repo is public + CI-green. The honest path from here is below —
ranked by *what it costs vs what it unlocks*, not by ladder position.

### What's shipped (recap)

- ✓ **L1 statevector**, **L2 MPS** (incl. GPU MPS through Phase 6 v1
  with χ ≤ 64), **L3 kernel fusion** (Tier B/C/D — 4.18× headline),
  **L6 chemistry** (full quantum-chemistry stack)
- ✓ **DMRG** with Lanczos + MPO; ITensor cross-checked at N = 8 to f64
- ✓ **Phase B**: TFIM/Heisenberg N = 128 in browser, validated vs Pfeuty/Bethe
- ✓ **Phase C/D/E1-5**: HF / MP2 / FCI / CCSD / CCSD(T) on
  H₂ → LiH → BeH₂ → H₂O → CH₄ in STO-3G; **cc-pVDZ CCSD(T) on H₂O in 106 s**
- ✓ 309 unit tests, 11 e2e specs, all green; CI live

### Next: chemistry-track tier roadmap

Ranked by ROI. Cost in "sessions" assumes one focused session = a few
hours. The tier framework exists because each subsequent feature stops
being free physics ladder-rungs and starts being grungy engineering — so
order matters.

#### Tier 1 — Quick wins (~1 session bundled)

Each takes ≤1 hour, kills a real gap, ships in one combined commit:

| feature | LOC | unlocks |
|---|---:|---|
| **f, g, h orbital integrals** | ~50 | cc-pVTZ basis (need only Boys F_n table extension to n ≤ 12 + EMSL constants — `integrals-cg.ts` already handles arbitrary L) |
| **Spherical-harmonic basis** | ~150 | bit-exact match to PySCF (kills the 4 mHa Cartesian-d slack) |
| **DIIS SCF accelerator** | ~100 | 5-10× HF speedup (H₂O cc-pVDZ: 101 iter → ~10) |
| **Frozen core** | ~30 | 2-3× CCSD(T) speedup |
| **Diffuse functions** | ~30 | aug-cc-pVDZ — anions, excited states |
| **Schwarz integral screening** | ~50 | 2-5× ERI speedup |

After this single bundle: **bit-exact PySCF agreement at every level**,
plus 5-10× across-the-board speedup. Highest-ROI session in the project.

#### Tier 2 — Major capability (~10 sessions total)

| feature | sessions | unlocks |
|---|---:|---|
| **DFT (LDA + B3LYP + Lebedev grids)** | 2 | **~90% of all real chemistry** uses DFT |
| **HF analytical gradients + BFGS** | 2 | **geometry optimization** — find equilibrium structures |
| **WebGPU port of (T) kernel** | 3 | 10-100× speedup on (T); cc-pVTZ CCSD(T) becomes routine |
| **EOM-CCSD (excited states)** | 1-2 | UV-vis, photochemistry — reuses CCSD intermediates |
| **UHF + open-shell CCSD** | 2 | radicals, transition-metal complexes |
| **Density fitting (RI)** | 1 | 3-5× speedup + half memory → cc-pVTZ becomes routine |

After Tier 2: **a genuinely useful undergrad chemistry tool**. Drug-style
geometry optimizations + DFT vibrational analysis + UV-vis spectra in a
browser tab.

#### Tier 3 — Substantial (~25 sessions)

CCSDT (full triples), CASSCF (multi-reference), TD-DFT, MP2/CCSD
gradients (Z-vector), PCM solvent, coupled-perturbed HF (NMR /
polarizabilities), WebGPU integral parallelization. Each is well-defined
but takes a few sessions of careful work.

#### Tier 4 — Genuinely hard (a season each)

CASPT2 / NEVPT2 (multi-ref perturbation, intruder states), periodic DFT
(k-points, Brillouin zone), spin-orbit / X2C (two-component spinors),
analytical CC gradients (Lagrangian per method), QM/MM. Possible but
expensive; do later.

### Deferred: the original moonshots

Still on the table, but lower priority than chemistry depth right now:

- **Phase D (WebRTC swarm)** — distributed 1D chain across browsers.
  ~3-5 sessions. Foundation for any multi-machine moonshot. Reuse
  `webgpu-p2p-evolution`'s relay.
- **E.1 — Verify Sycamore** — 2D PEPS primitive + Sycamore gate set
  + distributed contraction via Phase D. Reproduce Pan & Zhang 2021 in
  a browser. ~3-5 sessions on top of Phase D.
- **E.2 — Fault-tolerant qubit** — stabilizer sim + surface code +
  syndrome decoder + threshold curve. ~4-6 sessions, no Phase D needed.
- **E.3 — Browser-native lattice QCD** — 4D lattice + Wilson Dirac +
  fused CG solver. ~6-10 sessions, hardest port.

### Cleanest near-term path

**Tier 1 bundle → HF gradients → DFT → WebGPU (T) → EOM-CCSD**.
~8-9 sessions to a "real chemistry tool in a browser tab." Every step
ships a publishable artifact; if you stop early you still have a
strictly more useful repo than yesterday.

The unifying thesis stays the same: *"every advanced physics
simulation in the world ships as a URL"*. webgpu-q is the proof point;
the chemistry track is its highest-leverage demonstration.

---

## Current state (2026-05-06)

**Latest milestone: Tier 2 stage 8 — CIS / TDA excited states.**
First excited-state capability on top of HF. Diagonalizes the
CIS (Tamm-Dancoff) Hamiltonian on the singles excitation manifold:

  A^singlet_{ia,jb} = (ε_a − ε_i)·δ_ij·δ_ab + 2·(ia|jb) − (ij|ab)
  A^triplet_{ia,jb} = (ε_a − ε_i)·δ_ij·δ_ab            − (ij|ab)

with chemist-notation (pq|rs) MO ERIs from `transformERIToMO`.

What got built:
- **`src/chemistry/cis.ts`**: `runCIS(integrals, hf, opts)`
  builds the singlet + triplet A blocks separately and dense-
  diagonalizes via `eigsymmetric`. Returns excitation energies
  + amplitudes per spin sector. Optional `nRoots` and
  `spin: "singlet" | "triplet" | "both"` filters.
- 6 tests: triplet > 0 (ground-state stability) and singlet >
  triplet (Hund's rule) for H₂ / H₂O STO-3G; H₂ first singlet
  HOMO→LUMO at 0.947 Ha ≈ 25.7 eV (textbook reference); CIS
  amplitudes are normalized to 1e-10; `S₀ − T₀ = 2·(ia|ia)`
  internal consistency check.

Limitations / scope notes shipped:
- Dense eigsymm only — fine for n_occ·n_virt ≤ a few hundred.
  Larger systems would want a Davidson iterative solver.
- TDA / no full RPA — only A is diagonalized. Full TDDFT
  diagonalizes the (A, B) 2×2 block; modest follow-up.
- TDA-DFT (KS orbitals + hybrid mix) is a one-line extension
  once the XC kernel is plumbed; deferred.
- No symmetry adaptation — common literature values quoted with
  point-group labels (e.g. H₂O ¹B₁) need symmetry projection
  we don't ship; the raw C₁ HOMO→LUMO excitations we compute
  match what other codes report when symmetry is disabled.

**Tier 2 stage 7 — Lebedev angular quadrature.**
The DFT angular grid is now Lebedev-Laikov by default (order 110,
exact for spherical harmonics up to L = 17). Replaced the older
12 × 24 = 288-point Gauss-Legendre × uniform-φ product rule for a
**2.6× point reduction at strictly better algebraic accuracy**.
Available orders: 50, 110, 302 — `LEBEDEV_AVAILABLE_ORDERS`.
Pass `nLebedev: null` to fall back to the product rule.

What got built:
- **`src/chemistry/dft/lebedev.ts`**: `genOh` orbit expander
  (octahedral group Oh on 6 symmetry classes — axis, face,
  corner, (a,a,b), (a,b,0), (a,b,c)) plus tabulated parameters
  for orders 50, 110, 302. Tables sourced from the Lebedev-
  Laikov 1999 Fortran routine (Christoph van Wuellen translation,
  via PySCF's `dft/LebedevGrid.py`). Cross-validated to fp:
  Σw = 4π exactly, |r|=1 to 1e-16 per point, ∫x²y²z² = 4π/105
  to 1e-15 on every order.
- **`molecularGrid`** refactored to take `nLebedev?: LebedevOrder`
  and use it as the angular path; default `nLebedev: 110`.
  Legacy product rule still reachable via `nLebedev: null` and
  `nTheta` / `nPhi` for cross-checks.

H₂O STO-3G timings (M2 Pro, BLYP):
- Grid:        43200 → 16500 points (2.6× fewer).
- SCF:         119 ms → 55 ms (2.2× faster).
- Gradient:    389 ms → 356 ms (1.1× faster — grad is dominated
               by the n⁴ ERI derivative loop, which is angular-
               independent).
- Energy:      −75.27725 → −75.27722 (31 µHa difference, well
               below chemical accuracy).
- ρ-integration error: 1e-4 e (was 0 with the product rule; the
               product rule is exact for any 2π-periodic finite
               φ-Fourier mode, so uniform-φ + Gauss-Legendre on
               cos θ trivially conserves charge to fp). Both are
               far below the 0.01 e test pass bar.

Full vitest chemistry track: 60.7 s → 49.8 s (~17% faster, dominated
by the DFT-heavy tests). DFT energy tests alone: 2.5 s → 0.9 s.

**Tier 2 stage 6b — DFT gradients (GGA + hybrids).**
Analytical RKS-DFT geometry optimization is now end-to-end for
the full functional ladder: `lda-svwn`, `bvwn5`, `blyp`, `b3vwn5`,
`b3lyp5`. The GGA path adds the ∂γ/∂R term using a new basis-
Hessian evaluator on the molecular grid:
  ∂(∇ρ)_a/∂R_N^k = −2·Σ_{μ on N} { (∂_k φ_μ)·(P·∂_a φ)_μ
                                  + (∂_k ∂_a φ_μ)·(Pφ)_μ }
  ∂γ/∂R_N^k     = 2·Σ_a (∇ρ)_a · ∂(∇ρ)_a/∂R_N^k
  contribution   = −4·w·v_γ · Σ_{μ on N} {…}

What got built:
- **`evalBasisHessianOnGrid`** in `src/chemistry/dft/density.ts`:
  6 unique Hessian components (xx, yy, zz, xy, xz, yz) per (μ, p).
  Same shifted-L recursion as the gradient evaluator with extra
  ±2 polynomial powers on each axis. FD-validated to 2e-8 against
  central-FD of the gradient.
- **`dftGradient`** extended with the GGA path: pre-computes
  (Pφ), (P·∂_a φ) for a∈{x,y,z}, and contracts with the basis
  Hessian. LDA is now a code-path simplification rather than a
  separate function.
- 20 new test cases: FD-vs-analytical to **1e-3 Ha/Bohr** for
  every (functional, molecule) ∈ {lda-svwn, bvwn5, blyp, b3vwn5,
  b3lyp5} × {H₂, H₂O, BeH₂} STO-3G. Translational invariance
  also at 1e-3.

H₂O STO-3G gradient timings (M2 Pro, single-thread TS):
- LDA:     374 ms (vs 98 ms SCF).
- BVWN5:   409 ms.
- BLYP:    411 ms.
- B3LYP5:  511 ms.
GGA only adds ~10% on top of LDA — Hessian build is cheap.

The remaining honest negative is the **weights-fixed approximation**:
∂(Becke-partition weights)/∂R is still not computed. The residual
on Σ_atoms ∇E is ~1e-3 Ha/Bohr on H₂O, sub-mHa/Bohr per component.
Eliminating it is the immediate follow-up.

**Tier 2 stage 6 — DFT analytical gradients (LDA).** The Pulay
HF gradient was reused with `kFactor = hfMix`, and a first-pass
LDA XC contribution shipped before the GGA-Hessian work. H₂O
LDA STO-3G geom-opt: 55.4 s FD → 7.6 s analytical (7.3× faster).

**Tier 2 stage 5b — HF gradient speedup.**
The Pulay-1969 analytical gradient now actually beats FD. Three
optimizations on top of the stage-5a correctness implementation:
- **8-fold canonical ERI loop**: iterate (μ ≥ ν, λ ≥ σ, (μν) ≥
  (λσ)) only, computing one set of three derivative ERIs (∂A, ∂B,
  ∂C; ∂D from translational invariance) per canonical quartet.
  J + K combine via a unified Γ-coupling sum over the 8 ERI-
  symmetric permutations (deduplicated for low-multiplicity
  canonicals where μ = ν, λ = σ, or (μν) = (λσ)). 16× fewer
  derivative ERI evaluations than the naive loop.
- **Schwarz screening**: precompute Q_μν = √|⟨μν|μν⟩| and skip
  canonical quartets with `Q_μν · Q_λσ · |Γ| < 1e-10`.
- **1-electron pair symmetry**: μ ≥ ν loop with sym = 2 for off-
  diagonals, sym = 1 for the diagonal (where bra and ket sides
  go to the same atom — the conditional matters and was the
  source of the only bug introduced during this stage).

H₂O STO-3G headline (M2 Pro, single-threaded TS):
- HF energy: 86 ms.
- Analytical gradient: **4500 ms → 440 ms** (10× speedup).
- Geometry optimization: **52 s → 6.6 s analytical** (2.2× faster
  than FD's 14.5 s; was 3× SLOWER than FD before this stage).
- Same final energy E = −74.96590049 to 8 decimals as the FD path.
- Full vitest suite: 47 s → 22 s on the chemistry track.

The 8-fold canonical loop has a defensive moat: per-pair Γ-coef
is computed by enumerating the 8 permutations and deduplicating
on the fly, so it handles every (μ = ν, λ = σ, (μν) = (λσ))
multiplicity case without case analysis.

**Tier 2 stage 5a — analytical HF gradients (correctness).**
Pulay 1969 gradient via integral derivatives. FD-validated to
1e-5 Ha/Bohr per component on H₂ / H₂O / BeH₂ STO-3G.
Translational invariance Σ ∇E = 0 holds to 1e-9. Integral
derivatives via bra-side Hellmann-Feynman shift
`2α·prim(I+1) − I_axis·prim(I−1)` at the primitive level;
translational invariance recovers the partner-center derivatives.

**Tier 2 stage 4 — LYP correlation + B3LYP5.**
Two functionals shipped on top of the Tier 2 stage 3 GGA + hybrid
infrastructure:
- `blyp`: Slater + B88 GGA exchange + LYP GGA correlation. The
  classic "BLYP" most chemists mean by "GGA-DFT".
- `b3lyp5`: Becke 1993 hybrid with VWN5 — the published B3LYP, with
  VWN5 in place of VWN_RPA (i.e. PySCF's "B3LYP5"):
    E_xc = 0.20·E_x^HF + 0.80·E_x^Slater + 0.72·ΔE_x^B88
         + 0.81·E_c^LYP + 0.19·E_c^VWN5

How the LYP closed-shell bug from the prior attempt was avoided:
- Closed-shell collapse cross-referenced against the canonical
  libxc Maple source (`maple/gga_exc/gga_c_lyp.mpl`). The libxc
  per-particle ε at z = 0 simplifies to:
    ε^closed = −a/h − a·b·C_F·E/h
             + a·b·E·(3 + 7δ)·γ / (72·h·ρ^(8/3))
  with u = ρ^(−1/3), h = 1+d·u, E = exp(−c·u), δ = c·u + d·u/h.
- The previous attempt's hand-collapsed Miehlich form gave a γ
  coefficient of (73 + 11δ)/144 — about 10× too large with the
  wrong δ-coefficient. That's exactly the sign-error-grade bug the
  prior attempt shipped (30–240 mHa off PySCF B3LYP). The libxc
  cross-check is what caught it.
- Defensive moat: `tests/chemistry/lyp.test.ts` is a 20-test FD
  self-test on (ρ·ε_LYP) — analytic v_ρ and v_γ must match central-
  FD to 1e-6 across (ρ, γ) ∈ {0.01–2, 1e-6–4} sample grid, plus
  closed-form γ = 0 UEG match to 1e-10. Catches sign + magnitude
  errors at the kernel level before they hit any molecule.

H₂ STO-3G energies (Ha): HF = −1.117, BLYP = −1.155, B3LYP5 =
−1.159. H₂O STO-3G: HF = −74.96, BLYP = −75.28, B3LYP5 = −75.28.
Within ~10 mHa of published references (literature B3LYP/H₂ ≈
−1.166; PySCF B3LYP5/H₂O ≈ −75.31). The hybrid hierarchy is not
strictly bracketed (small minimal-basis molecules can have B3LYP5
slightly below BLYP) — that's a physical feature, not a bug.

**Tier 2 stage 3 — GGA + hybrid DFT.** Three functionals on top
of LDA: `bvwn5` (Slater + B88 + VWN5), `b3vwn5` (Becke3 hybrid w/
VWN5). What got built:
- Density gradients on the grid: `evalBasisGradOnGrid` (∇φ_μ),
  `evalDensityAndGradient` (∇ρ + γ = |∇ρ|²) — same O(n²·nGrid) cost.
- B88 GGA exchange — Becke 1988, ε_x^B88 = ε_x^Slater
  − 2^(−1/3) β ρ^(1/3) F(u), F(u) = u²/(1 + 6β u arcsinh u).
  Analytical v_ρ + v_γ.
- GGA Fock build: V_xc[μν] = ∫{v_ρ φ_μ φ_ν + 2 v_γ ∇ρ·(∇φ_μ φ_ν +
  φ_μ ∇φ_ν)} dr. Hybrid path subtracts ½ × hfMix × K from F.

H₂O / STO-3G timings: LDA 75 ms / 8 iter, BVWN5 86 ms / 6 iter,
B3VWN5 97 ms / 7 iter, BLYP 90 ms / 8 iter, B3LYP5 105 ms / 8 iter.

**Tier 2 stage 2 — DFT/LDA.** Becke-partitioned molecular grid
(Becke M3 radial × Gauss-Chebyshev 2nd-kind × Gauss-Legendre ×
uniform-φ angular). Default 50r × 12θ × 24φ per atom integrates
ρ to 10⁻⁵–10⁻⁷ e. DFT/STO-3G LDA matches PySCF SVWN5 within ~5 mHa.
Modules: `src/chemistry/dft/{grid,density,functional,rks-scf}.ts`.

**Tier 2 stage 1 — geometry optimization.** `optimizeGeometry(atoms,
opts)` minimizes E_HF over atomic positions with central-FD
gradients + L-BFGS line search. Validated on H₂ / H₂O / BeH₂
STO-3G to sub-mÅ + sub-degree agreement with PySCF references
(R_OH = 0.9894 Å vs 0.9893; ∠HOH = 100.02° vs 100.04; R_BeH =
1.291 Å). FD gradients keep it basis-/level-agnostic; analytical
swap is a transparent follow-up.

**Tier 1 bundle.** Six chemistry-track quick wins shipped earlier:
- **DIIS** SCF accelerator — H₂O cc-pVDZ HF: 101 → 14 iter (7.2×
  speedup), bit-identical energy.
- **Frozen-core** option on MP2 / CCSD / CCSD(T) (zeroes T1, T2 in core
  blocks every iter; canonical 1s-frozen for first-row chemistry).
- **Spherical-harmonic d-shell** basis (`{ spherical: true }` opt on
  `computeMolecularIntegrals`). cc-pVDZ HF/H₂O matches PySCF to **35
  µHa** vs 340 µHa Cartesian — kills the documented Cartesian-d slack.
- **f/g/h orbital integrals**: rewrote `boysAll` with per-n Taylor
  inside the recurrence-stability threshold + closed-form-anchored
  upward outside. Max relative error at n=12 dropped from 1.5e-2 to
  8e-10. Unblocks cc-pVTZ (and beyond) basis sets.
- **aug-cc-pVDZ** diffuse functions (H + O wired). HF/H₂O matches
  PySCF to 50 µHa; 14 mHa lower than cc-pVDZ as expected.
- **Schwarz integral screening** in the AO ERI build (Q[μ,ν] =
  √⟨μν|μν⟩, skip pairs with Q_μν · Q_λσ < 1e-10).

**Previously:** Phase E stage 5 — cc-pVDZ CCSD(T) on H₂O in 106 s
wall-clock in a browser tab. HF / MP2 / CCSD / CCSD(T) all ship; CH₄
full-STO-3G FCI works via sparse-CSR Hsec (Phase C v5).

**Headline numbers:**
- L1 statevector: F ≥ 0.999999 vs CPU; 4-experiment ladder (E1–E4) green.
- L2 MPS / DMRG: TFIM & Heisenberg N=128 in browser, χ=32, validated to
  Pfeuty/Bethe limits at 1/N. ITensor cross-checked at N=8 to f64.
- L3 kernel fusion: **4.18× headline** (Tier C, 8×8 cascade); Tier D plateau
  is the documented honest negative.
- L6 chemistry: HF (≤ 0.05 mHa vs PySCF, ≤ 50 µHa with spherical d) →
  MP2 → FCI (CH₄ to 0.76 mHa) → CCSD (≥ 99% capture) → **CCSD(T)** (≤
  0.25 mHa vs FCI). aug-cc-pVDZ now wired alongside cc-pVDZ.

**Test surface:** `npm run test` → **407/407** (was 401) + 1 opt-in
(cc-pVDZ CCSD(T), gated on `PHASE_E5_CCPVDZ=1`). `npx tsc --noEmit`
clean. `npm run lint` clean (2 pre-existing unused-disable warnings).
`npx playwright test` → **11/11 specs**, all 4 levels e2e.

**Live:** https://webgpu-q.vercel.app — landing, `/viz.html` (4D
hyperscope), `/experiments/` (E1–E16+ dashboard). **Standing preference:
do NOT auto-deploy** — deploy only when the user explicitly asks.

**For per-phase detail** (Phase A → E5, Tier 1, B v0/v1, C v0–v5, D,
E1–E5, viz extensions, public-repo polish, hardened-SVD fix, Tier B/C/D
fusion): read `git log` — every phase shipped its own commit with full
benchmarks in the message body. Don't replicate that history here.

**Next up (per the roadmap above):** the natural follow-up to
CIS is **TDA-DFT / TDDFT** — same machinery as CIS but with KS
orbitals + the XC kernel f_xc, and (for full TDDFT) the (A, B)
2×2 block instead of just A. Cleanly stages: TDA-DFT first
(½ session, reuses CIS A matrix with hybrid hfMix scaling on the
exchange piece + DFT XC kernel on the diagonal-perturbation
piece), then full TDDFT (½ session more). After that, the
bigger lever is still **WebGPU port of the (T) kernel** for
cc-pVTZ CCSD(T) (~3 sessions) or **EOM-CCSD** for correlated
excited states (~1–2 sessions). After Tier 2 the project
becomes a "real undergrad chemistry tool in a browser tab."

---

## Research-grade discipline (non-negotiable)

These come from `RESEARCH.md`. Every experiment enforces them.

### Reproducibility

- No `Math.random()` in any experiment path. Every random draw uses a named
  seed from `experiments/lib/seeds.ts` via `mulberry32(seed)`.
- Every JSON artifact records: git SHA (when available), `navigator.userAgent`,
  `adapter.info`, WebGPU limits, UTC ISO8601 timestamp, and echoes back
  `protocol`, `hypothesis`, `passBar`, `seed`, `warmup`, `trials`. See
  `experiments/lib/env.ts → captureEnv(device, adapter)`.
- Artifact shape is locked: `{ meta, env, rows, status, diagnosis }`. Do not
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
  truncation + accumulated Jacobi error, ~9 digits is realistic at χ = 64).
- Secondary: TVD, L1, L2, max|Δp|, ‖ψ_ref‖², ‖ψ_test‖² — always reported.

### Honest negative results

- If an experiment fails its pass bar, still commit the JSON with
  `"status": "fail"` and a `"diagnosis"` string naming the first failing
  cell and the smoking gun. **Failures are the evidence.** No silent
  rerunning until it passes.
- Example (MPS canonical-form bug, 2026-04-22): brick-wall F = 0.25 at depth
  2. Diagnosis: "non-monotonic two-site gate order breaks mixed-canonical
  invariant, local Frobenius norm ≠ global norm, renormalization distorts."
  Fix: `_canonicalizeBond(q)` before every `applyTwoSite`.

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
                     # Each level is also reachable via window.__webgpuq.runLevelN()
                     # in devtools at /experiments/.
npm run test:e2e:headed   # Same, but with a visible browser window.
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

tests/
  gates.test.ts          # Bell, GHZ, XX=I, HH=I, T⁴=Z, …
  fidelity.test.ts       # stateMetrics unit tests
  stats.test.ts          # median / percentile / IQR
  linalg.test.ts         # SVD round-trip, orthonormality, diagonal
  mps.test.ts            # Bell / GHZ / brick-wall / canonical / truncation

experiments/
  index.html             # Research dashboard (run buttons, result tables)
  runner.ts              # Dashboard entry point — wires each level's run-all
  lib/
    seeds.ts             # Named deterministic seeds (no Math.random)
    runner.ts            # timedRun harness + Artifact / ArtifactMeta schema
    env.ts               # captureEnv(device, adapter) → EnvBlock
    fidelity.ts          # stateMetrics, FIDELITY_PASS_BAR
    stats.ts             # stats() — median, p10/p90/p99, std, IQR
  level-1-statevector/
    protocol.md
    E1-gate-fidelity.ts
    E2-bandwidth-roofline.ts
    E3-scaling-law.ts
    E4-dispatch-overhead.ts
    run-all.ts           # runLevel1() + wireRunAllButton()
  level-2-mps/
    protocol.md
    E5-mps-correctness.ts
    E6-qubit-ceiling.ts
    E7-chi-scaling.ts
    E18-tfim-pfeuty.ts
    E19-heisenberg-bethe.ts
    run-all.ts
  level-3-fusion/      # E8–E13 shipped (Tiers A/B/C/D fusion)
  level-6-chemistry/   # E16, E20–E31 shipped (H₂ → CCSD(T)/cc-pVDZ)
  results/                 # JSON artifacts, organized YYYY-MM-DD/level-N/
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

---

## License

MIT (simulation). Research protocol and experiment artifacts: MIT.
