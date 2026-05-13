# Limitations

What webgpu-q **cannot** do, what it has **not yet been tested on**,
and what is **known broken** — all in one place. Updated 2026-05.

Honesty is more credible than completeness. We list every limitation a
reviewer or chemist would discover anyway.

---

## 1. System size ceilings

### Tested and confirmed working

| basis | system | NSO | wall time | notes |
|---|---|---|---|---|
| STO-3G | H₂, LiH, BeH₂, H₂O, CH₄ | ≤ 18 | sub-second | full pipeline incl. CCSD(T) + EOM |
| 6-31G* | H₂O, BeH₂ | ≤ 28 | seconds | spot-checked |
| cc-pVDZ | H₂O | 48 | 5.05 s (GPU CCSD(T)) | headline case |
| aug-cc-pVDZ | H₂O | 64 | minutes | diffuse functions wired |

### Tested and confirmed working but unbenchmarked

- cc-pVDZ CCSD(T) on CH₄, BeH₂, LiH — methods work; not in CI on every commit.
- STO-3G EOM-CCSD on systems up to ~12 occupied SOs — algorithm scales as
  O(N⁶) and gets slow above that.

### Not tested

- **cc-pVTZ on anything** — would push past ~ 100 SOs; no measurement.
- **Benzene / pyridine / formaldehyde** at any basis above STO-3G.
- **Transition metals** of any kind (no ECP / pseudopotential support yet).
- **Heavy elements (Z > 18)** — no scalar-relativistic correction.

### Hard ceilings

- **WebGPU buffer max ~4 GB per buffer** on Chrome / desktop, less on
  Firefox. A naive (i,j,k,a,b,c)-laid-out f32 partial-sum buffer for
  CCSD(T) at ~100 SOs is already 100⁶ · 4 = a lot — we'd need to tile,
  which is unimplemented.
- **Single-tab heap ceiling** of ~4 GB on Chrome desktop. JavaScript-side
  arrays for ERIs at large basis hit this before WebGPU does.

### Basis-set atom coverage

| basis | atoms wired |
|---|---|
| STO-3G | H, Li, Be, C, N, O |
| 6-31G* | H, C, N, O (spot-checked) |
| **cc-pVDZ** | **H, O only** (Phase E v2 limit; everything else throws) |
| **aug-cc-pVDZ** | **H, O only** |

Tier 3 follow-up: port the EMSL Basis Set Exchange tables for Li, Be,
C, N, F into cc-pVDZ / aug-cc-pVDZ so LiH / BeH₂ / CH₄ / formaldehyde
become first-class cc-pVDZ targets.

---

## 2. Browser / GPU vendor matrix

| browser | GPU | status |
|---|---|---|
| Chromium / Chrome | Apple M2 Pro | **primary test target** · everything works |
| Chromium / Chrome | NVIDIA discrete | untested |
| Chromium / Chrome | AMD discrete | untested |
| Chromium / Chrome | Intel iGPU | untested |
| Firefox (Nightly WebGPU) | M2 Pro | untested |
| Safari (Tech Preview WebGPU) | M2 Pro | untested |
| Edge | any | likely OK (Chromium engine) |
| Mobile (any) | any | unlikely to work — buffer limits + power |

We **assume** WebGPU 1.0 conformance and the published `requiredLimits`.
Adapters that don't report the expected `maxStorageBufferBindingSize`
will silently truncate large dispatches.

---

## 3. Known SCF / CC failure modes

- **HF doesn't converge** — we return `converged: false` and stop. No
  level-shift, no damping, no quadratically-convergent second-order SCF
  fallback. Stretched bonds and near-degenerate HOMO-LUMO gaps fail.
- **CCSD T-amplitude divergence** — for systems with multi-reference
  character (broken bonds, biradicals), the spin-orbital CCSD residual
  doesn't converge. We catch and report; no DIIS for amplitudes (only
  for the Fock matrix in SCF).
- **EOM-CCSD with R₂ dominant states** — algorithm runs and returns
  energies, but the literature accuracy for doubly-excited states is
  ~1 eV vs FCI (vs ~0.1–0.2 eV for singly-excited). Don't trust
  shake-up / two-electron states from this implementation.
- **DFT grid quadrature instability** — Becke-partitioned grid weights
  derivatives have a ~10⁻³ Ha/Bohr translational-invariance residual.
  Documented in CLAUDE.md. Don't trust DFT gradients beyond that scale.
- **TDA-DFT / DFT-gradient with spherical-d** — refuses with a clear
  error today; proper fix is Cartesian → spherical transform on grid
  values. Documented.

---

## 4. Missing features that researchers expect

| missing | impact | roadmap |
|---|---|---|
| Counterpoise correction (BSSE) | Can't quote noncovalent interaction energies | Tier 3 — additive on top of HF/CCSD |
| Frozen-core in EOM / CCSD(T) | We freeze core for HF but **don't verify** propagation through (T) and EOM | Tier 3 audit needed |
| ⟨S²⟩ post-CC spin diagnostics | UHF reports ⟨S²⟩; UCCSD doesn't | Tier 3 — trivial extension |
| Level-shift / damping | Stretched-bond HF fails | Tier 3 — standard SCF technique |
| Quadratically-convergent SCF | Hard cases bail out | Tier 3 |
| Molden / HDF5 / Cube output | No interop with Jmol / Avogadro / IboView | Tier 3 — orbital plot would unblock visualization |
| ccData / QC-Schema compatibility | Output not consumable by external tooling | Tier 3 |
| FAIR / Zenodo DOIs per release | Citations point at GitHub tag, not DOI | Tier 3 — set up CI workflow |
| Aux-basis density fitting | We have CD-DF but not JKFIT / RIFIT integral path | Tier 3 — needs 3-index ERI routine |
| Davidson / Krylov eigensolver | OK for n_occ·n_virt ≤ ~500; dense above. Measured 2026-05: EOM-CCSD on CH₂O / HCN at STO-3G (dim 3488 / 2660) takes 15-30 min per molecule on M2 Pro because the Hessenberg + Wilkinson QR is dense. Davidson would make these ~minutes. | Tier 3 · high impact |
| Multi-node parallel | One tab, one GPU | Tier 4 — substrate is Phase D WebRTC |
| Periodic boundary conditions | No solids, no surfaces | Tier 4 |
| Spin-orbit coupling / X2C / DKH | No heavy elements | Tier 4 |
| QM/MM | No biomolecules | Tier 4 |
| Anharmonic VPT2 | Harmonic only | Tier 4 |

---

## 5. Honest precision disclosures (carried from CLAUDE.md)

- **CCSD(T) GPU 39.3×** is a **single-run measurement** on M2 Pro. Not
  through the warmup+20-trials harness. The correctness (|Δ| = 2.4×10⁻¹⁰
  Ha) is reproducible; the specific 39.3× number is ±20% on different
  hardware and ±10% run-to-run.
- **EOM-CCSD ≡ FCI at 10⁻⁵ Ha** is **algorithmic precision on H₂
  STO-3G only**, where T̂² = 0 makes EOM-CCSD = FCI by construction
  (2-electron limit). E35 cross-validation against PySCF EOM-CCSD
  on LiH / BeH₂ / H₂O / NH₃ / CH₄ STO-3G is more nuanced than the
  first cut suggested. The gap is **not uniform across spin sectors**:
  - **Triplet excitations agree well**: LiH lowest triplet matches
    PySCF to **7 meV**, BeH₂ degenerate triplet matches to 1.3 meV.
    H₂O / NH₃ / CH₄ triplets show ~0.5–1.0 eV gap (worsening with
    system size).
  - **Singlet excitations show a consistent ~2–3 eV gap** across
    LiH, BeH₂, H₂O, NH₃, CH₄.
  - **HF + CCSD energies agree to 10⁻⁷ Ha** throughout.
  This pattern (triplets mostly correct, singlets systematically off)
  is now **isolated to one missing term in σ_1**. The LiH brute-force
  diagnostic (`tests/chemistry/eom-ccsd-bruteforce-lih.test.ts`)
  builds H̄ = e⁻ᵀ̂ H eᵀ̂ explicitly in the 64-state 4-electron Fock
  space, projects onto the (R_1 + antisym R_2) basis, and diagonalizes.
  CCSD energy matches FCI exactly on LiH STO-3G (CCSD = FCI for this
  system). The exact M_exact projection matches PySCF EOM-CCSD on
  all triplets AND singlets — so PySCF is correct.

  Our σ-equation matches M_exact on triplets to 7 meV but disagrees
  on singlets by ~2.57 eV in opposite directions. After extending
  the LiH brute-force test to diff the FULL 14×14 M_mine vs M_exact
  element-by-element (not just R_1×R_1), the bug structure is now:

  | block | max \|Δ\| | nature |
  |-------|----------:|--------|
  | R_1 × R_1 | 0.53 eV  | diagonal patch artifact (cosmetic) |
  | R_1 × R_2 | 5.84 eV  | cross-coupling — major |
  | R_2 × R_1 | 4.04 eV  | cross-coupling — major |
  | R_2 × R_2 | 7.26 eV  | self-coupling — dominant bug |

  The R_1×R_1 off-diagonal couplings ARE correct (initial hypothesis
  about missing ⟨iα jβ ‖ aα bβ⟩·R_1 was wrong — that coupling is
  in W_mbej and the diff confirms it). The singlet eigenvalue gap
  flows from R_2 contamination via R_1 ↔ R_2 mixing, not from R_1
  itself. The dominant offending entry is
  [R₂[0<3,0<1], R₂[0<1,0<1]] = 7.26 eV — an R_2 ↔ R_2 coupling
  between two doubles sharing the (a=0, b=1) virtual pair but
  different occupied pairs. That kind of coupling flows through
  Σ_mn W̄_mnij R_2[m,n,a,b] in σ_2 — so our W_mnij contraction or
  the W̄_mnij intermediate itself is the next thing to audit.

  Scope: was a real σ_2 bug, partially closed via a sign correction.

  Tested-and-rejected hypotheses (2026-05-13):
  - "Stage 32c patches over-correct on multi-electron — revert them
    and see if singlets improve." Reverting made the LiH lowest
    triplet WORSE (7 meV → 540 meV gap) and did NOT shrink singlets.
    Patches restored.

  Stage 32i: diagnostic basis-ordering correction — the prior R_2 ×
  R_2 "off-diagonal 7.26 eV bug" was diagnostic permutation noise,
  not a real bug. After correction the R_2 × R_2 off-diagonals went
  to ~10⁻¹⁵ Ha.

  Stage 32k (the actual fix): the σ_1 ← R_2 W̄_amef term had a
  sign-flip. Code used `+½ ⟨ma||ef⟩` where Stanton-Bartlett 1993
  Eq 41 requires `+½ ⟨am||ef⟩` (= −½ ⟨ma||ef⟩ by antisymmetry).
  One-line fix:
    V(m, a+VO, e+VO, f+VO)  →  V(a+VO, m, e+VO, f+VO)
  RESULT: LiH singlet gap collapsed 2.57 eV → **0.27 eV** (10×
  better), within the literature EOM-CCSD ↔ FCI accuracy bar
  (0.1–0.2 eV per Stanton-Bartlett). Triplet 6.77 meV (essentially
  exact).

  Stages 32j, 32l: added T1·T1 + linear-T1 dressings on W̄_abej,
  W̄_mbij, W̄_mnie, W̄_amef per Crawford-Schaefer 2000. Each closed
  ~10–25% of the remaining gap on bigger systems.

  Current state:
  - LiH STO-3G: triplets exact (7 meV); singlets 0.27 eV (method
    precision limit reached).
  - BeH₂: triplets within 0.1 eV; singlets not separately tested.
  - H₂O / NH₃ / CH₄: singlets still ~0.5–1.9 eV off — more
    structural T-dressings missing (PySCF's woVoO has ~8 dressings;
    we have 3-4 of them).

  Closure path: the PySCF port (MIGRATION.md) lands all remaining
  T-dressings at once. The brute-force diagnostic
  (`tests/chemistry/eom-ccsd-bruteforce-lih.test.ts`) is the
  permanent verifier — every fix attempt makes the M_mine − M_exact
  diff shrink or doesn't.
  See `experiments/results/2026-05-13/level-6/E35-comparison.md`.
- **IP-EOM-CCSD R₂ satellites** have a known **~2 Ha (~60 eV)
  over-count** on H₂ STO-3G. Documented in `ip-eom-ccsd.ts`. Affects
  R₂-dominated Auger / shake-up states only; physical lowest IPs are
  validated exact against brute-force.
- **Stage 30 eigenvectors** for degenerate eigenvalues set the
  zero-denominator entry to 0, picking one representative from the
  degenerate eigenspace. The 3 returned vectors are individually
  unit-normalized but NOT guaranteed mutually orthogonal. Callers that
  need an orthonormal degenerate basis must Gram-Schmidt explicitly.
- **DF-HF / DF-MP2 machine-precision** is validated on **STO-3G**
  (H₂O, BeH₂). cc-pVDZ DF is expected to be equally clean by
  construction (same Cholesky algorithm) but not separately
  benchmarked.

---

## 6. Software engineering

- **No CI vendor-matrix** — GitHub Actions runs Linux Chromium only.
  Apple Silicon, Windows, AMD, NVIDIA, Intel — manual spot checks.
- **No mutation testing** — `npm run test` is 401 cases; we don't
  measure whether they'd catch real mutations.
- **No formal coverage metric** — we know which files have tests; we
  don't track lines/branches.
- **No fuzz testing** of the integral / SCF / CC paths.
- **No memory safety audit** of WGSL kernels beyond manual review.
  WebGPU bounds-checks shader memory accesses by spec, so the blast
  radius is bounded, but undefined behavior is still possible.

---

## 7. Not in scope (now or ever)

- **Force fields (MM)** — this is an electronic-structure engine.
- **Molecular dynamics propagators** (Verlet, RESPA, etc.) — out of
  scope unless used by BOMD/NAMD, which is Tier 4.
- **Docking / drug discovery** — wrong layer of abstraction.
- **Crystal structure prediction** — needs periodic + force fields.
- **4-component Dirac with QED corrections** — specialist territory
  (DIRAC, BERTHA, BAGEL).

---

## When in doubt

Run a smaller analog through PySCF or ORCA first, then check ours
matches before relying on the result. Issue a GitHub issue with the
input + both outputs and we'll add a regression test.
