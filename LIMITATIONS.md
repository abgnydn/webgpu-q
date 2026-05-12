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
| Davidson / Krylov eigensolver | OK for n_occ·n_virt ≤ ~500; dense above | Tier 3 |
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
- **EOM-CCSD ≡ FCI at 10⁻⁵ Ha** is **algorithmic precision** on H₂
  STO-3G (the only system small enough to brute-force a reference).
  EOM-CCSD literature accuracy vs FCI on real singlet excitations is
  **0.1–0.2 eV (~3.7–7.4 mHa)**, dominated by missing R₃ / R₄
  excitations in the operator. That's a method limitation, not ours.
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
