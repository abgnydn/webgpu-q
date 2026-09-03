# Benchmarks

Standardized benchmark sets are the language of computational
chemistry papers. This document tracks which sets we **have run**,
which are **queued**, and the priority order.

Every set, once run, ships as a research-grade experiment with:
named seed, warmup, 20 trials, full env capture, fidelity-based pass
bar, JSON artifact in `experiments/results/`.

Updated 2026-05.

---

## Current scoreboard

| set | category | size | status | notes |
|---|---|---:|---|---|
| H₂ STO-3G analytical | FCI cross-check | 1 | ✅ shipped (E20–E33) | bedrock validation |
| LiH / BeH₂ / H₂O / CH₄ STO-3G | mini-molecule ladder | 4 | ✅ shipped | HF → MP2 → CCSD → (T) → EOM |
| H₂O cc-pVDZ | headline single point | 1 | ✅ shipped (E31, E32) | CCSD(T) GPU 13.8× median (28.4× p10) |
| TFIM N = 128 (browser) | many-body L2 | 1 | ✅ shipped (E18) | matches Pfeuty |
| Heisenberg N = 128 | many-body L2 | 1 | ✅ shipped (E19) | matches Bethe |
| **GMTKN55** (thermochem + kinetics + noncov) | DFT benchmark | 1505 | 🛣️ Tier 3 queued | the universal DFT bar |
| **Thiel-style EOM-CCSD cross-validation** (STO-3G subset) | EOM-CCSD vs PySCF | 5 small organics × 5 roots | ✅ shipped (E35) · **singlet-sector gap closed** | HF + CCSD match to 10⁻⁷ Ha. Original singlet gap came from hand-derived σ_2; closed 2026-05 by porting EE/IP/EA σ-equations from PySCF `eom_gccsd`, validated by LiH brute-force projection < 1e-10 Ha element-wise. |
| **Full Thiel / QUEST** (cc-pVTZ benchmark) | EOM / TDDFT | 28 / 472 | 🛣️ Tier 3 queued | blocked on cc-pVDZ → cc-pVTZ basis for C, N, F |
| **W4-11 / W4-17** (atomization) | high-accuracy thermo | 140 / 200 | 🛣️ Tier 3 queued | sub-kcal/mol thermochem |
| **S66 / S66x8** (noncovalent) | noncovalent | 66 / 528 | 🛣️ Tier 3 queued | dispersion + H-bond |
| **HEAT-345** (atomization) | gold-standard atomization | 31 | 🛣️ Tier 3 queued | sub-0.1 kcal/mol |
| **SIE4x4 / SIE11** (self-interaction) | DFT failure modes | 4×4 / 11 | 🛣️ Tier 3 queued | exposes SIE artifacts |
| **Schreiber** (vertical excitations) | TDDFT / EOM benchmark | 28 | 🛣️ Tier 3 queued | smaller than Thiel, faster |
| MP2-F12 / CCSD(T)-F12 vs CBS | basis-set convergence | various | 🛣️ Tier 3 queued | needs F12 implementation |
| Cross-vendor parity | GPU vendor matrix | n/a | 🛣️ Tier 3 queued | NVIDIA / AMD / Intel / Apple |
| **Wall-clock vs PySCF (CPU)** | head-to-head timing | 4 mol × 2 basis × 5 methods | ✅ shipped (E34) | see comparison artifact + honest summary below |
| **Wall-clock vs gpu4pyscf** | head-to-head GPU timing | same | 🛣️ Tier 3 next | needs gpu4pyscf install + PySCF script flag |

✅ shipped · 🛣️ Tier 3 · ⏳ Tier 4

---

## What each standardized set tests

### GMTKN55 (Goerigk, Hansen, Bauer et al., PCCP 2017)
1505 reactions across 55 subsets. The universal DFT benchmark — every
new functional gets a WTMAD-2 score on it. Current 2024-2025 leader is
ωB97M(2) at **WTMAD-2 = 2.19 kcal/mol**. We'd report:
- WTMAD-2 for each implemented functional (B3LYP5, BLYP, BVWN5, LSDA)
- Subset breakdown (TC, BH, NCI for thermochem, barrier heights, noncov)
- Cross-check against published reference numbers

### Thiel / QUEST excited states
Published vertical excitation energies for 28 organic molecules
(Thiel) / 472 transitions (QUEST). Reference values from FCI / CC3 /
NEVPT2-F12. Our EOM-CCSD numbers should land within ~0.2 eV per state.

### W4-11 (Karton et al.)
140 atomization energies with sub-0.1 kcal/mol reference accuracy via
W4 protocol (CCSDT(Q) extrapolated). Tests our CCSD(T) + basis-set
extrapolation. Likely needs cc-pVTZ minimum + extrapolation.

### S66 / S66x8 (Hobza)
66 noncovalent dimers (H-bond, dispersion, mixed), each at 8 distances.
S66x8 is the gold standard for noncovalent benchmarking. Tests
counterpoise + DFT-D / MP2 / CCSD(T).
**Blocker:** counterpoise (BSSE) correction shipped at `src/chemistry/counterpoise.ts`; now blocked on cc-pVTZ basis data.

### HEAT-345 (Tajti et al.)
31 small molecules with sub-0.1 kcal/mol atomization references.
Includes CCSDT(Q) + relativistic + DBOC corrections. We'd report
deltas at each correction level.

### Cross-vendor parity
Take 3 reference molecules (H₂O / BeH₂ / CH₄ STO-3G), run identical
input through identical code on:
- Apple M1 / M2 / M3
- NVIDIA RTX 30xx / 40xx / 50xx
- AMD Radeon
- Intel Arc / iGPU

Report HF, CCSD, CCSD(T) numbers + wall-clock. Pass = all match to
1×10⁻⁹ Ha; performance is informational.

### Wall-clock vs PySCF / gpu4pyscf  · 🟡 INFRASTRUCTURE SHIPPED (E34)

Reviewer's #1 question. **Infrastructure is now in place**:

- `experiments/level-6-chemistry/E34-wallclock-vs-pyscf.ts` runs the
  webgpu-q side (4 molecules × 2 basis × 5 methods) and emits a JSON
  artifact via the standard research-grade harness (named seed, env
  capture, full SCF / CCSD / (T) pipeline, identical thresholds).
- `scripts/run-pyscf-reference.py` runs the PySCF side with **matching
  JSON schema** so the two can be merged offline.
- `e2e/wallclock-vs-pyscf.spec.ts` runs E34 in headless WebGPU
  Chromium. Run it locally — no workflow names this spec, and hosted
  runners have no WebGPU adapter, so this row has no automated
  regression gate. (Some non-GPU swarm specs *do* run on PRs via
  `swarm-benches.yml`; this is not one of them.)

**To complete the comparison** (once PySCF env is available):

```bash
# webgpu-q side (local only — not run by CI):
npm run test:e2e -- wallclock-vs-pyscf

# PySCF side (run locally / on a server with Python):
pip install pyscf==2.13.0
python3 scripts/run-pyscf-reference.py --out experiments/results/<date>/level-6/E34-pyscf.json

# Optionally: gpu4pyscf side for GPU vs WebGPU
pip install gpu4pyscf-cuda12x
# (modify run-pyscf-reference.py to use df.RHF(mol).to_gpu(), etc.)
```

Honest expectation: PySCF is faster on CPU due to BLAS, gpu4pyscf
is faster on big systems due to cuBLAS; we win on small systems
because we have zero startup / JIT cost, and on "no-install" UX
plus the cc-pVDZ CCSD(T) WGSL kernel.

### Headline result (2026-05-12, Apple M2 Pro, PySCF 2.13.0 CPU)

[Full comparison →](./experiments/results/2026-05-12/level-6/E34-comparison.md)

**Energy agreement** — 19 directly comparable cells:
- max **\|ΔE\| = 1.00×10⁻⁴ Ha** (H₂O STO-3G CCSD(T), 100 µHa — below
  chemical accuracy of 1.594 mHa; the bulk of this is in the (T)
  correction itself, possibly frozen-core defaults differ)
- mean **\|ΔE\| = 8.13×10⁻⁶ Ha** across all cells

**Wall-clock** — 19 cells, **webgpu-q faster on 11 / 19**:

| we win | we lose |
|---|---|
| HF on small systems (105× on H₂ STO-3G, 12× on H₂O STO-3G, 1.09× on H₂O cc-pVDZ) — no Python startup cost | MP2 / CCSD on cc-pVDZ — PySCF uses NumPy/BLAS, we use TS loops (e.g. H₂O cc-pVDZ CCSD: **41.6 s for us vs 87 ms for PySCF = 480× slower**) |
| Small CCSD (e.g. LiH STO-3G: 0.8 ms vs PySCF 32 ms = 40× faster) | CPU CCSD(T) on medium systems (BeH₂ STO-3G: 42 ms vs PySCF 2 ms = 20× slower) |
| **CCSD(T)-GPU at cc-pVDZ — 4.2 s on H₂O**. PySCF without `gpu4pyscf` can't do this at all. | Anything bandwidth-bound where BLAS vectorization dominates |

**The honest story**: webgpu-q is **not always faster** — and the
README updated to reflect this. We win on (a) the no-install /
no-startup edge, (b) HF up through medium systems, (c) GPU (T) at
cc-pVDZ. PySCF wins on CPU-BLAS-bound MP2 / CCSD at production
basis. gpu4pyscf comparison is the next step.

### Known method-level residuals to investigate (Tier 3)

- BeH₂ STO-3G CCSD(T): |ΔE| = 3.5×10⁻⁵ Ha (35 µHa)
- H₂O STO-3G CCSD(T):  |ΔE| = 1.0×10⁻⁴ Ha (100 µHa)
- Likely candidates: frozen-core defaults, (T) prefactor convention,
  spin-orbital ordering. All below chemical accuracy but worth
  closing the loop on.

---

## Priority order (recommended)

1. **Thiel/QUEST excited states** — directly tests our EOM-CCSD which
   is novel for this codebase. ~28 systems, 1-2 sessions.
2. **Wall-clock vs PySCF** — reviewer's first question. ~1 session.
3. **GMTKN55 subsets** (start with thermochem-only) — universal DFT bar.
   Full set is 1505 reactions; even a 100-reaction subset has reviewer
   weight. ~2-3 sessions.
4. **Cross-vendor parity** — establishes WebGPU portability claim.
   Depends on getting access to non-Apple hardware. ~1 session on
   each platform.
5. **S66 noncovalent** — needs counterpoise first. ~1 session for
   counterpoise wiring + 1 session for S66 run.
6. **W4-11** — needs cc-pVTZ + extrapolation. ~2-3 sessions.

---

## How to add a new benchmark

1. Create `experiments/level-6-chemistry/E<NN>-<setname>.ts` following
   the E32/E33 template (env capture, named seed, warmup+trials,
   pass bar based on RMSE vs reference values).
2. Add reference numbers to a separate `<setname>-reference.ts`
   module — never inline.
3. Wire into `experiments/runner.ts` as `window.__webgpuq.runE<NN>`.
4. Add Playwright e2e at `e2e/<setname>.spec.ts` that runs in
   headless WebGPU Chromium.
5. Commit the **first** artifact (pass or fail) as the baseline.
   If it fails, commit it anyway with `status: "fail"` and a
   diagnosis.
6. Update this file with the result.

---

## What we will NOT chase

- **Random GitHub molecule lists** — not reproducible reference data.
- **Cherry-picked vs PySCF** — only side-by-side on identical inputs.
- **Single-shot timings** as primary results — must go through
  `timedRun` with 5 warmup + 20 trials.
- **Fictional speedups** — no apples-to-oranges (e.g., GPU f32 vs CPU
  f64) unless explicitly flagged.
