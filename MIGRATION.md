# Migration: hand-derived → ported from reference implementations

**Premise**: webgpu-q's differentiator is the browser/WebGPU layer, not
the chemistry methods. Methods are textbook and have peer-reviewed
reference implementations (PySCF, Psi4, ORCA, libxc, ITensor). For
those modules, **a careful port from the reference is more reliable
and faster than re-deriving from papers**. The May 2026 E35
EOM-CCSD finding (multi-eV gap vs PySCF on closed-shell singlets)
made this explicit: we were re-deriving Stanton-Bartlett σ-equations
and our hand-derived version has bugs that take weeks to find. PySCF's
implementation has been used by hundreds of papers and is correct.

**Going forward**: hand-write only what's genuinely novel; port
everything else with proper attribution.

---

## What's novel (we ARE the reference) — keep hand-written

| module | why we own it |
|---|---|
| `src/shaders/*.wgsl` | WGSL kernels are unique to this stack |
| WebGPU dispatch + sync glue (`quantum.ts`, `ccsd-t-gpu.ts`) | browser-specific |
| `src/mps.ts` browser canonical-form bookkeeping | no reference targets browser memory |
| Kernel fusion experiments (`level-3-fusion/`) | research-novel |
| `experiments/lib/*` research-grade harness | research-novel |
| `src/cpu-reference.ts` (statevector CPU ground truth) | trivially correct |

## What has a reference — should be ported

Status legend: 🟢 ported · 🟡 ported in flight · 🔴 hand-derived (current) · ⚫ partial

| module | reference | license | status | notes |
|---|---|---|---|---|
| `eom-ccsd.ts` σ_2 | PySCF `pyscf/cc/eom_gccsd.py` | Apache 2.0 | 🟢 ported | Wang-Tu-Wang Eq. (10) on PySCF intermediates. E35 bug closed; LiH brute-force diff < 1e-10 Ha element-wise |
| `eom-ccsd.ts` σ_1 | PySCF same | Apache 2.0 | 🟢 ported | Eq. (9); validated with σ_2 by the same LiH verifier |
| `ccsd.ts` residual + amplitudes | PySCF `pyscf/cc/ccsd.py` | Apache 2.0 | 🔴 | passing tests but worth re-validating |
| `ccsd-t.ts` (T) correction | PySCF `pyscf/cc/ccsd_t.py` | Apache 2.0 | 🔴 | E34 surfaced ≤ 100 µHa discrepancy |
| `uccsd.ts` open-shell | PySCF `pyscf/cc/uccsd.py` | Apache 2.0 | 🔴 | shares core with `ccsd.ts` |
| `hf-scf.ts` Fock build + DIIS | PySCF `pyscf/scf/hf.py` | Apache 2.0 | 🔴 | E34 says we match to 7×10⁻⁷ Ha |
| `mp2.ts` + DF-MP2 | PySCF `pyscf/mp/mp2.py` | Apache 2.0 | 🔴 | E34 says we match |
| `df.ts` Cholesky DF | PySCF `pyscf/df/incore.py` | Apache 2.0 | 🔴 | machine-precision agreement |
| `cis-tda.ts` / `tddft.ts` | PySCF `pyscf/tdscf/*.py` | Apache 2.0 | 🔴 | spin-pol functionals hand-derived |
| `dft.ts` functionals | libxc | MPL 2.0 / LGPL | 🔴 | currently hand-coded LDA/B88/LYP etc. |
| DFT angular grids | Lebedev tables | public domain | ⚫ | tables hand-typed |
| `hf-gradient.ts` HF analytical ∇ | PySCF `pyscf/grad/rhf.py` | Apache 2.0 | 🔴 | Pulay 1969 hand-derived |
| `dft-gradient.ts` DFT analytical ∇ | PySCF `pyscf/grad/rks.py` | Apache 2.0 | 🔴 | Pulay 1969 hand-derived |
| `vibrations.ts` Hessian + thermo | PySCF `pyscf/hessian/*.py` | Apache 2.0 | 🔴 | H₂O matches NIST entropy |
| `eom-ccsd-bruteforce-lih.test.ts` | none (we built it) | MIT | 🟢 | regression verifier for the port |
| `properties.ts` dipole, α, β | PySCF `pyscf/prop/*.py` | Apache 2.0 | 🔴 | finite-field hand-derived |
| Basis-set primitives | EMSL Basis Set Exchange | CC-BY 4.0 | ⚫ | 5 atoms wired; full JSON port unblocks Thiel |
| FCI / CASCI small | PySCF `pyscf/fci/*.py` | Apache 2.0 | 🔴 | hand-derived |
| Cholesky / Jacobi SVD | (textbook) | public domain | 🟢 | textbook algorithm, OK hand-derived |
| Hessenberg + Wilkinson QR | LAPACK / textbook | public domain | 🟢 | textbook algorithm |

---

## Priority order

Rank by *closes an open bug* > *unblocks a benchmark* > *codebase health*:

1. ~~**`eom-ccsd.ts` σ_2** ← PySCF `eom_gccsd.py`~~ — **DONE.** All three EOM
   variants (EE / IP / EA) are now patch-free PySCF ports on shared dressed
   intermediates, each with a multi-electron LiH brute-force verifier (T̂² ≠ 0):
   EE and IP agree with the explicit H̄ projection to < 1e-10 Ha element-wise,
   EA to ~5e-13 Ha. EA was the only variant that ever carried an empirical
   patch, removed 2026-06-16. This entry sat at "🔴 open bug, biggest priority"
   for months after the work landed — MIGRATION.md is what README points
   researchers at for port provenance, so a stale 🔴 here understates the repo
   to exactly the audience that checks.

2. **Basis-set primitives** ← EMSL JSON dump. Currently LiH, BeH₂, H₂O,
   NH₃, CH₄ work at STO-3G; cc-pVDZ is wired for H–Ar, all 18 elements
   (this line previously said "blocked for everything except H, O", which went
   stale once the basis tables were completed — and that stale belief was also
   hardcoded into E34's element gate, silently dropping 10 artifact rows).
   Port unblocks the **full Thiel/QUEST 28-molecule benchmark** plus
   GMTKN55 thermochemistry. Pure data-table port, no math.

3. **DFT functionals** ← libxc. Currently hand-coded LDA / BVWN5 / BLYP /
   B3LYP5. Porting libxc formulas (or interfacing to libxc.js if it
   exists) unblocks modern functionals: ωB97X-V, M06-2X, SCAN-D3,
   ωB97M(2), revDSD-PBEP86-D4.

4. **`ccsd-t.ts` (T) correction** ← PySCF `ccsd_t.py`. Closes the
   E34-surfaced 35-100 µHa BeH₂/H₂O residuals (likely frozen-core or
   prefactor convention).

5. **`hf-scf.ts`, `mp2.ts`, `ccsd.ts`** — currently passing tests, lower
   priority. Port if/when a comparison surfaces a residual.

---

## Attribution policy

For every ported module:

1. **File header**:

```typescript
// Ported from PySCF (https://github.com/pyscf/pyscf), Apache 2.0 license.
// Source: pyscf/cc/eom_rccsd.py at commit <SHA>
// Original authors: <PySCF developers, see PySCF/AUTHORS>
// Adaptations for webgpu-q (spin-orbital antisym basis, typed-array layout):
//   - <list of substantive changes>
// See LICENSE-PYSCF at repo root for the Apache 2.0 notice.
```

2. **Repo-level files**:
   - `LICENSE-PYSCF` at root (Apache 2.0 verbatim from PySCF/LICENSE)
   - `LICENSE` (MIT for original webgpu-q code)
   - Cite PySCF in `CITATION.cff` ✓ already done
   - Mention port in `README.md`'s "References" or "Built on" section

3. **CHANGELOG**: each port commit names the source file + commit SHA
   so reviewers can diff.

4. **No license confusion**: MIT and Apache 2.0 are compatible.
   Ported code retains Apache 2.0 obligation for *the ported portion*
   (provide notice + state changes). Rest of repo stays MIT.

---

## How to port (the recipe)

For each module:

1. Identify the **reference function(s)** in PySCF / libxc / etc.
2. Identify the **regression target** — what test or benchmark verifies
   correctness?
   - EOM-CCSD: `tests/chemistry/eom-ccsd-bruteforce-lih.test.ts` and `E35`
   - HF: cross-check against PySCF energy to 10⁻⁹ Ha
   - DFT functional: numerical match to libxc value tables
3. **Translate**, not retype. Read the reference function, understand
   the math, translate to our index conventions (TS strict typed arrays,
   spin-orbital P = 2p + σ for chemistry, etc.). Inline einsums into
   explicit loops.
4. **Add the attribution header** (see above).
5. **Replace** the hand-derived module behind a feature flag or in a
   parallel file (`*-ported.ts`), so we can A/B test before switching.
6. **Run the verifier**. The success criterion is bit-for-bit (or
   numerical-tolerance) agreement with the reference.
7. **Switch**: delete the hand-derived module, rename the ported one,
   update imports.
8. **Update this MIGRATION.md** to flip the row from 🔴 to 🟢.

---

## What porting changes about the project narrative

- **Headline**: "browser-native quantum chemistry via WebGPU" — unchanged.
- **Architecture pitch**: "hand-built WebGPU layer + carefully-ported
  chemistry methods from PySCF/libxc, all running in a browser tab" —
  more credible than "re-derived everything from textbooks".
- **JOSS paper** strengthens: methods are PySCF-validated, novelty is
  in the browser/GPU stack, not in re-deriving theory.
- **Reproducibility**: PySCF + our brute-force diagnostics give two
  independent grounds of truth.

---

## Concrete port-verification findings (2026-05-13 session)

Built `tests/chemistry/eom-ccsd-imds-vs-pyscf.test.ts` as the
per-intermediate verifier. Loads `experiments/results/2026-05-13/level-6/E36-pyscf-imds-lih.json`
and diffs our spin-orbital intermediates against PySCF's spatial
ones on same-spin (closed-shell RHF) pairs.

Results on LiH STO-3G:

| intermediate | max \|Δ\| vs PySCF | status |
|---|---:|---|
| F_me ↔ F_ov | 3.3×10⁻⁹ Ha | ✅ exact (within SCF tol) |
| F_ae ↔ F_vv + ε_a δ_ab | 8.7×10⁻⁵ Ha | 🟡 87 µHa — missing F_ov · T1 dressing |
| F_mi ↔ F_oo + ε_i δ_mi | 8.7×10⁻⁵ Ha | 🟡 87 µHa — same root cause |

The identical 87 µHa magnitude on F_ae and F_mi confirms a single
missing term: PySCF's make_imds tail adds 0.5 · F_ov · T1 to both
F_oo and F_vv. Our `makeF_ae` / `makeF_mi` don't include this
dressing because they're shared with the CCSD residual iteration.
Adding it for EOM requires either splitting the helpers into
`*_ccsd` / `*_eom` variants or passing F_me as an argument.

This finding is documented in the diagnostic test as a tolerance
loose at 1e-3 — tight gate (1e-8) is the goal for a successful port.
W intermediates (woOoO, woVoO, etc.) are next to verify.

## Status as of 2026-05-13

Migration framework shipped (LICENSE-PYSCF, this doc, attribution
policy). First port (EOM-CCSD σ_2) scheduled for next dedicated
session. Diagnostic test (`eom-ccsd-bruteforce-lih.test.ts`) ready
as the verifier. Repo philosophy now explicit: port where references
exist, hand-write only the novel WebGPU/WGSL/browser layer.
