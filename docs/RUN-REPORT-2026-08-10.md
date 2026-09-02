# Run report — periodic-table expansion, 2026-08-10

Branch: `feat/periodic-table-expansion`. Nothing pushed, nothing deployed.

---

## 1. What could NOT be verified

Listed first deliberately — this is the most useful section.

### 1.1 The Level-A bar in the plan is unreachable, and was never run

`docs/RUN-PLAN-24H-ELEMENTS.md` Gate 0.3 specified a two-level scheme:
Level A (our primitives fed to PySCF via `gto.basis.parse`, bar 1e-9 Ha)
to isolate engine bugs, Level B (our tables vs PySCF's) to isolate data
errors.

**Only Level B was built.** The committed harness
(`tests/chemistry/elements/reference-agreement.test.ts`) compares our
engine against PySCF's own basis at 0.1 mHa. That catches wrong data and
wrong integrals together but cannot separate them.

Level A's 1e-9 Ha bar is also unreachable as specified. `boys0` in
`integrals-cg.ts` uses the Abramowitz–Stegun 7.1.26 rational fit for
erf, whose max ABSOLUTE error is **1.394e-7** at x = 0.045, measured against
mpmath at 40 dps over x ∈ [1e-6, 8] (the code
admits ~1.5e-7 at `integrals-cg.ts:195`). Row 3 makes it worse: Ar's
STO-3G 1s exponent is 674.45 against carbon's 71.62, pushing more
core-core pairs onto the approximate branch. A real Level A needs either
an f64-accurate `erf` or a bar set from measurement.

### 1.2 Correlated methods are unvalidated for every new element

All 108 committed reference cells are **RHF only**. MP2, CCSD, CCSD(T),
DFT and EOM have no per-element reference for B, Ne, or any of Na–Ar.
Wrong basis data would show up in HF, so this is not blind — but
"HF agrees" is not "CCSD(T) agrees".

### 1.3 The row-3 frozen-core path is untested

`FROZEN_CORE_FOR` now returns 5 for Na–Ar (the full neon core) and
`defaultFrozenCore` sums it. No test exercises it.
`tests/chemistry/frozen-core-audit.test.ts` hardcodes `nFrozenCore: 1`
and only uses H2O/CH4, where 1 is correct — so it passes and will keep
passing while the row-3 path stays uncovered.

Known interaction: `rhf-auto.ts:450` throws (in `runUMP2Auto`) for `nFrozenCore > 0` on the
exact-ERI UMP2 path. Any caller that inherits 5 from `defaultFrozenCore`
on a small row-3 molecule will hit that throw. Not triggered by anything
currently in the suite.

### 1.4 Gradients were not hardened (Phase 2 not started)

> **SUPERSEDED 2026-09-02.** Phase 2 landed in `acf7b05`:
> `tests/chemistry/elements/gradient-agreement.test.ts` covers all ten new
> elements against central FD at 1.5e-6 Ha/Bohr, plus translational
> invariance at 1e-12. The rest of this section is kept as the record of
> what was true when the report was written.

No analytic-vs-finite-difference check was run for any new element.

### 1.5 No geometry validation

Reference geometries for the new hydrides (NaH 1.8874, MgH2 1.7297,
AlH3 1.5840, SiH4 1.4798, PH3 1.42/93.5, H2S 1.3356/92.11, HCl 1.2746,
BH3 1.19 A) were taken as experimental-ish values and used identically
on both sides. Since PySCF and webgpu-q consume the same coordinates,
a wrong geometry cannot cause a test failure — it just means the cell
validates a molecule that isn't quite the real one.

---

### 1.6 The 6-31G* claim was false, and had been for a long time

`LIMITATIONS.md` listed 6-31G* as "wired, spot-checked" for H/C/N/O.
The string "6-31g" appears nowhere in `src/`; `BasisName` is exactly
`"sto-3g" | "cc-pvdz" | "aug-cc-pvdz"`. Corrected in this run. Worth
noting *where* it was found: the honest-limitations document was itself
carrying a false capability claim.

---

## 2. What landed

18 elements, H through Ar, complete. 109/109 agreement cells green
(18 elements x 3 bases x 2 d-conventions, plus a convention sanity
check), bar 0.1 mHa. Full suite 140 files / 1065 passed / 1 skipped.

The point of the exercise was curriculum reach, so that is checked
directly in `tests/chemistry/elements/curriculum-molecules.test.ts`:

| molecule | why it matters | vs PySCF |
|---|---|---|
| CH3Cl | the SN2 substrate — Cl was the blocking element | < 0.1 mHa |
| CH3SH | real organosulfur, not a toy hydride | < 0.1 mHa |
| H2S | the "why is H2S a gas and H2O a liquid" comparison | < 0.1 mHa |
| PH3 | phosphorus; entry point to phosphates and DNA | < 0.1 mHa |

All four converge in ~1.2 s combined at STO-3G.

---

## 3. Defects found in existing code

See commit messages for full detail. Summary:

| defect | worst measured impact |
|---|---|
| Li cc-pVDZ 2p coefficients + diffuse exponent | 1.293 mHa on LiH |
| Be cc-pVDZ 1s/2s/2p coefficients | 0.367 mHa on BeH2 |
| aug-cc-pVDZ diffuse: only H and C were correct of 8 | 0.171 mHa on HF |
| CCPVDZ_H exponents non-canonical | ~5 uHa, every H molecule |
| CCPVDZ_N_1S coefficients | ~0.03 uHa |
| BECKE_XI missing F; He/C/O off-by-one | 0.64 uHa |
| import-formats: "Fe" silently imported as fluorine | wrong element, silent |

---

## 4. Corrections to the plan itself

- Na and Mg carry d functions in cc-pVDZ, so the Cartesian/spherical
  convention difference applies from **Na**, not from Al as written.
- STO-3G row 3 is **9** Cartesian functions (3s + 2p), not 13.
- The plan said "fix the generator — never the existing constants".
  Inverted: the constants were wrong and PySCF was the authority.

---

## 5. What the next run should start with

1. Phase 2 — gradient hardening (analytic vs FD) for the 10 new elements.
2. Correlated-method references (MP2/CCSD/CCSD(T)) per element.
3. A row-3 frozen-core test, plus resolving the `rhf-auto.ts:450` throw.
4. The loose-assertion cleanup, starting with
   `e2e/swarm-hf-anthracene-ccpvdz.spec.ts`, whose `(-1500, -100)` window
   certifies a documented-wrong -880 Ha result as green.
5. Re-source the ~15 hardcoded literals from
   `experiments/results/2026-07-06/level-6/E34-pyscf.json`, which holds
   40 full-precision references that no test currently uses.
6. Wire `scripts/check-basis-vs-pyscf.py` into CI — it is the only
   artifact that catches bad basis digits at the source, and it runs
   from no workflow today.
