# 24h autonomous run plan — break the periodic-table wall

**Status:** executed 2026-08-10 → 2026-09-02 on `feat/periodic-table-expansion` (PR #36).
Outcome in `docs/RUN-REPORT-2026-08-10.md`, which is itself stale on gradients —
Phase 2 landed after it was written.
**Branch:** `feat/periodic-table-expansion` (create from `main`)
**Written:** 2026-08-10

---

## Mission

Take `SUPPORTED_SYMBOLS` from 8 elements to 18, with every new element
validated against PySCF to sub-milliHartree and locked behind a
regression test.

Current: `H, He, Li, Be, C, N, O, F`
Target: add `B, Ne, Na, Mg, Al, Si, P, S, Cl, Ar`

This is a **data** problem, not an engine problem. The integral core
already handles d-shells (cc-pVDZ carbon carries `CCPVDZ_C_3D`), so
third-row elements need no new angular-momentum machinery — only more
contracted shells at L ≤ 2.

## Non-goals

Do **not** do any of these, even if they look tempting mid-run:

- Refactor working code. The repo was audited 2026-08-08; it is not the problem.
- Add new methods, new UI, new pages, or new experiments.
- Deploy anything. Not Vercel, not the HF Space. (Standing user preference.)
- Push to `main` or open a PR. Leave the branch for human review.
- Chase performance.

## Hard rules

1. **Every step ends in a number a machine can compare.** No "looks
   right", no visual inspection, no prose verdicts. If a step cannot be
   expressed as `assert |ours − reference| < tol`, it is not a step.
2. **Hard-fail on `converged: false`.** Every harness must treat a
   non-converged SCF/CCSD as a *failure*, never as a datum. This is the
   single most likely way a long run produces confident garbage.
3. **One element per commit.** A bad stretch must be revertable without
   losing the good ones. Trailer:
   `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
4. **Serialize test runs.** Do not run vitest suites concurrently with
   PySCF sweeps. The machine is a 32 GB M2 Max; saturating it produces
   timeout flakes that look exactly like real failures and will send the
   run chasing phantoms.
5. **Never widen a tolerance to make a test pass.** If a bar cannot be
   met, record the failure in the Phase 3 report and move to the next
   element. A loosened tolerance is an undetectable lie.
6. **`uv run` inside `~/dev/ml-research` for anything PySCF.** Never bare
   `pip` / `python`.

---

## Phase 0 — the harness (do not skip; nothing after this is safe without it)

### 0.1 Resolve the d-shell convention first

`tests/chemistry/ccpvdz-spherical.test.ts` documents that the Cartesian
(6d) path sits **~0.4 mHa below** PySCF's default spherical (5d) result
for H₂O/cc-pVDZ — a basis-set convention difference, *not* a bug.

Every element from Al onward has d functions in cc-pVDZ. If the harness
compares the Cartesian path against default PySCF, **every new element
will show a ~0.4 mHa phantom error**, and the run will spend hours
"fixing" correct code.

**Gate 0.1:** the harness must either (a) compare the spherical path
against default PySCF, or (b) compare the Cartesian path against PySCF
with `mol.cart = True`. Prove it by reproducing the existing H₂O/cc-pVDZ
agreement *before* touching any new element. Write down which convention
was chosen at the top of the harness file.

### 0.2 Generate basis tables — do not hand-transcribe

Write `scripts/gen-basis-tables.py`: read PySCF's own basis library
(`pyscf.gto.basis.load`) and emit TypeScript constants in the exact
shape `src/chemistry/integrals.ts` already uses:

```ts
export const STO3G_C_2P = {
  alpha: [2.9412494, 0.6834831, 0.2222899] as const,
  c: STO3G_L_2P_C,
};
```

Hand-transcribing 10 elements × 3 basis sets is ~100 blocks of decimal
digits. That is precisely the task where a long autonomous run silently
introduces a typo that surfaces as a 3 mHa discrepancy 14 hours later.
Generating it removes the failure mode entirely.

**Gate 0.2 (the important one):** run the generator against the **eight
elements that already exist** and diff against the current hand-written
constants. If it reproduces `C`, `N`, `O`, `F` to the last digit, the
generator is trustworthy for `S`, `P`, `Cl`. If it does not, fix the
generator — never the existing constants — until it does.

This is what makes the rest of the run autonomous-safe: the tool is
validated against known-good data before it is trusted with new data.

### 0.3 Two-level validation harness

For each (element, basis), the harness runs **two** independent checks
that isolate the two distinct failure modes:

| Level | What it compares | Catches |
|---|---|---|
| **A — engine** | our energy vs PySCF fed *our own primitives* via `gto.basis.parse` | bugs in our integrals/SCF for this element |
| **B — data** | our primitives vs PySCF's built-in basis table | wrong/garbled basis data |

`scripts/run-pyscf-reference.py` already uses `gto.basis.parse` for the
matched s-only Li basis — same mechanism, reuse it.

Interpretation is unambiguous:
- A passes, B fails → basis data is wrong.
- B passes, A fails → engine bug for this element.
- Both fail → start with B.

**Gate 0.3:** both levels green for all 8 existing elements before
proceeding. Bars: HF ≤ 1e-9 Ha for Level A (same primitives should agree
to machine precision, not chemistry precision), ≤ 0.1 mHa for Level B.

---

## Phase 1 — elements, in this order

The order is deliberate: **structurally-familiar elements first**, so
that an early failure means "the harness is wrong", not "this element is
hard". Only after the pipeline is proven does it meet new shell
structure.

### 1a. Second row — proves the pipeline (structure already supported)

| El | Z | STO-3G shells | Why |
|---|---|---|---|
| **B** | 5 | 1s, 2s, 2p | BH₃/BF₃ — the canonical empty-orbital Lewis acid lesson. Also fills the one gap in row 2. |
| **Ne** | 10 | 1s, 2s, 2p | Completes row 2; noble-gas reference for the dispersion labs (`dispersion.ts` already exists). |

These are byte-for-byte the same shell pattern as C/N/O/F. If either
fails, **stop and fix the harness** — do not proceed to row 3.

### 1b. Third row — the actual unlock (new: 3s/3p shells)

Ordered by curriculum value, highest first.

| El | Z | Unlocks |
|---|---|---|
| **Cl** | 17 | **SN2.** The single most-taught mechanism in undergraduate organic. Also HCl, CH₃Cl, Cl⁻ as leaving group. |
| **S** | 16 | Thiols, H₂S, disulfide bridges, SO₂. Biochemistry entry point. |
| **P** | 15 | Phosphates, PH₃ — the DNA/ATP backbone. |
| **Si** | 14 | Silanes, SiO₂; the "why doesn't Si behave like C" comparison. |
| **Na** | 11 | Ionic bonding; NaCl with Cl already landed. |
| **Mg** | 12 | Mg²⁺, chlorophyll hand-wave, second ionic case. |
| **Al** | 13 | AlCl₃ Lewis acid; completes the row. |
| **Ar** | 18 | Van der Waals / dispersion labs; noble-gas dimer. |

If the run stalls, **Cl, S, P are the three that matter.** Everything
after Si is completeness, not capability.

### Per-element loop (identical for every element)

1. Generate constants with the Phase-0 generator; append to `integrals.ts`.
2. Add the `case` to both switch blocks in `atoms.ts` (STO-3G and the
   `heavyShells` cc-pVDZ path). Follow the existing formatting exactly.
3. Add the symbol to `SYMBOL_BY_Z` and `SUPPORTED_SYMBOLS` in `xyz.ts`.
4. Run Level A + Level B for all three basis sets.
5. Write `tests/chemistry/elements/<sym>.test.ts` pinning the validated
   HF energy for at least one real molecule per basis set.
6. Run the **full** existing unit suite. A new element must not perturb
   any existing number. If it does, that is a shared-state bug — stop
   and report it, do not paper over it.
7. Commit. Move on.

**Per-element gate:** steps 4–6 all green, or the element is reverted and
logged as failed. No partial elements on the branch.

### Molecule choices for validation

Prefer molecules with published reference geometries and non-trivial
bonding, one per element:

`BH₃`, `Ne` (atom), `HCl` + `CH₃Cl`, `H₂S`, `PH₃`, `SiH₄`, `NaH`,
`MgH₂`, `AlH₃`, `Ar` (atom).

Atoms alone are a weak test — they exercise no two-center integrals. Use
a hydride wherever one exists.

---

## Phase 2 — gradient hardening

`hf-gradient.ts`, `dft-gradient.ts`, `cphf.ts` and `optimizer.ts` all
exist. Nothing verifies they stay correct as elements are added.

For every (new element, basis, method) combination: compare the analytic
gradient against a central finite-difference gradient of the energy.

**Gate 2:** max component difference ≤ 1e-6 Ha/Bohr. Use a step of
1e-4 Bohr; if the FD gradient is itself noisy at that step, record the
noise floor rather than tightening the bar.

This is a perfect autonomous oracle — the reference is generated from
the code's own energy function, so it needs no external tool and cannot
be argued with.

---

## Phase 3 — the report (reserve the final hour)

Write `docs/RUN-REPORT-<date>.md` containing, in this order:

1. **What could not be verified.** Put it first. This is the most
   valuable output of the whole run — more valuable than the elements.
2. Elements landed, with the measured Level A / Level B deltas per basis.
3. Elements attempted and reverted, with the actual failure.
4. Any tolerance that was *considered* for loosening and why it wasn't.
5. Any existing test whose number moved, and the explanation.
6. What the next run should start with.

Then update `LIMITATIONS.md` with the new element coverage and — this is
outstanding from the last session either way — the **anthracene n=246
wrong-basin result**, which currently lives only in a spec comment.

---

## Known traps (pre-empted here so the run doesn't rediscover them)

- **The 0.4 mHa d-convention phantom.** See Gate 0.1. This will look
  exactly like a real error on every element from Al onward.
- **`converged: false` treated as data.** Bit a lab script in the
  2026-08-08 session: the library reported honestly, the consumer code
  printed unconverged CCSD numbers as results. Assert the flag.
- **Load-induced timeout flakes.** Three "failures" in the last full run
  were a saturated machine, all passing in isolation. Before declaring a
  test broken, re-run it alone.
- **`sed` on macOS is BSD sed.** `\?` silently does nothing. Use Python
  for any regex rewriting of files.
- **Vite HMR reloads on artifacts.** `e2e/.artifacts/**` and friends are
  already in `server.watch.ignored`; do not write scratch files anywhere
  else under the repo root during a run.
- **`fciState` is H₂-only** (imports `buildH2Dense`). Not a general FCI
  solver. Do not use it to validate anything else.

---

## What this run does not accomplish

It makes webgpu-q *capable* of the standard curriculum — SN2, phosphates,
ionic bonding, dispersion. It does not put the project in front of a
single teacher. Those are separate problems, and only the second one is
genuinely uncertain. Spend the 24h here because it is cheap in human time
and unblocks the education lane; do not mistake it for validating that
lane.
