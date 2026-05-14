# Research-grade engineering standards

**Canonical document. Mirrored across four sibling WebGPU/WGSL research
projects:**

- [`webgpu-q`](https://github.com/abgnydn/webgpu-q) — quantum chemistry
- [`webgpu-dna`](https://github.com/abgnydn/webgpu-dna) — radiation track-structure / radiobiology
- [`zero-tvm`](https://github.com/abgnydn/zero-tvm) — Phi-3 LLM inference (hand-written WGSL, head-to-head vs WebLLM)
- [`neuropulse`](https://github.com/abgnydn/neuropulse) — live 1:1 LLM forward-pass visualization (Phi-3, 3.8B params)

Edit any one and propagate. Project-specific examples in §§ 1, 6, 7, 8, 10
diverge per repo; sections 2–5, 9, 11–15 are universal.

This is the discipline that makes the work publishable in JOSS, citable
years later, and reproducible by reviewers on different hardware. The
patterns matured in different repos and back-port / forward-port between
them (research-grade artifact discipline first in `webgpu-dna`, the
"falsify before shipping" CPU pre-screen in `zero-tvm`, automated
doc-vs-code drift detection in `neuropulse`, full porting framework in
`webgpu-q`). Future siblings inherit the union.

**Umbrella thesis**: every advanced physics simulation in the world
should ship as a URL. The browser/WebGPU layer is what's novel; the
chemistry/physics/data is textbook. **Hand-write only the novel
layer; port everything with a peer-reviewed reference.**

---

## 1. Single source of truth for quantitative claims

All measured numbers live in **one** canonical section per repo:

- `webgpu-dna` → `README.md` § Numbers
- `webgpu-q` → `README.md` "Key numbers — single source of truth" table

Anywhere else (CLAUDE.md, slide decks, blog posts, hero SVG, README
headlines) may *summarize* numbers but never *introduce* new ones.

**If a number isn't in the SoT, it isn't measured.**

Before stating a measurement anywhere:

  protocol → run experiment → commit JSON artifact → add SoT row → quote

Not the other way around.

---

## 2. Falsifiable JSON artifacts back every claim

Path: `experiments/results/YYYY-MM-DD/level-N/<id>.json`.

Shape (locked; don't add top-level keys without updating the runner):

```json
{
  "meta":     { "protocol": "...", "hypothesis": "...", "passBar": "...",
                "seed": "named-seed-id", "warmup": 5, "trials": 20 },
  "env":      { "gitSha": "...", "userAgent": "...", "adapter": {...},
                "limits": {...}, "timestamp": "2026-05-14T...",
                "shaderHashes": {"helpers_wgsl": "...", ...} },
  "rows":     [ { /* per-cell measurements */ } ],
  "status":   "pass" | "fail" | "noisy" | "partial",
  "diagnosis": "first-failing-cell + smoking-gun explanation"
}
```

`npm run experiments -- <id>` re-runs deterministically. Same machine
+ same seed + same shader hash = bit-exact. fp32 `atomicAdd` is NOT
order-deterministic across GPU vendors — same input on different
hardware yields statistically equivalent but not bit-exact results;
shaderHashes lets reviewers group rows correctly.

---

## 3. Status labels are first-class

- **`pass`** — meets the protocol's pass bar.
- **`fail`** — doesn't. Commit anyway with a `diagnosis` field naming
  the first failing cell and the smoking gun. **Never silently rerun
  until pass.**
- **`noisy`** — `std/median > 0.1` on any cell. Informational, not
  pass/fail.
- **`partial`** — some cells pass, others don't; explicit `N of M`
  count in the diagnosis.
- **`honest negative`** — failures that are evidence. The two sister
  documents (`LIMITATIONS.md` for webgpu-q, `PHYSICS_DIAGNOSIS.md` for
  webgpu-dna) cite the artifact and the rejected hypothesis.

Honest negatives become the project's evidence base. They are not
bugs to fix; they are findings.

---

## 4. Reproducibility (no randomness left to chance)

- `Math.random()` is **banned** in any experiment path. Every random
  draw uses a named seed from `experiments/lib/seeds.ts` (webgpu-q)
  or `experiments/lib/seeds.mjs` (webgpu-dna) via `mulberry32(seed)`.
- Every JSON artifact records: git SHA (when available), full
  `navigator.userAgent`, `adapter.info`, WebGPU `limits`, UTC
  ISO8601 timestamp, **shader-file SHA-256 / git-rev-parse hashes**.
- 5 warmup samples are discarded; 20 trials retained.
- Report **median + p10/p90/p99 + std + IQR** — never single-shot.
- If `std/median > 0.1` on any cell → label the artifact `"noisy"`.

---

## 5. GPU timing requires a forced sync

`performance.now()` deltas around `queue.submit` alone are fiction —
WebGPU is asynchronous. **Mandatory pattern**: a mapped readback of a
tiny buffer before AND after the work. The `timedRun()` helper in
`experiments/lib/runner.ts` / `experiments/lib/runner.mjs` does this
correctly; use it.

---

## 6. Multi-level correctness verification

Match against more than one reference frame. Listed in increasing
sophistication / decreasing strength:

1. **Analytical limits** where they exist: H₂ FCI on STO-3G,
   Pfeuty's exact TFIM at criticality, Bethe ansatz for the
   Heisenberg chain, ICRU 31 W-value, Sackur-Tetrode entropy.
2. **Brute-force diagnostic in a small basis**: explicit Fock-space
   construction in TypeScript (`eom-ccsd-bruteforce-lih.test.ts`,
   `eom-ccsd-bruteforce.test.ts`). Diff matrix elements
   element-by-element, not just eigenvalues.
3. **Peer-reviewed reference packages**: PySCF, libxc, ITensor for
   webgpu-q; Geant4-DNA + G4EMLOW data tables, Karamitros 2011
   IRT, Friedland 2011 / PARTRAC for webgpu-dna.
4. **Experiment**: NIST CCCBDB (IPs, vibrational frequencies, gas
   entropy), Sackur-Tetrode reference, PARTRAC for DSB/SSB
   ratios, ICRU references.

Multiple independent reference frames > one. Each artifact should
state which it's checking against.

---

## 7. Port from references; hand-write only the novel layer

This is the architectural rule. The differentiator of either project
is the WebGPU/WGSL/browser stack — not the physics formulas. So:

- **Hand-written and owned**: WGSL kernels, WebGPU dispatch glue,
  Web Worker IRT scheduling, browser memory bookkeeping, MPS
  canonical-form, kernel fusion, the research-grade harness.
- **Ported from peer-reviewed source with attribution**:
  - webgpu-q: HF, MP2, CCSD, UCCSD, CCSD(T), EOM-CCSD, DFT
    functionals (libxc), HF/DFT gradients, density-fitting,
    basis-set tables (EMSL Basis Set Exchange).
  - webgpu-dna: G4EMLOW cross-section tables, Karamitros 2011 IRT
    reaction rates, Geant4 angular distributions, dissociation
    branching ratios, scoring conventions.

**Per-file header** for ported code:

```
// Ported from <upstream> (<upstream-url>), <license> license.
// Source: <relative-path> at commit <SHA>
// Original authors: <upstream/AUTHORS>
// Adaptations for <this-project>:
//   - <substantive change 1>
//   - ...
// See LICENSE-<UPSTREAM> at repo root for the <license> notice.
```

**Repo-level**: `LICENSE-<UPSTREAM>` at root (verbatim from upstream).
Per-module status table in `MIGRATION.md`:

| module | reference | license | status |
|---|---|---|---|
| `eom-ccsd.ts` σ_2 | PySCF `pyscf/cc/eom_rccsd.py` | Apache 2.0 | 🔴 → 🟡 → 🟢 |

License compatibility: MIT + Apache 2.0 + BSD-like (Geant4) work
together — the ported portion keeps its upstream license
obligations (notice + state changes); the rest of the repo stays
MIT.

---

## 8. No fudge factors without a citation

Any tunable scalar in production code that isn't backed by a
peer-reviewed source is:

1. **Labeled empirical** in the code comment at point of use.
2. **Documented in `LIMITATIONS.md` / `PHYSICS_DIAGNOSIS.md`** with
   the magnitude of the empirical correction and what observable it
   was tuned against.
3. **Queued for removal** once the structural fix lands.
4. **Tracked in CHANGELOG / commit messages** when added and when
   removed.

Examples carried at the time of writing:

- webgpu-q: stage-32c diagonal patches in `eom-ccsd.ts` σ_1 / σ_2
  (+0.5·E_corr·R_1, −E_corr·R_2). Necessary for H₂ FCI exactness
  and net-positive on multi-electron triplets; queued for the
  PySCF port.
- webgpu-dna: `SIGMA_EXC_SCALE = 0.5` and `RECOMB_BOOST = 2.0` in
  `helpers.wgsl`. Empirical joint fix improves chem6 agreement;
  `RECOMB_BOOST` has been publicly refuted as having no physical
  basis after Geant4 source archaeology. Queued for cross-primary
  IRT.

Tested-and-rejected hypotheses go into the same documents so
future sessions don't re-test them.

---

## 9. Shader byte-hashing for reproducibility

Every artifact records the SHA-256 (or git-rev-parse short hash) of
each WGSL shader file the experiment depended on. Old artifacts get
retrofitted via `tools/retrofit-shader-hashes.mjs` (webgpu-dna's
pattern; adoptable to webgpu-q). This lets reviewers group rows by
shader version when a tunable scale shifts the baseline.

The `env` block carries `shaderHashes: { helpers_wgsl: "...",
primary_wgsl: "...", ... }`.

---

## 10. Living open-gaps document

- `webgpu-q`: `LIMITATIONS.md` at root
- `webgpu-dna`: `PHYSICS_DIAGNOSIS.md` at root

Each entry has three parts:

```
## N. The <observable> deficit vs <reference> (<artifact>, <date>)

Observed.  <quantitative gap with σ-significance>

Hypothesis A — <candidate root cause>
Hypothesis B — <alternative>

Falsification experiment: <what would distinguish them>
```

Entries are removed when the underlying gap closes; the artifact
references stay in `CHANGELOG.md`. Tested-and-rejected hypotheses
get a strikethrough entry with the refutation artifact link, so
the same hypothesis isn't tried twice.

---

## 11. Honest self-corrections

When a prior claim turns out wrong, revise it **in the same commit
that surfaces the data**, with the full arc preserved. Examples:

- "G(e⁻aq) V-shape — was claimed as ~40σ without backing → 126σ via
  primary-bootstrap" (webgpu-dna E10b).
- "EOM-CCSD ≡ FCI at 10⁻⁵ Ha" was true only for H₂ STO-3G (2-electron
  T̂² = 0 limit); multi-electron systems show 1–3 eV gap. Public
  surfaces updated (`readme-numbers.svg` card, validation matrix,
  scorecard, README SoT table) after stage 32k diagnosis (webgpu-q).
- "Stage 32f: missing σ_1 cross-spin coupling" — rejected after the
  full 14×14 diff in 32f-2 showed R₁ × R₁ off-diagonals were correct;
  4 rejected hypotheses (32f, 32f-2, 32g, 32h) preceded the true
  fix (32k σ_1 sign-flip). All documented (webgpu-q).

This is publication-grade transparency. **Wrong hypotheses become
part of the public scientific record, not an embarrassment to
hide.**

---

## 12. Citation infrastructure per release

Each minor release ships:

1. Git tag (`v0.X.Y`)
2. GitHub Release with notes drawn from CHANGELOG
3. **Zenodo DOI** minted via the GitHub-Zenodo integration
4. `CITATION.cff` `preferred-citation` block updated with the real
   DOI

Patch releases (doc-only, refactor, etc.) skip the Zenodo step.

---

## 13. WebGPU gotchas (carry forward across all projects)

- `initGPU()` MUST pass `requiredLimits` for
  `maxStorageBufferBindingSize` and `maxBufferSize`. The default
  128 MiB cap silently truncates large dispatches.
- `atomicAdd` works only on `u32` — not f32. Use fixed-point
  encoding (×100 units/eV worked in webgpu-dna) for f32 reductions.
- No recursion in WGSL. All shaders are single-pass.
- Uniform buffers must be 16-byte aligned.
- No subgroup intrinsics in WebGPU 1.0 spec (out for now, in
  future revisions).

---

## 14. Test discipline (non-negotiable)

- TypeScript `strict` + `noUncheckedIndexedAccess`. No exceptions.
- ESLint clean — 0 errors. Warnings tracked, ideally 0.
- CI green. Every PR runs unit + e2e + typecheck + lint.
- Each method has paired test coverage by **intent**, not by
  metric:
  - **Analytical** (FCI / Bethe / Pfeuty / ICRU) where it exists.
  - **Peer-package** (PySCF / Geant4 / libxc / ITensor) on a fixed
    cell.
  - **Brute-force** in a small basis where feasible.
- Honest negatives (status: "fail" tests) live alongside passes;
  they don't break CI but they're surfaced in the suite output.

---

## 15. Release cadence

- **Minor releases** (`v0.X.0`) for substantive features or
  scientific findings. Tag + GitHub Release + Zenodo DOI.
- **Patch releases** (`v0.X.Y`) for doc-only, refactor, SVG
  refresh, narrative updates. Tag + GitHub Release, no DOI.
- **CHANGELOG** follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
  format: `### Added / Changed / Fixed / Documented / Honest negatives`.
- **CITATION.cff version** matches `package.json` version matches
  Git tag matches GitHub Release tag, all pinned per release.

---

## On adding a new sibling project

Inherit these 15 principles from day one. Copy this file verbatim
into the new repo. Replace project-specific references in
sections 1, 6, 7, 8, 10 with the new project's analogs. Cross-link
sibling projects in §0 (umbrella thesis).

The discipline is the product.

---

*Last revised: 2026-05-14. Canonical document; siblings are
mirrors of this one. Edit either and propagate.*
