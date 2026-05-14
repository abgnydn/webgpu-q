# Contributing to webgpu-q

Thanks for your interest. This is a research-grade project — corrections,
benchmark additions, and honest negative results are all welcome.

## Quick start

```bash
git clone https://github.com/abgnydn/webgpu-q
cd webgpu-q
npm install
npm run test         # 401 vitest, all green
npm run typecheck    # tsc --noEmit, strict + noUncheckedIndexedAccess
npm run lint         # ESLint flat config
npm run test:e2e     # Playwright headless WebGPU
```

Before you commit, all four of the above MUST pass. CI enforces it.

## What's most valuable

Ranked by impact-per-effort:

1. **Reference-implementation ports** — see [`MIGRATION.md`](./MIGRATION.md).
   Methods like HF, CCSD, EOM-CCSD have peer-reviewed PySCF implementations.
   Porting (with proper attribution) is more reliable and faster than
   re-deriving from textbooks. EOM-CCSD σ_2 is the first scheduled port.
2. **Cross-checks against PySCF 2.13 / ORCA 6.1 / Psi4 1.10** on
   identical inputs. We claim agreement; a side-by-side artifact in
   `experiments/results/` cements it.
3. **Standardized benchmark runs** — see `BENCHMARKS.md` for the queue
   (Thiel/QUEST, GMTKN55, W4-11, S66, HEAT-345). Each new set ships as
   a research-grade experiment with named seed, warmup, trials.
4. **Cross-browser / cross-vendor verification** — we test on M2 Pro +
   Chromium. NVIDIA / AMD / Intel adapters, Firefox / Safari WebGPU
   support — all open.
5. **Modern functional implementations** — see Tier 3 roadmap
   (ωB97M(2), revDSD-PBEP86-D4, ωB97X-V, SCAN-D3). Port from libxc.
6. **Bug reports** with a reproducer JSON artifact, ideally.

## Research-grade engineering standards

All contributions follow the 15-principle canonical document at
[`RESEARCH_STANDARDS.md`](./RESEARCH_STANDARDS.md), mirrored in our
sibling [`webgpu-dna`](https://github.com/abgnydn/webgpu-dna).
Short version: single source of truth for numbers, falsifiable
JSON artifacts, honest negatives committed, no `Math.random()`,
GPU-sync timing, multi-level correctness verification, no fudge
factors without citation, shader byte-hashing, per-release Zenodo
DOI. See the canonical doc for full text.

## Porting policy (NEW — 2026-05-13)

webgpu-q's differentiator is the browser/WebGPU layer, not the chemistry
methods. **Hand-write only what's novel** (WGSL kernels, dispatch glue,
research harness). **Port everything with a working reference** (PySCF,
libxc, ITensor, EMSL Basis Set Exchange) with full attribution.

See [`MIGRATION.md`](./MIGRATION.md) for the per-module status table,
attribution recipe, and priority order. License compatibility:
- Original webgpu-q code: MIT.
- Ported from PySCF: Apache 2.0 (`LICENSE-PYSCF` at root).
- Both kept side-by-side; per-file headers state provenance.

## Coding conventions

- TypeScript `strict` + `noUncheckedIndexedAccess` is non-negotiable.
- No `Math.random()` anywhere in experiment paths — always seed via
  `experiments/lib/seeds.ts`.
- Honest negatives **commit** as JSON with `status: "fail"` and a
  `diagnosis` field. Don't silently rerun until passing.
- New experiments live under `experiments/level-N-<slug>/` and emit
  to `experiments/results/YYYY-MM-DD/level-N/`.
- New methods live under `src/chemistry/` or `src/manybody/` with a
  paired `tests/<area>/<method>.test.ts`.
- WGSL kernels in `src/shaders/` paired with an async TS wrapper.

## Validation discipline

Every experiment must include:

- A named seed from `seeds.ts` (no `Math.random()`).
- Forced GPU sync before AND after timing — a mapped readback of a
  tiny buffer. `queue.submit` alone is non-blocking.
- 5 warmup samples, 20 retained trials. Report median, p10, p90,
  p99, std, IQR.
- A fidelity-based pass bar — `F = |⟨ψ_ref|ψ_test⟩|²`, not `max|Δp|`.
- Full env capture: git SHA, `navigator.userAgent`, `adapter.info`,
  WebGPU limits, UTC ISO8601 timestamp.

If `std/median > 0.1`, mark the artifact `"status": "noisy"` —
don't pretend it isn't.

## Pull request checklist

- [ ] All four scripts pass: `test`, `typecheck`, `lint`, `test:e2e`
- [ ] If the change touches numbers in any SVG hero / scorecard /
      validation matrix, update `public/readme-*.svg` and the
      "Key numbers" SoT table in `README.md`.
- [ ] New methods include a unit test AND a cross-check against
      either an analytical limit, an external package, or a
      brute-force enumeration.
- [ ] New experiments include `protocol.md` + JSON artifact +
      one-paragraph diagnosis (pass or fail).
- [ ] If your PR introduces a new dependency, justify it in the PR
      description. Edge-of-browser code has a tight dependency
      budget.

## Authorship & attribution

webgpu-q is single-authored; substantial contributions earn
co-authorship on subsequent papers / releases. By default contributions
ship under the project MIT license; sign-off lines are not required
but appreciated for non-trivial work.

## Code of conduct

This project follows the [Contributor Covenant 2.1](./CODE_OF_CONDUCT.md).

## Questions

Open a GitHub issue. For sensitive matters: <abgunaydin94@gmail.com>.
