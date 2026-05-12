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

1. **Cross-checks against PySCF 2.13 / ORCA 6.1 / Psi4 1.10** on
   identical inputs. We claim agreement; a side-by-side artifact in
   `experiments/results/` cements it.
2. **Standardized benchmark runs** — see `BENCHMARKS.md` for the queue
   (Thiel/QUEST, GMTKN55, W4-11, S66, HEAT-345). Each new set ships as
   a research-grade experiment with named seed, warmup, trials.
3. **Cross-browser / cross-vendor verification** — we test on M2 Pro +
   Chromium. NVIDIA / AMD / Intel adapters, Firefox / Safari WebGPU
   support — all open.
4. **Modern functional implementations** — see Tier 3 roadmap
   (ωB97M(2), revDSD-PBEP86-D4, ωB97X-V, SCAN-D3).
5. **Bug reports** with a reproducer JSON artifact, ideally.

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
