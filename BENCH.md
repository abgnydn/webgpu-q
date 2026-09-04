# Performance gate policy

## (a) Rule

A timing cell may gate merges **only** when its `std/median <= 0.1` is measured
with the repo's 5-warmup / 20-trial harness (`experiments/lib/runner.ts`
`timedRun`; noisy flag is set in `experiments/lib/stats.ts`).

Noisy cells **warn, never fail**. The CCSD(T) WGSL kernel reports
`std/median ≈ 42%` and is therefore `status: "noisy"` under this repo's own
rule; it **must not gate** merges.

## (b) Provisional single-sample baseline (M2 Pro)

These numbers come from **one vitest run** and are **indicative only — NOT gate
thresholds**.

| suite | seconds |
|---|---|
| `tests/chemistry/dft-gradient.test.ts` | ~92 |
| `tests/chemistry/df-streaming-accuracy.test.ts` | ~91 |
| `tests/chemistry/ccpvdz-spherical.test.ts` | ~80 |
| `tests/chemistry/aug-ccpvdz-firstrow.test.ts` | ~57 |
| `tests/manybody/dmrg-pfeuty-bethe.test.ts` | ~52 |
| `tests/chemistry/aug-ccpvdz.test.ts` | ~39 |
| `tests/chemistry/vibrations.test.ts` | ~39 |
| `tests/chemistry/frozen-core-audit.test.ts` | ~31 |
| `npm run test:fast` subset (132 files / ~900 tests) | ~32 wall |

## (c) Promotion rule

A cell graduates to gated only after all of the following are recorded in
`bench/results.json`:

1. 5-warmup / 20-trial harness numbers from `experiments/lib/runner.ts`.
2. `std/median <= 0.1` for that cell.
3. At least 20 historical samples (`samples >= 20`).
4. A maintainer promotes the entry into `bench/gated.json` and adds
   `noise: { stdOverMedian, samples }`.

Until then every number in this file and in `bench/baseline.json` is
**provisional and tracking-only**.
