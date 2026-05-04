# experiments/results/

Every experiment run emits a JSON artifact here, laid out as:

```
results/<YYYY-MM-DD>/level-<N>/<experiment-id>-<slug>.json
```

e.g. `results/2026-04-22/level-1/E1-gate-fidelity.json`.

## Schema

All artifacts share the same envelope (see
`experiments/lib/runner.ts → Artifact<Row>`):

```ts
{
  meta: {
    protocol: string,       // e.g. "E1-gate-fidelity"
    hypothesis: string,     // falsifiable claim being tested
    passBar: string,        // quantitative threshold for "pass"
    seed: string,           // name from experiments/lib/seeds.ts
    warmup: number,         // samples discarded before timing
    trials: number,         // samples retained for stats
  },
  env: {
    gitSha: string,
    timestamp: string,      // UTC ISO8601
    userAgent: string,
    platform: string,
    adapter: { vendor, architecture, device, description },
    limits: { maxBufferSize, maxStorageBufferBindingSize,
              maxComputeWorkgroupsPerDimension,
              maxComputeInvocationsPerWorkgroup,
              maxComputeWorkgroupStorageSize },
    hardwareConcurrency: number,
    devicePixelRatio: number,
  },
  rows: [ /* experiment-specific row records */ ],
  status: "pass" | "fail" | "noisy",
  diagnosis: string,        // required if status ≠ "pass"
}
```

## Committing results

1. Run experiments in a browser (`npm run dev` → open
   `http://localhost:*/experiments/index.html`).
2. Click "Run E1–E4" (Level 1) and save each downloaded JSON into
   `results/<today>/level-1/`.
3. Commit. **Negative results are committed too**, with `status: "fail"`
   and a non-empty `diagnosis` string. A failure is evidence — do not
   silently re-run.

## Index

| Date | Adapter | L1 | L2 | L3 | L4 | L5 | L6 | Notes |
|------|---------|----|----|----|----|----|----|-------|
| _tbd_ | _tbd_ | — | — | — | — | — | — | — |

Add a row on each publish-worthy run. The `adapter.description` string
goes in the Adapter column. Levels 2–6 are protocol-only until their
source code lands.
