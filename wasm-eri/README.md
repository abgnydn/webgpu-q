# wasm-eri

Rust port of webgpu-q's ERI primitive kernel, built to wasm32 for
4× speedup over JIT'd TypeScript on the n⁴ ERI tensor build.

## Build

```bash
wasm-pack build --target web --release
```

Output lands in `pkg/`. Wired into `src/chemistry/wasm-eri.ts` via
a relative dynamic import.

## What's in here

Algorithm-identical port of `src/chemistry/integrals-cg.ts`:
- Boys function (Taylor / upward recurrence)
- McMurchie-Davidson 1D E-coefficient tables
- R_{tuv} auxiliary integrals
- Primitive ERI with pair-table caching (matches the TS pair-cache win)
- Schwarz screening + 8-fold symmetry fill

No SIMD intrinsics yet — pure native compile. Layering wasm-simd128
is open follow-up work.
