# wasm-gemm

Cache-blocked, WASM-SIMD (`simd128`, f64x2) **double-precision GEMM** —
the reusable fast-CPU matrix-multiply kernel intended to replace the naive
TypeScript triple/quad loops that dominate MP2/CCSD tensor contractions
(those contractions are BLAS-bound, so a real blocked-SIMD `dgemm` is the lever).

**Not yet wired into the chemistry paths** — this crate proves the lever and
ships an integrable kernel. Integration into MP2/CCSD is done separately.

## API (wasm-bindgen)

Row-major, f64. `C = alpha·op(A)·B + beta·C`.

Ergonomic (copy-in / copy-out — what TS calls):
- `dgemm(m,n,k, A, B, C, alpha, beta) -> Float64Array` — `op(A)=A`, A is m×k.
- `dgemm_at(m,n,k, A, B, C, alpha, beta) -> Float64Array` — `op(A)=Aᵀ`, A is k×m
  (common in ERI→MO and CC contractions).

Both cover `beta=0` (C=A·B) and `beta=1` (C+=A·B), and any `alpha`.

Raw zero-copy (heavy-loop / fair-bench path — operate directly on
Float64Array-backed wasm memory, no per-call marshalling):
- `alloc_f64(len) -> ptr`, `free_f64(ptr, len)`
- `dgemm_raw(m,n,k, aPtr, bPtr, cPtr, alpha, beta)`
- `dgemm_at_raw(m,n,k, aPtr, bPtr, cPtr, alpha, beta)`

TS loader (typed, lazy, unused-by-design): `src/chemistry/wasm-gemm.ts`.

## Implementation

- **SIMD**: `core::arch::wasm32` f64x2 (v128, 2 lanes), vectorized over the
  contiguous n dimension of row-major B/C.
- **Register blocking**: 4 rows × 2 columns micro-kernel (8 v128 accumulators),
  each B load reused across 4 A broadcasts. Scalar cleanup for odd rows/columns.
- **Cache blocking**: k-block `KC=256`, m-block `MC=64`, so the reused B panel
  stays warm in L2 while the C row-tiles cycle over it.
- `Aᵀ·B` packs Aᵀ into a contiguous m×k buffer (O(m·k), negligible) then reuses
  the NN kernel — one optimized micro-kernel, not two.

## Build

```bash
cd wasm-gemm
wasm-pack build --target web --release   # RUSTFLAGS simd128 via .cargo/config.toml
```

Output lands in `pkg/` (committed). Mirrors `wasm-eri`'s approach
(wasm-bindgen, `crate-type=cdylib`, opt-level 3 + LTO + codegen-units 1).

## Benchmark

```bash
mise exec -- node tools/gemm-bench/bench.mjs
```

Compares, on the SAME seeded random f64 matrices:
(a) naive-TS triple loop, (b) cache-blocked TS loop, (c) WASM-SIMD `dgemm`
(zero-copy raw path), plus (d) the ergonomic copy-in/out WASM API.

### Measured — Apple M2 Max (arm64), Node v22.23.1, single-thread, seed `0x5eed9e33`

GFLOP/s (2·m·n·k / time), warmup + median of trials:

| shape (A·B)              | naive-TS | blocked-TS | WASM-SIMD | WASM(copy) | ×naive | ×block | max rel |
|--------------------------|---------:|-----------:|----------:|-----------:|-------:|-------:|--------:|
| 64×64 · 64×64            |     1.70 |       1.63 |      4.21 |      11.85 |   2.5× |   2.6× |    0.0  |
| 128×128 · 128×128        |     1.14 |       1.05 |     10.98 |      10.53 |   9.7× |  10.4× |    0.0  |
| 256×256 · 256×256        |     1.01 |       1.05 |     11.72 |      10.73 |  11.6× |  11.1× |    0.0  |
| 512×512 · 512×512        |     1.29 |       1.27 |     13.82 |      13.39 |  10.8× |  10.9× | 1.6e-15 |
| 100×1444 · 1444×100      |     1.29 |       1.26 |     14.57 |      14.03 |  11.3× |  11.5× | 3.0e-15 |
| 1444×100 · 100×1444      |     1.30 |       1.27 |     12.38 |      12.11 |   9.5× |   9.8× |    0.0  |

- The two rectangular rows are CC-contraction-shaped (H₂O/cc-pVDZ:
  n≈24 AOs, NOCC≈5, NVIRT≈19 → (NOCC·NVIRT)×n² style panels).
- Correctness pass bar ≤ 1e-10 rel; worst residual **3e-15** across all shapes,
  incl. `beta=1` (C+=A·B) and `dgemm_at` (Aᵀ·B). PASS.

### Honest read

- **~9–13× over naive-TS** for real (≥128) sizes; 64³ is only 2.5× (kernel /
  block overhead dominates a tiny matrix). Peak ~14.6 GFLOP/s.
- **Blocking in JS buys ~nothing** (blocked-TS ≈ naive-TS, ~1 GFLOP/s). V8
  already JITs the ikj loop reasonably and can't emit f64 SIMD. The *entire*
  win is native compile + `simd128` vectorization — that is the lever, not
  cache blocking per se.
- **Marshalling is cheap**: the copy-in/out ergonomic API tracks the zero-copy
  raw path within noise (compute is O(n³), copies O(n²)), so integration won't
  pay a large tax. (At 64³ the copy path even looks faster — small-size noise.)
- **Ceiling**: WASM SIMD is 128-bit = 2×f64, and stable `simd128` has no fused
  f64 multiply-add (this kernel is mul+add), so ~14 GFLOP/s is ~half the
  mul+add NEON ceiling and far below native AVX2/512 BLAS. This is a
  **naive-TS → WASM-SIMD** result, NOT a "vs PySCF/BLAS" claim.

## Method-scale proof (the real MP2/CCSD contraction)

`tools/gemm-bench/eri-transform.mjs` runs the kernel on the *actual* O(N⁵)
hot loop — the ERI→MO quarter-transform `t[p,νλσ] = Σ_μ C[μ,p]·eri[μ,νλσ]`,
a single fat GEMM `Cᵀ(p×μ) · eri(μ × n³)` — at cc-pVDZ H₂O size (n=24):

| impl | ms | GFLOP/s |
|---|--:|--:|
| naive-TS | 7.43 | 2.14 |
| WASM-SIMD | 1.46 | 10.87 |
| **speedup** | | **5.1×** |

Bit-identical output (max\|Δ\| = 0). ~5× (not the microbench's ~10×) because at
n=24 the naive quadruple loop still caches well (2.1 vs ~1.0 GFLOP/s); the win
grows with system size. This is the real per-contraction, chemistry-grade lever.

## Integration status — deliberately deferred (scope-honest)

The kernel is proven and integration-ready (typed lazy loader in
`src/chemistry/wasm-gemm.ts`, opt-in), but it is **NOT wired into `mp2.ts` /
`ccsd.ts` yet, on purpose:**

1. **It's off the primary strategy.** The GEMM lever pays off on cc-pVDZ MP2/CCSD
   — exactly the production-basis workload where we lose to BLAS (136×/480×) and
   which we've explicitly chosen *not* to chase (see README "How fast"). The
   product's real workloads — education (`learn.html`) and screening — run
   STO-3G small molecules, where we already win and GEMM is not the bottleneck.
2. **It would risk the tested core for that off-niche gain.** The full 4-index
   transform is 4 quarter-transforms; keeping every pass a fat GEMM needs an
   inter-pass transpose (cheap), and the naive per-`p`/`pq` batching devolves
   passes 3–4 into tiny (n×1) GEMMs that *lose* to call overhead — so a correct
   integration is real work against 872 passing chemistry tests.
3. **The honest ceiling doesn't change the strategy.** Even fully integrated and
   worker-parallel, browser CPU stays ~10–30× slower than a multithreaded MKL
   PySCF at cc-pVDZ (128-bit SIMD, no f64 FMA, vs AVX-512 + threads). This
   *narrows* the gap; it never closes it.

**Decision:** land the proven kernel + benchmarks as reusable infrastructure and
a shareable systems result; wire it into the correlation methods only if/when
bigger in-browser systems become an actual goal. The lever is measured and ready.
