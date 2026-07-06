// wasm-gemm benchmark — naive-TS vs blocked-TS vs WASM-SIMD f64 dgemm.
//
// Measures GFLOP/s and speedups on the SAME seeded random f64 matrices for
// three implementations of C = A*B (row-major, double precision):
//   (a) naive TS triple loop      — the current MP2/CCSD baseline shape
//   (b) cache-blocked TS loop      — isolates blocking-vs-SIMD
//   (c) WASM-SIMD dgemm (raw path) — the kernel under test (zero-copy)
// Plus (d) the ergonomic copy-in/out WASM API, to show marshalling cost.
//
// Node supports WASM SIMD, so no browser is needed. Run:
//   mise exec -- node tools/gemm-bench/bench.mjs
//
// Honest framing: WASM SIMD is 128-bit (2×f64), so the arithmetic ceiling is
// inherently ~half native AVX2 / quarter AVX-512 BLAS. This measures
// "naive-TS → WASM-SIMD", NOT "vs PySCF/BLAS".

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { cpus } from "node:os";

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkgDir = resolve(__dirname, "../../wasm-gemm/pkg");

// ── Reproducible RNG (mulberry32, matches the repo's seeding discipline) ──
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randMatrix(rng, len) {
  const m = new Float64Array(len);
  for (let i = 0; i < len; i++) m[i] = rng() * 2 - 1;
  return m;
}

// ── Reference implementations ──────────────────────────────────────────
// (a) Naive triple loop, row-major, ikj order (the common TS baseline).
function gemmNaive(m, n, k, a, b) {
  const c = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const ai = i * k;
    const ci = i * n;
    for (let p = 0; p < k; p++) {
      const aip = a[ai + p];
      const bp = p * n;
      for (let j = 0; j < n; j++) {
        c[ci + j] += aip * b[bp + j];
      }
    }
  }
  return c;
}

// (b) Cache-blocked TS loop (no SIMD; isolates blocking from SIMD).
const BM = 64,
  BN = 64,
  BK = 256;
function gemmBlocked(m, n, k, a, b) {
  const c = new Float64Array(m * n);
  for (let ii = 0; ii < m; ii += BM) {
    const iMax = Math.min(ii + BM, m);
    for (let pp = 0; pp < k; pp += BK) {
      const pMax = Math.min(pp + BK, k);
      for (let jj = 0; jj < n; jj += BN) {
        const jMax = Math.min(jj + BN, n);
        for (let i = ii; i < iMax; i++) {
          const ai = i * k;
          const ci = i * n;
          for (let p = pp; p < pMax; p++) {
            const aip = a[ai + p];
            const bp = p * n;
            for (let j = jj; j < jMax; j++) {
              c[ci + j] += aip * b[bp + j];
            }
          }
        }
      }
    }
  }
  return c;
}

// ── Timing helpers ─────────────────────────────────────────────────────
function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
function timeIt(fn, warmup, trials) {
  for (let i = 0; i < warmup; i++) fn();
  const ts = [];
  for (let i = 0; i < trials; i++) {
    const t0 = performance.now();
    fn();
    ts.push(performance.now() - t0);
  }
  return median(ts); // ms
}
const gflops = (m, n, k, ms) => (2 * m * n * k) / (ms * 1e6);

function maxRelResidual(x, ref) {
  let maxRel = 0;
  let refNorm = 0;
  for (let i = 0; i < ref.length; i++) refNorm = Math.max(refNorm, Math.abs(ref[i]));
  const denom = refNorm > 0 ? refNorm : 1;
  for (let i = 0; i < ref.length; i++) {
    const rel = Math.abs(x[i] - ref[i]) / denom;
    if (rel > maxRel) maxRel = rel;
  }
  return maxRel;
}

// ── Load WASM (wasm-pack --target web glue; default() returns exports) ──
async function loadWasm() {
  const mod = await import(resolve(pkgDir, "wasm_gemm.js"));
  const bytes = readFileSync(resolve(pkgDir, "wasm_gemm_bg.wasm"));
  const wasm = await mod.default({ module_or_path: bytes });
  return { mod, wasm };
}

async function main() {
  const seed = 0x5eed_9e33;
  const rng = mulberry32(seed);
  const { mod, wasm } = await loadWasm();

  // Sizes: squares + CC-contraction-shaped rectangles. For H2O/cc-pVDZ,
  // n≈24 AOs, NOCC≈5, NVIRT≈19 (24 basis fns) — the ERI→MO / CC contractions
  // reshape to (NOCC·NVIRT)×n² style panels. We use the prompt's representative
  // shapes m=100,k=1444,n=100 and m=1444,k=100,n=1444.
  const shapes = [
    { m: 64, n: 64, k: 64 },
    { m: 128, n: 128, k: 128 },
    { m: 256, n: 256, k: 256 },
    { m: 512, n: 512, k: 512 },
    { m: 100, k: 1444, n: 100 }, // NOCC·NVIRT panel × small n
    { m: 1444, k: 100, n: 1444 }, // large outer, thin contraction
  ];

  const rows = [];
  for (const { m, n, k } of shapes) {
    const a = randMatrix(rng, m * k);
    const b = randMatrix(rng, k * n);

    // Correctness reference.
    const ref = gemmNaive(m, n, k, a, b);

    // Adaptive trial counts: fewer for the big grids.
    const flops = 2 * m * n * k;
    const heavy = flops > 1e8;
    const warmup = 3;
    const trials = heavy ? 5 : 15;
    const naiveTrials = heavy ? 3 : 10; // naive is slow at 512³

    // (a) naive TS
    const tNaive = timeIt(() => gemmNaive(m, n, k, a, b), 2, naiveTrials);

    // (b) blocked TS
    let cBlocked;
    const tBlocked = timeIt(() => {
      cBlocked = gemmBlocked(m, n, k, a, b);
    }, warmup, trials);
    const resBlocked = maxRelResidual(cBlocked, ref);

    // (c) WASM-SIMD raw (zero-copy). Allocate device buffers once, reuse.
    const aPtr = wasm.alloc_f64(m * k);
    const bPtr = wasm.alloc_f64(k * n);
    const cPtr = wasm.alloc_f64(m * n);
    // Views must be re-taken if memory grew; take after all allocs.
    const memF64 = () => new Float64Array(wasm.memory.buffer);
    memF64().set(a, aPtr / 8);
    memF64().set(b, bPtr / 8);
    const tWasmRaw = timeIt(() => {
      wasm.dgemm_raw(m, n, k, aPtr, bPtr, cPtr, 1.0, 0.0);
    }, warmup, trials);
    const cWasm = memF64().slice(cPtr / 8, cPtr / 8 + m * n);
    const resWasm = maxRelResidual(cWasm, ref);
    wasm.free_f64(aPtr, m * k);
    wasm.free_f64(bPtr, k * n);
    wasm.free_f64(cPtr, m * n);

    // (d) WASM ergonomic (copy-in/out) — realistic integration cost.
    const cZero = new Float64Array(m * n);
    let cErg;
    const tWasmErg = timeIt(() => {
      cErg = mod.dgemm(m, n, k, a, b, cZero, 1.0, 0.0);
    }, warmup, trials);
    const resErg = maxRelResidual(cErg, ref);

    rows.push({
      shape: `${m}×${k}·${k}×${n}`,
      m,
      n,
      k,
      naive: gflops(m, n, k, tNaive),
      blocked: gflops(m, n, k, tBlocked),
      wasm: gflops(m, n, k, tWasmRaw),
      wasmErg: gflops(m, n, k, tWasmErg),
      spVsNaive: tNaive / tWasmRaw,
      spVsBlocked: tBlocked / tWasmRaw,
      res: Math.max(resWasm, resBlocked, resErg),
      resWasm,
    });
  }

  // ── A^T*B correctness spot-check (dgemm_at) ──
  {
    const m = 96,
      n = 80,
      k = 128;
    const aT = randMatrix(rng, k * m); // A is k×m
    const b = randMatrix(rng, k * n);
    // reference: (A^T)[i,p] = aT[p*m+i]
    const ref = new Float64Array(m * n);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let p = 0; p < k; p++) s += aT[p * m + i] * b[p * n + j];
        ref[i * n + j] = s;
      }
    const cErg = mod.dgemm_at(m, n, k, aT, b, new Float64Array(m * n), 1.0, 0.0);
    var atRes = maxRelResidual(cErg, ref);
  }

  // ── beta=1 (C += A*B) correctness spot-check ──
  {
    const m = 64,
      n = 64,
      k = 64;
    const a = randMatrix(rng, m * k);
    const b = randMatrix(rng, k * n);
    const c0 = randMatrix(rng, m * n);
    const prod = gemmNaive(m, n, k, a, b);
    const ref = new Float64Array(m * n);
    for (let i = 0; i < m * n; i++) ref[i] = c0[i] + prod[i];
    const cErg = mod.dgemm(m, n, k, a, b, c0, 1.0, 1.0);
    var betaRes = maxRelResidual(cErg, ref);
  }

  // ── Report ──
  const nCpu = cpus()[0]?.model ?? "unknown";
  console.log("\n=== wasm-gemm benchmark (f64, row-major C = A·B) ===");
  console.log(`machine: ${nCpu} (arm64, ${cpus().length} cores) — single-threaded bench`);
  console.log(`node: ${process.version} · seed: 0x${seed.toString(16)}`);
  console.log(
    "WASM SIMD is 128-bit (2×f64): ceiling ~½ AVX2 / ¼ AVX-512 BLAS. Framing: naive-TS → WASM-SIMD.\n",
  );

  const h = [
    "shape (A·B)".padEnd(20),
    "naive".padStart(8),
    "block".padStart(8),
    "WASM".padStart(8),
    "WASMcpy".padStart(9),
    "×naive".padStart(8),
    "×block".padStart(8),
    "maxRel".padStart(10),
  ].join(" ");
  console.log(h);
  console.log("-".repeat(h.length));
  for (const r of rows) {
    console.log(
      [
        r.shape.padEnd(20),
        r.naive.toFixed(2).padStart(8),
        r.blocked.toFixed(2).padStart(8),
        r.wasm.toFixed(2).padStart(8),
        r.wasmErg.toFixed(2).padStart(9),
        (r.spVsNaive.toFixed(1) + "×").padStart(8),
        (r.spVsBlocked.toFixed(1) + "×").padStart(8),
        r.resWasm.toExponential(1).padStart(10),
      ].join(" "),
    );
  }
  console.log("\nGFLOP/s columns: naive-TS · blocked-TS · WASM-SIMD(raw) · WASM(copy-in/out)");
  console.log("×naive / ×block = WASM-SIMD(raw) speedup vs naive-TS / blocked-TS");
  console.log(`\ncorrectness (max relative residual vs naive f64 reference):`);
  const worst = Math.max(...rows.map((r) => r.resWasm), atRes, betaRes);
  console.log(`  C=A·B  (all shapes) : ${Math.max(...rows.map((r) => r.resWasm)).toExponential(2)}`);
  console.log(`  C+=A·B (beta=1)     : ${betaRes.toExponential(2)}`);
  console.log(`  C=Aᵀ·B (dgemm_at)   : ${atRes.toExponential(2)}`);
  console.log(`  worst-case          : ${worst.toExponential(2)}  (pass bar ≤ 1e-10)  ${worst <= 1e-10 ? "PASS" : "FAIL"}`);
  console.log("");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
