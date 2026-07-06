#!/usr/bin/env node
// eri-transform.mjs — method-scale proof of the WASM-SIMD GEMM lever.
//
// The kernel microbench (bench.mjs) shows ~10× on raw GEMM. This measures the
// SAME win on the actual O(N⁵) contraction that dominates MP2 (and CCSD setup):
// the ERI→MO quarter-transform  t[p, νλσ] = Σ_μ C[μ,p] · eri[μ, νλσ].
//
// That's a single fat GEMM: Cᵀ (p×μ) · eri (μ × n³) → t (p × n³). We run it two
// ways on cc-pVDZ-sized dimensions (n=24, H₂O), assert bit-for-bit-close output,
// and report the speedup. This is the real method-level number, not a microbench.
//
// Honest scope: one quarter-transform. The full 4-index transform is 4 of these;
// a production integration keeps each pass a fat GEMM via an inter-pass transpose
// (cheap vs the GEMM). Passes that devolve into tiny (n×1) GEMMs must NOT be
// naively batched — they'd lose to overhead. See the integration note.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const pkgDir = join(here, "../../wasm-gemm/pkg");
const mod = await import(join(pkgDir, "wasm_gemm.js"));
await mod.default({ module_or_path: readFileSync(join(pkgDir, "wasm_gemm_bg.wasm")) });

const n = 24;                 // cc-pVDZ H₂O ≈ 24 basis functions
const N3 = n * n * n;         // columns of the fat GEMM
const rand = (len) => { const a = new Float64Array(len); for (let i = 0; i < len; i++) a[i] = Math.random() * 2 - 1; return a; };
const C = rand(n * n);        // MO coefficients, row-major [μ*n + p]
const eri = rand(n * n * n * n); // (μν|λσ), row-major

// Naive quarter-transform: t1[p, ν, λ, σ] = Σ_μ C[μ,p] · eri[μ, ν, λ, σ]
function naive() {
  const t1 = new Float64Array(n * N3);
  for (let p = 0; p < n; p++)
    for (let col = 0; col < N3; col++) {
      let s = 0;
      for (let mu = 0; mu < n; mu++) s += C[mu * n + p] * eri[mu * N3 + col];
      t1[p * N3 + col] = s;
    }
  return t1;
}

// GEMM quarter-transform: t1 (p × N3) = Cᵀ (p×μ) · eri (μ × N3).
// dgemm_at computes Aᵀ·B with A row-major k×m; here A=C (μ×p, k=μ=n, m=p=n).
function viaGemm() {
  return mod.dgemm_at(n, N3, n, C, eri, new Float64Array(n * N3), 1, 0);
}

// Correctness
const a = naive(), b = viaGemm();
let maxAbs = 0, maxRel = 0;
for (let i = 0; i < a.length; i++) {
  const d = Math.abs(a[i] - b[i]);
  maxAbs = Math.max(maxAbs, d);
  maxRel = Math.max(maxRel, d / (Math.abs(a[i]) + 1e-30));
}

// Timing (median of several runs, a few warmups)
function time(fn, warm = 2, runs = 7) {
  for (let i = 0; i < warm; i++) fn();
  const ts = [];
  for (let i = 0; i < runs; i++) { const t = performance.now(); fn(); ts.push(performance.now() - t); }
  ts.sort((x, y) => x - y);
  return ts[ts.length >> 1];
}
const flops = 2 * n * N3 * n;                 // 2·m·N·k
const tN = time(naive), tG = time(viaGemm);

console.log(`ERI→MO quarter-transform, n=${n} (cc-pVDZ H₂O size), fat GEMM ${n}×${N3}, k=${n}`);
console.log(`  correctness: max|Δ|=${maxAbs.toExponential(2)}  max rel=${maxRel.toExponential(2)}  (bar 1e-10) → ${maxRel < 1e-10 ? "PASS" : "FAIL"}`);
console.log(`  naive-TS   : ${tN.toFixed(2)} ms   ${(flops / tN / 1e6).toFixed(2)} GFLOP/s`);
console.log(`  WASM-SIMD  : ${tG.toFixed(2)} ms   ${(flops / tG / 1e6).toFixed(2)} GFLOP/s`);
console.log(`  speedup    : ${(tN / tG).toFixed(1)}× on the real contraction`);
if (maxRel >= 1e-10) process.exit(1);
