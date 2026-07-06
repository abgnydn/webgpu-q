// wasm-gemm — cache-blocked, SIMD (wasm-simd128 f64x2) double-precision GEMM.
//
// The reusable fast-CPU kernel meant to replace the naive TypeScript
// triple/quad loops that dominate MP2/CCSD tensor contractions. Those
// contractions are BLAS-bound (matrix multiply), so a proper register +
// L1-blocked SIMD dgemm is the lever.
//
// C = alpha * op(A) * B + beta * C, row-major, f64.
//   - gemm_nn:  op(A) = A          (A is m x k)
//   - gemm_tn:  op(A) = A^T        (A is k x m)  — common in ERI->MO / CC
//
// SIMD: core::arch::wasm32 f64x2 (v128, two f64 lanes). Vectorized over the
// contiguous n (column) dimension of row-major B/C. Register-blocked 4 rows
// x 2 columns (8 v128 accumulators); cache-blocked over k (KC) and m (MC) so
// the reused B panel stays warm in L2. Built with -C target-feature=+simd128.
//
// WASM SIMD is 128-bit = 2 x f64, so the arithmetic ceiling is inherently
// half of native AVX2 (4-wide) and a quarter of AVX-512 BLAS. The point here
// is "naive-TS -> WASM-SIMD", not "vs native BLAS".

use core::arch::wasm32::*;
use wasm_bindgen::prelude::*;

// Cache-block tile sizes. KC * n panel of B should sit in L2; MC rows of C
// worth of accumulators cycle over that warm panel.
const MC: usize = 64;
const KC: usize = 256;

#[inline(always)]
fn scale_c(c: &mut [f64], beta: f64) {
    if beta == 0.0 {
        for x in c.iter_mut() {
            *x = 0.0;
        }
    } else if beta != 1.0 {
        for x in c.iter_mut() {
            *x *= beta;
        }
    }
}

/// Micro-kernel: 4 rows x full-n, accumulating the k-slice [pc, pc+kc) of a
/// row-major A (m x k) times B (k x n) into C (m x n). C already holds beta*C.
#[inline(always)]
unsafe fn micro_4xn(
    n: usize,
    kc: usize,
    pc: usize,
    i: usize,
    k: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    alpha: f64,
) {
    let alpha_v = f64x2_splat(alpha);
    let a0 = i * k;
    let a1 = (i + 1) * k;
    let a2 = (i + 2) * k;
    let a3 = (i + 3) * k;
    let c0 = i * n;
    let c1 = (i + 1) * n;
    let c2 = (i + 2) * n;
    let c3 = (i + 3) * n;

    let mut j = 0;
    while j + 2 <= n {
        let mut acc0 = f64x2_splat(0.0);
        let mut acc1 = f64x2_splat(0.0);
        let mut acc2 = f64x2_splat(0.0);
        let mut acc3 = f64x2_splat(0.0);
        let mut p = 0;
        while p < kc {
            let pp = pc + p;
            let bb = v128_load(b.add(pp * n + j) as *const v128);
            acc0 = f64x2_add(acc0, f64x2_mul(f64x2_splat(*a.add(a0 + pp)), bb));
            acc1 = f64x2_add(acc1, f64x2_mul(f64x2_splat(*a.add(a1 + pp)), bb));
            acc2 = f64x2_add(acc2, f64x2_mul(f64x2_splat(*a.add(a2 + pp)), bb));
            acc3 = f64x2_add(acc3, f64x2_mul(f64x2_splat(*a.add(a3 + pp)), bb));
            p += 1;
        }
        let p0 = c.add(c0 + j) as *mut v128;
        let p1 = c.add(c1 + j) as *mut v128;
        let p2 = c.add(c2 + j) as *mut v128;
        let p3 = c.add(c3 + j) as *mut v128;
        v128_store(p0, f64x2_add(v128_load(p0), f64x2_mul(alpha_v, acc0)));
        v128_store(p1, f64x2_add(v128_load(p1), f64x2_mul(alpha_v, acc1)));
        v128_store(p2, f64x2_add(v128_load(p2), f64x2_mul(alpha_v, acc2)));
        v128_store(p3, f64x2_add(v128_load(p3), f64x2_mul(alpha_v, acc3)));
        j += 2;
    }
    // Odd trailing column (scalar), for all 4 rows.
    if j < n {
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let mut s3 = 0.0;
        let mut p = 0;
        while p < kc {
            let pp = pc + p;
            let bv = *b.add(pp * n + j);
            s0 += *a.add(a0 + pp) * bv;
            s1 += *a.add(a1 + pp) * bv;
            s2 += *a.add(a2 + pp) * bv;
            s3 += *a.add(a3 + pp) * bv;
            p += 1;
        }
        *c.add(c0 + j) += alpha * s0;
        *c.add(c1 + j) += alpha * s1;
        *c.add(c2 + j) += alpha * s2;
        *c.add(c3 + j) += alpha * s3;
    }
}

/// Micro-kernel for a single row (row-remainder cleanup). SIMD over n.
#[inline(always)]
unsafe fn micro_1xn(
    n: usize,
    kc: usize,
    pc: usize,
    i: usize,
    k: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    alpha: f64,
) {
    let alpha_v = f64x2_splat(alpha);
    let a0 = i * k;
    let c0 = i * n;
    let mut j = 0;
    while j + 2 <= n {
        let mut acc = f64x2_splat(0.0);
        let mut p = 0;
        while p < kc {
            let pp = pc + p;
            let bb = v128_load(b.add(pp * n + j) as *const v128);
            acc = f64x2_add(acc, f64x2_mul(f64x2_splat(*a.add(a0 + pp)), bb));
            p += 1;
        }
        let p0 = c.add(c0 + j) as *mut v128;
        v128_store(p0, f64x2_add(v128_load(p0), f64x2_mul(alpha_v, acc)));
        j += 2;
    }
    if j < n {
        let mut s = 0.0;
        let mut p = 0;
        while p < kc {
            let pp = pc + p;
            s += *a.add(a0 + pp) * *b.add(pp * n + j);
            p += 1;
        }
        *c.add(c0 + j) += alpha * s;
    }
}

/// C = alpha * A * B + beta * C, all row-major. A: m x k, B: k x n, C: m x n.
fn gemm_nn(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64], alpha: f64, beta: f64) {
    scale_c(c, beta);
    if m == 0 || n == 0 || k == 0 || alpha == 0.0 {
        return;
    }
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();
    let mut pc = 0;
    while pc < k {
        let kc = core::cmp::min(KC, k - pc);
        let mut ic = 0;
        while ic < m {
            let mc = core::cmp::min(MC, m - ic);
            let row_end = ic + mc;
            let mut i = ic;
            while i + 4 <= row_end {
                unsafe { micro_4xn(n, kc, pc, i, k, ap, bp, cp, alpha) };
                i += 4;
            }
            while i < row_end {
                unsafe { micro_1xn(n, kc, pc, i, k, ap, bp, cp, alpha) };
                i += 1;
            }
            ic += MC;
        }
        pc += KC;
    }
}

/// C = alpha * A^T * B + beta * C. A: k x m (row-major), B: k x n, C: m x n.
/// Implemented by packing A^T into a contiguous m x k buffer then reusing the
/// optimized NN kernel. The transpose is O(m*k), negligible vs O(m*n*k).
fn gemm_tn(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64], alpha: f64, beta: f64) {
    let mut at = vec![0.0f64; m * k];
    // a[p*m + i] -> at[i*k + p]
    for p in 0..k {
        let row = p * m;
        for i in 0..m {
            at[i * k + p] = a[row + i];
        }
    }
    gemm_nn(m, n, k, &at, b, c, alpha, beta);
}

// ─────────────────────────── Public wasm-bindgen API ───────────────────────
// Ergonomic (copy-in / copy-out) — this is what the TS chemistry paths call.
// Float64Arrays are marshalled into wasm linear memory by wasm-bindgen; the
// fresh result C is returned as a new Float64Array. Marshalling is O(n^2),
// dominated by the O(n^3) compute at the sizes that matter.

/// C = alpha * A * B + beta * C. Returns the resulting C (length m*n).
#[wasm_bindgen]
pub fn dgemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    b: &[f64],
    c: &[f64],
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let mut out = c.to_vec();
    gemm_nn(m, n, k, a, b, &mut out, alpha, beta);
    out
}

/// C = alpha * A^T * B + beta * C. A is k x m. Returns C (length m*n).
#[wasm_bindgen]
pub fn dgemm_at(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    b: &[f64],
    c: &[f64],
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let mut out = c.to_vec();
    gemm_tn(m, n, k, a, b, &mut out, alpha, beta);
    out
}

// ─────────────────────────── Raw zero-copy API ─────────────────────────────
// For fair kernel benchmarking (and heavy-loop integration): allocate f64
// buffers inside wasm memory once, write A/B/C via a Float64Array view over
// `memory`, and call the raw kernels with pointer offsets — no per-call
// marshalling. These functions have no JS-heap dependencies, so the module
// can be instantiated directly (no wasm-bindgen glue needed) in Node.

/// Allocate `len` f64 slots in wasm memory; returns the byte pointer.
#[wasm_bindgen]
pub fn alloc_f64(len: usize) -> *mut f64 {
    let mut v = vec![0.0f64; len];
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

/// Free a buffer previously returned by `alloc_f64`.
#[wasm_bindgen]
pub fn free_f64(ptr: *mut f64, len: usize) {
    unsafe {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// Raw C = alpha*A*B + beta*C on wasm-memory pointers (element offsets).
#[wasm_bindgen]
pub fn dgemm_raw(
    m: usize,
    n: usize,
    k: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    alpha: f64,
    beta: f64,
) {
    unsafe {
        let a = core::slice::from_raw_parts(a, m * k);
        let b = core::slice::from_raw_parts(b, k * n);
        let c = core::slice::from_raw_parts_mut(c, m * n);
        gemm_nn(m, n, k, a, b, c, alpha, beta);
    }
}

/// Raw C = alpha*A^T*B + beta*C (A is k x m) on wasm-memory pointers.
#[wasm_bindgen]
pub fn dgemm_at_raw(
    m: usize,
    n: usize,
    k: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    alpha: f64,
    beta: f64,
) {
    unsafe {
        let a = core::slice::from_raw_parts(a, k * m);
        let b = core::slice::from_raw_parts(b, k * n);
        let c = core::slice::from_raw_parts_mut(c, m * n);
        gemm_tn(m, n, k, a, b, c, alpha, beta);
    }
}
