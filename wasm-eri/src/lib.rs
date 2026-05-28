// wasm-eri — Rust port of the ERI primitive kernel.
//
// Mirrors src/chemistry/integrals-cg.ts (Boys, eCoefTable, rAuxTable,
// primERIWithPairs, ERI_cg) but compiles to native CPU instructions via
// wasm32-unknown-unknown. Goal: 3-5× per-thread speedup over JIT'd TS
// on the n⁴ ERI build.
//
// First pass: algorithm-identical port, no SIMD intrinsics yet. The
// native-compile-vs-JIT delta alone is usually ~1.5-2× on tight FP
// loops. We layer wasm-simd128 on top in a later pass.

use wasm_bindgen::prelude::*;

const SQRT_PI: f64 = 1.7724538509055160272;

/// Thread-local scratch for build_jk_df_slice. WASM is single-threaded
/// per Worker so static mut is race-free. Resized lazily to fit the
/// largest call so far; never shrinks. See use-site for rationale.
static mut SCRATCH_X: Vec<f64> = Vec::new();
static mut SCRATCH_XT: Vec<f64> = Vec::new();

// ── Per-primitive Cartesian-Gaussian normalization ─────────────
fn double_fact_odd(n: i32) -> f64 {
    if n <= 0 { return 1.0; }
    let mut r: f64 = 1.0;
    for k in 1..=n {
        r *= (2 * k - 1) as f64;
    }
    r
}

fn norm_cg(alpha: f64, ix: i32, iy: i32, iz: i32) -> f64 {
    let l = ix + iy + iz;
    let dfx = double_fact_odd(ix);
    let dfy = double_fact_odd(iy);
    let dfz = double_fact_odd(iz);
    let radial = (2.0 * alpha / std::f64::consts::PI).powf(0.75);
    let four_alpha_l = (4.0 * alpha).powi(l);
    let angular = (four_alpha_l / (dfx * dfy * dfz)).sqrt();
    radial * angular
}

// ── Boys function F_n(t) for n = 0..=nMax ──────────────────────
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let tt = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * (-ax * ax).exp();
    sign * y
}

fn boys0(t: f64) -> f64 {
    if t < 1e-12 { return 1.0 - t / 3.0 + (t * t) / 10.0; }
    let s = t.sqrt();
    0.5 * SQRT_PI / s * erf_approx(s)
}

fn boys_n_taylor(n: i32, t: f64) -> f64 {
    let mut sum = 0.0_f64;
    let mut term = 1.0 / (2.0 * (n as f64) + 1.0);
    for k in 0..200 {
        sum += term;
        let nf = n as f64;
        let kf = k as f64;
        term *= -t / (kf + 1.0) * (2.0 * nf + 2.0 * kf + 1.0) / (2.0 * nf + 2.0 * kf + 3.0);
        if term.abs() < 1e-18 * sum.abs() { break; }
    }
    sum
}

fn boys_all(n_max: i32, t: f64, out: &mut [f64]) {
    if t < 1e-12 {
        for n in 0..=n_max {
            out[n as usize] = 1.0 / (2.0 * (n as f64) + 1.0);
        }
        return;
    }
    let upward_threshold = (2.0 * (n_max as f64) - 1.0) / 2.0;
    if t < upward_threshold {
        for n in 0..=n_max {
            out[n as usize] = boys_n_taylor(n, t);
        }
        return;
    }
    out[0] = boys0(t);
    if n_max == 0 { return; }
    let exp_mt = (-t).exp();
    for n in 1..=n_max {
        let n_us = n as usize;
        out[n_us] = ((2.0 * (n as f64) - 1.0) * out[n_us - 1] - exp_mt) / (2.0 * t);
    }
}

// ── McMurchie-Davidson 1D E-coefficients ───────────────────────
struct PairData {
    p: f64,
    p_xyz: [f64; 3],
    ex: Vec<f64>,
    ey: Vec<f64>,
    ez: Vec<f64>,
    i_max_x: i32, j_max_x: i32,
    i_max_y: i32, j_max_y: i32,
    i_max_z: i32, j_max_z: i32,
}

fn e_coef_table(i_max: i32, j_max: i32, pa: f64, pb: f64, p: f64, k: f64) -> Vec<f64> {
    let t_dim = (i_max + j_max + 1) as usize;
    let i_dim = (i_max + 1) as usize;
    let j_dim = (j_max + 1) as usize;
    let mut tab = vec![0.0_f64; i_dim * j_dim * t_dim];
    let idx = |a: i32, b: i32, t: i32| -> usize {
        ((a as usize) * j_dim + (b as usize)) * t_dim + (t as usize)
    };
    tab[idx(0, 0, 0)] = k;
    let inv2p = 1.0 / (2.0 * p);

    for b in 0..j_max {
        for t in 0..=(b + 1) {
            let left = if t > 0 { tab[idx(0, b, t - 1)] } else { 0.0 };
            let mid = if t <= b { tab[idx(0, b, t)] } else { 0.0 };
            let right = if t + 1 <= b { tab[idx(0, b, t + 1)] } else { 0.0 };
            tab[idx(0, b + 1, t)] = inv2p * left + pb * mid + ((t + 1) as f64) * right;
        }
    }
    for a in 0..i_max {
        for b in 0..=j_max {
            for t in 0..=(a + b + 1) {
                let left = if t > 0 { tab[idx(a, b, t - 1)] } else { 0.0 };
                let mid = if t <= a + b { tab[idx(a, b, t)] } else { 0.0 };
                let right = if t + 1 <= a + b { tab[idx(a, b, t + 1)] } else { 0.0 };
                tab[idx(a + 1, b, t)] = inv2p * left + pa * mid + ((t + 1) as f64) * right;
            }
        }
    }
    tab
}

fn e_at(tab: &[f64], j_max: i32, i_max: i32, i: i32, j: i32, t: i32) -> f64 {
    if t < 0 || t > i + j { return 0.0; }
    let t_dim = (i_max + j_max + 1) as usize;
    let j_dim = (j_max + 1) as usize;
    tab[((i as usize) * j_dim + (j as usize)) * t_dim + (t as usize)]
}

fn build_pair(
    alpha: f64, ax: f64, ay: f64, az: f64, ix: i32, iy: i32, iz: i32,
    beta: f64, bx: f64, by: f64, bz: f64, jx: i32, jy: i32, jz: i32,
) -> PairData {
    let p = alpha + beta;
    let mu = alpha * beta / p;
    let px = (alpha * ax + beta * bx) / p;
    let py = (alpha * ay + beta * by) / p;
    let pz = (alpha * az + beta * bz) / p;
    let kx = (-mu * (ax - bx) * (ax - bx)).exp();
    let ky = (-mu * (ay - by) * (ay - by)).exp();
    let kz = (-mu * (az - bz) * (az - bz)).exp();
    PairData {
        p,
        p_xyz: [px, py, pz],
        ex: e_coef_table(ix, jx, px - ax, px - bx, p, kx),
        ey: e_coef_table(iy, jy, py - ay, py - by, p, ky),
        ez: e_coef_table(iz, jz, pz - az, pz - bz, p, kz),
        i_max_x: ix, j_max_x: jx,
        i_max_y: iy, j_max_y: jy,
        i_max_z: iz, j_max_z: jz,
    }
}

// ── R_{tuv} auxiliary table for ERI ─────────────────────────────
//
// Writes into caller-provided buffers `f_buf` (Boys table) and
// `r_buf` (R-aux table). Buffers are re-used across the 81 prim_eri
// calls per ERI quartet and across 26 M ERI quartets per benzene
// build — avoiding ~2 B malloc/free pairs that the original
// Vec-returning version forced. Both buffers are `clear()` +
// `resize(0.0)` to keep capacity but zero-fill the active prefix.
fn r_aux_table(
    t_max: i32, u_max: i32, v_max: i32,
    p: f64, rx: f64, ry: f64, rz: f64,
    f_buf: &mut Vec<f64>,
    r_buf: &mut Vec<f64>,
) -> (usize, usize, usize) {
    let n_max = t_max + u_max + v_max;
    f_buf.clear();
    f_buf.resize((n_max + 1) as usize, 0.0);
    boys_all(n_max, p * (rx * rx + ry * ry + rz * rz), f_buf);
    let t_dim = (t_max + 1) as usize;
    let u_dim = (u_max + 1) as usize;
    let v_dim = (v_max + 1) as usize;
    let n_dim = (n_max + 1) as usize;
    r_buf.clear();
    r_buf.resize(n_dim * t_dim * u_dim * v_dim, 0.0);
    let r = r_buf.as_mut_slice();
    let idx = |n: i32, t: i32, u: i32, v: i32| -> usize {
        (((n as usize) * t_dim + (t as usize)) * u_dim + (u as usize)) * v_dim + (v as usize)
    };
    // Seed.
    let mut neg2p_n = 1.0;
    for n in 0..=n_max {
        r[idx(n, 0, 0, 0)] = neg2p_n * f_buf[n as usize];
        neg2p_n *= -2.0 * p;
    }
    for n in (0..n_max).rev() {
        for t in 0..=t_max {
            for u in 0..=u_max {
                for v in 0..=v_max {
                    if t == 0 && u == 0 && v == 0 { continue; }
                    let mut acc = 0.0_f64;
                    if v > 0 {
                        acc += rz * r[idx(n + 1, t, u, v - 1)];
                        if v >= 2 { acc += ((v - 1) as f64) * r[idx(n + 1, t, u, v - 2)]; }
                    } else if u > 0 {
                        acc += ry * r[idx(n + 1, t, u - 1, v)];
                        if u >= 2 { acc += ((u - 1) as f64) * r[idx(n + 1, t, u - 2, v)]; }
                    } else {
                        acc += rx * r[idx(n + 1, t - 1, u, v)];
                        if t >= 2 { acc += ((t - 1) as f64) * r[idx(n + 1, t - 2, u, v)]; }
                    }
                    r[idx(n, t, u, v)] = acc;
                }
            }
        }
    }
    (t_dim, u_dim, v_dim)
}

// ── Primitive ERI given pair tables ─────────────────────────────
//
// Hot path. Optimizations relative to the textbook 6-loop:
//   - Hoist invariant index bases (ex1_base, ey1_base, …) out of the
//     hottest inner loops so each level adds at most one mul-add.
//   - Hoist partial-products (xyz1 = ex1·ey1·ez1, etc.) so the inner
//     loop is 2 mults + 1 multiply-by-sign + 1 load + 1 add.
//   - Branch-free `sign = 1 - 2·parity` (kills a div-by-2 branch
//     mispredict pattern at the innermost loop level).
//
// Earlier attempt to prefetch the 6 E-coef slices into [f64; 13]
// stack arrays *regressed* benzene single-thread by 55% — the
// stack zero-fill cost across ~130M primitive-ERI calls swamped the
// vectorization gain. Direct indexing into pd1.ex (heap, cached by
// the surrounding ERI_cg quartet) is faster.
//
// Branch-free hot path also makes the `if eN == 0.0 continue`
// branches go away. For cc-pVDZ those branches rarely fired anyway
// (loop bounds keep us in the recurrence-filled region of the E
// table), so dropping them is correctness-neutral.

#[allow(clippy::too_many_arguments)]
fn prim_eri_with_pairs(
    pd1: &PairData, ix: i32, iy: i32, iz: i32, jx: i32, jy: i32, jz: i32,
    pd2: &PairData, kx: i32, ky: i32, kz: i32, lx: i32, ly: i32, lz: i32,
    f_buf: &mut Vec<f64>,
    r_buf: &mut Vec<f64>,
) -> f64 {
    let p = pd1.p;
    let q = pd2.p;
    let alpha_pair = (p * q) / (p + q);
    let px = pd1.p_xyz[0]; let py = pd1.p_xyz[1]; let pz = pd1.p_xyz[2];
    let qx = pd2.p_xyz[0]; let qy = pd2.p_xyz[1]; let qz = pd2.p_xyz[2];

    let t_max = (ix + jx) as usize;
    let u_max = (iy + jy) as usize;
    let v_max = (iz + jz) as usize;
    let tau_max = (kx + lx) as usize;
    let nu_max = (ky + ly) as usize;
    let phi_max = (kz + lz) as usize;
    let t_dim_sum = (t_max + tau_max) as i32;
    let u_dim_sum = (u_max + nu_max) as i32;
    let v_dim_sum = (v_max + phi_max) as i32;
    let (_, u_dim_r, v_dim_r) = r_aux_table(
        t_dim_sum, u_dim_sum, v_dim_sum, alpha_pair, px - qx, py - qy, pz - qz,
        f_buf, r_buf,
    );
    let rt = r_buf.as_slice();

    let p1_jdim_x = (pd1.j_max_x + 1) as usize;
    let p1_jdim_y = (pd1.j_max_y + 1) as usize;
    let p1_jdim_z = (pd1.j_max_z + 1) as usize;
    let p1_tdim_x = (pd1.i_max_x + pd1.j_max_x + 1) as usize;
    let p1_tdim_y = (pd1.i_max_y + pd1.j_max_y + 1) as usize;
    let p1_tdim_z = (pd1.i_max_z + pd1.j_max_z + 1) as usize;
    let p2_jdim_x = (pd2.j_max_x + 1) as usize;
    let p2_jdim_y = (pd2.j_max_y + 1) as usize;
    let p2_jdim_z = (pd2.j_max_z + 1) as usize;
    let p2_tdim_x = (pd2.i_max_x + pd2.j_max_x + 1) as usize;
    let p2_tdim_y = (pd2.i_max_y + pd2.j_max_y + 1) as usize;
    let p2_tdim_z = (pd2.i_max_z + pd2.j_max_z + 1) as usize;

    let ex1_base = ((ix as usize) * p1_jdim_x + jx as usize) * p1_tdim_x;
    let ey1_base = ((iy as usize) * p1_jdim_y + jy as usize) * p1_tdim_y;
    let ez1_base = ((iz as usize) * p1_jdim_z + jz as usize) * p1_tdim_z;
    let ex2_base = ((kx as usize) * p2_jdim_x + lx as usize) * p2_tdim_x;
    let ey2_base = ((ky as usize) * p2_jdim_y + ly as usize) * p2_tdim_y;
    let ez2_base = ((kz as usize) * p2_jdim_z + lz as usize) * p2_tdim_z;

    let ex1_slice = &pd1.ex[ex1_base..ex1_base + t_max + 1];
    let ey1_slice = &pd1.ey[ey1_base..ey1_base + u_max + 1];
    let ez1_slice = &pd1.ez[ez1_base..ez1_base + v_max + 1];
    let ex2_slice = &pd2.ex[ex2_base..ex2_base + tau_max + 1];
    let ey2_slice = &pd2.ey[ey2_base..ey2_base + nu_max + 1];
    let ez2_slice = &pd2.ez[ez2_base..ez2_base + phi_max + 1];

    let mut sum = 0.0_f64;
    for t in 0..=t_max {
        let ex1_t = ex1_slice[t];
        for u in 0..=u_max {
            let ex_ey_1 = ex1_t * ey1_slice[u];
            for v in 0..=v_max {
                let xyz1 = ex_ey_1 * ez1_slice[v];
                for tau in 0..=tau_max {
                    let xyz1_x2 = xyz1 * ex2_slice[tau];
                    let r_row_base = (t + tau) * u_dim_r * v_dim_r;
                    let tau_parity = tau & 1;
                    for nu in 0..=nu_max {
                        let xyz1_x2_y2 = xyz1_x2 * ey2_slice[nu];
                        let r_phi0 = r_row_base + (u + nu) * v_dim_r + v;
                        let nu_parity_xor_tau = (nu & 1) ^ tau_parity;
                        for phi in 0..=phi_max {
                            let parity = (phi & 1) ^ nu_parity_xor_tau;
                            let sign = 1.0 - 2.0 * (parity as f64);
                            sum += xyz1_x2_y2 * ez2_slice[phi] * sign * rt[r_phi0 + phi];
                        }
                    }
                }
            }
        }
    }
    2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt()) * sum
}

// ── Shell representation (flat for JS interop) ─────────────────
//
// JS passes shells as parallel arrays:
//   nShells: number
//   nPrimsPerShell: Uint32Array of length nShells (sum = total prim count)
//   alphaFlat: Float64Array of all alphas concatenated
//   cFlat: Float64Array of all contraction coefficients concatenated
//   centerFlat: Float64Array, 3 entries per shell, length 3*nShells
//   angularFlat: Int32Array, 3 entries per shell, length 3*nShells

// ── ERI between four shells (extracted for reuse) ──────────────
//
// PairTable stores the precomputed primitive pair tables for ONE
// ordered (a, b) shell pair. Computed once per (a, b), reused across
// the full (μ, ν, λ, σ) sweep: bra uses (mu, nu) → table; ket uses
// (la, si) → table. Without caching, eri_cg_for_shells rebuilds the
// pair tables 8M× on benzene cc-pVDZ (once per ERI call).
struct PairTable {
    pairs: Vec<PairData>,
    coef: Vec<f64>,
    /// Angular momenta of the bra shell (a).
    ax_ang: [i32; 3],
    /// Angular momenta of the ket shell (b).
    bx_ang: [i32; 3],
}

#[allow(clippy::too_many_arguments)]
fn build_pair_table(
    a: usize, b: usize,
    n_prims_per_shell: &[u32],
    prim_offsets: &[u32],
    alpha_flat: &[f64],
    c_flat: &[f64],
    center_flat: &[f64],
    angular_flat: &[i32],
) -> PairTable {
    let n_a = n_prims_per_shell[a] as usize;
    let n_b = n_prims_per_shell[b] as usize;
    let a_off = prim_offsets[a] as usize;
    let b_off = prim_offsets[b] as usize;
    let ax = center_flat[a * 3]; let ay = center_flat[a * 3 + 1]; let az = center_flat[a * 3 + 2];
    let bx = center_flat[b * 3]; let by = center_flat[b * 3 + 1]; let bz = center_flat[b * 3 + 2];
    let aix = angular_flat[a * 3]; let aiy = angular_flat[a * 3 + 1]; let aiz = angular_flat[a * 3 + 2];
    let bix = angular_flat[b * 3]; let biy = angular_flat[b * 3 + 1]; let biz = angular_flat[b * 3 + 2];

    let mut pairs = Vec::with_capacity(n_a * n_b);
    let mut coef = Vec::with_capacity(n_a * n_b);
    for i in 0..n_a {
        let ai = alpha_flat[a_off + i];
        let ci = c_flat[a_off + i] * norm_cg(ai, aix, aiy, aiz);
        for j in 0..n_b {
            let bj = alpha_flat[b_off + j];
            let cj = c_flat[b_off + j] * norm_cg(bj, bix, biy, biz);
            coef.push(ci * cj);
            pairs.push(build_pair(
                ai, ax, ay, az, aix, aiy, aiz,
                bj, bx, by, bz, bix, biy, biz,
            ));
        }
    }
    PairTable {
        pairs, coef,
        ax_ang: [aix, aiy, aiz],
        bx_ang: [bix, biy, biz],
    }
}

/// Compute one (μν|λσ) ERI from precomputed pair tables.
/// `f_buf` / `r_buf` are scratch buffers owned by the caller — reused
/// across all 81 primitive-pair calls within this ERI quartet and
/// across all ERI calls in the parent build loop.
fn eri_from_pair_tables(
    bra: &PairTable, ket: &PairTable,
    f_buf: &mut Vec<f64>, r_buf: &mut Vec<f64>,
) -> f64 {
    let [aix, aiy, aiz] = bra.ax_ang;
    let [bix, biy, biz] = bra.bx_ang;
    let [cix, ciy, ciz] = ket.ax_ang;
    let [dix, diy, diz] = ket.bx_ang;
    let mut s = 0.0_f64;
    for ij in 0..bra.pairs.len() {
        let pd1 = &bra.pairs[ij];
        let c_bra = bra.coef[ij];
        for kl in 0..ket.pairs.len() {
            s += c_bra * ket.coef[kl] * prim_eri_with_pairs(
                pd1, aix, aiy, aiz, bix, biy, biz,
                &ket.pairs[kl], cix, ciy, ciz, dix, diy, diz,
                f_buf, r_buf,
            );
        }
    }
    s
}

/// Precompute all canonical (a, b) pair tables with a ≤ b. Index in
/// the returned Vec is `a * n + b` (only the upper triangle is filled;
/// the lower-triangle slots remain unused, sentinelled with an empty
/// PairTable). Memory: ~4.5 MB on benzene cc-pVDZ (n=120) — tiny vs
/// the n⁴ ERI tensor.
fn precompute_pair_tables(
    n: usize,
    n_prims_per_shell: &[u32],
    prim_offsets: &[u32],
    alpha_flat: &[f64],
    c_flat: &[f64],
    center_flat: &[f64],
    angular_flat: &[i32],
) -> Vec<PairTable> {
    let mut tables = Vec::with_capacity(n * n);
    for _ in 0..(n * n) {
        tables.push(PairTable {
            pairs: Vec::new(), coef: Vec::new(),
            ax_ang: [0, 0, 0], bx_ang: [0, 0, 0],
        });
    }
    for a in 0..n {
        for b in a..n {
            tables[a * n + b] = build_pair_table(
                a, b,
                n_prims_per_shell, prim_offsets,
                alpha_flat, c_flat, center_flat, angular_flat,
            );
        }
    }
    tables
}

#[allow(clippy::too_many_arguments)]
fn eri_cg_for_shells(
    a: usize, b: usize, c: usize, d: usize,
    n_prims_per_shell: &[u32],
    prim_offsets: &[u32],
    alpha_flat: &[f64],
    c_flat: &[f64],
    center_flat: &[f64],
    angular_flat: &[i32],
) -> f64 {
    // Slow-path entry used by the legacy `eri_build` and the diagonal
    // (μν|μν) Q-table calls. Builds bra+ket tables fresh each time —
    // callers in tight loops should use eri_from_pair_tables + a
    // shared `precompute_pair_tables` cache.
    let bra = build_pair_table(
        a, b,
        n_prims_per_shell, prim_offsets,
        alpha_flat, c_flat, center_flat, angular_flat,
    );
    let ket = build_pair_table(
        c, d,
        n_prims_per_shell, prim_offsets,
        alpha_flat, c_flat, center_flat, angular_flat,
    );
    let mut f_buf = Vec::with_capacity(32);
    let mut r_buf = Vec::with_capacity(1024);
    eri_from_pair_tables(&bra, &ket, &mut f_buf, &mut r_buf)
}

#[wasm_bindgen]
pub fn eri_build(
    n_shells: u32,
    n_prims_per_shell: &[u32],
    prim_offsets: &[u32],
    alpha_flat: &[f64],
    c_flat: &[f64],
    center_flat: &[f64],
    angular_flat: &[i32],
    schwarz_tol: f64,
) -> Vec<f64> {
    let n = n_shells as usize;
    let mut eri = vec![0.0_f64; n * n * n * n];

    // Precompute all canonical (a, b) pair tables once. Reused for
    // both Schwarz Q-table (n² calls) and the main 8-fold ERI build
    // (~n⁴/8 calls). Saves O(n²) redundant primitive-pair construction.
    let pair_tables = precompute_pair_tables(
        n, n_prims_per_shell, prim_offsets,
        alpha_flat, c_flat, center_flat, angular_flat,
    );

    // Scratch buffers shared across all r_aux_table calls in this
    // build. Capacities sized for L_max=2 (cc-pVDZ) with slack;
    // higher-L bases will grow them on first use.
    let mut f_buf: Vec<f64> = Vec::with_capacity(32);
    let mut r_buf: Vec<f64> = Vec::with_capacity(1024);

    // Schwarz Q table: (μν|μν) reuses bra=ket=(mu, nu).
    let mut q = vec![0.0_f64; n * n];
    for mu in 0..n {
        for nu in mu..n {
            let t = &pair_tables[mu * n + nu];
            let v = eri_from_pair_tables(t, t, &mut f_buf, &mut r_buf);
            let qv = v.abs().sqrt();
            q[mu * n + nu] = qv;
            q[nu * n + mu] = qv;
        }
    }

    // Build ERI tensor with 8-fold symmetry + Schwarz screening.
    for mu in 0..n {
        for nu in mu..n {
            let q_mu_nu = q[mu * n + nu];
            let bra = &pair_tables[mu * n + nu];
            for la in 0..n {
                for si in la..n {
                    if mu * n + nu > la * n + si { continue; }
                    if q_mu_nu < schwarz_tol || q_mu_nu * q[la * n + si] < schwarz_tol { continue; }
                    let ket = &pair_tables[la * n + si];
                    let v = eri_from_pair_tables(bra, ket, &mut f_buf, &mut r_buf);
                    let n2 = n * n;
                    let n3 = n2 * n;
                    eri[mu * n3 + nu * n2 + la * n + si] = v;
                    eri[nu * n3 + mu * n2 + la * n + si] = v;
                    eri[mu * n3 + nu * n2 + si * n + la] = v;
                    eri[nu * n3 + mu * n2 + si * n + la] = v;
                    eri[la * n3 + si * n2 + mu * n + nu] = v;
                    eri[si * n3 + la * n2 + mu * n + nu] = v;
                    eri[la * n3 + si * n2 + nu * n + mu] = v;
                    eri[si * n3 + la * n2 + nu * n + mu] = v;
                }
            }
        }
    }
    eri
}

/// Compute the canonical ERIs (μν|λσ) for μ ∈ mus only.
///
/// Returns a packed flat array: [μ, ν, λ, σ, v, μ, ν, λ, σ, v, ...] of
/// length 5K where K is the number of unique non-screened ERIs in this
/// slice. Indices are stored as f64 (n ≤ 2^53 fits exactly).
///
/// Q-table is precomputed by the caller (cheap, n² ERIs) so multiple
/// workers can share it via postMessage clone. Schwarz screening is
/// applied identically to the full-build path.
///
/// Worker-side: the caller is responsible for writing the 8 symmetric
/// positions for each (μ, ν, λ, σ, v) into the shared output buffer.
#[wasm_bindgen]
pub fn eri_build_slice(
    mus: &[u32],
    n_shells: u32,
    n_prims_per_shell: &[u32],
    prim_offsets: &[u32],
    alpha_flat: &[f64],
    c_flat: &[f64],
    center_flat: &[f64],
    angular_flat: &[i32],
    q_table: &[f64],
    schwarz_tol: f64,
) -> Vec<f64> {
    let n = n_shells as usize;
    let mut out: Vec<f64> = Vec::with_capacity(mus.len() * n * n);

    // Precompute all (a, b) pair tables once. Reused for every (μ, ν,
    // λ, σ) ERI in this slice — eliminates ~470 M redundant primitive
    // pair builds on benzene cc-pVDZ.
    let pair_tables = precompute_pair_tables(
        n, n_prims_per_shell, prim_offsets,
        alpha_flat, c_flat, center_flat, angular_flat,
    );

    // Scratch buffers for r_aux_table — reused across all primitive
    // ERI calls in this slice. Eliminates ~2 B malloc/free pairs.
    let mut f_buf: Vec<f64> = Vec::with_capacity(32);
    let mut r_buf: Vec<f64> = Vec::with_capacity(1024);

    for &mu_u in mus {
        let mu = mu_u as usize;
        for nu in mu..n {
            let q_mu_nu = q_table[mu * n + nu];
            if q_mu_nu < schwarz_tol { continue; }
            let pair_mn = mu * n + nu;
            let bra = &pair_tables[mu * n + nu];
            for la in 0..n {
                for si in la..n {
                    if pair_mn > la * n + si { continue; }
                    if q_mu_nu * q_table[la * n + si] < schwarz_tol { continue; }
                    let ket = &pair_tables[la * n + si];
                    let v = eri_from_pair_tables(bra, ket, &mut f_buf, &mut r_buf);
                    out.push(mu as f64);
                    out.push(nu as f64);
                    out.push(la as f64);
                    out.push(si as f64);
                    out.push(v);
                }
            }
        }
    }
    out
}

/// Compute just the Schwarz Q table (diagonal-pair ERIs sqrt-abs).
/// Cheap, but JS-side construction is also slow on TS — expose this for
/// workers that want to skip the postMessage clone.
#[wasm_bindgen]
pub fn schwarz_q_table(
    n_shells: u32,
    n_prims_per_shell: &[u32],
    prim_offsets: &[u32],
    alpha_flat: &[f64],
    c_flat: &[f64],
    center_flat: &[f64],
    angular_flat: &[i32],
) -> Vec<f64> {
    let n = n_shells as usize;
    let mut q = vec![0.0_f64; n * n];
    for mu in 0..n {
        for nu in mu..n {
            let v = eri_cg_for_shells(
                mu, nu, mu, nu,
                n_prims_per_shell, prim_offsets,
                alpha_flat, c_flat, center_flat, angular_flat,
            );
            let qv = v.abs().sqrt();
            q[mu * n + nu] = qv;
            q[nu * n + mu] = qv;
        }
    }
    q
}

/// Inner-product helper for the JK σ-loop. Sums
///   Σ_si d[si] · (j[si] − 0.5 · k[si])
/// across 3 length-n slices. Uses wasm-simd128 intrinsics
/// (2-lane f64) when compiled with `-C target-feature=+simd128`;
/// falls back to scalar otherwise.
#[inline(always)]
fn jk_dot(d: &[f64], j: &[f64], k: &[f64], n: usize) -> f64 {
    #[cfg(target_feature = "simd128")]
    {
        use std::arch::wasm32::*;
        unsafe {
            let mut acc = f64x2_splat(0.0);
            let half = f64x2_splat(0.5);
            let dp = d.as_ptr();
            let jp = j.as_ptr();
            let kp = k.as_ptr();
            let mut si = 0;
            while si + 2 <= n {
                let dv = v128_load(dp.add(si) as *const v128);
                let jv = v128_load(jp.add(si) as *const v128);
                let kv = v128_load(kp.add(si) as *const v128);
                let term = f64x2_sub(jv, f64x2_mul(half, kv));
                acc = f64x2_add(acc, f64x2_mul(dv, term));
                si += 2;
            }
            let mut s = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);
            while si < n {
                s += *dp.add(si) * (*jp.add(si) - 0.5 * *kp.add(si));
                si += 1;
            }
            s
        }
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        let mut s = 0.0_f64;
        for si in 0..n {
            s += d[si] * (j[si] - 0.5 * k[si]);
        }
        s
    }
}

/// Single-μ variant of fock_build_slice. Computes G[μ, :] for one μ.
///
///   G[μ, ν] = Σ_{λ, σ} D[λ, σ] · ( (μν|λσ) − ½ (μλ|νσ) )
///
/// `eri_mu_row` is the n³ slab eri[μ, :, :, :], laid out row-major as
/// eri_mu_row[a · n² + b · n + c] = eri[μ, a, b, c].
///
/// Used by the per-μ WASM JK kernel: the worker copies only this μ's
/// n³ slab into WASM linear memory per call (rather than caching the
/// full per-worker slab of |mus|·n³ entries, which doubles browser
/// memory pressure on benzene cc-pVDZ). The copy amortizes against
/// the ~10ms WASM compute per μ at n=120. Inner σ-loop uses the
/// 2-lane f64 SIMD `jk_dot` helper above.
#[wasm_bindgen]
pub fn fock_one_mu_row(
    n: u32,
    eri_mu_row: &[f64],
    d: &[f64],
) -> Vec<f64> {
    let n = n as usize;
    let n2 = n * n;
    let mut g_row = vec![0.0_f64; n];
    for nu in 0..n {
        let mut s = 0.0_f64;
        let nu_n2 = nu * n2;
        let nu_n = nu * n;
        for la in 0..n {
            let d_la_base = la * n;
            let j_base = nu_n2 + la * n;
            let k_base = la * n2 + nu_n;
            // 2-lane f64 SIMD over the σ-inner loop. n is 25-200 for
            // typical bases — long enough to amortize SIMD setup and
            // benefit from the 2× lane throughput.
            let d_slice = &d[d_la_base..d_la_base + n];
            let j_slice = &eri_mu_row[j_base..j_base + n];
            let k_slice = &eri_mu_row[k_base..k_base + n];
            s += jk_dot(d_slice, j_slice, k_slice, n);
        }
        g_row[nu] = s;
    }
    g_row
}

/// Compute the Fock matrix G slice G[μ, ν] for μ ∈ `mus` (a subset of
/// rows), from the AO ERI tensor and the density matrix D.
///
///   G[μ, ν] = Σ_{λ, σ} D[λ, σ] · ( (μν|λσ) − ½ (μλ|νσ) )
///
/// Inputs:
///   - `mus`: which global μ indices this worker owns (length K).
///   - `n`: AO basis size.
///   - `eri_slab`: a flat `K · n³` chunk of the ERI tensor laid out as
///     `eri_slab[k * n³ + a * n² + b * n + c] = eri[mus[k], a, b, c]`,
///     i.e. row-major over (k = local μ index, a, b, c). Caller is
///     responsible for gathering this slab from the full n⁴ ERI before
///     the per-iteration SCF loop and reusing it across iterations.
///   - `d`: the full n × n density matrix, row-major.
///
/// Output: `K · n` Fock entries, `g_slice[k * n + nu] = G[mus[k], nu]`.
/// The caller scatters these back into the full G via the same `mus`
/// indices.
///
/// Why slab-not-tensor: WASM linear memory is separate from the JS
/// SAB, and copying the full n⁴ ERI (1.65 GB on benzene cc-pVDZ) into
/// WASM would dominate the kernel. The slab is 8 × smaller per
/// worker on N=8 and changes never during SCF — copy once, reuse.
#[wasm_bindgen]
pub fn fock_build_slice(
    mus: &[u32],
    n: u32,
    eri_slab: &[f64],
    d: &[f64],
) -> Vec<f64> {
    let n = n as usize;
    let k_count = mus.len();
    let n2 = n * n;
    let n3 = n2 * n;
    let mut g_slice = vec![0.0_f64; k_count * n];
    for k in 0..k_count {
        let slab_base = k * n3;
        for nu in 0..n {
            let mut s = 0.0_f64;
            // J = (μν|λσ): inner stride-1 over σ, outer over λ.
            // K = (μλ|νσ): inner stride-1 over σ, outer over λ.
            // Both share the σ inner loop — combine to reduce
            // memory pressure.
            let nu_n2 = nu * n2;
            let nu_n = nu * n;
            for la in 0..n {
                let d_la_base = la * n;
                let j_base = slab_base + nu_n2 + la * n;
                let k_base = slab_base + la * n2 + nu_n;
                // Fused inner loop: one D load, one J load, one K
                // load, three mults, one add per σ.
                let mut s_inner = 0.0_f64;
                for si in 0..n {
                    let dls = d[d_la_base + si];
                    let j_val = eri_slab[j_base + si];
                    let k_val = eri_slab[k_base + si];
                    s_inner += dls * (j_val - 0.5 * k_val);
                }
                s += s_inner;
            }
            g_slice[k * n + nu] = s;
        }
    }
    g_slice
}

// ─────────────────────────────────────────────────────────────
// 3-index ERI kernel for aux-basis density fitting (RI).
//
// Computes (μν|P) where μ, ν are orbital basis functions (the same
// shell list used in `eri_build_slice`) and P is a single auxiliary
// basis function. The ket has no partner function — the Hermite
// expansion on the ket side reduces to a simpler 1D centered
// recursion (PA = 0). Inner loop is 3-deep instead of 6-deep.
//
// For a single Gaussian at center C with exponent α and angular
// (kx, ky, kz), the Hermite expansion coefficients satisfy:
//   E_0^0    = 1
//   E_t^{k+1} = (1/(2α)) · E_{t-1}^k + (t+1) · E_{t+1}^k
// (the PA-shift term vanishes because the "product center" of a
// single Gaussian is at its own center). One table per (α, k_max);
// since exp(-α r²) factorizes, the same table indexes for x, y, z.
//
// The 3-index integral via McMurchie-Davidson:
//   (μν|P) = (2π^(5/2)) / (p · q · √(p+q))
//              · Σ_{t,u,v}  E^bra_{t,u,v}(p, μν-pair)
//              · E^aux_{kx,ky,kz}_signed(q)
//              · R_{t+kx, u+ky, v+kz}(α', P_bra − C)
// where α' = p · q / (p + q) and the parity sign (-1)^(kx+ky+kz)
// is absorbed into the closed-form factor.

/// E-coefficient table for a single (un-paired) Gaussian with exponent
/// α and angular momentum up to `k_max` in one dimension. Layout:
///   tab[k * dim + t] = E_t for angular momentum k (0 ≤ t ≤ k).
/// Higher-t entries are 0 (recurrence boundary). `dim = k_max + 1`.
fn aux_e_table(alpha: f64, k_max: i32) -> Vec<f64> {
    let dim = (k_max + 1) as usize;
    let mut tab = vec![0.0_f64; dim * dim];
    tab[0] = 1.0;
    let inv2a = 1.0 / (2.0 * alpha);
    for k in 0..k_max {
        for t in 0..=(k + 1) {
            let k_us = k as usize;
            let t_us = t as usize;
            let left = if t > 0 { tab[k_us * dim + (t_us - 1)] } else { 0.0 };
            let right = if (t + 1) <= k { tab[k_us * dim + (t_us + 1)] } else { 0.0 };
            tab[(k_us + 1) * dim + t_us] = inv2a * left + ((t + 1) as f64) * right;
        }
    }
    tab
}

/// Single auxiliary primitive Gaussian: exponent + center + angular.
/// Contraction coefficient and Cartesian normalization are baked into
/// the caller-side `c_norm` constant.
struct AuxPrim {
    alpha: f64,
    cx: f64, cy: f64, cz: f64,
    kx: i32, ky: i32, kz: i32,
}

/// Primitive 3-index ERI: (bra_pair | aux_prim).
/// Bra side reuses PairData. Aux side is a single-function Hermite
/// expansion. The `aux_ex/ey/ez` are 1D E-coef tables sized to k_max
/// for the aux exponent; only the (kx, ky, kz) row of each is read.
#[allow(clippy::too_many_arguments)]
fn prim_eri_3idx(
    pd_bra: &PairData, ix: i32, iy: i32, iz: i32, jx: i32, jy: i32, jz: i32,
    aux: &AuxPrim,
    aux_e: &[f64], aux_k_dim: usize,
    f_buf: &mut Vec<f64>, r_buf: &mut Vec<f64>,
) -> f64 {
    let p = pd_bra.p;
    let q = aux.alpha;
    let alpha_pair = (p * q) / (p + q);
    let [px, py, pz] = pd_bra.p_xyz;
    let cx = aux.cx; let cy = aux.cy; let cz = aux.cz;

    let t_max = (ix + jx) as usize;
    let u_max = (iy + jy) as usize;
    let v_max = (iz + jz) as usize;
    let kxu = aux.kx as usize;
    let kyu = aux.ky as usize;
    let kzu = aux.kz as usize;

    // R-aux dims account for both bra (t, u, v) and aux (kx, ky, kz)
    // contributions: index range is t + kx ∈ [0, t_max + kx].
    let t_dim_total = (t_max + kxu) as i32;
    let u_dim_total = (u_max + kyu) as i32;
    let v_dim_total = (v_max + kzu) as i32;
    let (_, u_dim_r, v_dim_r) = r_aux_table(
        t_dim_total, u_dim_total, v_dim_total, alpha_pair, px - cx, py - cy, pz - cz,
        f_buf, r_buf,
    );
    let rt = r_buf.as_slice();

    // Bra E-coef slices (same layout as 4-index path).
    let p_jdim_x = (pd_bra.j_max_x + 1) as usize;
    let p_jdim_y = (pd_bra.j_max_y + 1) as usize;
    let p_jdim_z = (pd_bra.j_max_z + 1) as usize;
    let p_tdim_x = (pd_bra.i_max_x + pd_bra.j_max_x + 1) as usize;
    let p_tdim_y = (pd_bra.i_max_y + pd_bra.j_max_y + 1) as usize;
    let p_tdim_z = (pd_bra.i_max_z + pd_bra.j_max_z + 1) as usize;
    let ex_base = ((ix as usize) * p_jdim_x + jx as usize) * p_tdim_x;
    let ey_base = ((iy as usize) * p_jdim_y + jy as usize) * p_tdim_y;
    let ez_base = ((iz as usize) * p_jdim_z + jz as usize) * p_tdim_z;
    let bra_ex = &pd_bra.ex[ex_base..ex_base + t_max + 1];
    let bra_ey = &pd_bra.ey[ey_base..ey_base + u_max + 1];
    let bra_ez = &pd_bra.ez[ez_base..ez_base + v_max + 1];

    // Aux E-coef rows: aux_e[k * aux_k_dim + tau] = E_tau for angular k.
    // We need to iterate (tau, nu, phi) ∈ [0..=kx] × [0..=ky] × [0..=kz]
    // since the aux Hermite expansion has nonzero coefs across the
    // whole 0..=k range (not just t = k as the 4-index ket would).
    let aux_ex_row = &aux_e[kxu * aux_k_dim..kxu * aux_k_dim + kxu + 1];
    let aux_ey_row = &aux_e[kyu * aux_k_dim..kyu * aux_k_dim + kyu + 1];
    let aux_ez_row = &aux_e[kzu * aux_k_dim..kzu * aux_k_dim + kzu + 1];

    // 6-deep loop reduced to 3 + 3 (bra t/u/v, aux tau/nu/phi).
    let mut sum = 0.0_f64;
    for t in 0..=t_max {
        let ex = bra_ex[t];
        for u in 0..=u_max {
            let exy = ex * bra_ey[u];
            for v in 0..=v_max {
                let exyz = exy * bra_ez[v];
                for tau in 0..=kxu {
                    let aex = aux_ex_row[tau];
                    let tau_par = tau & 1;
                    for nuv in 0..=kyu {
                        let aey = aex * aux_ey_row[nuv];
                        let nu_par_xor_tau = (nuv & 1) ^ tau_par;
                        let r_row_base = (t + tau) * u_dim_r * v_dim_r;
                        let r_phi0 = r_row_base + (u + nuv) * v_dim_r + v;
                        for phi in 0..=kzu {
                            let parity = (phi & 1) ^ nu_par_xor_tau;
                            let sign = 1.0 - 2.0 * (parity as f64);
                            sum += exyz * aey * aux_ez_row[phi] * sign * rt[r_phi0 + phi];
                        }
                    }
                }
            }
        }
    }
    2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt()) * sum
}

/// Compute the contracted 3-index ERI (μν|P) between a bra PairTable
/// and an auxiliary contracted shell. Sums over all primitive
/// combinations.
#[allow(clippy::too_many_arguments)]
fn eri_cg_3idx(
    bra: &PairTable,
    aux_alpha: &[f64], aux_c: &[f64],
    aux_center: [f64; 3], aux_angular: [i32; 3],
    f_buf: &mut Vec<f64>, r_buf: &mut Vec<f64>,
) -> f64 {
    let [aix, aiy, aiz] = bra.ax_ang;
    let [bix, biy, biz] = bra.bx_ang;
    let [kx, ky, kz] = aux_angular;
    let n_aux_prim = aux_alpha.len();
    let k_max = kx.max(ky).max(kz);
    let k_dim = (k_max + 1) as usize;

    let mut s = 0.0_f64;
    for ka in 0..n_aux_prim {
        let ap = aux_alpha[ka];
        let cp = aux_c[ka] * norm_cg(ap, kx, ky, kz);
        let aux_e = aux_e_table(ap, k_max);
        let aux_prim = AuxPrim {
            alpha: ap,
            cx: aux_center[0], cy: aux_center[1], cz: aux_center[2],
            kx, ky, kz,
        };
        for ij in 0..bra.pairs.len() {
            let pd_bra = &bra.pairs[ij];
            let c_bra = bra.coef[ij];
            s += c_bra * cp * prim_eri_3idx(
                pd_bra, aix, aiy, aiz, bix, biy, biz,
                &aux_prim, &aux_e, k_dim,
                f_buf, r_buf,
            );
        }
    }
    s
}

/// Build the 3-index AO ERI tensor V[μν, P] = (μν|P) for aux-basis
/// density fitting. Returns a flat `n² · n_aux` Float64Array with
/// layout V[(μ · n + ν) · n_aux + P].
///
/// For prototype, the aux basis is just another shell list; production
/// would pass a separate jkfit/auxiliary set. Currently the orbital
/// and aux shell representations are identical (CGShell-shaped).
///
/// Cost: n_bra_pairs · n_aux · prim_eri_3idx (much cheaper than
/// 4-index since the inner loop is 3+3 deep instead of 6 deep).
#[wasm_bindgen]
pub fn eri_3idx_build(
    n_orbital: u32,
    n_aux: u32,
    /* orbital shells */
    n_prims_per_orb: &[u32],
    prim_offsets_orb: &[u32],
    alpha_orb: &[f64],
    c_orb: &[f64],
    center_orb: &[f64],
    angular_orb: &[i32],
    /* aux shells */
    n_prims_per_aux: &[u32],
    prim_offsets_aux: &[u32],
    alpha_aux: &[f64],
    c_aux: &[f64],
    center_aux: &[f64],
    angular_aux: &[i32],
) -> Vec<f64> {
    let n = n_orbital as usize;
    let nx = n_aux as usize;
    let mut v = vec![0.0_f64; n * n * nx];

    let pair_tables = precompute_pair_tables(
        n, n_prims_per_orb, prim_offsets_orb,
        alpha_orb, c_orb, center_orb, angular_orb,
    );

    let mut f_buf: Vec<f64> = Vec::with_capacity(32);
    let mut r_buf: Vec<f64> = Vec::with_capacity(1024);

    for mu in 0..n {
        for nu in mu..n {
            let bra = &pair_tables[mu * n + nu];
            for px_idx in 0..nx {
                let na = n_prims_per_aux[px_idx] as usize;
                let off = prim_offsets_aux[px_idx] as usize;
                let alpha_slice = &alpha_aux[off..off + na];
                let c_slice = &c_aux[off..off + na];
                let center = [
                    center_aux[px_idx * 3],
                    center_aux[px_idx * 3 + 1],
                    center_aux[px_idx * 3 + 2],
                ];
                let angular = [
                    angular_aux[px_idx * 3],
                    angular_aux[px_idx * 3 + 1],
                    angular_aux[px_idx * 3 + 2],
                ];
                let val = eri_cg_3idx(
                    bra,
                    alpha_slice, c_slice,
                    center, angular,
                    &mut f_buf, &mut r_buf,
                );
                v[(mu * n + nu) * nx + px_idx] = val;
                if mu != nu {
                    v[(nu * n + mu) * nx + px_idx] = val;
                }
            }
        }
    }
    v
}

/// Worker-parallel slice of form_b_tensor: computes B for μ rows
/// in `mus`. Returns flat (|mus| · n · n_aux) packed contiguously
/// by (k, ν, P) where k is the local μ index.
///
/// Caller scatters the returned slice into a shared full-B SAB at
/// the correct μ-offsets. Each worker owns disjoint μ rows so no
/// atomics needed.
#[wasm_bindgen]
pub fn form_b_tensor_slice(
    mus: &[u32],
    n: u32,
    n_aux: u32,
    v: &[f64],
    u: &[f64],
    inv_sqrt_lam: &[f64],
) -> Vec<f64> {
    let n = n as usize;
    let n_aux = n_aux as usize;
    let k_mus = mus.len();
    let mut b = vec![0.0_f64; k_mus * n * n_aux];
    let mut t = vec![0.0_f64; n_aux];

    for (k, &mu_u) in mus.iter().enumerate() {
        let mu = mu_u as usize;
        for nu in 0..n {
            let v_base = (mu * n + nu) * n_aux;
            // T[i] = (V[μν, :] · U[:, i]) · λ_i^(-1/2)
            for i in 0..n_aux {
                let isl = inv_sqrt_lam[i];
                if isl == 0.0 {
                    t[i] = 0.0;
                    continue;
                }
                let u_col_base = i * n_aux;
                let mut s = 0.0_f64;
                #[cfg(target_feature = "simd128")]
                unsafe {
                    use std::arch::wasm32::*;
                    let mut acc = f64x2_splat(0.0);
                    let vp = v.as_ptr().add(v_base);
                    let up = u.as_ptr().add(u_col_base);
                    let mut q = 0;
                    while q + 2 <= n_aux {
                        let vv = v128_load(vp.add(q) as *const v128);
                        let uv = v128_load(up.add(q) as *const v128);
                        acc = f64x2_add(acc, f64x2_mul(vv, uv));
                        q += 2;
                    }
                    s = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);
                    while q < n_aux {
                        s += *vp.add(q) * *up.add(q);
                        q += 1;
                    }
                }
                #[cfg(not(target_feature = "simd128"))]
                {
                    for q in 0..n_aux {
                        s += v[v_base + q] * u[u_col_base + q];
                    }
                }
                t[i] = s * isl;
            }
            // B[k, ν, P] = Σ_i T[i] · U[P, i]
            let b_base = (k * n + nu) * n_aux;
            for p in 0..n_aux { b[b_base + p] = 0.0; }
            for i in 0..n_aux {
                let ti = t[i];
                if ti == 0.0 { continue; }
                let u_row_base = i * n_aux;
                #[cfg(target_feature = "simd128")]
                unsafe {
                    use std::arch::wasm32::*;
                    let ti_v = f64x2_splat(ti);
                    let up = u.as_ptr().add(u_row_base);
                    let bp = b.as_mut_ptr().add(b_base);
                    let mut p = 0;
                    while p + 2 <= n_aux {
                        let uv = v128_load(up.add(p) as *const v128);
                        let bv = v128_load(bp.add(p) as *const v128);
                        let res = f64x2_add(bv, f64x2_mul(ti_v, uv));
                        v128_store(bp.add(p) as *mut v128, res);
                        p += 2;
                    }
                    while p < n_aux {
                        *bp.add(p) += ti * *up.add(p);
                        p += 1;
                    }
                }
                #[cfg(not(target_feature = "simd128"))]
                {
                    for p in 0..n_aux {
                        b[b_base + p] += ti * u[u_row_base + p];
                    }
                }
            }
        }
    }
    b
}

/// Form the density-fitting B tensor: B[μν, P] = Σ_Q V[μν, Q] · M⁻¹⸍²[Q, P]
/// where M⁻¹⸍² = U · diag(λ⁻¹⸍²) · Uᵀ from the eigendecomposition of M.
///
/// Inputs:
///   - `v`: 3-index tensor, shape (n·n, n_aux), row-major over (μν, Q)
///   - `u`: eigenvector matrix from `eigsymmetric`, column-major:
///          u[i·n_aux + Q] = U[Q, i] (Q-th component of i-th eigenvector)
///   - `inv_sqrt_lam`: 1/√λ_i for each eigenvalue (0 for regularized-out modes)
///   - n, n_aux: dimensions
///
/// Returns flat B of length n·n·n_aux, row-major over (μν, P).
///
/// Hot path: O(n² · n_aux²). For benzene cc-pVDZ (n=120, n_aux≈400)
/// that's ~4.6 B FLOPs — was ~5 s in TypeScript even with cache-
/// friendly loop reorder. WASM with f64x2 SIMD on the inner P loop
/// (length n_aux=400, perfect for vectorization) drops it to ~1 s.
///
/// Note: a Cholesky-back-sub variant (form_b_from_cholesky) was tried
/// in Rust+WASM and lost to TS (91 s vs 40 s on naphthalene).
/// wasm-bindgen copies the 312 MB V tensor per call, and the
/// pivot-indirect access into V/L is cache-unfriendly. See
/// `src/chemistry/df-aux.ts` for the active TS path.
#[wasm_bindgen]
pub fn form_b_tensor(
    n: u32,
    n_aux: u32,
    v: &[f64],
    u: &[f64],
    inv_sqrt_lam: &[f64],
) -> Vec<f64> {
    let n = n as usize;
    let n_aux = n_aux as usize;
    let mut b = vec![0.0_f64; n * n * n_aux];
    // Scratch T[i] for one (μν) row at a time.
    let mut t = vec![0.0_f64; n_aux];

    for mu in 0..n {
        for nu in 0..n {
            let v_base = (mu * n + nu) * n_aux;
            // Step 1: T[i] = (V[μν, :] · U[:, i]) · λ_i^(-1/2)
            for i in 0..n_aux {
                let isl = inv_sqrt_lam[i];
                if isl == 0.0 {
                    t[i] = 0.0;
                    continue;
                }
                let u_col_base = i * n_aux;
                let mut s = 0.0_f64;
                // Inner Q-loop: contiguous reads from V[μν, :] and U[:, i].
                // SIMD: f64x2 pairs of multiply-accumulate.
                #[cfg(target_feature = "simd128")]
                unsafe {
                    use std::arch::wasm32::*;
                    let mut acc = f64x2_splat(0.0);
                    let vp = v.as_ptr().add(v_base);
                    let up = u.as_ptr().add(u_col_base);
                    let mut q = 0;
                    while q + 2 <= n_aux {
                        let vv = v128_load(vp.add(q) as *const v128);
                        let uv = v128_load(up.add(q) as *const v128);
                        acc = f64x2_add(acc, f64x2_mul(vv, uv));
                        q += 2;
                    }
                    s = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);
                    while q < n_aux {
                        s += *vp.add(q) * *up.add(q);
                        q += 1;
                    }
                }
                #[cfg(not(target_feature = "simd128"))]
                {
                    for q in 0..n_aux {
                        s += v[v_base + q] * u[u_col_base + q];
                    }
                }
                t[i] = s * isl;
            }
            // Step 2: B[μν, P] = Σ_i T[i] · U[P, i] = Σ_i T[i] · u[i·n_aux + P]
            // Outer i, inner P. Zero the row first, then accumulate.
            let b_base = v_base;
            for p in 0..n_aux { b[b_base + p] = 0.0; }
            for i in 0..n_aux {
                let ti = t[i];
                if ti == 0.0 { continue; }
                let u_row_base = i * n_aux;
                #[cfg(target_feature = "simd128")]
                unsafe {
                    use std::arch::wasm32::*;
                    let ti_v = f64x2_splat(ti);
                    let up = u.as_ptr().add(u_row_base);
                    let bp = b.as_mut_ptr().add(b_base);
                    let mut p = 0;
                    while p + 2 <= n_aux {
                        let uv = v128_load(up.add(p) as *const v128);
                        let bv = v128_load(bp.add(p) as *const v128);
                        let res = f64x2_add(bv, f64x2_mul(ti_v, uv));
                        v128_store(bp.add(p) as *mut v128, res);
                        p += 2;
                    }
                    while p < n_aux {
                        *bp.add(p) += ti * *up.add(p);
                        p += 1;
                    }
                }
                #[cfg(not(target_feature = "simd128"))]
                {
                    for p in 0..n_aux {
                        b[b_base + p] += ti * u[u_row_base + p];
                    }
                }
            }
        }
    }
    b
}

/// Worker-parallel 3-index ERI build. Computes (μν|P) for μ ∈ `mus`
/// only and returns a packed flat array of length 4·K:
///   [μ, ν, P, value, μ, ν, P, value, …]
/// Indices stored as f64 (n ≤ 2^53 fits exactly). Caller scatters the
/// values into the n²·n_aux V-tensor at both (μν, P) and (νμ, P)
/// positions.
///
/// Cost per worker: O(|mus| · n · n_aux · prim_eri_3idx). For benzene
/// cc-pVDZ with n=120, n_aux≈400, |mus|=15: ~720K 3-index ERIs ≈ ~3 s
/// on M2 Pro. Parallel across 8 workers: <1 s wall.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn eri_3idx_build_slice(
    mus: &[u32],
    n_orbital: u32,
    n_aux: u32,
    n_prims_per_orb: &[u32],
    prim_offsets_orb: &[u32],
    alpha_orb: &[f64],
    c_orb: &[f64],
    center_orb: &[f64],
    angular_orb: &[i32],
    n_prims_per_aux: &[u32],
    prim_offsets_aux: &[u32],
    alpha_aux: &[f64],
    c_aux: &[f64],
    center_aux: &[f64],
    angular_aux: &[i32],
) -> Vec<f64> {
    let n = n_orbital as usize;
    let nx = n_aux as usize;
    let mut out: Vec<f64> = Vec::with_capacity(mus.len() * n * nx * 4);

    let pair_tables = precompute_pair_tables(
        n, n_prims_per_orb, prim_offsets_orb,
        alpha_orb, c_orb, center_orb, angular_orb,
    );

    let mut f_buf: Vec<f64> = Vec::with_capacity(32);
    let mut r_buf: Vec<f64> = Vec::with_capacity(1024);

    for &mu_u in mus {
        let mu = mu_u as usize;
        for nu in mu..n {
            let bra = &pair_tables[mu * n + nu];
            for px_idx in 0..nx {
                let na = n_prims_per_aux[px_idx] as usize;
                let off = prim_offsets_aux[px_idx] as usize;
                let alpha_slice = &alpha_aux[off..off + na];
                let c_slice = &c_aux[off..off + na];
                let center = [
                    center_aux[px_idx * 3],
                    center_aux[px_idx * 3 + 1],
                    center_aux[px_idx * 3 + 2],
                ];
                let angular = [
                    angular_aux[px_idx * 3],
                    angular_aux[px_idx * 3 + 1],
                    angular_aux[px_idx * 3 + 2],
                ];
                let val = eri_cg_3idx(
                    bra,
                    alpha_slice, c_slice,
                    center, angular,
                    &mut f_buf, &mut r_buf,
                );
                out.push(mu as f64);
                out.push(nu as f64);
                out.push(px_idx as f64);
                out.push(val);
            }
        }
    }
    out
}

/// Primitive 2-index ERI: (P|Q) where both P and Q are single
/// auxiliary Gaussians. Symmetric in (P, Q). Both Hermite expansions
/// are single-function (no PA-shift) — the inner loop is 3+3 deep,
/// same total complexity as `prim_eri_3idx` but without bra E-coef
/// fetches.
#[allow(clippy::too_many_arguments)]
fn prim_eri_2idx(
    p_alpha: f64, p_cx: f64, p_cy: f64, p_cz: f64, pkx: i32, pky: i32, pkz: i32,
    q_alpha: f64, q_cx: f64, q_cy: f64, q_cz: f64, qkx: i32, qky: i32, qkz: i32,
    p_e: &[f64], p_k_dim: usize,
    q_e: &[f64], q_k_dim: usize,
    f_buf: &mut Vec<f64>, r_buf: &mut Vec<f64>,
) -> f64 {
    let p = p_alpha;
    let q = q_alpha;
    let alpha_pair = (p * q) / (p + q);

    let pkxu = pkx as usize; let pkyu = pky as usize; let pkzu = pkz as usize;
    let qkxu = qkx as usize; let qkyu = qky as usize; let qkzu = qkz as usize;

    let t_dim_total = (pkxu + qkxu) as i32;
    let u_dim_total = (pkyu + qkyu) as i32;
    let v_dim_total = (pkzu + qkzu) as i32;
    let (_, u_dim_r, v_dim_r) = r_aux_table(
        t_dim_total, u_dim_total, v_dim_total, alpha_pair,
        p_cx - q_cx, p_cy - q_cy, p_cz - q_cz,
        f_buf, r_buf,
    );
    let rt = r_buf.as_slice();

    let p_ex_row = &p_e[pkxu * p_k_dim..pkxu * p_k_dim + pkxu + 1];
    let p_ey_row = &p_e[pkyu * p_k_dim..pkyu * p_k_dim + pkyu + 1];
    let p_ez_row = &p_e[pkzu * p_k_dim..pkzu * p_k_dim + pkzu + 1];
    let q_ex_row = &q_e[qkxu * q_k_dim..qkxu * q_k_dim + qkxu + 1];
    let q_ey_row = &q_e[qkyu * q_k_dim..qkyu * q_k_dim + qkyu + 1];
    let q_ez_row = &q_e[qkzu * q_k_dim..qkzu * q_k_dim + qkzu + 1];

    let mut sum = 0.0_f64;
    for t in 0..=pkxu {
        let pex = p_ex_row[t];
        for u in 0..=pkyu {
            let pey = pex * p_ey_row[u];
            for v in 0..=pkzu {
                let pez = pey * p_ez_row[v];
                for tau in 0..=qkxu {
                    let aex = q_ex_row[tau];
                    let tau_par = tau & 1;
                    for nuv in 0..=qkyu {
                        let aey = aex * q_ey_row[nuv];
                        let nu_par_xor_tau = (nuv & 1) ^ tau_par;
                        let r_phi0 = (t + tau) * u_dim_r * v_dim_r + (u + nuv) * v_dim_r + v;
                        for phi in 0..=qkzu {
                            let parity = (phi & 1) ^ nu_par_xor_tau;
                            let sign = 1.0 - 2.0 * (parity as f64);
                            sum += pez * aey * q_ez_row[phi] * sign * rt[r_phi0 + phi];
                        }
                    }
                }
            }
        }
    }
    2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt()) * sum
}

/// Build the 2-index AO ERI metric M[P, Q] = (P|Q) on the auxiliary
/// basis. Symmetric n_aux × n_aux matrix.
///
/// This is the Coulomb metric for density fitting: B = V · M^(-1/2)
/// where V is the 3-index tensor from `eri_3idx_build`.
#[wasm_bindgen]
pub fn eri_2idx_build(
    n_aux: u32,
    n_prims_per_aux: &[u32],
    prim_offsets_aux: &[u32],
    alpha_aux: &[f64],
    c_aux: &[f64],
    center_aux: &[f64],
    angular_aux: &[i32],
) -> Vec<f64> {
    let nx = n_aux as usize;
    let mut m = vec![0.0_f64; nx * nx];

    let mut f_buf: Vec<f64> = Vec::with_capacity(32);
    let mut r_buf: Vec<f64> = Vec::with_capacity(1024);

    // Precompute aux E-coef tables (per primitive) and norms.
    // Each aux shell has up to ~3 primitives; each primitive has its
    // own α → its own Hermite table.
    let total_prims: usize = n_prims_per_aux.iter().map(|&x| x as usize).sum();
    let mut aux_max_k_per_prim: Vec<i32> = Vec::with_capacity(total_prims);
    let mut aux_e_per_prim: Vec<Vec<f64>> = Vec::with_capacity(total_prims);
    let mut aux_norm_per_prim: Vec<f64> = Vec::with_capacity(total_prims);
    for sh in 0..nx {
        let na = n_prims_per_aux[sh] as usize;
        let off = prim_offsets_aux[sh] as usize;
        let kx = angular_aux[sh * 3];
        let ky = angular_aux[sh * 3 + 1];
        let kz = angular_aux[sh * 3 + 2];
        let k_max = kx.max(ky).max(kz);
        for p in 0..na {
            let ap = alpha_aux[off + p];
            aux_max_k_per_prim.push(k_max);
            aux_e_per_prim.push(aux_e_table(ap, k_max));
            aux_norm_per_prim.push(norm_cg(ap, kx, ky, kz));
        }
    }

    for p_idx in 0..nx {
        let pkx = angular_aux[p_idx * 3];
        let pky = angular_aux[p_idx * 3 + 1];
        let pkz = angular_aux[p_idx * 3 + 2];
        let na_p = n_prims_per_aux[p_idx] as usize;
        let off_p = prim_offsets_aux[p_idx] as usize;
        let p_cx = center_aux[p_idx * 3];
        let p_cy = center_aux[p_idx * 3 + 1];
        let p_cz = center_aux[p_idx * 3 + 2];
        // Aggregate primitive index into the global per-prim arrays.
        let p_prim_global_base: usize =
            (0..p_idx).map(|x| n_prims_per_aux[x] as usize).sum();
        for q_idx in p_idx..nx {
            let qkx = angular_aux[q_idx * 3];
            let qky = angular_aux[q_idx * 3 + 1];
            let qkz = angular_aux[q_idx * 3 + 2];
            let na_q = n_prims_per_aux[q_idx] as usize;
            let off_q = prim_offsets_aux[q_idx] as usize;
            let q_cx = center_aux[q_idx * 3];
            let q_cy = center_aux[q_idx * 3 + 1];
            let q_cz = center_aux[q_idx * 3 + 2];
            let q_prim_global_base: usize =
                (0..q_idx).map(|x| n_prims_per_aux[x] as usize).sum();

            let mut s = 0.0_f64;
            for ip in 0..na_p {
                let p_alpha = alpha_aux[off_p + ip];
                let p_c_norm = c_aux[off_p + ip] * aux_norm_per_prim[p_prim_global_base + ip];
                let p_e = &aux_e_per_prim[p_prim_global_base + ip];
                let p_k_dim = (aux_max_k_per_prim[p_prim_global_base + ip] + 1) as usize;
                for iq in 0..na_q {
                    let q_alpha = alpha_aux[off_q + iq];
                    let q_c_norm = c_aux[off_q + iq] * aux_norm_per_prim[q_prim_global_base + iq];
                    let q_e = &aux_e_per_prim[q_prim_global_base + iq];
                    let q_k_dim = (aux_max_k_per_prim[q_prim_global_base + iq] + 1) as usize;
                    s += p_c_norm * q_c_norm * prim_eri_2idx(
                        p_alpha, p_cx, p_cy, p_cz, pkx, pky, pkz,
                        q_alpha, q_cx, q_cy, q_cz, qkx, qky, qkz,
                        p_e, p_k_dim, q_e, q_k_dim,
                        &mut f_buf, &mut r_buf,
                    );
                }
            }
            m[p_idx * nx + q_idx] = s;
            if p_idx != q_idx {
                m[q_idx * nx + p_idx] = s;
            }
        }
    }
    m
}

/// DF Fock build: given B-tensor B[μν, P] and density D[μν], compute
///   J[μν] = Σ_P B[μν, P] · γ[P],  γ[P] = Σ_λσ B[λσ, P] · D[λσ]
///   K[μν] = Σ_P Σ_σ X[P, μ, σ] · B[νσ, P],
///           X[P, μ, σ] = Σ_λ B[μλ, P] · D[λ, σ]
///
/// Returns packed [J(0..N), K(0..N)] of length 2N where N = n².
/// Layout: out[0..N] = J, out[N..2N] = K.
///
/// Hot path: the K inner pass is O(n²·n_aux·n) = ~1 B FLOPs for
/// benzene cc-pVDZ at n_aux≈660. SIMD on the innermost contiguous
/// loops; the (la, si) and (P, si) inner pairs are both length-n
/// f64 reductions perfect for f64x2 wasm-simd128.
/// Worker-parallel slice of build_jk_df: computes J[μ, :] and K[μ, :]
/// for μ ∈ `mus`. Returns packed [J_slice; K_slice] of length 2·|mus|·n.
/// Each worker handles a disjoint μ-row range; combine on main thread.
///
/// The full J and K depend on the same γ[P] and X[P, μ', σ] precomputes,
/// but each row μ only needs:
///   J[μ, ν] = Σ_P B[μν, P] · γ[P]                    — local to row μ
///   K[μ, ν] = Σ_P Σ_σ X[P, μ, σ] · B[νσ, P]           — needs X[*, μ, *]
///
/// γ is shared (computed once per call, small: n_aux entries).
/// X[*, μ, *] for μ ∈ mus is the per-worker partition (n_aux · n entries
/// per μ). Build that per-worker locally.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn build_jk_df_slice(
    mus: &[u32],
    n: u32,
    n_aux: u32,
    b: &[f64],
    d: &[f64],
    gamma: &[f64],
) -> Vec<f64> {
    let n = n as usize;
    let n_aux = n_aux as usize;
    let k_mus = mus.len();
    let mut out = vec![0.0_f64; 2 * k_mus * n];
    // out[0..k_mus*n] = J slice; out[k_mus*n..] = K slice.

    // J[μν] = Σ_P B[μν, P] · γ[P]   (μ ∈ mus, ν ∈ 0..n)
    for (k, &mu_u) in mus.iter().enumerate() {
        let mu = mu_u as usize;
        for nu in 0..n {
            let base = (mu * n + nu) * n_aux;
            let mut s = 0.0_f64;
            for pp in 0..n_aux {
                s += b[base + pp] * gamma[pp];
            }
            out[k * n + nu] = s;
        }
    }

    // X[P, μ, σ] for μ ∈ mus only (local k_mus·n_aux·n footprint).
    // Reorder loops to (μ, λ, P, σ) for cache-friendly σ inner.
    //
    // Reuse thread-local scratch across SCF iters: naphthalene n=190
    // n_aux=1080 has each worker's X at ~38 MB. Allocating + freeing
    // 76 MB (X + X_T) per call × 13 iters × 8 workers = ~8 GB of
    // alloc/free per SCF. Reuse pays ~2 s on a typical SCF.
    //
    // WASM is single-threaded per Worker, so the static-mut Vec is
    // race-free. We zero only the used prefix.
    let x_len = k_mus * n_aux * n;
    let x: &mut [f64] = unsafe {
        // safety: WASM is single-threaded per Worker; no concurrent borrow.
        if SCRATCH_X.len() < x_len { SCRATCH_X.resize(x_len, 0.0); }
        for v in &mut SCRATCH_X[..x_len] { *v = 0.0; }
        &mut SCRATCH_X[..x_len]
    };
    for (k, &mu_u) in mus.iter().enumerate() {
        let mu = mu_u as usize;
        for la in 0..n {
            let d_row_base = la * n;
            for pp in 0..n_aux {
                let bv = b[(mu * n + la) * n_aux + pp];
                let x_row_base = (k * n_aux + pp) * n;
                #[cfg(target_feature = "simd128")]
                unsafe {
                    use std::arch::wasm32::*;
                    let bv_v = f64x2_splat(bv);
                    let dp = d.as_ptr().add(d_row_base);
                    let xp = x.as_mut_ptr().add(x_row_base);
                    let mut si = 0;
                    // 4-wide unroll: process 8 f64 per iter. Hides load
                    // and FMA latency better than the 2-wide loop on
                    // n=190 (n/8 = 23 full unrolled iters + 1 tail).
                    while si + 8 <= n {
                        let d0 = v128_load(dp.add(si)     as *const v128);
                        let d1 = v128_load(dp.add(si + 2) as *const v128);
                        let d2 = v128_load(dp.add(si + 4) as *const v128);
                        let d3 = v128_load(dp.add(si + 6) as *const v128);
                        let x0 = v128_load(xp.add(si)     as *const v128);
                        let x1 = v128_load(xp.add(si + 2) as *const v128);
                        let x2 = v128_load(xp.add(si + 4) as *const v128);
                        let x3 = v128_load(xp.add(si + 6) as *const v128);
                        let r0 = f64x2_add(x0, f64x2_mul(bv_v, d0));
                        let r1 = f64x2_add(x1, f64x2_mul(bv_v, d1));
                        let r2 = f64x2_add(x2, f64x2_mul(bv_v, d2));
                        let r3 = f64x2_add(x3, f64x2_mul(bv_v, d3));
                        v128_store(xp.add(si)     as *mut v128, r0);
                        v128_store(xp.add(si + 2) as *mut v128, r1);
                        v128_store(xp.add(si + 4) as *mut v128, r2);
                        v128_store(xp.add(si + 6) as *mut v128, r3);
                        si += 8;
                    }
                    while si + 2 <= n {
                        let dv = v128_load(dp.add(si) as *const v128);
                        let xv = v128_load(xp.add(si) as *const v128);
                        let res = f64x2_add(xv, f64x2_mul(bv_v, dv));
                        v128_store(xp.add(si) as *mut v128, res);
                        si += 2;
                    }
                    while si < n {
                        *xp.add(si) += bv * *dp.add(si);
                        si += 1;
                    }
                }
                #[cfg(not(target_feature = "simd128"))]
                {
                    for si in 0..n {
                        x[x_row_base + si] += bv * d[d_row_base + si];
                    }
                }
            }
        }
    }

    // K[μν] = Σ_P Σ_σ X[P, μ, σ] · B[νσ, P]
    // X_T[μ, σ, P] = X[(k_aux + pp) * n + si] transposed.
    // Reuse thread-local scratch — see SCRATCH_X comment above.
    let xt_len = k_mus * n * n_aux;
    let x_t: &mut [f64] = unsafe {
        if SCRATCH_XT.len() < xt_len { SCRATCH_XT.resize(xt_len, 0.0); }
        &mut SCRATCH_XT[..xt_len]
    };
    for k in 0..k_mus {
        for pp in 0..n_aux {
            for si in 0..n {
                x_t[(k * n + si) * n_aux + pp] = x[(k * n_aux + pp) * n + si];
            }
        }
    }
    // K is symmetric in HF: K[μ,ν] = K[ν,μ]. Compute only ν ≥ μ per
    // worker and zero-fill ν < μ; the orchestrator mirrors after
    // collecting. This halves K-build inner work (the dominant kernel
    // cost at n=190): ~22% reduction in SCF time. Bit-exact since we
    // mirror, not approximate.
    for (k, &mu_u) in mus.iter().enumerate() {
        let mu = mu_u as usize;
        // Lower-triangle entries are filled by the orchestrator from
        // the symmetric upper-triangle pair owned by some other worker.
        // Initialize to 0 so the orchestrator's mirror sees clean zeros.
        for nu in 0..mu {
            out[k_mus * n + k * n + nu] = 0.0;
        }
        for nu in mu..n {
            let mut s = 0.0_f64;
            for si in 0..n {
                let xt_base = (k * n + si) * n_aux;
                let b_base = (nu * n + si) * n_aux;
                #[cfg(target_feature = "simd128")]
                unsafe {
                    use std::arch::wasm32::*;
                    let mut acc = f64x2_splat(0.0);
                    let xp = x_t.as_ptr().add(xt_base);
                    let bp = b.as_ptr().add(b_base);
                    let mut pp = 0;
                    while pp + 2 <= n_aux {
                        let xv = v128_load(xp.add(pp) as *const v128);
                        let bv = v128_load(bp.add(pp) as *const v128);
                        acc = f64x2_add(acc, f64x2_mul(xv, bv));
                        pp += 2;
                    }
                    let mut s_inner = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);
                    while pp < n_aux {
                        s_inner += *xp.add(pp) * *bp.add(pp);
                        pp += 1;
                    }
                    s += s_inner;
                }
                #[cfg(not(target_feature = "simd128"))]
                {
                    for pp in 0..n_aux {
                        s += x_t[xt_base + pp] * b[b_base + pp];
                    }
                }
            }
            out[k_mus * n + k * n + nu] = s;
        }
    }

    out
}

/// Compute the gamma vector γ[P] = Σ_λσ B[λσ, P] · D[λσ] — shared
/// across workers in the parallel build_jk_df path.
#[wasm_bindgen]
pub fn build_gamma_df(
    n: u32,
    n_aux: u32,
    b: &[f64],
    d: &[f64],
) -> Vec<f64> {
    let n = n as usize;
    let n_aux = n_aux as usize;
    let big_n = n * n;
    let mut gamma = vec![0.0_f64; n_aux];
    for pp in 0..n_aux {
        let mut s = 0.0_f64;
        for i in 0..big_n {
            s += b[i * n_aux + pp] * d[i];
        }
        gamma[pp] = s;
    }
    gamma
}

#[wasm_bindgen]
pub fn build_jk_df(
    n: u32,
    n_aux: u32,
    b: &[f64],
    d: &[f64],
) -> Vec<f64> {
    let n = n as usize;
    let n_aux = n_aux as usize;
    let big_n = n * n;
    let mut j = vec![0.0_f64; big_n];
    let mut k = vec![0.0_f64; big_n];

    // γ[P] = Σ_i B[i, P] · D[i]  (i = μν)
    let mut gamma = vec![0.0_f64; n_aux];
    for pp in 0..n_aux {
        let mut s = 0.0_f64;
        for i in 0..big_n {
            s += b[i * n_aux + pp] * d[i];
        }
        gamma[pp] = s;
    }
    // J[i] = Σ_P B[i, P] · γ[P]
    for i in 0..big_n {
        let base = i * n_aux;
        let mut s = 0.0_f64;
        for pp in 0..n_aux {
            s += b[base + pp] * gamma[pp];
        }
        j[i] = s;
    }

    // X[P, μ, σ] = Σ_λ B[μλ, P] · D[λ, σ]   shape (n_aux, n, n)
    //
    // Reorder loops to (μ, λ, P, σ) with σ innermost. For fixed
    // (μ, λ, P), B[μλ, P] is a scalar (one load), D[λ, σ] varies
    // contiguously in σ (stride 1), and we accumulate into
    // X[(P, μ, σ)] which is also contiguous in σ. Pure SIMD-able
    // FMA pattern. Initialize X to zero, accumulate.
    let mut x = vec![0.0_f64; n_aux * big_n];
    for mu in 0..n {
        for la in 0..n {
            let d_row_base = la * n;  // D[λ, :] row
            for pp in 0..n_aux {
                let bv = b[(mu * n + la) * n_aux + pp];
                let x_row_base = (pp * n + mu) * n;
                #[cfg(target_feature = "simd128")]
                unsafe {
                    use std::arch::wasm32::*;
                    let bv_v = f64x2_splat(bv);
                    let dp = d.as_ptr().add(d_row_base);
                    let xp = x.as_mut_ptr().add(x_row_base);
                    let mut si = 0;
                    while si + 2 <= n {
                        let dv = v128_load(dp.add(si) as *const v128);
                        let xv = v128_load(xp.add(si) as *const v128);
                        let res = f64x2_add(xv, f64x2_mul(bv_v, dv));
                        v128_store(xp.add(si) as *mut v128, res);
                        si += 2;
                    }
                    while si < n {
                        *xp.add(si) += bv * *dp.add(si);
                        si += 1;
                    }
                }
                #[cfg(not(target_feature = "simd128"))]
                {
                    for si in 0..n {
                        x[x_row_base + si] += bv * d[d_row_base + si];
                    }
                }
            }
        }
    }

    // K[μν] = Σ_P Σ_σ X[P, μ, σ] · B[νσ, P]
    //
    // The "natural" (μ, ν, P, σ) loop kills the cache: B[(νσ)·n_aux + P]
    // has stride n_aux as σ varies. At n_aux=662 (benzene auto-aux),
    // each next-σ load misses L1 (32 KB) hard.
    //
    // Transpose X[P, μ, σ] → X_T[μ, σ, P] first. Then the (μ, ν)
    // double sum becomes Σ_σ ⟨X_T[μ, σ, :], B[ν, σ, :]⟩ — a clean
    // inner dot-product over P with stride-1 access on both sides.
    // Perfect for SIMD.
    let mut x_t = vec![0.0_f64; n_aux * big_n];
    for pp in 0..n_aux {
        for mu in 0..n {
            for si in 0..n {
                x_t[(mu * n + si) * n_aux + pp] = x[(pp * n + mu) * n + si];
            }
        }
    }
    for mu in 0..n {
        for nu in 0..n {
            let mut s = 0.0_f64;
            for si in 0..n {
                let xt_base = (mu * n + si) * n_aux;
                let b_base = (nu * n + si) * n_aux;
                // P-loop: both X_T and B contiguous, SIMD-friendly.
                #[cfg(target_feature = "simd128")]
                unsafe {
                    use std::arch::wasm32::*;
                    let mut acc = f64x2_splat(0.0);
                    let xp = x_t.as_ptr().add(xt_base);
                    let bp = b.as_ptr().add(b_base);
                    let mut pp = 0;
                    while pp + 2 <= n_aux {
                        let xv = v128_load(xp.add(pp) as *const v128);
                        let bv = v128_load(bp.add(pp) as *const v128);
                        acc = f64x2_add(acc, f64x2_mul(xv, bv));
                        pp += 2;
                    }
                    let mut s_inner = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);
                    while pp < n_aux {
                        s_inner += *xp.add(pp) * *bp.add(pp);
                        pp += 1;
                    }
                    s += s_inner;
                }
                #[cfg(not(target_feature = "simd128"))]
                {
                    for pp in 0..n_aux {
                        s += x_t[xt_base + pp] * b[b_base + pp];
                    }
                }
            }
            k[mu * n + nu] = s;
        }
    }

    // Pack J and K into one output buffer [J; K] of length 2N.
    let mut out = Vec::with_capacity(2 * big_n);
    out.extend_from_slice(&j);
    out.extend_from_slice(&k);
    out
}
