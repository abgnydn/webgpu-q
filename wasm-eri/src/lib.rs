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
