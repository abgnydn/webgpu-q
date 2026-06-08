// WebGPU integral build — 2-index DF metric in WGSL/f32.
//
// #1 did s-only (closed form). #2 (this) adds angular momentum (s/p/d) via the
// full McMurchie–Davidson machinery ported to WGSL with NO recursion and
// bounded per-thread scratch: single-Gaussian Hermite E-coefficients (hard-coded
// for L ≤ 2), the Hermite-Coulomb R-tensor recurrence, and a three-branch Boys
// evaluator (small-t series / Taylor / large-t upward recurrence with erf).
//
// Validated against the f64 WASM eri_2idx_build. Correctness first — the 625-f32
// R-tensor scratch per thread is unoptimized (perf is a later increment).

import type { CGShell } from "./integrals-cg.js";

const METRIC_WGSL = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const PI_2_5: f32 = 17.493418327624863;

fn erf_pos(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * x);
  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return 1.0 - poly * exp(-x * x);
}
fn boys0(t: f32) -> f32 {
  if (t < 1.0e-6) { return 1.0 - t / 3.0 + t * t / 10.0; }
  let s = sqrt(t);
  return 0.5 * sqrt(PI) / s * erf_pos(s);
}
// Single-Gaussian Hermite E-coefficient for Cartesian power k (0..2), hermite
// index tt, with inv2a = 1/(2 alpha). (Derived from the standard recurrence.)
fn ecoef(k: u32, tt: u32, inv2a: f32) -> f32 {
  if (k == 0u) { return 1.0; }
  if (k == 1u) { if (tt == 1u) { return inv2a; } return 0.0; }
  // k == 2
  if (tt == 0u) { return inv2a; }
  if (tt == 2u) { return inv2a * inv2a; }
  return 0.0;
}
fn ridx(nn: u32, t: u32, u: u32, v: u32) -> u32 { return ((nn * 5u + t) * 5u + u) * 5u + v; }

struct Params { n_aux: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> centers: array<f32>;
@group(0) @binding(2) var<storage, read> prim_off: array<u32>;
@group(0) @binding(3) var<storage, read> n_prim: array<u32>;
@group(0) @binding(4) var<storage, read> alpha: array<f32>;
@group(0) @binding(5) var<storage, read> coef_n: array<f32>;
@group(0) @binding(6) var<storage, read> ang: array<u32>;       // n_aux * 3
@group(0) @binding(7) var<storage, read_write> m_out: array<f32>;

@compute @workgroup_size(8, 8)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params.n_aux;
  let P = gid.x; let Q = gid.y;
  if (P >= n || Q >= n) { return; }
  let rx = centers[P * 3u] - centers[Q * 3u];
  let ry = centers[P * 3u + 1u] - centers[Q * 3u + 1u];
  let rz = centers[P * 3u + 2u] - centers[Q * 3u + 2u];
  let r2 = rx * rx + ry * ry + rz * rz;
  let pkx = ang[P * 3u]; let pky = ang[P * 3u + 1u]; let pkz = ang[P * 3u + 2u];
  let qkx = ang[Q * 3u]; let qky = ang[Q * 3u + 1u]; let qkz = ang[Q * 3u + 2u];
  let Tmax = pkx + qkx; let Umax = pky + qky; let Vmax = pkz + qkz;
  let Nmax = Tmax + Umax + Vmax;
  let offP = prim_off[P]; let nP = n_prim[P];
  let offQ = prim_off[Q]; let nQ = n_prim[Q];

  var acc: f32 = 0.0;
  for (var ip: u32 = 0u; ip < nP; ip = ip + 1u) {
    let a = alpha[offP + ip]; let ca = coef_n[offP + ip]; let inv2aP = 1.0 / (2.0 * a);
    for (var iq: u32 = 0u; iq < nQ; iq = iq + 1u) {
      let b = alpha[offQ + iq]; let cb = coef_n[offQ + iq]; let inv2aQ = 1.0 / (2.0 * b);
      let p = a * b / (a + b);
      let x = p * r2;

      // Boys F_0..F_Nmax (three-branch, matching the WASM).
      var f: array<f32, 5>;
      if (x < 1.0e-12) {
        for (var nn: u32 = 0u; nn <= Nmax; nn = nn + 1u) { f[nn] = 1.0 / (2.0 * f32(nn) + 1.0); }
      } else if (x < (2.0 * f32(Nmax) - 1.0) / 2.0) {
        for (var nn: u32 = 0u; nn <= Nmax; nn = nn + 1u) {
          var sum: f32 = 0.0;
          var term: f32 = 1.0 / (2.0 * f32(nn) + 1.0);
          for (var kk: u32 = 0u; kk < 60u; kk = kk + 1u) {
            sum = sum + term;
            term = term * (-x / (f32(kk) + 1.0)) * (2.0 * f32(nn) + 2.0 * f32(kk) + 1.0) / (2.0 * f32(nn) + 2.0 * f32(kk) + 3.0);
            if (abs(term) < 1.0e-8 * abs(sum)) { break; }
          }
          f[nn] = sum;
        }
      } else {
        f[0] = boys0(x);
        let em = exp(-x);
        for (var nn: u32 = 1u; nn <= Nmax; nn = nn + 1u) {
          f[nn] = ((2.0 * f32(nn) - 1.0) * f[nn - 1u] - em) / (2.0 * x);
        }
      }

      // R-tensor: R[n][0][0][0] = (-2p)^n F_n, then downward recurrence.
      var r: array<f32, 625>;
      var neg2p: f32 = 1.0;
      for (var nn: u32 = 0u; nn <= Nmax; nn = nn + 1u) { r[ridx(nn, 0u, 0u, 0u)] = neg2p * f[nn]; neg2p = neg2p * (-2.0 * p); }
      if (Nmax > 0u) {
        for (var nn: i32 = i32(Nmax) - 1; nn >= 0; nn = nn - 1) {
          let un = u32(nn);
          for (var t: u32 = 0u; t <= Tmax; t = t + 1u) {
            for (var u: u32 = 0u; u <= Umax; u = u + 1u) {
              for (var v: u32 = 0u; v <= Vmax; v = v + 1u) {
                if (t == 0u && u == 0u && v == 0u) { continue; }
                var a2: f32 = 0.0;
                if (v > 0u) {
                  a2 = a2 + rz * r[ridx(un + 1u, t, u, v - 1u)];
                  if (v >= 2u) { a2 = a2 + f32(v - 1u) * r[ridx(un + 1u, t, u, v - 2u)]; }
                } else if (u > 0u) {
                  a2 = a2 + ry * r[ridx(un + 1u, t, u - 1u, v)];
                  if (u >= 2u) { a2 = a2 + f32(u - 1u) * r[ridx(un + 1u, t, u - 2u, v)]; }
                } else {
                  a2 = a2 + rx * r[ridx(un + 1u, t - 1u, u, v)];
                  if (t >= 2u) { a2 = a2 + f32(t - 1u) * r[ridx(un + 1u, t - 2u, u, v)]; }
                }
                r[ridx(un, t, u, v)] = a2;
              }
            }
          }
        }
      }

      // Contract Hermite E-coefficients with R[0].
      var s: f32 = 0.0;
      for (var t: u32 = 0u; t <= pkx; t = t + 1u) {
        let pex = ecoef(pkx, t, inv2aP);
        for (var u: u32 = 0u; u <= pky; u = u + 1u) {
          let pey = pex * ecoef(pky, u, inv2aP);
          for (var v: u32 = 0u; v <= pkz; v = v + 1u) {
            let pez = pey * ecoef(pkz, v, inv2aP);
            for (var ta: u32 = 0u; ta <= qkx; ta = ta + 1u) {
              let qex = ecoef(qkx, ta, inv2aQ);
              for (var nv: u32 = 0u; nv <= qky; nv = nv + 1u) {
                let qey = qex * ecoef(qky, nv, inv2aQ);
                for (var ph: u32 = 0u; ph <= qkz; ph = ph + 1u) {
                  let parity = (ta + nv + ph) & 1u;
                  let sign = select(1.0, -1.0, parity == 1u);
                  s = s + pez * qey * ecoef(qkz, ph, inv2aQ) * sign * r[ridx(0u, t + ta, u + nv, v + ph)];
                }
              }
            }
          }
        }
      }
      acc = acc + ca * cb * (PI_2_5 * 2.0 / (a * b * sqrt(a + b))) * s;
    }
  }
  m_out[P * n + Q] = acc;
}
`;

function doubleFactOdd(i: number): number {
  let r = 1;
  for (let k = 2 * i - 1; k > 0; k -= 2) r *= k;
  return r;
}
function normCg(a: number, ix: number, iy: number, iz: number): number {
  const L = ix + iy + iz;
  const radial = Math.pow((2 * a) / Math.PI, 0.75);
  const angular = Math.sqrt(Math.pow(4 * a, L) / (doubleFactOdd(ix) * doubleFactOdd(iy) * doubleFactOdd(iz)));
  return radial * angular;
}

/** GPU 2-index metric M[P,Q]=(P|Q) for aux Gaussians up to angular momentum d.
 *  Returns f32 row-major n_aux × n_aux. Requires WebGPU (browser). */
export async function buildMetric2idxGPU(auxShells: readonly CGShell[]): Promise<Float32Array> {
  for (const sh of auxShells) {
    if (sh.angular[0] + sh.angular[1] + sh.angular[2] > 2) {
      throw new Error("buildMetric2idxGPU: angular momentum > 2 (d) not yet supported");
    }
  }
  const nAux = auxShells.length;
  const centers = new Float32Array(nAux * 3);
  const primOff = new Uint32Array(nAux);
  const nPrim = new Uint32Array(nAux);
  const ang = new Uint32Array(nAux * 3);
  let total = 0;
  for (let i = 0; i < nAux; i++) {
    const sh = auxShells[i]!;
    nPrim[i] = sh.alpha.length; primOff[i] = total; total += sh.alpha.length;
    centers[i * 3] = sh.center[0]; centers[i * 3 + 1] = sh.center[1]; centers[i * 3 + 2] = sh.center[2];
    ang[i * 3] = sh.angular[0]; ang[i * 3 + 1] = sh.angular[1]; ang[i * 3 + 2] = sh.angular[2];
  }
  const alpha = new Float32Array(total);
  const coefN = new Float32Array(total);
  let k = 0;
  for (let i = 0; i < nAux; i++) {
    const sh = auxShells[i]!;
    for (let p = 0; p < sh.alpha.length; p++) {
      const a = sh.alpha[p]!;
      alpha[k] = a;
      coefN[k] = sh.c[p]! * normCg(a, sh.angular[0], sh.angular[1], sh.angular[2]);
      k++;
    }
  }

  const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
  if (!gpu) throw new Error("WebGPU unavailable");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice();

  const R = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const mk = (data: Float32Array | Uint32Array, usage: number): GPUBuffer => {
    const buf = device.createBuffer({ size: Math.max(16, data.byteLength), usage });
    device.queue.writeBuffer(buf, 0, data as unknown as BufferSource);
    return buf;
  };
  const paramsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([nAux]));
  const buffers = [mk(centers, R), mk(primOff, R), mk(nPrim, R), mk(alpha, R), mk(coefN, R), mk(ang, R)];
  const outBuf = device.createBuffer({ size: nAux * nAux * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: nAux * nAux * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const mod = device.createShaderModule({ code: METRIC_WGSL });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "build" } });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      ...buffers.map((b, i) => ({ binding: i + 1, resource: { buffer: b } })),
      { binding: 7, resource: { buffer: outBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  const wg = Math.ceil(nAux / 8);
  pass.dispatchWorkgroups(wg, wg);
  pass.end();
  enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, nAux * nAux * 4);
  device.queue.submit([enc.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  device.destroy();
  return out;
}

/** s-only alias (increment #1 entry point). */
export const buildMetric2idxGPU_sOnly = buildMetric2idxGPU;

// ── Increment #3: 3-index tensor V[μν,P] = (μν|P), s-only ──────────────────────
// The new piece vs the metric is the BRA PAIR (μ,ν) — a product of two Gaussians
// on different centers. For s-only the McMurchie–Davidson bra E-coefficients
// collapse to the Gaussian-product factor K = exp(-μ |A-B|²) per axis, and the
// R-tensor to F_0, so this validates the bra-pair geometry + the 3-index dispatch
// and V layout before adding angular momentum (and the d-function R-memory work).

// Inputs packed into 2 buffers per set (f32: centers|alpha|coefN, u32: off|np) to
// stay under maxStorageBuffersPerShaderStage (default 8): 4 inputs + 1 output.
const V3IDX_S_WGSL = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const PI_2_5: f32 = 17.493418327624863;
fn erf_pos(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * x);
  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return 1.0 - poly * exp(-x * x);
}
fn boys0(t: f32) -> f32 {
  if (t < 1.0e-6) { return 1.0 - t / 3.0 + t * t / 10.0; }
  let s = sqrt(t);
  return 0.5 * sqrt(PI) / s * erf_pos(s);
}
// Params: n, n_aux, then section offsets into the packed buffers.
//   orb f32: centers @0, alpha @oa_off, coefN @ocn_off ; orb u32: off @0, np @onp_off
//   aux f32: centers @0, alpha @qa_off, coefN @qcn_off ; aux u32: off @0, np @qnp_off
struct Params {
  n: u32, n_aux: u32,
  oa_off: u32, ocn_off: u32, onp_off: u32,
  qa_off: u32, qcn_off: u32, qnp_off: u32,
};
@group(0) @binding(0) var<uniform> prm: Params;
@group(0) @binding(1) var<storage, read> of32: array<f32>;  // orb centers|alpha|coefN
@group(0) @binding(2) var<storage, read> ou32: array<u32>;  // orb off|np
@group(0) @binding(3) var<storage, read> qf32: array<f32>;  // aux centers|alpha|coefN
@group(0) @binding(4) var<storage, read> qu32: array<u32>;  // aux off|np
@group(0) @binding(5) var<storage, read_write> v_out: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = prm.n; let na = prm.n_aux;
  let mu = gid.x; let nu = gid.y; let P = gid.z;
  if (mu >= n || nu >= n || P >= na) { return; }
  let ax = of32[mu*3u]; let ay = of32[mu*3u+1u]; let az = of32[mu*3u+2u];
  let bx = of32[nu*3u]; let by = of32[nu*3u+1u]; let bz = of32[nu*3u+2u];
  let cx = qf32[P*3u]; let cy = qf32[P*3u+1u]; let cz = qf32[P*3u+2u];
  let dabx = ax-bx; let daby = ay-by; let dabz = az-bz;
  let muOff = ou32[mu]; let muNp = ou32[prm.onp_off + mu];
  let nuOff = ou32[nu]; let nuNp = ou32[prm.onp_off + nu];
  let pOff = qu32[P]; let pNp = qu32[prm.qnp_off + P];
  var acc: f32 = 0.0;
  for (var i: u32 = 0u; i < muNp; i = i + 1u) {
    let a = of32[prm.oa_off + muOff + i]; let ca = of32[prm.ocn_off + muOff + i];
    for (var j: u32 = 0u; j < nuNp; j = j + 1u) {
      let b = of32[prm.oa_off + nuOff + j]; let cb = of32[prm.ocn_off + nuOff + j];
      let p = a + b; let mab = a * b / p;
      let px = (a*ax + b*bx)/p; let py = (a*ay + b*by)/p; let pz = (a*az + b*bz)/p;
      let kk = exp(-mab * (dabx*dabx + daby*daby + dabz*dabz));
      for (var k: u32 = 0u; k < pNp; k = k + 1u) {
        let q = qf32[prm.qa_off + pOff + k]; let cq = qf32[prm.qcn_off + pOff + k];
        let rx = px-cx; let ry = py-cy; let rz = pz-cz;
        let ap = p*q/(p+q);
        let pref = PI_2_5 * 2.0 / (p*q*sqrt(p+q));
        acc = acc + ca*cb*cq*kk*pref*boys0(ap*(rx*rx+ry*ry+rz*rz));
      }
    }
  }
  v_out[(mu*n+nu)*na + P] = acc;
}
`;

// ── Increment #3b: general s/p/d 3-index ──────────────────────────────────────
// Adds angular momentum to the 3-index path. New piece: the two-center bra
// E-coefficient recurrence (e_coef_table). R-tensor/Boys/aux-ecoef reused from #2.
// R scratch sized 7^4 (covers L_μ+L_ν+L_P ≤ 6); a probe of whether that private
// array is viable on the GPU (else a 2-slab rewrite). dg = 3 (j_dim), tg = 5 (t_dim).
const V3IDX_WGSL = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const PI_2_5: f32 = 17.493418327624863;
fn erf_pos(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * x);
  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return 1.0 - poly * exp(-x * x);
}
fn boys0(t: f32) -> f32 {
  if (t < 1.0e-6) { return 1.0 - t / 3.0 + t * t / 10.0; }
  let s = sqrt(t);
  return 0.5 * sqrt(PI) / s * erf_pos(s);
}
// single-Gaussian Hermite E-coef (aux side), k in 0..2.
fn ecoef(k: u32, tt: u32, inv2a: f32) -> f32 {
  if (k == 0u) { return 1.0; }
  if (k == 1u) { if (tt == 1u) { return inv2a; } return 0.0; }
  if (tt == 0u) { return inv2a; }
  if (tt == 2u) { return inv2a * inv2a; }
  return 0.0;
}
// 2-center bra E-coefficients E^{im,jm}_t, t=0..im+jm, written to outp[0..].
// tab[i][j][t] flat with j_dim=3, t_dim=5 → idx(i,j,t)=(i*3+j)*5+t (i,j<=2,t<=4).
fn ecoef_pair(im: u32, jm: u32, pa: f32, pb: f32, inv2p: f32, kfac: f32, outp: ptr<function, array<f32, 5>>) {
  var tab: array<f32, 45>;
  for (var z: u32 = 0u; z < 45u; z = z + 1u) { tab[z] = 0.0; }
  tab[0] = kfac; // idx(0,0,0)
  // raise j at i=0
  for (var b: u32 = 0u; b < jm; b = b + 1u) {
    for (var t: u32 = 0u; t <= b + 1u; t = t + 1u) {
      var left: f32 = 0.0; if (t > 0u) { left = tab[(0u * 3u + b) * 5u + (t - 1u)]; }
      var mid: f32 = 0.0;  if (t <= b) { mid = tab[(0u * 3u + b) * 5u + t]; }
      var right: f32 = 0.0; if (t + 1u <= b) { right = tab[(0u * 3u + b) * 5u + (t + 1u)]; }
      tab[(0u * 3u + (b + 1u)) * 5u + t] = inv2p * left + pb * mid + f32(t + 1u) * right;
    }
  }
  // raise i
  for (var a: u32 = 0u; a < im; a = a + 1u) {
    for (var b: u32 = 0u; b <= jm; b = b + 1u) {
      for (var t: u32 = 0u; t <= a + b + 1u; t = t + 1u) {
        var left: f32 = 0.0; if (t > 0u) { left = tab[(a * 3u + b) * 5u + (t - 1u)]; }
        var mid: f32 = 0.0;  if (t <= a + b) { mid = tab[(a * 3u + b) * 5u + t]; }
        var right: f32 = 0.0; if (t + 1u <= a + b) { right = tab[(a * 3u + b) * 5u + (t + 1u)]; }
        tab[((a + 1u) * 3u + b) * 5u + t] = inv2p * left + pa * mid + f32(t + 1u) * right;
      }
    }
  }
  for (var t: u32 = 0u; t <= im + jm; t = t + 1u) { (*outp)[t] = tab[(im * 3u + jm) * 5u + t]; }
}
// (t,u,v) slab index (7^3); the R recurrence ping-pongs two of these over n.
fn sidx(t: u32, u: u32, v: u32) -> u32 { return (t * 7u + u) * 7u + v; }

struct Params {
  n: u32, n_aux: u32,
  oa_off: u32, ocn_off: u32, onp_off: u32, oang_off: u32,
  qa_off: u32, qcn_off: u32, qnp_off: u32, qang_off: u32,
};
@group(0) @binding(0) var<uniform> prm: Params;
@group(0) @binding(1) var<storage, read> of32: array<f32>;
@group(0) @binding(2) var<storage, read> ou32: array<u32>;   // off | np | angular(n*3)
@group(0) @binding(3) var<storage, read> qf32: array<f32>;
@group(0) @binding(4) var<storage, read> qu32: array<u32>;
@group(0) @binding(5) var<storage, read_write> v_out: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = prm.n; let na = prm.n_aux;
  let mu = gid.x; let nu = gid.y; let P = gid.z;
  if (mu >= n || nu >= n || P >= na) { return; }
  let ax = of32[mu*3u]; let ay = of32[mu*3u+1u]; let az = of32[mu*3u+2u];
  let bx = of32[nu*3u]; let by = of32[nu*3u+1u]; let bz = of32[nu*3u+2u];
  let cx = qf32[P*3u]; let cy = qf32[P*3u+1u]; let cz = qf32[P*3u+2u];
  let ix = ou32[prm.oang_off + mu*3u]; let iy = ou32[prm.oang_off + mu*3u+1u]; let iz = ou32[prm.oang_off + mu*3u+2u];
  let jx = ou32[prm.oang_off + nu*3u]; let jy = ou32[prm.oang_off + nu*3u+1u]; let jz = ou32[prm.oang_off + nu*3u+2u];
  let kx = qu32[prm.qang_off + P*3u]; let ky = qu32[prm.qang_off + P*3u+1u]; let kz = qu32[prm.qang_off + P*3u+2u];
  let muOff = ou32[mu]; let muNp = ou32[prm.onp_off + mu];
  let nuOff = ou32[nu]; let nuNp = ou32[prm.onp_off + nu];
  let pOff = qu32[P]; let pNp = qu32[prm.qnp_off + P];
  let Tb = ix + jx; let Ub = iy + jy; let Vb = iz + jz;

  var acc: f32 = 0.0;
  for (var i: u32 = 0u; i < muNp; i = i + 1u) {
    let a = of32[prm.oa_off + muOff + i]; let ca = of32[prm.ocn_off + muOff + i];
    for (var j: u32 = 0u; j < nuNp; j = j + 1u) {
      let b = of32[prm.oa_off + nuOff + j]; let cb = of32[prm.ocn_off + nuOff + j];
      let p = a + b; let mab = a * b / p; let inv2p = 1.0 / (2.0 * p);
      let px = (a*ax + b*bx)/p; let py = (a*ay + b*by)/p; let pz = (a*az + b*bz)/p;
      let kfx = exp(-mab*(ax-bx)*(ax-bx)); let kfy = exp(-mab*(ay-by)*(ay-by)); let kfz = exp(-mab*(az-bz)*(az-bz));
      var bex: array<f32, 5>; var bey: array<f32, 5>; var bez: array<f32, 5>;
      ecoef_pair(ix, jx, px-ax, px-bx, inv2p, kfx, &bex);
      ecoef_pair(iy, jy, py-ay, py-by, inv2p, kfy, &bey);
      ecoef_pair(iz, jz, pz-az, pz-bz, inv2p, kfz, &bez);
      for (var k: u32 = 0u; k < pNp; k = k + 1u) {
        let q = qf32[prm.qa_off + pOff + k]; let cq = qf32[prm.qcn_off + pOff + k]; let inv2q = 1.0 / (2.0 * q);
        let ap = p*q/(p+q);
        let rx = px-cx; let ry = py-cy; let rz = pz-cz;
        let Tt = Tb + kx; let Uu = Ub + ky; let Vv = Vb + kz; let Nmax = Tt + Uu + Vv;
        // Boys
        var f: array<f32, 7>;
        let xv = ap*(rx*rx+ry*ry+rz*rz);
        if (xv < 1.0e-12) {
          for (var nn: u32 = 0u; nn <= Nmax; nn = nn + 1u) { f[nn] = 1.0/(2.0*f32(nn)+1.0); }
        } else if (xv < (2.0*f32(Nmax)-1.0)/2.0) {
          for (var nn: u32 = 0u; nn <= Nmax; nn = nn + 1u) {
            var sm: f32 = 0.0; var tm: f32 = 1.0/(2.0*f32(nn)+1.0);
            for (var c2: u32 = 0u; c2 < 80u; c2 = c2 + 1u) {
              sm = sm + tm;
              tm = tm * (-xv/(f32(c2)+1.0)) * (2.0*f32(nn)+2.0*f32(c2)+1.0)/(2.0*f32(nn)+2.0*f32(c2)+3.0);
              if (abs(tm) < 1.0e-8*abs(sm)) { break; }
            }
            f[nn] = sm;
          }
        } else {
          f[0] = boys0(xv); let em = exp(-xv);
          for (var nn: u32 = 1u; nn <= Nmax; nn = nn + 1u) { f[nn] = ((2.0*f32(nn)-1.0)*f[nn-1u]-em)/(2.0*xv); }
        }
        // R-tensor via 2-slab downward recurrence (memory: 2*7^3 not 7^4).
        // g[n] = (-2p)^n F_n are the R[n][0][0][0] diagonal values.
        var g: array<f32, 7>;
        var n2p: f32 = 1.0;
        for (var nn: u32 = 0u; nn <= Nmax; nn = nn + 1u) { g[nn] = n2p*f[nn]; n2p = n2p*(-2.0*ap); }
        var sa: array<f32, 343>; // hi slab = R[n+1]
        var sb: array<f32, 343>; // lo slab = R[n]
        for (var z: u32 = 0u; z < 343u; z = z + 1u) { sa[z] = 0.0; }
        sa[0] = g[Nmax];
        if (Nmax > 0u) {
          for (var nn: i32 = i32(Nmax)-1; nn >= 0; nn = nn - 1) {
            for (var z: u32 = 0u; z < 343u; z = z + 1u) { sb[z] = 0.0; }
            sb[0] = g[u32(nn)];
            for (var t: u32 = 0u; t <= Tt; t = t + 1u) {
              for (var u: u32 = 0u; u <= Uu; u = u + 1u) {
                for (var v: u32 = 0u; v <= Vv; v = v + 1u) {
                  if (t == 0u && u == 0u && v == 0u) { continue; }
                  var a2: f32 = 0.0;
                  if (v > 0u) { a2 = a2 + rz*sa[sidx(t,u,v-1u)]; if (v >= 2u) { a2 = a2 + f32(v-1u)*sa[sidx(t,u,v-2u)]; } }
                  else if (u > 0u) { a2 = a2 + ry*sa[sidx(t,u-1u,v)]; if (u >= 2u) { a2 = a2 + f32(u-1u)*sa[sidx(t,u-2u,v)]; } }
                  else { a2 = a2 + rx*sa[sidx(t-1u,u,v)]; if (t >= 2u) { a2 = a2 + f32(t-1u)*sa[sidx(t-2u,u,v)]; } }
                  sb[sidx(t,u,v)] = a2;
                }
              }
            }
            for (var z: u32 = 0u; z < 343u; z = z + 1u) { sa[z] = sb[z]; }
          }
        }
        // sa now holds R[0][t][u][v].
        // contract bra E (t,u,v) × aux ecoef (tau,nv,ph)
        var s: f32 = 0.0;
        for (var t: u32 = 0u; t <= Tb; t = t + 1u) {
          for (var u: u32 = 0u; u <= Ub; u = u + 1u) {
            let be2 = bex[t]*bey[u];
            for (var v: u32 = 0u; v <= Vb; v = v + 1u) {
              let be3 = be2*bez[v];
              for (var ta: u32 = 0u; ta <= kx; ta = ta + 1u) {
                let ae1 = ecoef(kx, ta, inv2q);
                for (var nvv: u32 = 0u; nvv <= ky; nvv = nvv + 1u) {
                  let ae2 = ae1*ecoef(ky, nvv, inv2q);
                  for (var ph: u32 = 0u; ph <= kz; ph = ph + 1u) {
                    let parity = (ta + nvv + ph) & 1u;
                    let sign = select(1.0, -1.0, parity == 1u);
                    s = s + be3*ae2*ecoef(kz, ph, inv2q)*sign*sa[sidx(t+ta, u+nvv, v+ph)];
                  }
                }
              }
            }
          }
        }
        acc = acc + ca*cb*cq*(PI_2_5*2.0/(p*q*sqrt(p+q)))*s;
      }
    }
  }
  v_out[(mu*n+nu)*na + P] = acc;
}
`;

/** GPU 3-index tensor V[μν,P]=(μν|P) up to angular momentum d. Returns f32,
 *  layout V[(μ·n+ν)·n_aux + P] (matches WASM eri_3idx_build). */
export async function buildV3idxGPU(
  orbShells: readonly CGShell[], auxShells: readonly CGShell[],
): Promise<Float32Array> {
  for (const s of [...orbShells, ...auxShells]) {
    if (s.angular[0] + s.angular[1] + s.angular[2] > 2) throw new Error("buildV3idxGPU: angular > d unsupported");
  }
  const pack = (sh: readonly CGShell[]): { f32: Float32Array; u32: Uint32Array; aOff: number; cnOff: number; npOff: number; angOff: number } => {
    const m = sh.length;
    let tot = 0; for (let i = 0; i < m; i++) tot += sh[i]!.alpha.length;
    const f32 = new Float32Array(m * 3 + tot * 2);
    const u32 = new Uint32Array(m * 2 + m * 3);
    const aOff = m * 3, cnOff = m * 3 + tot, npOff = m, angOff = m * 2;
    let pk = 0;
    for (let i = 0; i < m; i++) {
      f32[i * 3] = sh[i]!.center[0]; f32[i * 3 + 1] = sh[i]!.center[1]; f32[i * 3 + 2] = sh[i]!.center[2];
      u32[i] = pk; u32[npOff + i] = sh[i]!.alpha.length;
      u32[angOff + i * 3] = sh[i]!.angular[0]; u32[angOff + i * 3 + 1] = sh[i]!.angular[1]; u32[angOff + i * 3 + 2] = sh[i]!.angular[2];
      for (let p = 0; p < sh[i]!.alpha.length; p++) {
        const a = sh[i]!.alpha[p]!;
        f32[aOff + pk] = a;
        f32[cnOff + pk] = sh[i]!.c[p]! * normCg(a, sh[i]!.angular[0], sh[i]!.angular[1], sh[i]!.angular[2]);
        pk++;
      }
    }
    return { f32, u32, aOff, cnOff, npOff, angOff };
  };
  const orb = pack(orbShells); const aux = pack(auxShells);
  const n = orbShells.length; const nAux = auxShells.length;

  const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
  if (!gpu) throw new Error("WebGPU unavailable");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const R = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const mk = (data: Float32Array | Uint32Array): GPUBuffer => {
    const buf = device.createBuffer({ size: Math.max(16, data.byteLength), usage: R });
    device.queue.writeBuffer(buf, 0, data as unknown as BufferSource); return buf;
  };
  const paramsBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([
    n, nAux, orb.aOff, orb.cnOff, orb.npOff, orb.angOff, aux.aOff, aux.cnOff, aux.npOff, aux.angOff,
  ]));
  const ins = [orb.f32, orb.u32, aux.f32, aux.u32].map(mk);
  const outLen = n * n * nAux;
  const outBuf = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const mod = device.createShaderModule({ code: V3IDX_WGSL });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "build" } });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      ...ins.map((b, i) => ({ binding: i + 1, resource: { buffer: b } })),
      { binding: 5, resource: { buffer: outBuf } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / 4), Math.ceil(n / 4), Math.ceil(nAux / 4));
  pass.end();
  enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outLen * 4);
  device.queue.submit([enc.finish()]);
  const err = await device.popErrorScope();
  if (err) { device.destroy(); throw new Error(`WebGPU validation: ${err.message}`); }
  await readBuf.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap(); device.destroy();
  return out;
}

/** GPU 3-index tensor V[μν,P]=(μν|P), s-only orbitals AND aux. Returns f32,
 *  layout V[(μ·n+ν)·n_aux + P] (matches WASM eri_3idx_build). */
export async function buildV3idxGPU_sOnly(
  orbShells: readonly CGShell[], auxShells: readonly CGShell[],
): Promise<Float32Array> {
  for (const s of [...orbShells, ...auxShells]) {
    if (s.angular[0] + s.angular[1] + s.angular[2] !== 0) throw new Error("buildV3idxGPU_sOnly: non-s shell");
  }
  // Pack a shell set into f32 = [centers(m*3) | alpha(tot) | coefN(tot)] and
  // u32 = [off(m) | np(m)]. Returns the buffers + the section offsets.
  const pack = (sh: readonly CGShell[]): { f32: Float32Array; u32: Uint32Array; aOff: number; cnOff: number; npOff: number } => {
    const m = sh.length;
    let tot = 0; for (let i = 0; i < m; i++) tot += sh[i]!.alpha.length;
    const f32 = new Float32Array(m * 3 + tot * 2);
    const u32 = new Uint32Array(m * 2);
    const aOff = m * 3, cnOff = m * 3 + tot, npOff = m;
    let pk = 0;
    for (let i = 0; i < m; i++) {
      f32[i * 3] = sh[i]!.center[0]; f32[i * 3 + 1] = sh[i]!.center[1]; f32[i * 3 + 2] = sh[i]!.center[2];
      u32[i] = pk; u32[npOff + i] = sh[i]!.alpha.length;
      for (let p = 0; p < sh[i]!.alpha.length; p++) {
        const a = sh[i]!.alpha[p]!;
        f32[aOff + pk] = a;
        f32[cnOff + pk] = sh[i]!.c[p]! * Math.pow((2 * a) / Math.PI, 0.75);
        pk++;
      }
    }
    return { f32, u32, aOff, cnOff, npOff };
  };
  const orb = pack(orbShells); const aux = pack(auxShells);
  const n = orbShells.length; const nAux = auxShells.length;

  const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
  if (!gpu) throw new Error("WebGPU unavailable");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const R = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const mk = (data: Float32Array | Uint32Array): GPUBuffer => {
    const buf = device.createBuffer({ size: Math.max(16, data.byteLength), usage: R });
    device.queue.writeBuffer(buf, 0, data as unknown as BufferSource); return buf;
  };
  const paramsBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([
    n, nAux, orb.aOff, orb.cnOff, orb.npOff, aux.aOff, aux.cnOff, aux.npOff,
  ]));
  const ins = [orb.f32, orb.u32, aux.f32, aux.u32].map(mk);
  const outLen = n * n * nAux;
  const outBuf = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const mod = device.createShaderModule({ code: V3IDX_S_WGSL });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "build" } });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      ...ins.map((b, i) => ({ binding: i + 1, resource: { buffer: b } })),
      { binding: 5, resource: { buffer: outBuf } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / 4), Math.ceil(n / 4), Math.ceil(nAux / 4));
  pass.end();
  enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outLen * 4);
  device.queue.submit([enc.finish()]);
  const err = await device.popErrorScope();
  if (err) { device.destroy(); throw new Error(`WebGPU validation: ${err.message}`); }
  await readBuf.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap(); device.destroy();
  return out;
}
