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
