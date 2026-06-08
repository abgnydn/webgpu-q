// WebGPU integral build — increment #1: the 2-index DF metric, s-only, f32.
//
// First real GPU integral. Computes M[P,Q] = (P|Q) for s-type auxiliary
// Gaussians entirely in a WGSL compute shader, to answer the make-or-break for
// the whole GPU-integral effort: does f32 *evaluation* (Boys function via an
// erf approximation, in single precision) hold chemical accuracy vs the f64
// WASM reference? If yes, extend to p/d and the 3-index tensor.
//
// s-only reduces McMurchie–Davidson to a closed form: the Hermite expansion is
// trivial (E_0 = 1), so (P|Q) = Σ_ij (c_i N_i)(c_j N_j) · 2π^{5/2}/(a_i a_j √(a_i+a_j))
// · F_0(a_i a_j/(a_i+a_j) · R_PQ²), with the s norm N = (2a/π)^{3/4}.

import type { CGShell } from "./integrals-cg.js";

const METRIC_WGSL = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const PI_2_5: f32 = 17.493418327624863;   // PI^2.5

// erf, Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7), x >= 0.
fn erf_pos(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * x);
  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return 1.0 - poly * exp(-x * x);
}
// Boys F_0(t).
fn boys0(t: f32) -> f32 {
  if (t < 1.0e-6) { return 1.0 - t / 3.0 + t * t / 10.0; }
  let s = sqrt(t);
  return 0.5 * sqrt(PI) / s * erf_pos(s);
}

struct Params { n_aux: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> centers: array<f32>;    // n_aux * 3
@group(0) @binding(2) var<storage, read> prim_off: array<u32>;   // n_aux
@group(0) @binding(3) var<storage, read> n_prim: array<u32>;     // n_aux
@group(0) @binding(4) var<storage, read> alpha: array<f32>;      // total prims
@group(0) @binding(5) var<storage, read> coef_n: array<f32>;     // c * norm
@group(0) @binding(6) var<storage, read_write> m_out: array<f32>; // n_aux * n_aux

@compute @workgroup_size(8, 8)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = params.n_aux;
  let p = gid.x;
  let q = gid.y;
  if (p >= n || q >= n) { return; }
  let dx = centers[p * 3u] - centers[q * 3u];
  let dy = centers[p * 3u + 1u] - centers[q * 3u + 1u];
  let dz = centers[p * 3u + 2u] - centers[q * 3u + 2u];
  let r2 = dx * dx + dy * dy + dz * dz;
  let offP = prim_off[p]; let nP = n_prim[p];
  let offQ = prim_off[q]; let nQ = n_prim[q];
  var sum: f32 = 0.0;
  for (var ip: u32 = 0u; ip < nP; ip = ip + 1u) {
    let a = alpha[offP + ip];
    let ca = coef_n[offP + ip];
    for (var iq: u32 = 0u; iq < nQ; iq = iq + 1u) {
      let b = alpha[offQ + iq];
      let cb = coef_n[offQ + iq];
      let ab = a * b / (a + b);
      let pref = PI_2_5 * 2.0 / (a * b * sqrt(a + b));
      sum = sum + ca * cb * pref * boys0(ab * r2);
    }
  }
  m_out[p * n + q] = sum;
}
`;

/** GPU 2-index metric for s-type aux Gaussians. Returns M[P,Q] as f32 (row-major
 *  n_aux × n_aux). Requires WebGPU (browser). */
export async function buildMetric2idxGPU_sOnly(auxShells: readonly CGShell[]): Promise<Float32Array> {
  for (const sh of auxShells) {
    if (sh.angular[0] !== 0 || sh.angular[1] !== 0 || sh.angular[2] !== 0) {
      throw new Error("buildMetric2idxGPU_sOnly: non-s aux shell (increment #1 is s-only)");
    }
  }
  const nAux = auxShells.length;

  // Pack.
  const centers = new Float32Array(nAux * 3);
  const primOff = new Uint32Array(nAux);
  const nPrim = new Uint32Array(nAux);
  let total = 0;
  for (let i = 0; i < nAux; i++) {
    const sh = auxShells[i]!;
    nPrim[i] = sh.alpha.length;
    primOff[i] = total;
    total += sh.alpha.length;
    centers[i * 3] = sh.center[0]; centers[i * 3 + 1] = sh.center[1]; centers[i * 3 + 2] = sh.center[2];
  }
  const alpha = new Float32Array(total);
  const coefN = new Float32Array(total);
  let k = 0;
  for (let i = 0; i < nAux; i++) {
    const sh = auxShells[i]!;
    for (let p = 0; p < sh.alpha.length; p++) {
      const a = sh.alpha[p]!;
      alpha[k] = a;
      coefN[k] = sh.c[p]! * Math.pow((2 * a) / Math.PI, 0.75); // s normalization
      k++;
    }
  }

  const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
  if (!gpu) throw new Error("WebGPU unavailable");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice();

  const mk = (data: Float32Array | Uint32Array, usage: number): GPUBuffer => {
    const buf = device.createBuffer({ size: Math.max(16, data.byteLength), usage });
    device.queue.writeBuffer(buf, 0, data as unknown as BufferSource);
    return buf;
  };
  const R = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const paramsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([nAux]));
  const centersBuf = mk(centers, R);
  const primOffBuf = mk(primOff, R);
  const nPrimBuf = mk(nPrim, R);
  const alphaBuf = mk(alpha, R);
  const coefBuf = mk(coefN, R);
  const outBuf = device.createBuffer({ size: nAux * nAux * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: nAux * nAux * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const mod = device.createShaderModule({ code: METRIC_WGSL });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "build" } });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: centersBuf } },
      { binding: 2, resource: { buffer: primOffBuf } },
      { binding: 3, resource: { buffer: nPrimBuf } },
      { binding: 4, resource: { buffer: alphaBuf } },
      { binding: 5, resource: { buffer: coefBuf } },
      { binding: 6, resource: { buffer: outBuf } },
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
