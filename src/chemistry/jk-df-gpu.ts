// GPU density-fitted JK build — the per-SCF-iteration Fock contraction on the
// GPU, straight from the (small) DF B-tensor. Complements the GPU integral build:
// GPU forms B once (df-gpu.ts), then GPU does J/K every iteration here, so a
// DF-HF SCF loop runs without ever materializing the full 4-index ERI.
//
// Mirrors buildJK_DF (df.ts), f32:
//   gamma[P] = Σ_i B[i,P] D[i]              (i = μν flat, N = n²)
//   J[i]     = Σ_P B[i,P] gamma[P]
//   X[P,μ,σ] = Σ_λ B[μλ,P] D[λσ]            (exchange half-transform)
//   K[μν]    = Σ_P Σ_σ X[P,μ,σ] B[νσ,P]
// Four contraction passes; B stays resident across them.

import type { DFResult } from "./df.js";

const WGSL = /* wgsl */ `
struct P2 { n: u32, n_aux: u32 };
@group(0) @binding(0) var<uniform> prm: P2;
@group(0) @binding(1) var<storage, read> B: array<f32>;     // [i*n_aux + P], i = μν
@group(0) @binding(2) var<storage, read> D: array<f32>;     // [i]
@group(0) @binding(3) var<storage, read_write> gamma: array<f32>; // [P]
@group(0) @binding(4) var<storage, read_write> J: array<f32>;     // [i]
@group(0) @binding(5) var<storage, read_write> X: array<f32>;     // [(P*n+μ)*n+σ]
@group(0) @binding(6) var<storage, read_write> K: array<f32>;     // [μ*n+ν]

@compute @workgroup_size(64)
fn gamma_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let P = gid.x; let na = prm.n_aux; let N = prm.n * prm.n;
  if (P >= na) { return; }
  var s: f32 = 0.0;
  for (var i: u32 = 0u; i < N; i = i + 1u) { s = s + B[i * na + P] * D[i]; }
  gamma[P] = s;
}
@compute @workgroup_size(64)
fn j_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let na = prm.n_aux; let N = prm.n * prm.n;
  if (i >= N) { return; }
  var s: f32 = 0.0;
  for (var P: u32 = 0u; P < na; P = P + 1u) { s = s + B[i * na + P] * gamma[P]; }
  J[i] = s;
}
@compute @workgroup_size(64)
fn x_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = prm.n; let na = prm.n_aux;
  let idx = gid.x; if (idx >= na * n * n) { return; }
  let P = idx / (n * n); let rem = idx % (n * n); let mu = rem / n; let si = rem % n;
  var s: f32 = 0.0;
  for (var la: u32 = 0u; la < n; la = la + 1u) { s = s + B[(mu * n + la) * na + P] * D[la * n + si]; }
  X[idx] = s;
}
@compute @workgroup_size(64)
fn k_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = prm.n; let na = prm.n_aux;
  let idx = gid.x; if (idx >= n * n) { return; }
  let mu = idx / n; let nu = idx % n;
  var s: f32 = 0.0;
  for (var P: u32 = 0u; P < na; P = P + 1u) {
    for (var si: u32 = 0u; si < n; si = si + 1u) {
      s = s + X[(P * n + mu) * n + si] * B[(nu * n + si) * na + P];
    }
  }
  K[idx] = s;
}
`;

/** A persistent GPU DF-JK builder: uploads B and creates the device/pipelines
 *  ONCE, then `jk(D)` does the Fock contraction each call (reusing them). This is
 *  what makes a fully-GPU DF-HF SCF loop efficient — no per-iteration device
 *  setup or B re-upload. Call `dispose()` when the SCF finishes. */
export interface GpuDFJK {
  jk(D: Float64Array): Promise<{ J: Float32Array; K: Float32Array }>;
  dispose(): void;
}

export async function makeGpuDFJK(df: DFResult): Promise<GpuDFJK> {
  const { B, nAux, n } = df;
  const N = n * n;
  const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
  if (!gpu) throw new Error("WebGPU unavailable");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice();

  const Bf = new Float32Array(B.length); for (let i = 0; i < B.length; i++) Bf[i] = B[i]!;
  const buf = (bytes: number, usage: number): GPUBuffer => device.createBuffer({ size: Math.max(16, bytes), usage });
  const S = GPUBufferUsage.STORAGE; const SD = S | GPUBufferUsage.COPY_DST; const SC = S | GPUBufferUsage.COPY_SRC;
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([n, nAux]));
  const bB = buf(Bf.byteLength, SD); device.queue.writeBuffer(bB, 0, Bf); // uploaded once
  const bD = buf(N * 4, SD);
  const bGamma = buf(nAux * 4, S);
  const bJ = buf(N * 4, SC);
  const bX = buf(nAux * N * 4, S);
  const bK = buf(N * 4, SC);
  const rJ = buf(N * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const rK = buf(N * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

  const mod = device.createShaderModule({ code: WGSL });
  const C = GPUShaderStage.COMPUTE;
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: C, buffer: { type: "uniform" } },
      { binding: 1, visibility: C, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: C, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: C, buffer: { type: "storage" } },
      { binding: 4, visibility: C, buffer: { type: "storage" } },
      { binding: 5, visibility: C, buffer: { type: "storage" } },
      { binding: 6, visibility: C, buffer: { type: "storage" } },
    ],
  });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const bg = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: bB } },
      { binding: 2, resource: { buffer: bD } }, { binding: 3, resource: { buffer: bGamma } },
      { binding: 4, resource: { buffer: bJ } }, { binding: 5, resource: { buffer: bX } },
      { binding: 6, resource: { buffer: bK } },
    ],
  });
  const mk = (entryPoint: string): GPUComputePipeline => device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint } });
  const pGamma = mk("gamma_pass"), pJ = mk("j_pass"), pX = mk("x_pass"), pK = mk("k_pass");

  return {
    async jk(D: Float64Array): Promise<{ J: Float32Array; K: Float32Array }> {
      const Df = new Float32Array(N); for (let i = 0; i < N; i++) Df[i] = D[i]!;
      device.queue.writeBuffer(bD, 0, Df);
      const enc = device.createCommandEncoder();
      const run = (pipe: GPUComputePipeline, groups: number): void => {
        const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(groups); p.end();
      };
      run(pGamma, Math.ceil(nAux / 64)); run(pJ, Math.ceil(N / 64));
      run(pX, Math.ceil((nAux * N) / 64)); run(pK, Math.ceil(N / 64));
      enc.copyBufferToBuffer(bJ, 0, rJ, 0, N * 4);
      enc.copyBufferToBuffer(bK, 0, rK, 0, N * 4);
      device.queue.submit([enc.finish()]);
      await Promise.all([rJ.mapAsync(GPUMapMode.READ), rK.mapAsync(GPUMapMode.READ)]);
      const J = new Float32Array(rJ.getMappedRange().slice(0));
      const K = new Float32Array(rK.getMappedRange().slice(0));
      rJ.unmap(); rK.unmap();
      return { J, K };
    },
    dispose(): void { device.destroy(); },
  };
}

/** One-shot GPU DF-JK (validation convenience). Prefer makeGpuDFJK for SCF loops. */
export async function buildJK_DF_GPU(df: DFResult, D: Float64Array): Promise<{ J: Float32Array; K: Float32Array }> {
  const g = await makeGpuDFJK(df);
  try { return await g.jk(D); } finally { g.dispose(); }
}
