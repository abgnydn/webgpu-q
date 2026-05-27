// jk-gpu.ts — WebGPU port of the Fock matrix JK build.
//
// Per HF SCF iteration, compute
//   G[μ, ν] = Σ_{λ, σ} D[λ, σ] · ( (μν|λσ) − ½ (μλ|νσ) )
// on the GPU. ERI tensor is uploaded once per HF call (~830 MB on
// benzene cc-pVDZ f32) and reused across all SCF iterations. Only the
// D matrix changes per iter (small, ~58 KB on benzene).
//
// f32 precision: f32 has ~7 decimal digits. For HF SCF the f32 noise
// in G is typically below the energy tolerance (1e-8 Ha) on cc-pVDZ
// organic molecules. For tighter convergence requirements, use the
// WASM JK path instead via `useWasmJK = true`.
//
// Buffer sizes (benzene cc-pVDZ, n=120, f32):
//   ERI tensor: n⁴·4 = 830 MB    ← biggest, needs raised maxBufferSize
//   D matrix:   n²·4 = 58 KB
//   G matrix:   n²·4 = 58 KB
//   Dims uniform: 16 B
// Total: ~830 MB. M2 Pro typically allows up to ~2 GB per buffer.

import wgslSource from "../shaders/fock-build.wgsl?raw";

interface CachedState {
  device: GPUDevice;
  n: number;
  /** Identity of the ERI tensor — different ERI ⇒ rebuild buffers. */
  eriIdentity: Float64Array | null;
  dimsBuf: GPUBuffer;
  eriBuf: GPUBuffer;
  dBuf: GPUBuffer;
  gBuf: GPUBuffer;
  readbackBuf: GPUBuffer;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  numWorkgroupsX: number;
  numWorkgroupsY: number;
}

let cached: CachedState | null = null;

function buildState(
  device: GPUDevice,
  eri: Float64Array,
  n: number,
): CachedState {
  // ── f64 → f32 conversion for GPU storage. ────────────────
  const eri_f32 = new Float32Array(eri.length);
  for (let i = 0; i < eri.length; i++) eri_f32[i] = eri[i]!;

  const n2 = n * n;
  const n3 = n2 * n;
  const eriBytes = eri_f32.byteLength;
  const dBytes = n2 * 4;
  const gBytes = n2 * 4;

  // ── Allocate buffers. ────────────────────────────────────
  const dimsBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "fock-dims",
  });
  device.queue.writeBuffer(dimsBuf, 0, new Uint32Array([n, n2, n3, 0]));

  const eriBuf = device.createBuffer({
    size: eriBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "fock-eri",
  });
  device.queue.writeBuffer(eriBuf, 0, eri_f32);

  const dBuf = device.createBuffer({
    size: dBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "fock-d",
  });

  const gBuf = device.createBuffer({
    size: gBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: "fock-g",
  });

  const readbackBuf = device.createBuffer({
    size: gBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: "fock-readback",
  });

  // ── Compile shader + pipeline. ───────────────────────────
  const module = device.createShaderModule({ code: wgslSource, label: "fock" });
  const pipeline = device.createComputePipeline({
    label: "fock-pipeline",
    layout: "auto",
    compute: { module, entryPoint: "buildG" },
  });
  const bindGroup = device.createBindGroup({
    label: "fock-bg",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dimsBuf } },
      { binding: 1, resource: { buffer: eriBuf } },
      { binding: 2, resource: { buffer: dBuf } },
      { binding: 3, resource: { buffer: gBuf } },
    ],
  });

  // workgroup_size = (8, 8) → ceil(n / 8) × ceil(n / 8) workgroups.
  const numWorkgroupsX = Math.ceil(n / 8);
  const numWorkgroupsY = Math.ceil(n / 8);

  return {
    device, n, eriIdentity: eri,
    dimsBuf, eriBuf, dBuf, gBuf, readbackBuf,
    pipeline, bindGroup,
    numWorkgroupsX, numWorkgroupsY,
  };
}

function disposeState(s: CachedState): void {
  s.dimsBuf.destroy();
  s.eriBuf.destroy();
  s.dBuf.destroy();
  s.gBuf.destroy();
  s.readbackBuf.destroy();
}

/**
 * Compute the Fock G matrix on the GPU. Pre-uploads the ERI tensor
 * on first call (or when `eri_AO` changes); subsequent calls reuse it.
 *
 * Returns a fresh n×n Float64Array (f32 GPU values are widened to
 * f64 on CPU before return).
 *
 * Caller responsibilities:
 *   - Provide a ready `GPUDevice`. Use `initGPU()` from `src/quantum.ts`.
 *   - Ensure `eri_AO` is the SAME Float64Array across all calls within
 *     one SCF cycle (cache keys on identity). When the ERI changes
 *     (new molecule, new geometry step), the buffers are rebuilt.
 *   - Call `disposeJKGpu()` after SCF to free GPU memory.
 */
export async function buildGGpu(
  device: GPUDevice,
  D: Float64Array,
  eri_AO: Float64Array,
  n: number,
): Promise<Float64Array> {
  // Cache check: same ERI identity → reuse buffers.
  let s: CachedState;
  if (cached && cached.device === device && cached.n === n && cached.eriIdentity === eri_AO) {
    s = cached;
  } else {
    if (cached) disposeState(cached);
    s = buildState(device, eri_AO, n);
    cached = s;
  }

  // ── Upload D (per-iter, small). ──────────────────────────
  const d_f32 = new Float32Array(D.length);
  for (let i = 0; i < D.length; i++) d_f32[i] = D[i]!;
  device.queue.writeBuffer(s.dBuf, 0, d_f32);

  // ── Dispatch + readback. ─────────────────────────────────
  const encoder = device.createCommandEncoder({ label: "fock-encoder" });
  const pass = encoder.beginComputePass({ label: "fock-pass" });
  pass.setPipeline(s.pipeline);
  pass.setBindGroup(0, s.bindGroup);
  pass.dispatchWorkgroups(s.numWorkgroupsX, s.numWorkgroupsY);
  pass.end();
  encoder.copyBufferToBuffer(s.gBuf, 0, s.readbackBuf, 0, n * n * 4);
  device.queue.submit([encoder.finish()]);

  await s.readbackBuf.mapAsync(GPUMapMode.READ);
  const g_f32 = new Float32Array(s.readbackBuf.getMappedRange().slice(0));
  s.readbackBuf.unmap();

  // f32 → f64 widening.
  const G = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) G[i] = g_f32[i]!;
  return G;
}

/** Free GPU buffers. Call after HF SCF completes. */
export function disposeJKGpu(): void {
  if (cached) {
    disposeState(cached);
    cached = null;
  }
}
