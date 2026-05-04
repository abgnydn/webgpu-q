// ─────────────────────────────────────────────────────────────
// jacobi-svd.ts — WebGPU one-sided Jacobi SVD for small (n ≤ 32)
// complex matrices. Phase 1B of the GPU MPS port.
//
// Returns the standard SVD triple {U, σ, Vh} where:
//   M = U · diag(σ) · V^H
//   U: m × n with orthonormal columns (here m = n; rectangular
//      support is Phase 1C)
//   σ: length n, sorted descending
//   Vh: n × n unitary, V^H = conj(V)^T
// ─────────────────────────────────────────────────────────────

import shaderSmall from "../shaders/jacobi-svd-small.wgsl?raw";
import shaderMedium from "../shaders/jacobi-svd-medium.wgsl?raw";
import shaderLarge from "../shaders/jacobi-svd-large.wgsl?raw";
import shaderStorage from "../shaders/jacobi-svd-storage.wgsl?raw";
import { type CMatF32, cmatZeros } from "./types.js";
import { poolFor } from "./buffer-pool.js";

// Bytes of workgroup storage each shader needs (sA + sV + sRed, vec2<f32>=8B, vec4<f32>=16B).
const SMALL_BYTES = 24 * 24 * 8 * 2 + 32 * 16;     // ≈ 9 KB
const MEDIUM_BYTES = 32 * 32 * 8 * 2 + 32 * 16;    // ≈ 16.9 KB
const LARGE_BYTES = 48 * 48 * 8 * 2 + 32 * 16;     // ≈ 37 KB
// Storage-mode keeps A, V in global memory and only shuttles 4 columns of
// length n=64 into shared memory per (p, q) rotation. Workgroup cost is
// constant regardless of n: 4 · 64 · 8 + 32 · 16 = 2.5 KB.
const STORAGE_BYTES = 4 * 64 * 8 + 32 * 16;        // ≈ 2.5 KB

interface PipelineSet {
  small: GPUComputePipeline;
  medium: GPUComputePipeline | null;   // null = device too small for n=32 path
  large: GPUComputePipeline | null;    // null = device too small for n=48 path
  storage: GPUComputePipeline;          // n ≤ 64 — works on every adapter
  /** Largest n the device supports given its workgroup storage limit. */
  maxN: number;
}

const pipelineCache: WeakMap<GPUDevice, PipelineSet> = new WeakMap();

function getPipelines(device: GPUDevice): PipelineSet {
  const cached = pipelineCache.get(device);
  if (cached) return cached;

  const wgStorage = device.limits.maxComputeWorkgroupStorageSize;

  const smallMod = device.createShaderModule({ label: "jacobi-svd-small", code: shaderSmall });
  const small = device.createComputePipeline({
    label: "jacobi-svd-small-pipeline",
    layout: "auto",
    compute: { module: smallMod, entryPoint: "jacobi_svd" },
  });

  let medium: GPUComputePipeline | null = null;
  if (wgStorage >= MEDIUM_BYTES) {
    const mediumMod = device.createShaderModule({ label: "jacobi-svd-medium", code: shaderMedium });
    medium = device.createComputePipeline({
      label: "jacobi-svd-medium-pipeline",
      layout: "auto",
      compute: { module: mediumMod, entryPoint: "jacobi_svd" },
    });
  }

  let large: GPUComputePipeline | null = null;
  if (wgStorage >= LARGE_BYTES) {
    const largeMod = device.createShaderModule({ label: "jacobi-svd-large", code: shaderLarge });
    large = device.createComputePipeline({
      label: "jacobi-svd-large-pipeline",
      layout: "auto",
      compute: { module: largeMod, entryPoint: "jacobi_svd" },
    });
  }

  // Storage-mode kernel always fits — its workgroup footprint is constant.
  const storageMod = device.createShaderModule({ label: "jacobi-svd-storage", code: shaderStorage });
  const storage = device.createComputePipeline({
    label: "jacobi-svd-storage-pipeline",
    layout: "auto",
    compute: { module: storageMod, entryPoint: "jacobi_svd" },
  });

  // Storage-mode handles n ≤ 64 on every device, so that's now the floor.
  const maxN = 64;
  void SMALL_BYTES; // referenced in comments
  void STORAGE_BYTES; // documented above
  const set: PipelineSet = { small, medium, large, storage, maxN };
  pipelineCache.set(device, set);
  return set;
}

/** Largest n the device's GPU SVD path will accept (24 or 32). */
export function gpuJacobiSvdMaxN(device: GPUDevice): number {
  return getPipelines(device).maxN;
}

export interface GpuSvdResult {
  /** m × n with orthonormal columns. */
  U: CMatF32;
  /** Length n, sorted descending. */
  sigma: Float32Array;
  /** n × n; V^H = conj(V)^T. */
  Vh: CMatF32;
}

/** GPU one-sided Jacobi SVD; restricted to square n×n matrices with n ≤ 32. */
export async function gpuJacobiSvd(
  device: GPUDevice,
  M: CMatF32,
  opts: { sweeps?: number } = {},
): Promise<GpuSvdResult> {
  if (M.rows !== M.cols) {
    throw new Error(`gpuJacobiSvd v1: square matrices only, got ${M.rows}×${M.cols}`);
  }
  const set = getPipelines(device);
  if (M.rows > set.maxN) {
    throw new Error(
      `gpuJacobiSvd: n=${M.rows} exceeds device max of ${set.maxN} ` +
      `(maxComputeWorkgroupStorageSize=${device.limits.maxComputeWorkgroupStorageSize} B)`,
    );
  }
  const n = M.rows;
  // Jacobi has quadratic convergence near the fixed point. 15 sweeps
  // brings random Gaussian matrices below the f32 floor; bumping to
  // 30 buys nothing measurable for n ≤ 32. Override via opts.sweeps
  // for adversarial matrices where the default is too aggressive.
  const sweeps = opts.sweeps ?? 15;
  // Pick the smallest shared-memory pipeline that fits, falling back to
  // storage-mode (n ≤ 64) when the matrix is bigger than any shared kernel
  // can host. Shared-memory variants are faster per-rotation; storage-mode
  // is the guaranteed-available large-n path.
  const pipeline =
    n <= 24 ? set.small :
    n <= 32 && set.medium ? set.medium :
    n <= 48 && set.large ? set.large :
    set.storage;

  const pool = poolFor(device);
  const byteLen = M.data.byteLength;

  const aBuf = pool.acquire(
    byteLen,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    "svd-A",
  );
  const vBuf = pool.acquire(
    byteLen,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    "svd-V",
  );
  const ab = new ArrayBuffer(byteLen);
  new Float32Array(ab).set(M.data);
  device.queue.writeBuffer(aBuf, 0, ab);

  const paramsBuf = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "svd-params");
  device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([n, sweeps, 0, 0]));

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: vBuf } },
      { binding: 2, resource: { buffer: paramsBuf } },
    ],
  });

  const enc = device.createCommandEncoder({ label: "svd-frame" });
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(1);
  pass.end();

  const aStage = pool.acquire(byteLen, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "svd-A-readback");
  const vStage = pool.acquire(byteLen, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "svd-V-readback");
  enc.copyBufferToBuffer(aBuf, 0, aStage, 0, byteLen);
  enc.copyBufferToBuffer(vBuf, 0, vStage, 0, byteLen);
  device.queue.submit([enc.finish()]);

  await Promise.all([aStage.mapAsync(GPUMapMode.READ), vStage.mapAsync(GPUMapMode.READ)]);
  const aOrth = new Float32Array(aStage.getMappedRange().slice(0));
  const vRaw = new Float32Array(vStage.getMappedRange().slice(0));
  aStage.unmap(); vStage.unmap();

  pool.release(aBuf);
  pool.release(vBuf);
  pool.release(paramsBuf);
  pool.release(aStage);
  pool.release(vStage);

  // Extract σ = ‖A[:,i]‖, U = A / σ. Sort descending.
  const sigmaUnsorted = new Float32Array(n);
  for (let col = 0; col < n; col++) {
    let s = 0;
    for (let row = 0; row < n; row++) {
      const idx = (row * n + col) * 2;
      const re = aOrth[idx]!;
      const im = aOrth[idx + 1]!;
      s += re * re + im * im;
    }
    sigmaUnsorted[col] = Math.sqrt(s);
  }
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => sigmaUnsorted[b]! - sigmaUnsorted[a]!);

  const U = cmatZeros(n, n);
  const sigma = new Float32Array(n);
  const Vh = cmatZeros(n, n);
  for (let i = 0; i < n; i++) {
    const src = order[i]!;
    const s = sigmaUnsorted[src]!;
    sigma[i] = s;
    const inv = s > 0 ? 1 / s : 0;
    for (let row = 0; row < n; row++) {
      const aIdx = (row * n + src) * 2;
      const uIdx = (row * n + i) * 2;
      U.data[uIdx] = aOrth[aIdx]! * inv;
      U.data[uIdx + 1] = aOrth[aIdx + 1]! * inv;
    }
    // Vh[i, c] = conj(V[c, src]).
    for (let col = 0; col < n; col++) {
      const vIdx = (col * n + src) * 2;
      const vhIdx = (i * n + col) * 2;
      Vh.data[vhIdx] = vRaw[vIdx]!;
      Vh.data[vhIdx + 1] = -vRaw[vIdx + 1]!;
    }
  }

  return { U, sigma, Vh };
}

/** Reconstruct M = U · diag(σ) · V^H, useful for cross-validation. */
export function svdReconstruct(svd: GpuSvdResult): CMatF32 {
  const { U, sigma, Vh } = svd;
  const m = U.rows, k = sigma.length, n = Vh.cols;
  const out = cmatZeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let re = 0, im = 0;
      for (let r = 0; r < k; r++) {
        const uIdx = (i * k + r) * 2;
        const vIdx = (r * n + j) * 2;
        const ur = U.data[uIdx]!, ui = U.data[uIdx + 1]!;
        const vr = Vh.data[vIdx]!, vi = Vh.data[vIdx + 1]!;
        // (U · σ)[i,r] = (ur, ui) · σ[r]; (· Vh)[i,j] = sum (Uσ)[i,r] · Vh[r,j]
        const sr = ur * sigma[r]!, si = ui * sigma[r]!;
        re += sr * vr - si * vi;
        im += sr * vi + si * vr;
      }
      const oIdx = (i * n + j) * 2;
      out.data[oIdx] = re;
      out.data[oIdx + 1] = im;
    }
  }
  return out;
}
