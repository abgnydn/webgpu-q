// ─────────────────────────────────────────────────────────────
// matmul.ts — WebGPU complex matrix multiplication.
//
// Each call uploads A and B, dispatches the cmatmul shader,
// reads back C. Pipeline + buffer pools could amortize uploads
// across many calls; for the v1 prototype we keep it simple.
// ─────────────────────────────────────────────────────────────

import shaderSrc from "../shaders/complex-matmul.wgsl?raw";
import { type CMatF32, cmatZeros } from "./types.js";
import { poolFor } from "./buffer-pool.js";

const pipelineCache: WeakMap<GPUDevice, GPUComputePipeline> = new WeakMap();

function getPipeline(device: GPUDevice): GPUComputePipeline {
  let p = pipelineCache.get(device);
  if (p) return p;
  const mod = device.createShaderModule({ label: "complex-matmul", code: shaderSrc });
  p = device.createComputePipeline({
    label: "complex-matmul-pipeline",
    layout: "auto",
    compute: { module: mod, entryPoint: "cmatmul" },
  });
  pipelineCache.set(device, p);
  return p;
}

function alignBufSize(bytes: number): number {
  return Math.max(16, Math.ceil(bytes / 4) * 4);
}

function uploadToPooled(device: GPUDevice, label: string, m: CMatF32, usage: GPUBufferUsageFlags): GPUBuffer {
  const buf = poolFor(device).acquire(alignBufSize(m.data.byteLength), usage | GPUBufferUsage.COPY_DST, label);
  const ab = new ArrayBuffer(m.data.byteLength);
  new Float32Array(ab).set(m.data);
  device.queue.writeBuffer(buf, 0, ab);
  return buf;
}

/** Compute C = A · B on the GPU. */
export async function gpuCMatMul(device: GPUDevice, A: CMatF32, B: CMatF32): Promise<CMatF32> {
  if (A.cols !== B.rows) {
    throw new Error(`gpuCMatMul shape: ${A.rows}×${A.cols} · ${B.rows}×${B.cols}`);
  }
  const C = cmatZeros(A.rows, B.cols);
  const pipeline = getPipeline(device);
  const pool = poolFor(device);

  const aBuf = uploadToPooled(device, "matmul-A", A, GPUBufferUsage.STORAGE);
  const bBuf = uploadToPooled(device, "matmul-B", B, GPUBufferUsage.STORAGE);
  const cBuf = pool.acquire(
    alignBufSize(C.data.byteLength),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    "matmul-C",
  );

  const paramsBuf = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "matmul-params");
  device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([A.rows, A.cols, B.cols, 0]));

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: paramsBuf } },
    ],
  });

  const wgX = Math.max(1, Math.ceil(A.rows / 8));
  const wgY = Math.max(1, Math.ceil(B.cols / 8));

  const enc = device.createCommandEncoder({ label: "matmul-frame" });
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  const stage = pool.acquire(
    cBuf.size,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    "matmul-readback",
  );
  enc.copyBufferToBuffer(cBuf, 0, stage, 0, cBuf.size);
  device.queue.submit([enc.finish()]);

  await stage.mapAsync(GPUMapMode.READ);
  const view = new Float32Array(stage.getMappedRange());
  C.data.set(view.subarray(0, C.data.length));
  stage.unmap();

  // Return everything to the pool for the next call.
  pool.release(aBuf);
  pool.release(bBuf);
  pool.release(cBuf);
  pool.release(paramsBuf);
  pool.release(stage);

  return C;
}
