// ─────────────────────────────────────────────────────────────
// ccsd-t-gpu.ts — WebGPU port of the CCSD(T) perturbative-triples
// kernel. Tier 2 stage 27.
//
// Design:
//   • Parallel decomposition: 1 GPU thread per (i, j, k) occupied
//     spin-orbital triple. Each thread sums over all (a, b, c)
//     virtual triples internally and emits a single f32 partial
//     sum. CPU reduces the NOCC³ partial sums in f64 for final
//     E_(T).
//   • f32 storage on GPU, f64 CPU reduction. Per-thread sum
//     magnitude is bounded by ~µHa to mHa for closed-shell
//     molecules near equilibrium; f32 → f64 rollup error is at
//     the ~10⁻⁹ Ha level even on millions of terms (Kahan-free).
//   • Workgroup size 64. For STO-3G H₂O (NOCC=10) only 16
//     workgroups dispatch — under-utilization at small basis is
//     expected. cc-pVDZ H₂O (NOCC=10) dispatches the same outer
//     count but each thread does NVIRT³·9 ≈ 500k ops, so total
//     compute is ~0.5 GFLOPS — well-suited to GPU.
//
// API matches the CPU `runCCSDT`. Returns the same `CCSDTResult`
// shape with `seconds` reflecting GPU dispatch + readback time.
//
// Validation: requires browser / Playwright (WebGPU not in
// node-vitest). The kernel implements the exact same Stanton-
// Bartlett spin-orbital sigma equations as `runCCSDT`; a
// per-system e2e check (E_(T)_GPU ≈ E_(T)_CPU within f32 noise)
// is the canonical validation.
//
// BENCHMARK DISCLOSURE: the speedup numbers reported in
// CLAUDE.md (e.g. "39.3× on H₂O cc-pVDZ", "13.9× on H₂O STO-3G")
// come from a single e2e run on Apple M2 Pro. They are NOT
// produced by the warmup+20-trials research harness used by the
// E1–E16 experiments — run-to-run variation is unmeasured.
// They validate that the GPU port is correct and substantially
// faster than CPU on production-relevant basis sizes; they do
// NOT meet the project's normal research-grade timing standard
// (see RESEARCH.md §Timing). Future work: route through
// `experiments/lib/runner.ts → timedRun` with the standard
// warmup + trials methodology.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";
import type { CCSDResult } from "./ccsd.js";
import type { CCSDTResult, CCSDTOpts } from "./ccsd-t.js";
import { transformERIToMO } from "./mp2.js";
import { buildSpinOrbitalERI } from "./ccsd.js";

import wgslSource from "../shaders/ccsd-t.wgsl?raw";

export interface CCSDTGPUResult extends CCSDTResult {
  /** GPU-side compute time only (excludes ERI build / buffer setup). */
  readonly gpuSeconds: number;
  /** Total buffer footprint on GPU (bytes). */
  readonly gpuMemoryBytes: number;
}

/**
 * Run CCSD(T) perturbative-triples on a WebGPU device.
 * Returns the same numeric answer as `runCCSDT` to f32 reduction
 * precision (~10⁻⁹ Ha on STO-3G H₂O class systems).
 *
 * Caller must provide a ready `GPUDevice`. Use `initGPU()` from
 * `src/quantum.ts` or `navigator.gpu.requestAdapter` / `requestDevice`
 * pattern. Buffer-size limits: ERI tensor is NSO⁴·4 bytes — for
 * cc-pVDZ H₂O (NSO=48) that's 21 MB, well within typical 256 MB
 * caps.
 */
export async function runCCSDT_GPU(
  ccsd: CCSDResult,
  hf: HFResult,
  integrals: MolecularIntegrals,
  device: GPUDevice,
  opts: CCSDTOpts = {},
): Promise<CCSDTGPUResult> {
  const tStartTotal = performance.now();
  const n = integrals.n;
  const NSO = 2 * n;
  const NOCC = 2 * hf.nOccupied;
  const NVIRT = NSO - NOCC;
  const nFrozenSO = 2 * (opts.nFrozenCore ?? 0);

  if (NOCC - nFrozenSO < 3 || NVIRT < 3) {
    return {
      tripleCorrection: 0,
      totalEnergy: ccsd.totalEnergy,
      seconds: 0,
      gpuSeconds: 0,
      gpuMemoryBytes: 0,
    };
  }

  // ── Build spin-orbital ERI + ε (same as CPU runCCSDT). ───
  const eri_MO = transformERIToMO(integrals.eri_AO, hf.C_MO, n);
  const eri_f64 = buildSpinOrbitalERI(eri_MO, n);
  const eps_f64 = new Float64Array(NSO);
  for (let P = 0; P < NSO; P++) eps_f64[P] = hf.orbitalEnergies[P >> 1]!;

  // ── f64 → f32 conversion for GPU storage. ────────────────
  const eri_f32 = new Float32Array(eri_f64.length);
  for (let i = 0; i < eri_f64.length; i++) eri_f32[i] = eri_f64[i]!;
  const T1_f32 = new Float32Array(ccsd.T1.length);
  for (let i = 0; i < ccsd.T1.length; i++) T1_f32[i] = ccsd.T1[i]!;
  const T2_f32 = new Float32Array(ccsd.T2.length);
  for (let i = 0; i < ccsd.T2.length; i++) T2_f32[i] = ccsd.T2[i]!;
  const eps_f32 = new Float32Array(NSO);
  for (let i = 0; i < NSO; i++) eps_f32[i] = eps_f64[i]!;

  // ── Allocate GPU buffers. ────────────────────────────────
  const eriBytes = eri_f32.byteLength;
  const t1Bytes = T1_f32.byteLength;
  const t2Bytes = T2_f32.byteLength;
  const epsBytes = eps_f32.byteLength;
  const numTriples = NOCC * NOCC * NOCC;
  const partialBytes = numTriples * 4;
  const totalBytes = eriBytes + t1Bytes + t2Bytes + epsBytes + partialBytes + 16;

  // Dimensions uniform: { NSO, NOCC, NVIRT, nFrozenSO } as u32×4.
  const dimsBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "ccsd-t-dims",
  });
  device.queue.writeBuffer(
    dimsBuf,
    0,
    new Uint32Array([NSO, NOCC, NVIRT, nFrozenSO]),
  );

  const eriBuf = device.createBuffer({
    size: eriBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "ccsd-t-eri",
  });
  device.queue.writeBuffer(eriBuf, 0, eri_f32);

  const t1Buf = device.createBuffer({
    size: t1Bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "ccsd-t-T1",
  });
  device.queue.writeBuffer(t1Buf, 0, T1_f32);

  const t2Buf = device.createBuffer({
    size: t2Bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "ccsd-t-T2",
  });
  device.queue.writeBuffer(t2Buf, 0, T2_f32);

  const epsBuf = device.createBuffer({
    size: epsBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "ccsd-t-eps",
  });
  device.queue.writeBuffer(epsBuf, 0, eps_f32);

  const partialBuf = device.createBuffer({
    size: partialBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: "ccsd-t-partial",
  });

  const readbackBuf = device.createBuffer({
    size: partialBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: "ccsd-t-readback",
  });

  // ── Compile shader + pipeline. ───────────────────────────
  const module = device.createShaderModule({ code: wgslSource, label: "ccsd-t" });
  const pipeline = device.createComputePipeline({
    label: "ccsd-t-pipeline",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    label: "ccsd-t-bg",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dimsBuf } },
      { binding: 1, resource: { buffer: eriBuf } },
      { binding: 2, resource: { buffer: t1Buf } },
      { binding: 3, resource: { buffer: t2Buf } },
      { binding: 4, resource: { buffer: epsBuf } },
      { binding: 5, resource: { buffer: partialBuf } },
    ],
  });

  // ── Dispatch + readback. ─────────────────────────────────
  const workgroupSize = 64;
  const numWorkgroups = Math.ceil(numTriples / workgroupSize);

  const tStartGPU = performance.now();
  const encoder = device.createCommandEncoder({ label: "ccsd-t-encoder" });
  const pass = encoder.beginComputePass({ label: "ccsd-t-pass" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(numWorkgroups);
  pass.end();
  encoder.copyBufferToBuffer(partialBuf, 0, readbackBuf, 0, partialBytes);
  device.queue.submit([encoder.finish()]);

  await readbackBuf.mapAsync(GPUMapMode.READ);
  const partials = new Float32Array(readbackBuf.getMappedRange().slice(0));
  readbackBuf.unmap();
  const gpuSeconds = (performance.now() - tStartGPU) / 1000;

  // ── f64 reduction on CPU. ────────────────────────────────
  let E_T = 0;
  for (let i = 0; i < partials.length; i++) E_T += partials[i]!;

  // Cleanup.
  dimsBuf.destroy();
  eriBuf.destroy();
  t1Buf.destroy();
  t2Buf.destroy();
  epsBuf.destroy();
  partialBuf.destroy();
  readbackBuf.destroy();

  const seconds = (performance.now() - tStartTotal) / 1000;
  return {
    tripleCorrection: E_T,
    totalEnergy: ccsd.totalEnergy + E_T,
    seconds,
    gpuSeconds,
    gpuMemoryBytes: totalBytes,
  };
}
