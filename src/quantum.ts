// ─────────────────────────────────────────────────────────────
// QuantumCircuit — WebGPU statevector simulator.
//
// Holds a statevector of 2^N complex amplitudes on the GPU as a
// storage buffer of vec2<f32>. Gates dispatch compute shaders
// that mutate the statevector in place.
//
// Memory footprint:
//   N qubits → 2^N amplitudes × 8 bytes = 2^(N+3) bytes
//   20 qubits =   8 MB
//   25 qubits = 256 MB   ← needs requiredLimits up-bump
//   26 qubits = 512 MB   ← at the edge of current WebGPU max
//
// Gates currently available:
//   single-qubit : apply(matrix, qubit)
//   two-qubit    : applyControlled(matrix, control, target)
//                  → CNOT = applyControlled(X, c, t)
// ─────────────────────────────────────────────────────────────

import singleQubitSrc from "./shaders/single-qubit.wgsl?raw";
import twoQubitSrc from "./shaders/two-qubit.wgsl?raw";
import twoQubitDenseSrc from "./shaders/two-qubit-dense.wgsl?raw";
import threeQubitDenseSrc from "./shaders/three-qubit-dense.wgsl?raw";
import fourQubitDenseSrc from "./shaders/four-qubit-dense.wgsl?raw";
import { X, type Matrix2, matrixFloats } from "./gates.js";
import { type Matrix4, matrix4Floats } from "./two-qubit-dense.js";
import { type Matrix8, matrix8Floats } from "./three-qubit-dense.js";
import { type Matrix16, matrix16Floats } from "./four-qubit-dense.js";
import { FUSED_CHAIN_MAX_K } from "./fusion-max-k.js";
import { FusedPipelineCache } from "./fusion-jit.js";

export { FUSED_CHAIN_MAX_K };

// Fire-and-forget: read the shader module's compilation info and loudly
// log any error-severity messages. Without this, WGSL mistakes (reserved
// keywords, bad types) are silently fatal — the pipeline is created but
// every dispatch becomes a no-op.
function reportCompilationErrors(mod: GPUShaderModule, label: string): void {
  mod.getCompilationInfo().then((info) => {
    for (const m of info.messages) {
      if (m.type === "error") {
         
        console.error(`[webgpu-q] ${label}:${m.lineNum}:${m.linePos} ${m.message}`);
      } else if (m.type === "warning") {
         
        console.warn(`[webgpu-q] ${label}:${m.lineNum}:${m.linePos} ${m.message}`);
      }
    }
  });
}

export interface QuantumCircuitInit {
  device: GPUDevice;
  nQubits: number;
}

export class QuantumCircuit {
  readonly device: GPUDevice;
  readonly nQubits: number;
  readonly nStates: number;

  private readonly psi: GPUBuffer;
  private readonly singlePipeline: GPUComputePipeline;
  private readonly twoPipeline: GPUComputePipeline;
  private readonly twoDensePipeline: GPUComputePipeline;
  private readonly threeDensePipeline: GPUComputePipeline;
  private readonly fourDensePipeline: GPUComputePipeline;
  private readonly fusedCache: FusedPipelineCache;
  // One uniform buffer recycled per dispatch (small, 48 B).
  private readonly singleParamsBuf: GPUBuffer;
  private readonly twoParamsBuf: GPUBuffer;
  // Dense 4×4: 16 vec2<f32> (128 B) + 4 u32 (16 B) = 144 B; round to 160.
  private readonly twoDenseParamsBuf: GPUBuffer;
  // Dense 8×8: 32 vec4<f32> (512 B, packed pairs) + 8 u32 (32 B) = 544 B.
  private readonly threeDenseParamsBuf: GPUBuffer;
  // Dense 16×16: 128 vec4<f32> (2048 B, packed pairs) + 8 u32 (32 B) = 2080 B.
  private readonly fourDenseParamsBuf: GPUBuffer;
  // Fused chain: 16 B header + MAX_K × 32 B gate slots.
  private readonly fusedParamsBuf: GPUBuffer;

  private gateCount = 0;

  constructor({ device, nQubits }: QuantumCircuitInit) {
    if (nQubits < 1 || nQubits > 28) {
      throw new Error(`nQubits must be in [1, 28]. Got ${nQubits}.`);
    }
    this.device = device;
    this.nQubits = nQubits;
    this.nStates = 1 << nQubits;

    // ── statevector buffer: 2^N × vec2<f32> = 2^(N+3) bytes ──
    const psiBytes = this.nStates * 8;
    this.psi = device.createBuffer({
      size: psiBytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    // Initialize to |0...0⟩
    const init = new Float32Array(this.psi.getMappedRange());
    init[0] = 1.0; // |0⟩ amplitude = 1 + 0i
    this.psi.unmap();

    // ── pipelines ────────────────────────────────────────────
    const singleMod = device.createShaderModule({
      label: "single-qubit",
      code: singleQubitSrc,
    });
    reportCompilationErrors(singleMod, "single-qubit.wgsl");
    this.singlePipeline = device.createComputePipeline({
      label: "single-qubit-pipeline",
      layout: "auto",
      compute: { module: singleMod, entryPoint: "apply" },
    });

    const twoMod = device.createShaderModule({
      label: "two-qubit",
      code: twoQubitSrc,
    });
    reportCompilationErrors(twoMod, "two-qubit.wgsl");
    this.twoPipeline = device.createComputePipeline({
      label: "two-qubit-pipeline",
      layout: "auto",
      compute: { module: twoMod, entryPoint: "apply" },
    });

    const twoDenseMod = device.createShaderModule({
      label: "two-qubit-dense",
      code: twoQubitDenseSrc,
    });
    reportCompilationErrors(twoDenseMod, "two-qubit-dense.wgsl");
    this.twoDensePipeline = device.createComputePipeline({
      label: "two-qubit-dense-pipeline",
      layout: "auto",
      compute: { module: twoDenseMod, entryPoint: "apply" },
    });

    const threeDenseMod = device.createShaderModule({
      label: "three-qubit-dense",
      code: threeQubitDenseSrc,
    });
    reportCompilationErrors(threeDenseMod, "three-qubit-dense.wgsl");
    this.threeDensePipeline = device.createComputePipeline({
      label: "three-qubit-dense-pipeline",
      layout: "auto",
      compute: { module: threeDenseMod, entryPoint: "apply" },
    });

    const fourDenseMod = device.createShaderModule({
      label: "four-qubit-dense",
      code: fourQubitDenseSrc,
    });
    reportCompilationErrors(fourDenseMod, "four-qubit-dense.wgsl");
    this.fourDensePipeline = device.createComputePipeline({
      label: "four-qubit-dense-pipeline",
      layout: "auto",
      compute: { module: fourDenseMod, entryPoint: "apply" },
    });

    // Fused chains use JIT-compiled, loop-unrolled shaders: one pipeline
    // per chain length k, cached on the device. See fusion-jit.ts for why
    // (runtime-bounded loops don't unroll cleanly and leave a non-amortizing
    // residual C in the α_eff(k) = α_raw/k + C measurement).
    this.fusedCache = sharedFusedCache(device);

    // ── uniform buffers ──────────────────────────────────────
    // Single-qubit: 4 vec2<f32> (matrix) + 4 u32 (qubit, n_qubits, _, _) = 48 B
    this.singleParamsBuf = device.createBuffer({
      label: "single-params",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Two-qubit: 4 vec2<f32> + 4 u32 = 48 B
    this.twoParamsBuf = device.createBuffer({
      label: "two-params",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Dense 4×4: 16 vec2<f32> (128) + 4 u32 (16) = 144 → 160 (16-byte aligned).
    this.twoDenseParamsBuf = device.createBuffer({
      label: "two-dense-params",
      size: 160,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Dense 8×8: 32 vec4<f32> (512) + 8 u32 (32) = 544 (16-byte aligned).
    this.threeDenseParamsBuf = device.createBuffer({
      label: "three-dense-params",
      size: 544,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Dense 16×16: 128 vec4<f32> (2048) + 8 u32 (32) = 2080 (16-byte aligned).
    this.fourDenseParamsBuf = device.createBuffer({
      label: "four-dense-params",
      size: 2080,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Fused chain: 16 B header + 64 × 32 B = 2064 B.
    this.fusedParamsBuf = device.createBuffer({
      label: "fused-chain-params",
      size: 16 + FUSED_CHAIN_MAX_K * 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  // ─── single-qubit gate ─────────────────────────────────────
  apply(matrix: Matrix2, qubit: number): void {
    this.assertQubit(qubit);

    // One thread per pair; dispatch 2D to fit within maxComputeWorkgroupsPerDimension.
    const nPairs = this.nStates >> 1;
    const { wgX, wgY, strideInv } = this.planDispatch(nPairs);

    const u = new ArrayBuffer(48);
    new Float32Array(u, 0, 8).set(matrixFloats(matrix));
    new Uint32Array(u, 32, 4).set([qubit, this.nQubits, strideInv, 0]);
    this.device.queue.writeBuffer(this.singleParamsBuf, 0, u);

    const bindGroup = this.device.createBindGroup({
      layout: this.singlePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.psi } },
        { binding: 1, resource: { buffer: this.singleParamsBuf } },
      ],
    });

    const enc = this.device.createCommandEncoder({ label: `gate-${this.gateCount++}` });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.singlePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  // ─── fused single-qubit chain ─────────────────────────────
  //
  // Apply matrices[0], matrices[1], …, matrices[k−1] to `qubit` in
  // order, all in one dispatch. Algebraically identical to calling
  // `apply(matrices[i], qubit)` in a loop — each thread holds its
  // (i, j) pair in registers and runs k 2×2 matmuls between the
  // single load and single store.
  //
  // Caps at FUSED_CHAIN_MAX_K (64). For longer streams, callers
  // should chunk into multiple dispatches or lift MAX_K in both
  // the shader and here in lockstep.
  applyFusedSingle(matrices: readonly Matrix2[], qubit: number): void {
    this.assertQubit(qubit);
    const k = matrices.length;
    if (k === 0) return;
    if (k > FUSED_CHAIN_MAX_K) {
      throw new Error(
        `applyFusedSingle: chain length ${k} > FUSED_CHAIN_MAX_K (${FUSED_CHAIN_MAX_K}).`,
      );
    }

    const nPairs = this.nStates >> 1;
    const { wgX, wgY, strideInv } = this.planDispatch(nPairs);

    // Header: qubit, n_qubits, k, stride_inv (16 B).
    // Body:   MAX_K × Gate = MAX_K × 4 × vec2<f32> = MAX_K × 32 B.
    // Gates beyond k are don't-care.
    const size = 16 + FUSED_CHAIN_MAX_K * 32;
    const u = new ArrayBuffer(size);
    new Uint32Array(u, 0, 4).set([qubit, this.nQubits, k, strideInv]);
    const floats = new Float32Array(u, 16, FUSED_CHAIN_MAX_K * 8);
    for (let g = 0; g < k; g++) {
      floats.set(matrixFloats(matrices[g]!), g * 8);
    }
    this.device.queue.writeBuffer(this.fusedParamsBuf, 0, u);

    const pipeline = this.fusedCache.get(k);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.psi } },
        { binding: 1, resource: { buffer: this.fusedParamsBuf } },
      ],
    });

    const enc = this.device.createCommandEncoder({ label: `fused-${this.gateCount++}(k=${k})` });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Pre-compile JIT pipelines for a set of chain lengths. Pipeline creation
   * is expensive (shader compile + Metal/SPIR-V lowering), so benchmarks
   * that want to measure only warm-dispatch cost should warm the cache first.
   */
  warmFusedCache(ks: Iterable<number>): void {
    this.fusedCache.warm(ks);
  }

  /** Cold-compile wall time per k, in ms. Useful for E9's "cold-start α" diagnostic. */
  fusedColdCompileMs(): ReadonlyMap<number, number> {
    return this.fusedCache.coldCompileMs;
  }

  // ─── dense 4×4 two-qubit gate ─────────────────────────────
  //
  // Apply an arbitrary 4×4 complex unitary U to the qubit pair
  // (qLo, qHi). One dispatch reads/writes the full statevector.
  //
  // Use this to fuse a brick-wall layer's local block — the
  // single-qubit gate on qLo, the single-qubit gate on qHi, and
  // the CNOT(qLo, qHi) collapse into one 4×4 matrix on the CPU
  // (`fuseBrickwallPair` in two-qubit-dense.ts) and one dispatch
  // here. 3 gates → 1 dispatch and a single round-trip through
  // the statevector instead of three.
  //
  // qLo must be strictly less than qHi (the shader assumes ordered
  // bit insertion). Caller can use min/max to normalize.
  applyDense4x4(matrix: Matrix4, qLo: number, qHi: number): void {
    this.assertQubit(qLo);
    this.assertQubit(qHi);
    if (qLo >= qHi) throw new Error(`applyDense4x4: need qLo < qHi, got (${qLo}, ${qHi})`);

    // One thread per amplitude block; each block touches 4 amps.
    const nBlocks = this.nStates >> 2;
    const { wgX, wgY, strideInv } = this.planDispatch(nBlocks);

    const u = new ArrayBuffer(160);
    new Float32Array(u, 0, 32).set(matrix4Floats(matrix));
    new Uint32Array(u, 128, 4).set([qLo, qHi, this.nQubits, strideInv]);
    this.device.queue.writeBuffer(this.twoDenseParamsBuf, 0, u);

    const bindGroup = this.device.createBindGroup({
      layout: this.twoDensePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.psi } },
        { binding: 1, resource: { buffer: this.twoDenseParamsBuf } },
      ],
    });

    const enc = this.device.createCommandEncoder({ label: `dense4-${this.gateCount++}` });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.twoDensePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  // ─── dense 8×8 three-qubit gate (Tier C) ──────────────────
  //
  // Apply an arbitrary 8×8 complex unitary U to the qubit triple
  // (qLo, qMid, qHi) with qLo < qMid < qHi. One dispatch reads /
  // writes the full statevector; each thread processes an 8-amp
  // block via a 64-cmul matvec held in registers.
  //
  // Use this to fuse 5-op cascades (3 singles + 2 CNOTs) or two
  // adjacent brick-wall sub-layers into a single dispatch. The
  // wider tile collapses more ops than `applyDense4x4` per dispatch
  // — at the cost of 8× the per-block FLOPs (still bandwidth-bound).
  applyDense8x8(
    matrix: Matrix8,
    qLo: number,
    qMid: number,
    qHi: number,
  ): void {
    this.assertQubit(qLo);
    this.assertQubit(qMid);
    this.assertQubit(qHi);
    if (!(qLo < qMid && qMid < qHi)) {
      throw new Error(
        `applyDense8x8: need qLo < qMid < qHi, got (${qLo}, ${qMid}, ${qHi})`,
      );
    }
    if (this.nQubits < 3) throw new Error(`applyDense8x8: need n ≥ 3, got ${this.nQubits}`);

    const nBlocks = this.nStates >> 3;
    const { wgX, wgY, strideInv } = this.planDispatch(nBlocks);

    const u = new ArrayBuffer(544);
    new Float32Array(u, 0, 128).set(matrix8Floats(matrix));
    new Uint32Array(u, 512, 8).set([qLo, qMid, qHi, this.nQubits, strideInv, 0, 0, 0]);
    this.device.queue.writeBuffer(this.threeDenseParamsBuf, 0, u);

    const bindGroup = this.device.createBindGroup({
      layout: this.threeDensePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.psi } },
        { binding: 1, resource: { buffer: this.threeDenseParamsBuf } },
      ],
    });

    const enc = this.device.createCommandEncoder({ label: `dense8-${this.gateCount++}` });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.threeDensePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  // ─── dense 16×16 four-qubit gate (Tier D) ─────────────────
  //
  // Apply an arbitrary 16×16 complex unitary U to the qubit
  // quadruple (qLo, qMid1, qMid2, qHi) with strict ordering.
  // One dispatch reads / writes the full statevector; each
  // thread processes a 16-amp block via a 256-cmul matvec held
  // in registers.
  //
  // Folds 7-op cascades (4 singles + 3 cascading CNOTs) into
  // a single dispatch — same thesis as Tier B/C with a wider
  // window for more dispatch collapse per tile.
  applyDense16x16(
    matrix: Matrix16,
    qLo: number,
    qMid1: number,
    qMid2: number,
    qHi: number,
  ): void {
    this.assertQubit(qLo);
    this.assertQubit(qMid1);
    this.assertQubit(qMid2);
    this.assertQubit(qHi);
    if (!(qLo < qMid1 && qMid1 < qMid2 && qMid2 < qHi)) {
      throw new Error(
        `applyDense16x16: need qLo < qMid1 < qMid2 < qHi, got (${qLo}, ${qMid1}, ${qMid2}, ${qHi})`,
      );
    }
    if (this.nQubits < 4) throw new Error(`applyDense16x16: need n ≥ 4, got ${this.nQubits}`);

    const nBlocks = this.nStates >> 4;
    const { wgX, wgY, strideInv } = this.planDispatch(nBlocks);

    const u = new ArrayBuffer(2080);
    new Float32Array(u, 0, 512).set(matrix16Floats(matrix));
    new Uint32Array(u, 2048, 8).set([qLo, qMid1, qMid2, qHi, this.nQubits, strideInv, 0, 0]);
    this.device.queue.writeBuffer(this.fourDenseParamsBuf, 0, u);

    const bindGroup = this.device.createBindGroup({
      layout: this.fourDensePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.psi } },
        { binding: 1, resource: { buffer: this.fourDenseParamsBuf } },
      ],
    });

    const enc = this.device.createCommandEncoder({ label: `dense16-${this.gateCount++}` });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.fourDensePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  // ─── controlled-U (two-qubit) ─────────────────────────────
  applyControlled(matrix: Matrix2, control: number, target: number): void {
    this.assertQubit(control);
    this.assertQubit(target);
    if (control === target) {
      throw new Error(`control and target must differ, both = ${control}`);
    }

    // One thread per state; thread exits if (control, target) bits don't qualify.
    const { wgX, wgY, strideInv } = this.planDispatch(this.nStates);

    const u = new ArrayBuffer(48);
    new Float32Array(u, 0, 8).set(matrixFloats(matrix));
    new Uint32Array(u, 32, 4).set([control, target, this.nQubits, strideInv]);
    this.device.queue.writeBuffer(this.twoParamsBuf, 0, u);

    const bindGroup = this.device.createBindGroup({
      layout: this.twoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.psi } },
        { binding: 1, resource: { buffer: this.twoParamsBuf } },
      ],
    });

    const enc = this.device.createCommandEncoder({ label: `cgate-${this.gateCount++}` });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.twoPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  // ─── compute a 2D workgroup dispatch for `nThreads` invocations ──
  //
  // Both kernels use @workgroup_size(256). WebGPU caps workgroups per
  // dimension at 65535 — so for N ≥ ~21 qubits we need wgY > 1.
  //
  // Returns (wgX, wgY, strideInv) where
  //   strideInv = wgX * WG_SIZE
  // is the linear-invocation stride the shader uses to reconstruct
  // its thread id: `tid = gid.y * strideInv + gid.x`.
  private planDispatch(nThreads: number): { wgX: number; wgY: number; strideInv: number } {
    const WG_SIZE = 256;
    const totalWG = Math.max(1, Math.ceil(nThreads / WG_SIZE));
    const maxDim = this.device.limits.maxComputeWorkgroupsPerDimension;
    const wgX = Math.min(totalWG, maxDim);
    const wgY = Math.ceil(totalWG / wgX);
    return { wgX, wgY, strideInv: wgX * WG_SIZE };
  }

  cnot(control: number, target: number): void {
    this.applyControlled(X, control, target);
  }

  // ─── readback: full amplitudes (expensive) ────────────────
  async amplitudes(): Promise<Float32Array> {
    const size = this.nStates * 8;
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.psi, 0, staging, 0, size);
    this.device.queue.submit([enc.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return copy;
  }

  // ─── readback: probabilities only ─────────────────────────
  async probabilities(): Promise<Float32Array> {
    const amps = await this.amplitudes();
    const probs = new Float32Array(this.nStates);
    for (let i = 0; i < this.nStates; i++) {
      const re = amps[i * 2];
      const im = amps[i * 2 + 1];
      probs[i] = (re ?? 0) ** 2 + (im ?? 0) ** 2;
    }
    return probs;
  }

  /** Sum of |ψ_i|² — should stay at 1.0 within f32 precision. */
  async norm(): Promise<number> {
    const p = await this.probabilities();
    let s = 0;
    for (const x of p) s += x;
    return s;
  }

  /** Reset to |0…0⟩ without reallocating. */
  reset(): void {
    this.device.queue.writeBuffer(
      this.psi,
      0,
      new Float32Array(this.nStates * 2),
    );
    // |0⟩ amplitude = 1
    this.device.queue.writeBuffer(this.psi, 0, new Float32Array([1, 0]));
  }

  dispose(): void {
    this.psi.destroy();
    this.singleParamsBuf.destroy();
    this.twoParamsBuf.destroy();
    this.twoDenseParamsBuf.destroy();
    this.threeDenseParamsBuf.destroy();
    this.fourDenseParamsBuf.destroy();
    this.fusedParamsBuf.destroy();
  }

  private assertQubit(q: number): void {
    if (!Number.isInteger(q) || q < 0 || q >= this.nQubits) {
      throw new Error(`qubit index ${q} out of range [0, ${this.nQubits})`);
    }
  }
}

// ─────────────────────────────────────────────────────────────
// One FusedPipelineCache per GPUDevice, shared across circuits.
// Recompiling a loop-unrolled shader per QuantumCircuit would be
// wasteful — cache by device so every circuit on that device hits
// warm pipelines after the first use.
// ─────────────────────────────────────────────────────────────
const fusedCacheByDevice = new WeakMap<GPUDevice, FusedPipelineCache>();

function sharedFusedCache(device: GPUDevice): FusedPipelineCache {
  const cached = fusedCacheByDevice.get(device);
  if (cached) return cached;
  const fresh = new FusedPipelineCache(device);
  fusedCacheByDevice.set(device, fresh);
  return fresh;
}

// ─────────────────────────────────────────────────────────────
// GPU bootstrap — asks the adapter for its max buffer size so
// 25-qubit statevectors (256 MB) actually fit.
// ─────────────────────────────────────────────────────────────
export async function initGPU(): Promise<GPUDevice> {
  if (!("gpu" in navigator)) {
    throw new Error("WebGPU not available in this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw new Error("No GPU adapter.");

  const maxStorage = adapter.limits.maxStorageBufferBindingSize;
  const maxBuffer = adapter.limits.maxBufferSize;
  // Phase 1C of the GPU-MPS port wants ≥17 KB of workgroup storage to host
  // the 32×32 single-workgroup Jacobi SVD kernel. Most adapters expose
  // 32 KB; ask for whatever the adapter has so headless Chromium (which
  // defaults to the 16 KB spec floor) bumps up too.
  const maxWgStorage = adapter.limits.maxComputeWorkgroupStorageSize;

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: maxStorage,
      maxBufferSize: maxBuffer,
      maxComputeWorkgroupStorageSize: maxWgStorage,
    },
  });

  // Shout loud on every validation / OOM event. We learned the hard way
  // that silent WGSL compile errors (e.g. using the reserved keyword
  // `target` as a struct field) turn dispatches into no-ops — you get
  // a normalized statevector that silently disagrees with the CPU.
  device.addEventListener("uncapturederror", (ev) => {
    const e = (ev as unknown as { error?: GPUError }).error;
    const msg = e && "message" in e ? (e as { message: string }).message : String(e);
     
    console.error("[webgpu-q]", msg);
  });

  return device;
}
