// ─────────────────────────────────────────────────────────────
// mps-gpu.ts — Phase 4a of the GPU MPS port: GPU-resident
// tensor storage with a single-qubit-gate kernel and a
// statevector-contraction readback. Two-site gates are still
// CPU-routed in 4a; 4b moves them onto the GPU.
//
// Storage: each site has its own GPUBuffer holding the tensor
// data as `array<vec2<f32>>` (interleaved [re, im]). Bond
// dimensions are tracked CPU-side in `siteShape[]` since they
// drift during evolution. Buffers are sized for the maximum
// possible (χ_max · 2 · χ_max) shape so we never have to
// reallocate.
//
// The single-qubit gate dispatches a kernel per call but stays
// fully on the GPU — no roundtrip per single-qubit gate. For a
// circuit dominated by single-qubit gates (preparation layers,
// many circuits' warmups), this alone removes per-gate CPU
// overhead.
// ─────────────────────────────────────────────────────────────

import singleQubitSrc from "../shaders/mps-single-qubit.wgsl?raw";
import mergeSrc from "../shaders/mps-two-site-merge.wgsl?raw";
import colNormsSrc from "../shaders/column-norms.wgsl?raw";
import extractUSrc from "../shaders/mps-two-site-extract-u.wgsl?raw";
import extractUSSrc from "../shaders/mps-two-site-extract-uS.wgsl?raw";
import buildTjSrc from "../shaders/mps-two-site-build-tj.wgsl?raw";
import buildVhSrc from "../shaders/mps-two-site-build-vh.wgsl?raw";
import sigmaSortSrc from "../shaders/sigma-sort.wgsl?raw";
import jacobiSmallSrc from "../shaders/jacobi-svd-small.wgsl?raw";
import jacobiMediumSrc from "../shaders/jacobi-svd-medium.wgsl?raw";
import jacobiLargeSrc from "../shaders/jacobi-svd-large.wgsl?raw";
import jacobiStorageSrc from "../shaders/jacobi-svd-storage.wgsl?raw";
import type { Matrix2 } from "../gates.js";
import type { ComplexMatrix } from "../linalg.js";
import { poolFor } from "./buffer-pool.js";

interface SiteShape {
  chiL: number;
  chiR: number;
}

interface MpsPipelines {
  single: GPUComputePipeline;
  merge: GPUComputePipeline;
  colNorms: GPUComputePipeline;
  sigmaSort: GPUComputePipeline;
  extractU: GPUComputePipeline;
  extractUS: GPUComputePipeline;
  buildTj: GPUComputePipeline;
  buildVh: GPUComputePipeline;
  jacobiSmall: GPUComputePipeline;
  jacobiMedium: GPUComputePipeline | null;
  jacobiLarge: GPUComputePipeline | null;
  jacobiStorage: GPUComputePipeline;
  /** Largest n the SVD path supports — 64 with storage-mode kernel. */
  svdMaxN: number;
}

const SVD_MEDIUM_BYTES = 32 * 32 * 8 * 2 + 32 * 16;
const SVD_LARGE_BYTES = 48 * 48 * 8 * 2 + 32 * 16;

const pipelineCache: WeakMap<GPUDevice, MpsPipelines> = new WeakMap();

function getPipelines(device: GPUDevice): MpsPipelines {
  const cached = pipelineCache.get(device);
  if (cached) return cached;

  const mk = (label: string, code: string, entry: string): GPUComputePipeline => {
    const mod = device.createShaderModule({ label, code });
    return device.createComputePipeline({
      label: `${label}-pipeline`,
      layout: "auto",
      compute: { module: mod, entryPoint: entry },
    });
  };

  const single = mk("mps-single-qubit", singleQubitSrc, "single_qubit");
  const merge = mk("mps-two-site-merge", mergeSrc, "merge");
  const colNorms = mk("column-norms", colNormsSrc, "col_norms");
  const sigmaSort = mk("sigma-sort", sigmaSortSrc, "sort");
  const extractU = mk("mps-extract-u", extractUSrc, "extract_u");
  const extractUS = mk("mps-extract-uS", extractUSSrc, "extract_uS");
  const buildTj = mk("mps-build-tj", buildTjSrc, "build_tj");
  const buildVh = mk("mps-build-vh", buildVhSrc, "build_vh");
  const jacobiSmall = mk("jacobi-svd-small", jacobiSmallSrc, "jacobi_svd");

  let jacobiMedium: GPUComputePipeline | null = null;
  if (device.limits.maxComputeWorkgroupStorageSize >= SVD_MEDIUM_BYTES) {
    jacobiMedium = mk("jacobi-svd-medium", jacobiMediumSrc, "jacobi_svd");
  }
  let jacobiLarge: GPUComputePipeline | null = null;
  if (device.limits.maxComputeWorkgroupStorageSize >= SVD_LARGE_BYTES) {
    jacobiLarge = mk("jacobi-svd-large", jacobiLargeSrc, "jacobi_svd");
  }
  const jacobiStorage = mk("jacobi-svd-storage", jacobiStorageSrc, "jacobi_svd");

  // Storage-mode lifts the ceiling to n=64 (χ_max=32) on every adapter.
  const svdMaxN = 64;
  const set: MpsPipelines = {
    single, merge, colNorms, sigmaSort, extractU, extractUS, buildTj, buildVh,
    jacobiSmall, jacobiMedium, jacobiLarge, jacobiStorage,
    svdMaxN,
  };
  pipelineCache.set(device, set);
  return set;
}

/**
 * Pick the smallest SVD kernel that still covers `n`. The "small" kernel
 * caps at n=24 with 9.7 KB workgroup storage (always available); medium
 * at n=32 with 16.5 KB; large at n=48 with 37 KB. Smaller kernels run
 * faster per-sweep, so prefer the tightest fit.
 */
function pickSvdPipeline(set: MpsPipelines, n: number): GPUComputePipeline {
  if (n <= 24) return set.jacobiSmall;
  if (n <= 32 && set.jacobiMedium) return set.jacobiMedium;
  if (n <= 48 && set.jacobiLarge) return set.jacobiLarge;
  return set.jacobiStorage;
}

export interface MPSGpuInit {
  device: GPUDevice;
  nQubits: number;
  /** Maximum bond dimension. Determines per-site buffer size. */
  chiMax?: number;
}

/**
 * Phase 4a GPU-resident MPS. Same site-tensor convention as the
 * CPU MPS: T[i] has shape (χ_L · 2, χ_R), with χ_L = (rows / 2)
 * and χ_R = cols. Stored row-major as Float32 vec2 (re, im).
 *
 * v0 supports:
 *   • init to |0…0⟩
 *   • single-qubit gates via GPU kernel
 *   • statevector readback (contract all sites on CPU after readback)
 *
 * Two-site gates are deferred to Phase 4b.
 */
export class MPSGpu {
  readonly device: GPUDevice;
  readonly nQubits: number;
  readonly chiMax: number;
  private readonly tensors: GPUBuffer[];
  private readonly siteShape: SiteShape[];
  /** Bytes per tensor buffer = 2 · 2 · (χ_max · χ_max) f32 = 16 χ_max² bytes. */
  private readonly maxTensorBytes: number;
  /** Optional per-gate telemetry buffer for the truncation-renorm debug. */
  private _renormHistory: { renorm: number; sumKept: number; kKeep: number; n: number }[] = [];
  /** Persistent staging buffer for renorm readback when debug enabled. */
  private _renormStage: GPUBuffer | null = null;
  private _renormDebug = false;

  constructor({ device, nQubits, chiMax = 32 }: MPSGpuInit) {
    if (nQubits < 1) throw new Error("nQubits must be ≥ 1");
    this.device = device;
    this.nQubits = nQubits;
    this.chiMax = chiMax;
    // (χ_L · 2) × χ_R complex max = 2 χ_max² complex = 16 χ_max² bytes.
    this.maxTensorBytes = 16 * chiMax * chiMax;

    this.tensors = [];
    this.siteShape = [];
    const pool = poolFor(device);
    for (let i = 0; i < nQubits; i++) {
      const buf = pool.acquire(
        this.maxTensorBytes,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        `mps-gpu-T${i}`,
      );
      // Initialize to χ_L = χ_R = 1, |0⟩ amplitude = (1, 0); rest zero.
      const init = new ArrayBuffer(this.maxTensorBytes);
      const f32 = new Float32Array(init);
      f32[0] = 1; // T[0, 0, 0] = 1 (real part)
      device.queue.writeBuffer(buf, 0, init);
      this.tensors.push(buf);
      this.siteShape.push({ chiL: 1, chiR: 1 });
    }
  }

  /** Free the GPU buffers back to the pool. */
  dispose(): void {
    const pool = poolFor(this.device);
    for (const t of this.tensors) pool.release(t);
    if (this._renormStage) {
      this._renormStage.destroy();
      this._renormStage = null;
    }
  }

  /** Enable per-gate renorm telemetry readback (slow; debug only). */
  enableRenormDebug(): void {
    this._renormDebug = true;
    this._renormHistory = [];
    if (!this._renormStage) {
      this._renormStage = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: "mps-renorm-debug-stage",
      });
    }
  }

  /** Read the accumulated per-gate renorm history (after enableRenormDebug). */
  renormHistory(): ReadonlyArray<{ renorm: number; sumKept: number; kKeep: number; n: number }> {
    return this._renormHistory;
  }

  /** Bond dims χ_1..χ_{N-1}. */
  bondDimensions(): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.nQubits - 1; i++) out.push(this.siteShape[i]!.chiR);
    return out;
  }

  /** Apply a single-qubit gate U to qubit q via a GPU kernel. */
  apply(U: Matrix2, q: number): void {
    if (q < 0 || q >= this.nQubits) throw new Error(`apply q=${q} out of range`);
    const shape = this.siteShape[q]!;
    const pipeline = getPipelines(this.device).single;
    const pool = poolFor(this.device);

    // Params: 2 u32 + 4 vec2<f32> + 8 bytes pad. Layout matches WGSL struct.
    // WGSL aligns vec2<f32> to 8 bytes; 2 u32 = 8 bytes (also 8-aligned).
    // Total = 8 + 4·8 = 40 bytes. Round to 48 for buffer.
    const paramsBuf = pool.acquire(48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "mps-gate-params");
    const ab = new ArrayBuffer(48);
    const u32 = new Uint32Array(ab, 0, 2);
    u32[0] = shape.chiL;
    u32[1] = shape.chiR;
    const f32 = new Float32Array(ab, 8, 8);
    f32[0] = U[0][0]; f32[1] = U[0][1];
    f32[2] = U[1][0]; f32[3] = U[1][1];
    f32[4] = U[2][0]; f32[5] = U[2][1];
    f32[6] = U[3][0]; f32[7] = U[3][1];
    this.device.queue.writeBuffer(paramsBuf, 0, ab);

    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tensors[q]! } },
        { binding: 1, resource: { buffer: paramsBuf } },
      ],
    });

    const total = shape.chiL * shape.chiR;
    const wgX = Math.max(1, Math.ceil(total / 64));
    const enc = this.device.createCommandEncoder({ label: `mps-gate-${q}` });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(wgX);
    pass.end();
    this.device.queue.submit([enc.finish()]);

    pool.release(paramsBuf);
  }

  /**
   * Apply a two-site gate G (4×4) to bond (q, q+1) entirely on the GPU.
   *
   * Pipeline: merge T_q · T_{q+1} via the bond and apply G → SVD the
   * merged matrix in place → reduce A's column norms to extract σ →
   * read back σ (a few floats), CPU sorts + computes the truncation
   * permutation + renormalization → upload perm and σ-arrays → kernels
   * write the new T_q' and T_{q+1}' tensor buffers.
   *
   * Phase 4b limitations:
   *   • Square SVD only — merged matrix must be (chiL·2) = (2·chiR),
   *     so we require chiL == chiR. (Handled by clamping chiL during
   *     evolution; brick-wall + chi_max bookkeeping below ensures it.)
   *   • Caps at SVD-pipeline max n (24 or 32). Throws if exceeded.
   *   • No GPU-side canonicalization — caller is responsible for
   *     applying gates in an order where canonical-form invariants
   *     hold approximately (e.g. brick-wall with adequate chiMax).
   *
   * Phase A truncation renorm (this codepath): when kKeep < n the σ
   * spectrum loses sumAll − sumKept; sigma-sort publishes the scalar
   * √(sumAll/sumKept) and build-tj absorbs it cell-by-cell so the
   * post-truncation T_q · T_{q+1} keeps the merged matrix's pre-SVD
   * Frobenius norm. The chain stays unit-norm (verified at depths
   * 4–7, χ_max 4–8 by the Phase A e2e bench). When kKeep = n the
   * factor is 1.0, so the no-truncation hot path remains a single
   * fused multiply with no extra dispatch.
   *
   * Known caveat: without GPU-side canonicalization (Phase 4c+)
   * the σ values are SVD singular values of M, not Schmidt
   * coefficients of the bipartition. Truncating the smallest σ is
   * still norm-preserving but not error-optimal vs the CPU MPS
   * (which canonicalizes pre-gate). Fidelity vs canonical CPU MPS
   * degrades at heavy truncation; the *state norm* always stays
   * within f32 precision of 1.
   */
  async applyTwoSite(G: ComplexMatrix, q: number): Promise<void> {
    if (q < 0 || q >= this.nQubits - 1) {
      throw new Error(`applyTwoSite q=${q} out of range`);
    }
    if (G.rows !== 4 || G.cols !== 4) {
      throw new Error(`applyTwoSite: G must be 4×4`);
    }

    const left = this.siteShape[q]!;
    const right = this.siteShape[q + 1]!;
    if (left.chiR !== right.chiL) {
      throw new Error(`bond mismatch: T[${q}].chiR=${left.chiR}, T[${q + 1}].chiL=${right.chiL}`);
    }
    const chiL = left.chiL;
    const chiBond = left.chiR;
    const chiR = right.chiR;
    // Phase 5: allow non-square merged matrices by zero-padding to nMax × nMax.
    // The SVD kernels are square-only; padding columns/rows produce σ = 0
    // singular values which the host kKeep cap (≤ min(rows, cols)) discards.
    const mRows = chiL * 2;
    const mCols = 2 * chiR;
    const nMax = Math.max(mRows, mCols);
    const physicalRank = Math.min(mRows, mCols);
    const set = getPipelines(this.device);
    if (nMax > set.svdMaxN) {
      throw new Error(`applyTwoSite: padded dim nMax=${nMax} exceeds SVD max ${set.svdMaxN}`);
    }
    const n = nMax;
    const pool = poolFor(this.device);
    const device = this.device;

    // ── 1. Merge: M = (G · T_q ⊗ T_{q+1}) reshaped to n×n ──────
    // Convert G to f32 row-major (16 vec2 = 128 B).
    const gBytes = 16 * 8;
    const gBuf = pool.acquire(gBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "tg-G");
    const gAb = new ArrayBuffer(gBytes);
    const gF32 = new Float32Array(gAb);
    for (let i = 0; i < 16; i++) {
      gF32[i * 2] = G.data[i * 2]!;
      gF32[i * 2 + 1] = G.data[i * 2 + 1]!;
    }
    device.queue.writeBuffer(gBuf, 0, gAb);

    const mBytes = n * n * 8;
    const mBuf = pool.acquire(
      mBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      "tg-M",
    );
    // Pool may return a dirty buffer; SVD relies on padded cells being zero,
    // so wipe before the merge writes the live submatrix.
    device.queue.writeBuffer(mBuf, 0, new ArrayBuffer(mBytes));
    const mergeParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-merge-params");
    device.queue.writeBuffer(mergeParams, 0, new Uint32Array([chiL, chiBond, chiR, n]));

    const mergeBind = device.createBindGroup({
      layout: set.merge.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tensors[q]! } },
        { binding: 1, resource: { buffer: this.tensors[q + 1]! } },
        { binding: 2, resource: { buffer: gBuf } },
        { binding: 3, resource: { buffer: mBuf } },
        { binding: 4, resource: { buffer: mergeParams } },
      ],
    });

    // ── 2. SVD in place on mBuf with a fresh V buf ────────────
    const vBuf = pool.acquire(
      mBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      "tg-V",
    );
    const svdParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-svd-params");
    device.queue.writeBuffer(svdParams, 0, new Uint32Array([n, 15, 0, 0]));
    const svdPipeline = pickSvdPipeline(set, n);
    const svdBind = device.createBindGroup({
      layout: svdPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mBuf } },
        { binding: 1, resource: { buffer: vBuf } },
        { binding: 2, resource: { buffer: svdParams } },
      ],
    });

    // ── 3. Column-norms → σ ──────────────────────────────────
    const sigmaBytes = n * 4;
    const sigmaBuf = pool.acquire(
      Math.max(16, sigmaBytes),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      "tg-sigma",
    );
    const colNormsParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-colnorms-params");
    device.queue.writeBuffer(colNormsParams, 0, new Uint32Array([n, 0, 0, 0]));
    const colNormsBind = device.createBindGroup({
      layout: set.colNorms.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mBuf } },
        { binding: 1, resource: { buffer: sigmaBuf } },
        { binding: 2, resource: { buffer: colNormsParams } },
      ],
    });

    // ── 3.5 GPU-side sort: σ desc → perm[] (Phase 5 fast path) ─
    // Replaces the CPU readback + JS sort + writeBuffer round-trip
    // that used to be the per-gate sync point. The whole two-site
    // update now runs in a single command-encoder + single submit,
    // letting consecutive gates pipeline on the GPU.
    //
    // kKeep is purely shape-derived: caller's chiMax capped at
    // physicalRank = min(mRows, mCols). The σ-threshold trim
    // (drop columns with σ < 1e-12 · σ_max) is dropped here —
    // such columns get pushed to the bottom of the sort and only
    // matter for kKeep < physicalRank truncation. For unitary
    // gates within chiMax that's exact-by-construction.
    const kKeep = Math.min(this.chiMax, physicalRank);

    const permBytes = Math.max(16, n * 4);
    const permBuf = pool.acquire(permBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "tg-perm");
    const sortParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-sort-params");
    device.queue.writeBuffer(sortParams, 0, new Uint32Array([n, kKeep, 0, 0]));
    const renormBuf = pool.acquire(
      16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      "tg-renorm",
    );
    const sortBind = device.createBindGroup({
      layout: set.sigmaSort.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sigmaBuf } },
        { binding: 1, resource: { buffer: permBuf } },
        { binding: 2, resource: { buffer: sortParams } },
        { binding: 3, resource: { buffer: renormBuf } },
      ],
    });

    // ── 4. Extract U into T_q' and build T_{q+1}' ───────────
    const newTqRows = chiL * 2;
    const newTjRows = kKeep * 2;
    const extractParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-extract-params");
    device.queue.writeBuffer(extractParams, 0, new Uint32Array([newTqRows, kKeep, n, 0]));
    const extractBind = device.createBindGroup({
      layout: set.extractU.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mBuf } },
        { binding: 1, resource: { buffer: sigmaBuf } },
        { binding: 2, resource: { buffer: permBuf } },
        { binding: 3, resource: { buffer: this.tensors[q]! } },
        { binding: 4, resource: { buffer: extractParams } },
      ],
    });

    const buildParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-build-params");
    device.queue.writeBuffer(buildParams, 0, new Uint32Array([kKeep, chiR, n, 0]));
    const buildBind = device.createBindGroup({
      layout: set.buildTj.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: vBuf } },
        { binding: 1, resource: { buffer: sigmaBuf } },     // raw σ (gathered via perm in-kernel)
        { binding: 2, resource: { buffer: permBuf } },
        { binding: 3, resource: { buffer: this.tensors[q + 1]! } },
        { binding: 4, resource: { buffer: buildParams } },
        { binding: 5, resource: { buffer: renormBuf } },    // 1/√Σ_kept σ² (truncation renorm)
      ],
    });

    // Single command encoder, single submit — no readback.
    const enc = device.createCommandEncoder({ label: "two-site-fast" });
    const pass = enc.beginComputePass();
    pass.setPipeline(set.merge);
    pass.setBindGroup(0, mergeBind);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(chiL * chiR / 64)));
    pass.setPipeline(svdPipeline);
    pass.setBindGroup(0, svdBind);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(set.colNorms);
    pass.setBindGroup(0, colNormsBind);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(set.sigmaSort);
    pass.setBindGroup(0, sortBind);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(set.extractU);
    pass.setBindGroup(0, extractBind);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(newTqRows * kKeep / 64)));
    pass.setPipeline(set.buildTj);
    pass.setBindGroup(0, buildBind);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(newTjRows * chiR / 64)));
    pass.end();
    if (this._renormDebug && this._renormStage) {
      enc.copyBufferToBuffer(renormBuf, 0, this._renormStage, 0, 16);
    }
    device.queue.submit([enc.finish()]);

    // Update CPU-side bond bookkeeping.
    this.siteShape[q] = { chiL, chiR: kKeep };
    this.siteShape[q + 1] = { chiL: kKeep, chiR };

    if (this._renormDebug && this._renormStage) {
      await this._renormStage.mapAsync(GPUMapMode.READ);
      const view = new Float32Array(this._renormStage.getMappedRange().slice(0));
      this._renormStage.unmap();
      this._renormHistory.push({
        renorm: view[0]!,
        sumKept: view[1]!,
        kKeep: view[2]!,
        n: view[3]!,
      });
    }

    // Return all the scratch buffers to the pool.
    pool.release(gBuf);
    pool.release(mBuf);
    pool.release(mergeParams);
    pool.release(vBuf);
    pool.release(svdParams);
    pool.release(sigmaBuf);
    pool.release(colNormsParams);
    pool.release(permBuf);
    pool.release(sortParams);
    pool.release(renormBuf);
    pool.release(extractParams);
    pool.release(buildParams);
  }

  /**
   * Apply a two-site gate G with the symmetric (right-canonical) split:
   * T_q' ← U·σ (carries residual), T_{q+1}' ← V^H (right-canonical isometry).
   *
   * Same merge + SVD + sort as `applyTwoSite`; only the final extract step
   * differs. Used by `canonicalize(targetBond)` to push the orthogonality
   * center *leftward* across a chain — the existing `applyTwoSite` only
   * pushes it rightward.
   */
  async applyTwoSiteLeft(G: ComplexMatrix, q: number): Promise<void> {
    if (q < 0 || q >= this.nQubits - 1) {
      throw new Error(`applyTwoSiteLeft q=${q} out of range`);
    }
    if (G.rows !== 4 || G.cols !== 4) {
      throw new Error(`applyTwoSiteLeft: G must be 4×4`);
    }

    const left = this.siteShape[q]!;
    const right = this.siteShape[q + 1]!;
    if (left.chiR !== right.chiL) {
      throw new Error(`bond mismatch: T[${q}].chiR=${left.chiR}, T[${q + 1}].chiL=${right.chiL}`);
    }
    const chiL = left.chiL;
    const chiBond = left.chiR;
    const chiR = right.chiR;
    const mRows = chiL * 2;
    const mCols = 2 * chiR;
    const nMax = Math.max(mRows, mCols);
    const physicalRank = Math.min(mRows, mCols);
    const set = getPipelines(this.device);
    if (nMax > set.svdMaxN) {
      throw new Error(`applyTwoSiteLeft: padded dim nMax=${nMax} exceeds SVD max ${set.svdMaxN}`);
    }
    const n = nMax;
    const pool = poolFor(this.device);
    const device = this.device;

    // ── Merge (same as applyTwoSite) ──────────────────────────
    const gBytes = 16 * 8;
    const gBuf = pool.acquire(gBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "tg-G");
    const gAb = new ArrayBuffer(gBytes);
    const gF32 = new Float32Array(gAb);
    for (let i = 0; i < 16; i++) {
      gF32[i * 2] = G.data[i * 2]!;
      gF32[i * 2 + 1] = G.data[i * 2 + 1]!;
    }
    device.queue.writeBuffer(gBuf, 0, gAb);

    const mBytes = n * n * 8;
    const mBuf = pool.acquire(
      mBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      "tg-M",
    );
    device.queue.writeBuffer(mBuf, 0, new ArrayBuffer(mBytes));
    const mergeParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-merge-params");
    device.queue.writeBuffer(mergeParams, 0, new Uint32Array([chiL, chiBond, chiR, n]));
    const mergeBind = device.createBindGroup({
      layout: set.merge.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tensors[q]! } },
        { binding: 1, resource: { buffer: this.tensors[q + 1]! } },
        { binding: 2, resource: { buffer: gBuf } },
        { binding: 3, resource: { buffer: mBuf } },
        { binding: 4, resource: { buffer: mergeParams } },
      ],
    });

    // SVD + colNorms + sort (same as applyTwoSite)
    const vBuf = pool.acquire(
      mBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      "tg-V",
    );
    const svdParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-svd-params");
    device.queue.writeBuffer(svdParams, 0, new Uint32Array([n, 15, 0, 0]));
    const svdPipeline = pickSvdPipeline(set, n);
    const svdBind = device.createBindGroup({
      layout: svdPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mBuf } },
        { binding: 1, resource: { buffer: vBuf } },
        { binding: 2, resource: { buffer: svdParams } },
      ],
    });

    const sigmaBytes = n * 4;
    const sigmaBuf = pool.acquire(
      Math.max(16, sigmaBytes),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      "tg-sigma",
    );
    const colNormsParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-colnorms-params");
    device.queue.writeBuffer(colNormsParams, 0, new Uint32Array([n, 0, 0, 0]));
    const colNormsBind = device.createBindGroup({
      layout: set.colNorms.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mBuf } },
        { binding: 1, resource: { buffer: sigmaBuf } },
        { binding: 2, resource: { buffer: colNormsParams } },
      ],
    });

    const kKeep = Math.min(this.chiMax, physicalRank);
    const permBytes = Math.max(16, n * 4);
    const permBuf = pool.acquire(permBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "tg-perm");
    const sortParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-sort-params");
    device.queue.writeBuffer(sortParams, 0, new Uint32Array([n, kKeep, 0, 0]));
    const renormBuf = pool.acquire(
      16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      "tg-renorm",
    );
    const sortBind = device.createBindGroup({
      layout: set.sigmaSort.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sigmaBuf } },
        { binding: 1, resource: { buffer: permBuf } },
        { binding: 2, resource: { buffer: sortParams } },
        { binding: 3, resource: { buffer: renormBuf } },
      ],
    });

    // ── Right-canonical extract: T_q ← A_perm · renorm (carries σ),
    //    T_{q+1} ← V^H reshape (right-canonical isometry).
    const newTqRows = chiL * 2;
    const newTjRows = kKeep * 2;
    const extractParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-extractuS-params");
    device.queue.writeBuffer(extractParams, 0, new Uint32Array([newTqRows, kKeep, n, 0]));
    const extractBind = device.createBindGroup({
      layout: set.extractUS.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: mBuf } },
        { binding: 1, resource: { buffer: permBuf } },
        { binding: 2, resource: { buffer: this.tensors[q]! } },
        { binding: 3, resource: { buffer: extractParams } },
        { binding: 4, resource: { buffer: renormBuf } },
      ],
    });

    const buildParams = pool.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "tg-buildvh-params");
    device.queue.writeBuffer(buildParams, 0, new Uint32Array([kKeep, chiR, n, 0]));
    const buildBind = device.createBindGroup({
      layout: set.buildVh.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: vBuf } },
        { binding: 1, resource: { buffer: permBuf } },
        { binding: 2, resource: { buffer: this.tensors[q + 1]! } },
        { binding: 3, resource: { buffer: buildParams } },
      ],
    });

    const enc = device.createCommandEncoder({ label: "two-site-left-fast" });
    const pass = enc.beginComputePass();
    pass.setPipeline(set.merge);
    pass.setBindGroup(0, mergeBind);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(chiL * chiR / 64)));
    pass.setPipeline(svdPipeline);
    pass.setBindGroup(0, svdBind);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(set.colNorms);
    pass.setBindGroup(0, colNormsBind);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(set.sigmaSort);
    pass.setBindGroup(0, sortBind);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(set.extractUS);
    pass.setBindGroup(0, extractBind);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(newTqRows * kKeep / 64)));
    pass.setPipeline(set.buildVh);
    pass.setBindGroup(0, buildBind);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(newTjRows * chiR / 64)));
    pass.end();
    device.queue.submit([enc.finish()]);

    this.siteShape[q] = { chiL, chiR: kKeep };
    this.siteShape[q + 1] = { chiL: kKeep, chiR };

    pool.release(gBuf);
    pool.release(mBuf);
    pool.release(mergeParams);
    pool.release(vBuf);
    pool.release(svdParams);
    pool.release(sigmaBuf);
    pool.release(colNormsParams);
    pool.release(permBuf);
    pool.release(sortParams);
    pool.release(renormBuf);
    pool.release(extractParams);
    pool.release(buildParams);
  }

  /**
   * Bring the chain into mixed-canonical form with the orthogonality
   * center at the bond between sites `targetBond` and `targetBond+1`:
   *   • sites 0..targetBond  are left-canonical
   *   • sites targetBond+1..N-1 are right-canonical
   *
   * Implementation: full left-sweep first (applyTwoSite with G=I from
   * 0..N-2), then right-sweep (applyTwoSiteLeft from N-2..targetBond+1)
   * to walk the residual back. Both sweeps use the GPU-resident pipeline.
   */
  async canonicalize(targetBond: number): Promise<void> {
    if (targetBond < 0 || targetBond >= this.nQubits - 1) {
      throw new Error(`canonicalize: targetBond=${targetBond} out of range [0, ${this.nQubits - 1})`);
    }
    const I4: ComplexMatrix = {
      rows: 4, cols: 4,
      data: new Float64Array([
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
      ]),
    };
    // Left-sweep: orthogonality center walks rightward to site N-1.
    for (let q = 0; q < this.nQubits - 1; q++) {
      await this.applyTwoSite(I4, q);
    }
    // Right-sweep: walk residual back from N-2 down to targetBond+1.
    for (let q = this.nQubits - 2; q > targetBond; q--) {
      await this.applyTwoSiteLeft(I4, q);
    }
  }

  /** Read back all tensors and contract them into a flat statevector
   *  of length 2 · 2^N (interleaved [re, im]). For test/cross-check use. */
  async statevector(): Promise<Float64Array> {
    if (this.nQubits > 24) {
      throw new Error(`statevector(): N=${this.nQubits} too large to materialize`);
    }
    // Stage one readback per tensor.
    const pool = poolFor(this.device);
    const stages: GPUBuffer[] = [];
    const enc = this.device.createCommandEncoder({ label: "mps-gpu-readback" });
    for (let i = 0; i < this.nQubits; i++) {
      const stage = pool.acquire(
        this.maxTensorBytes,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        `mps-readback-${i}`,
      );
      enc.copyBufferToBuffer(this.tensors[i]!, 0, stage, 0, this.maxTensorBytes);
      stages.push(stage);
    }
    this.device.queue.submit([enc.finish()]);
    await Promise.all(stages.map((s) => s.mapAsync(GPUMapMode.READ)));

    // Rebuild local CPU copies of each tensor with its actual (chiL, chiR).
    const tensors: Float32Array[] = [];
    for (let i = 0; i < this.nQubits; i++) {
      const view = new Float32Array(stages[i]!.getMappedRange().slice(0));
      stages[i]!.unmap();
      pool.release(stages[i]!);
      const { chiL, chiR } = this.siteShape[i]!;
      const len = chiL * 2 * chiR * 2;
      const t = new Float32Array(len);
      // The tensor is stored at offset 0 of its max-sized buffer; rows are
      // packed contiguously of width chiR (not chiMax). Initial state has
      // chiR=1 so rows are 1-wide already.
      for (let row = 0; row < chiL * 2; row++) {
        for (let col = 0; col < chiR; col++) {
          const src = (row * chiR + col) * 2;
          t[(row * chiR + col) * 2]     = view[src]!;
          t[(row * chiR + col) * 2 + 1] = view[src + 1]!;
        }
      }
      tensors.push(t);
    }

    // Contract all sites: statevector[s_0 + 2 s_1 + …] = T_0[0, s_0, b_0] · T_1[b_0, s_1, b_1] · … · T_{N-1}[b_{N-2}, s_{N-1}, 0].
    // The CPU MPS does the same on f64; here we run on f32 then widen.
    return contractMPS(tensors, this.siteShape, this.nQubits);
  }
}

/** Walk the chain left-to-right multiplying matrices to grow the statevector. */
function contractMPS(
  tensors: Float32Array[],
  shape: ReadonlyArray<{ chiL: number; chiR: number }>,
  N: number,
): Float64Array {
  // current[basis_idx, b] holds the partial amplitude after sites 0..k.
  // Layout: row = basis_idx ∈ [0, 2^k), col = b ∈ [0, χ_R^k). Complex (re, im).
  let curRows = 1;     // 2^0
  let curCols = 1;     // initial χ_L = 1
  let cur = new Float64Array(2);
  cur[0] = 1; // |empty⟩

  for (let i = 0; i < N; i++) {
    const T = tensors[i]!;
    const { chiL, chiR } = shape[i]!;
    if (chiL !== curCols) {
      throw new Error(`contractMPS: bond mismatch at site ${i}: chiL=${chiL}, prev cols=${curCols}`);
    }
    const newRows = curRows * 2;
    const newCols = chiR;
    const next = new Float64Array(newRows * newCols * 2);

    // next[basis_new, b_new] = Σ_{basis_old, s, b_old}
    //                            cur[basis_old, b_old] · T[b_old · 2 + s, b_new]
    //   where basis_new = basis_old + 2^i · s.
    for (let basisOld = 0; basisOld < curRows; basisOld++) {
      for (let s = 0; s < 2; s++) {
        const basisNew = basisOld + (1 << i) * s;
        for (let bOld = 0; bOld < chiL; bOld++) {
          const cIdx = (basisOld * curCols + bOld) * 2;
          const cR = cur[cIdx]!, cI = cur[cIdx + 1]!;
          if (cR === 0 && cI === 0) continue;
          for (let bNew = 0; bNew < chiR; bNew++) {
            const tIdx = ((bOld * 2 + s) * chiR + bNew) * 2;
            const tR = T[tIdx]!, tI = T[tIdx + 1]!;
            const idx = (basisNew * newCols + bNew) * 2;
            next[idx] = (next[idx] ?? 0) + cR * tR - cI * tI;
            next[idx + 1] = (next[idx + 1] ?? 0) + cR * tI + cI * tR;
          }
        }
      }
    }
    cur = next;
    curRows = newRows;
    curCols = newCols;
  }

  // After all sites, curCols = chiR_final = 1 by MPS convention.
  if (curCols !== 1) {
    throw new Error(`contractMPS: trailing bond ≠ 1 (got ${curCols})`);
  }
  // cur is shape (2^N, 1, 2) = (2^N, 2). Already what we want.
  return cur;
}
