// ─────────────────────────────────────────────────────────────
// svd-bridge.ts — Phase 2 entry: a function with signature
//   (M: ComplexMatrix) => Promise<SVDResult>
// that runs the GPU one-sided Jacobi SVD when the matrix is
// square and ≤ 24×24 (the Phase 1B kernel's working envelope),
// and silently falls back to the CPU SVD otherwise.
//
// `MPS.applyTwoSiteAsync` consumes this. Caller stays oblivious
// to which path executed.
// ─────────────────────────────────────────────────────────────

import { type ComplexMatrix, type SVDResult, svd as cpuSvd, zeros } from "../linalg.js";
import { gpuJacobiSvd, gpuJacobiSvdMaxN } from "./jacobi-svd.js";
import { type CMatF32, cmatZeros } from "./types.js";

/** Build a Promise-returning SVD that uses the GPU when possible.
 *  The cap (24 or 32) is queried from the device's workgroup-storage
 *  limit so the same code unblocks χ=16 MPS bonds on capable hardware. */
export function makeAdaptiveSvd(device: GPUDevice): (M: ComplexMatrix) => Promise<SVDResult> {
  const maxN = gpuJacobiSvdMaxN(device);
  return async (M: ComplexMatrix): Promise<SVDResult> => {
    if (M.rows === M.cols && M.rows <= maxN) {
      const f32 = complexMatrixToCMatF32(M);
      const { U, sigma, Vh } = await gpuJacobiSvd(device, f32);
      return {
        U: cmatF32ToComplexMatrix(U),
        sigma: f32ToF64(sigma),
        Vh: cmatF32ToComplexMatrix(Vh),
      };
    }
    // Fallback: CPU svd. Wrap in a resolved Promise to match the async shape.
    return Promise.resolve(cpuSvd(M));
  };
}

function complexMatrixToCMatF32(M: ComplexMatrix): CMatF32 {
  const out: CMatF32 = { rows: M.rows, cols: M.cols, data: new Float32Array(M.data.length) };
  for (let i = 0; i < M.data.length; i++) out.data[i] = M.data[i]!;
  return out;
}

function cmatF32ToComplexMatrix(M: CMatF32): ComplexMatrix {
  const out = zeros(M.rows, M.cols);
  for (let i = 0; i < M.data.length; i++) out.data[i] = M.data[i]!;
  return out;
}

function f32ToF64(arr: Float32Array): Float64Array {
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i]!;
  return out;
}

void cmatZeros; // imported for type re-export consistency in future
