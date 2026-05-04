// ─────────────────────────────────────────────────────────────
// CPU reference statevector simulator.
//
// Pure TypeScript, no GPU. Mirrors QuantumCircuit's gate API so
// tests + browser can cross-check GPU results against ground
// truth. Slow by construction — do NOT run past ~20 qubits.
// ─────────────────────────────────────────────────────────────

import type { Matrix2 } from "./gates.js";
import { X } from "./gates.js";
import type { Matrix4 } from "./two-qubit-dense.js";
import type { Matrix8 } from "./three-qubit-dense.js";
import type { Matrix16 } from "./four-qubit-dense.js";

export class CpuCircuit {
  readonly nQubits: number;
  readonly nStates: number;
  /** Interleaved [re, im, re, im, ...]. */
  readonly psi: Float64Array;

  constructor(nQubits: number) {
    if (nQubits < 1 || nQubits > 24) {
      throw new Error(`CPU reference capped at 24 qubits. Got ${nQubits}.`);
    }
    this.nQubits = nQubits;
    this.nStates = 1 << nQubits;
    this.psi = new Float64Array(this.nStates * 2);
    this.psi[0] = 1; // |0…0⟩
  }

  apply(m: Matrix2, qubit: number): void {
    const q = qubit;
    if (q < 0 || q >= this.nQubits) {
      throw new Error(`qubit ${q} out of range`);
    }
    const maskLow = (1 << q) - 1;
    const nPairs = this.nStates >> 1;

    const m00 = m[0], m01 = m[1], m10 = m[2], m11 = m[3];
    const a00r = m00[0], a00i = m00[1];
    const a01r = m01[0], a01i = m01[1];
    const a10r = m10[0], a10i = m10[1];
    const a11r = m11[0], a11i = m11[1];

    const psi = this.psi;
    for (let t = 0; t < nPairs; t++) {
      const i = ((t & ~maskLow) << 1) | (t & maskLow);
      const j = i | (1 << q);
      const ii = i * 2;
      const jj = j * 2;
      const ar = psi[ii]!, ai = psi[ii + 1]!;
      const br = psi[jj]!, bi = psi[jj + 1]!;

      psi[ii]     = a00r * ar - a00i * ai + a01r * br - a01i * bi;
      psi[ii + 1] = a00r * ai + a00i * ar + a01r * bi + a01i * br;
      psi[jj]     = a10r * ar - a10i * ai + a11r * br - a11i * bi;
      psi[jj + 1] = a10r * ai + a10i * ar + a11r * bi + a11i * br;
    }
  }

  applyControlled(m: Matrix2, control: number, target: number): void {
    if (control === target) throw new Error("control === target");
    const c = control, tg = target;
    const n = this.nStates;
    const m00 = m[0], m01 = m[1], m10 = m[2], m11 = m[3];
    const a00r = m00[0], a00i = m00[1];
    const a01r = m01[0], a01i = m01[1];
    const a10r = m10[0], a10i = m10[1];
    const a11r = m11[0], a11i = m11[1];

    const psi = this.psi;
    for (let i = 0; i < n; i++) {
      const cOn = (i >>> c) & 1;
      const tOff = ((i >>> tg) & 1) === 0;
      if (!cOn || !tOff) continue;
      const j = i | (1 << tg);
      const ii = i * 2;
      const jj = j * 2;
      const ar = psi[ii]!, ai = psi[ii + 1]!;
      const br = psi[jj]!, bi = psi[jj + 1]!;

      psi[ii]     = a00r * ar - a00i * ai + a01r * br - a01i * bi;
      psi[ii + 1] = a00r * ai + a00i * ar + a01r * bi + a01i * br;
      psi[jj]     = a10r * ar - a10i * ai + a11r * br - a11i * bi;
      psi[jj + 1] = a10r * ai + a10i * ar + a11r * bi + a11i * br;
    }
  }

  cnot(control: number, target: number): void {
    this.applyControlled(X, control, target);
  }

  /**
   * Apply a dense 4×4 unitary U to the qubit pair (qLo, qHi). Basis
   * ordering inside the 4-vector: j = 2·s_qHi + s_qLo (matches the
   * GPU kernel in src/shaders/two-qubit-dense.wgsl). Used to fuse a
   * brick-wall layer's `(U_qLo ⊗ U_qHi)` and CNOT into one operation.
   */
  applyDense4x4(U: Matrix4, qLo: number, qHi: number): void {
    if (qLo >= qHi) throw new Error(`applyDense4x4: need qLo < qHi, got (${qLo}, ${qHi})`);
    if (qLo < 0 || qHi >= this.nQubits) throw new Error(`applyDense4x4: out-of-range (${qLo}, ${qHi})`);
    if (U.length !== 16) throw new Error(`applyDense4x4: matrix must have 16 entries, got ${U.length}`);

    const nBlocks = this.nStates >> 2;
    const psi = this.psi;
    const bitLo = 1 << qLo, bitHi = 1 << qHi;
    const mLo = bitLo - 1;
    const mHi = bitHi - 1;

    for (let block = 0; block < nBlocks; block++) {
      // Insert zero bits at qLo, then qHi (qLo < qHi).
      const bLo = (block & mLo) | ((block & ~mLo) << 1);
      const iBase = (bLo & mHi) | ((bLo & ~mHi) << 1);
      const i00 = iBase;
      const i01 = iBase | bitLo;
      const i10 = iBase | bitHi;
      const i11 = i01 | bitHi;

      const idx = [i00, i01, i10, i11];
      // Read all 4 amps.
      const inRe = idx.map((i) => psi[i * 2]!);
      const inIm = idx.map((i) => psi[i * 2 + 1]!);

      // Multiply U · in.
      for (let row = 0; row < 4; row++) {
        let outRe = 0, outIm = 0;
        for (let col = 0; col < 4; col++) {
          const m = U[row * 4 + col]!;
          const mRe = m[0], mIm = m[1];
          const aRe = inRe[col]!, aIm = inIm[col]!;
          outRe += mRe * aRe - mIm * aIm;
          outIm += mRe * aIm + mIm * aRe;
        }
        psi[idx[row]! * 2]     = outRe;
        psi[idx[row]! * 2 + 1] = outIm;
      }
    }
  }

  /**
   * Apply a dense 8×8 unitary U to the qubit triple (qLo, qMid, qHi),
   * qLo < qMid < qHi. Basis ordering inside the 8-vector:
   * j = 4·s_qHi + 2·s_qMid + s_qLo (matches the GPU kernel in
   * src/shaders/three-qubit-dense.wgsl).
   */
  applyDense8x8(U: Matrix8, qLo: number, qMid: number, qHi: number): void {
    if (!(qLo < qMid && qMid < qHi)) {
      throw new Error(`applyDense8x8: need qLo < qMid < qHi, got (${qLo}, ${qMid}, ${qHi})`);
    }
    if (qLo < 0 || qHi >= this.nQubits) {
      throw new Error(`applyDense8x8: out-of-range (${qLo}, ${qMid}, ${qHi}) for n=${this.nQubits}`);
    }
    if (U.length !== 64) throw new Error(`applyDense8x8: matrix must have 64 entries, got ${U.length}`);

    const nBlocks = this.nStates >> 3;
    const psi = this.psi;
    const bitLo = 1 << qLo, bitMid = 1 << qMid, bitHi = 1 << qHi;
    const mLo = bitLo - 1;
    const mMid = bitMid - 1;
    const mHi = bitHi - 1;

    const inRe = new Float64Array(8);
    const inIm = new Float64Array(8);

    for (let block = 0; block < nBlocks; block++) {
      // Insert zero bits at qLo, then qMid, then qHi.
      const bLo = (block & mLo) | ((block & ~mLo) << 1);
      const bMid = (bLo & mMid) | ((bLo & ~mMid) << 1);
      const iBase = (bMid & mHi) | ((bMid & ~mHi) << 1);

      const idx = [
        iBase,
        iBase | bitLo,
        iBase | bitMid,
        iBase | bitLo | bitMid,
        iBase | bitHi,
        iBase | bitLo | bitHi,
        iBase | bitMid | bitHi,
        iBase | bitLo | bitMid | bitHi,
      ];
      for (let k = 0; k < 8; k++) {
        inRe[k] = psi[idx[k]! * 2]!;
        inIm[k] = psi[idx[k]! * 2 + 1]!;
      }

      for (let row = 0; row < 8; row++) {
        let outRe = 0, outIm = 0;
        for (let col = 0; col < 8; col++) {
          const m = U[row * 8 + col]!;
          const mRe = m[0], mIm = m[1];
          const aRe = inRe[col]!, aIm = inIm[col]!;
          outRe += mRe * aRe - mIm * aIm;
          outIm += mRe * aIm + mIm * aRe;
        }
        psi[idx[row]! * 2] = outRe;
        psi[idx[row]! * 2 + 1] = outIm;
      }
    }
  }

  /**
   * Apply a dense 16×16 unitary U to the qubit quadruple
   * (qLo, qMid1, qMid2, qHi) with strict ordering. Basis ordering
   * inside the 16-vector: j = 8·s_qHi + 4·s_qMid2 + 2·s_qMid1 + s_qLo
   * (matches src/shaders/four-qubit-dense.wgsl).
   */
  applyDense16x16(
    U: Matrix16,
    qLo: number,
    qMid1: number,
    qMid2: number,
    qHi: number,
  ): void {
    if (!(qLo < qMid1 && qMid1 < qMid2 && qMid2 < qHi)) {
      throw new Error(
        `applyDense16x16: need qLo < qMid1 < qMid2 < qHi, got (${qLo}, ${qMid1}, ${qMid2}, ${qHi})`,
      );
    }
    if (qLo < 0 || qHi >= this.nQubits) {
      throw new Error(`applyDense16x16: out-of-range for n=${this.nQubits}`);
    }
    if (U.length !== 256) throw new Error(`applyDense16x16: matrix must have 256 entries, got ${U.length}`);

    const nBlocks = this.nStates >> 4;
    const psi = this.psi;
    const bitLo = 1 << qLo, bitMid1 = 1 << qMid1, bitMid2 = 1 << qMid2, bitHi = 1 << qHi;
    const mLo = bitLo - 1;
    const mMid1 = bitMid1 - 1;
    const mMid2 = bitMid2 - 1;
    const mHi = bitHi - 1;

    const idx = new Int32Array(16);
    const inRe = new Float64Array(16);
    const inIm = new Float64Array(16);

    for (let block = 0; block < nBlocks; block++) {
      const bLo = (block & mLo) | ((block & ~mLo) << 1);
      const bMid1 = (bLo & mMid1) | ((bLo & ~mMid1) << 1);
      const bMid2 = (bMid1 & mMid2) | ((bMid1 & ~mMid2) << 1);
      const iBase = (bMid2 & mHi) | ((bMid2 & ~mHi) << 1);

      for (let k = 0; k < 16; k++) {
        idx[k] = iBase
          | ((k & 1) ? bitLo : 0)
          | ((k & 2) ? bitMid1 : 0)
          | ((k & 4) ? bitMid2 : 0)
          | ((k & 8) ? bitHi : 0);
        inRe[k] = psi[idx[k]! * 2]!;
        inIm[k] = psi[idx[k]! * 2 + 1]!;
      }

      for (let row = 0; row < 16; row++) {
        let outRe = 0, outIm = 0;
        for (let col = 0; col < 16; col++) {
          const m = U[row * 16 + col]!;
          const mRe = m[0], mIm = m[1];
          const aRe = inRe[col]!, aIm = inIm[col]!;
          outRe += mRe * aRe - mIm * aIm;
          outIm += mRe * aIm + mIm * aRe;
        }
        psi[idx[row]! * 2] = outRe;
        psi[idx[row]! * 2 + 1] = outIm;
      }
    }
  }

  probabilities(): Float64Array {
    const p = new Float64Array(this.nStates);
    for (let i = 0; i < this.nStates; i++) {
      const re = this.psi[i * 2]!;
      const im = this.psi[i * 2 + 1]!;
      p[i] = re * re + im * im;
    }
    return p;
  }

  norm(): number {
    let s = 0;
    const p = this.probabilities();
    for (const x of p) s += x;
    return s;
  }
}
