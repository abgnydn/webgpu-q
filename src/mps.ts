// ─────────────────────────────────────────────────────────────
// mps.ts — Matrix Product State simulator (Level 2).
//
// Representation: tensors[i] is a ComplexMatrix of shape (χ_L · 2, χ_R),
// with entry T[l, s, r] stored at row = l·2 + s, col = r ("left-grouped").
// χ_L = tensors[i].rows / 2,  χ_R = tensors[i].cols.
//
// Canonical form: after each two-site SVD update the left tensor (U-side)
// has orthonormal columns and the right tensor (σ·V^H) carries the Schmidt
// coefficients. We renormalize σ after truncation so the global state
// stays unit-norm to machine precision (|norm − 1| < 1e-12 typical).
//
// API mirrors QuantumCircuit's — single-qubit on any site, controlled-U
// restricted to |c − t| = 1 (adjacency). Non-adjacent gates require a
// swap network and are deferred to a later iteration.
//
// Index convention: qubit 0 is the LEAST significant bit of the output
// statevector index (statevector[s_0 + 2·s_1 + 4·s_2 + …] = ψ_{s_0…s_{N−1}}).
// Site 0 of the MPS = qubit 0. For two-site gates on sites (q, q+1), the
// internal 4×4 matrix is indexed with site q as the HIGH bit within the
// 2-site tensor (i = s_q · 2 + s_{q+1}).
// ─────────────────────────────────────────────────────────────

import type { Matrix2, Complex } from "./gates.js";
import { X } from "./gates.js";
import {
  type ComplexMatrix,
  type SVDResult,
  zeros,
  clone,
  cset,
  matmul,
  svd,
} from "./linalg.js";

export interface MPSInit {
  nQubits: number;
  /** Max bond dimension. Defaults to 64. Truncation happens if SVD rank exceeds this. */
  chiMax?: number;
  /** Relative singular-value cutoff — drop σ_i below eps · σ_0. Defaults to 1e-12. */
  eps?: number;
}

export class MPS {
  readonly nQubits: number;
  readonly chiMax: number;
  readonly eps: number;
  /** tensors[i] has shape (χ_L · 2, χ_R). Initial state = |0…0⟩ ⇒ every T = ((1+0i, 0+0i))^T. */
  private readonly tensors: ComplexMatrix[] = [];
  /** Cumulative truncation loss: 1 − Σ kept σ² / Σ all σ², summed per update. */
  private _truncationLoss = 0;

  constructor(init: MPSInit) {
    if (init.nQubits < 1) throw new Error("nQubits must be ≥ 1");
    this.nQubits = init.nQubits;
    this.chiMax = Math.max(1, init.chiMax ?? 64);
    this.eps = init.eps ?? 1e-12;

    for (let i = 0; i < this.nQubits; i++) {
      // Every site begins χ_L = χ_R = 1. T[0, 0, 0] = 1 (|0⟩-amplitude),
      // T[0, 1, 0] = 0. Shape (2, 1).
      const t = zeros(2, 1);
      t.data[0] = 1;
      this.tensors.push(t);
    }
  }

  // ─── inspection ────────────────────────────────────────────

  /** Bond dimensions χ_1 … χ_{N−1}. χ_0 = χ_N = 1 always (boundary). */
  bondDimensions(): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.nQubits - 1; i++) out.push(this.tensors[i]!.cols);
    return out;
  }

  /** Accumulated truncation loss (sum of per-step `1 − Σ kept σ² / Σ all σ²`). */
  truncationLoss(): number {
    return this._truncationLoss;
  }

  /**
   * Compute ‖ψ‖² without materializing the full statevector.
   * Uses canonical-form sweep: after right-canonicalizing all sites
   * but the leftmost, the squared Frobenius norm of T_0 equals ‖ψ‖².
   *
   * Cheap on small N, exact for any size.
   */
  normSquared(): number {
    for (let i = this.nQubits - 1; i > 0; i--) this._rightCanonicalizeSite(i);
    const T = this.tensors[0]!;
    let s = 0;
    for (let i = 0; i < T.data.length; i++) s += T.data[i]! * T.data[i]!;
    return s;
  }

  /**
   * Renormalize the state to ‖ψ‖ = 1 by absorbing 1/√‖ψ‖² into the
   * leftmost tensor. Used after non-unitary gates (projectors,
   * imag-time evolution) to keep the represented state physical.
   */
  renormalize(): void {
    const n2 = this.normSquared();
    if (n2 <= 0) throw new Error("renormalize: state has zero norm");
    const factor = 1 / Math.sqrt(n2);
    const T = this.tensors[0]!;
    for (let i = 0; i < T.data.length; i++) T.data[i] = T.data[i]! * factor;
  }

  /**
   * Schmidt coefficients across the bond between site `cut` and `cut+1`.
   * Moves the orthogonality center to site `cut` (mixed canonical form:
   * sites [0..cut−1] left-canonical, sites [cut+1..N−1] right-canonical)
   * and SVDs the center tensor. The returned σ are the Schmidt coefficients
   * of the bipartition [0..cut] | [cut+1..N−1]; Σ σ² = ‖ψ‖².
   *
   * Mutates internal canonical form but leaves the represented state
   * invariant. Caller is free to apply more gates afterwards.
   */
  schmidtSpectrum(cut: number): Float64Array {
    if (cut < 0 || cut >= this.nQubits - 1) {
      throw new Error(`schmidtSpectrum cut=${cut} out of range [0, ${this.nQubits - 2}]`);
    }
    for (let i = 0; i < cut; i++) this._leftCanonicalizeSite(i);
    for (let i = this.nQubits - 1; i > cut; i--) this._rightCanonicalizeSite(i);
    const T = this.tensors[cut]!;
    const { sigma } = svd(T);
    return new Float64Array(sigma);
  }

  // ─── single-qubit gate ────────────────────────────────────
  apply(U: Matrix2, q: number): void {
    this.assertQubit(q);
    const T = this.tensors[q]!;
    const chiL = T.rows / 2;
    const chiR = T.cols;

    const u00r = U[0][0], u00i = U[0][1];
    const u01r = U[1][0], u01i = U[1][1];
    const u10r = U[2][0], u10i = U[2][1];
    const u11r = U[3][0], u11i = U[3][1];

    // T'[l, s', r] = Σ_s U[s', s] · T[l, s, r]. Iterate (l, r), update 2-vector.
    for (let l = 0; l < chiL; l++) {
      for (let r = 0; r < chiR; r++) {
        const k0 = ((l * 2 + 0) * chiR + r) * 2;
        const k1 = ((l * 2 + 1) * chiR + r) * 2;
        const t0r = T.data[k0]!, t0i = T.data[k0 + 1]!;
        const t1r = T.data[k1]!, t1i = T.data[k1 + 1]!;
        // new t0 = U[0,0]·t0 + U[0,1]·t1
        T.data[k0]     = u00r * t0r - u00i * t0i + u01r * t1r - u01i * t1i;
        T.data[k0 + 1] = u00r * t0i + u00i * t0r + u01r * t1i + u01i * t1r;
        T.data[k1]     = u10r * t0r - u10i * t0i + u11r * t1r - u11i * t1i;
        T.data[k1 + 1] = u10r * t0i + u10i * t0r + u11r * t1i + u11i * t1r;
      }
    }
  }

  // ─── two-site gate ────────────────────────────────────────
  /**
   * Apply a 4×4 unitary G to sites (q, q+1). G is a ComplexMatrix of shape 4×4,
   * indexed by i = s_q · 2 + s_{q+1} (site q = MSB within this 2-site index).
   * Truncates to χ_max via SVD and renormalizes.
   */
  applyTwoSite(G: ComplexMatrix, q: number): void {
    const { M, chiL, chiR } = this._buildTwoSiteMerged(G, q);
    const result = svd(M);
    this._absorbTwoSiteSvd(q, chiL, chiR, result);
  }

  /**
   * Same as applyTwoSite but the SVD is provided by an external (typically
   * GPU) async function. Lets a caller plug in `gpuJacobiSvd` for the bond-
   * merge SVD — which is the dominant cost on every two-site gate.
   *
   * The provider receives M (the full merged matrix) and must return an SVD
   * of it; downstream truncation + repacking is identical to the sync path.
   */
  async applyTwoSiteAsync(
    G: ComplexMatrix,
    q: number,
    svdProvider: (M: ComplexMatrix) => Promise<SVDResult>,
  ): Promise<void> {
    const { M, chiL, chiR } = this._buildTwoSiteMerged(G, q);
    const result = await svdProvider(M);
    this._absorbTwoSiteSvd(q, chiL, chiR, result);
  }

  /** Build the merged 4-leg-then-grouped matrix M = (G · (T_q ⊗ T_{q+1}))
   *  and reshape to (χ_L · 2, 2 · χ_R), the input to the SVD step. */
  private _buildTwoSiteMerged(
    G: ComplexMatrix,
    q: number,
  ): { M: ComplexMatrix; chiL: number; chiR: number } {
    if (q < 0 || q >= this.nQubits - 1) {
      throw new Error(`applyTwoSite q=${q} out of range [0, ${this.nQubits - 2}]`);
    }
    if (G.rows !== 4 || G.cols !== 4) {
      throw new Error(`applyTwoSite: G must be 4×4, got ${G.rows}×${G.cols}`);
    }

    // Restore mixed canonical form at bond (q, q+1) so |M|_F² = ‖ψ‖²
    // and σ-renormalization is faithful.
    this._canonicalizeBond(q);

    const Ti = this.tensors[q]!;
    const Tj = this.tensors[q + 1]!;
    const chiL = Ti.rows / 2;
    const chiBond = Ti.cols;
    if (Tj.rows / 2 !== chiBond) {
      throw new Error(`bond mismatch: T[${q}].χR=${chiBond}, T[${q + 1}].χL=${Tj.rows / 2}`);
    }
    const chiR = Tj.cols;

    // Build merged post-gate matrix M of shape (χ_L · 2, 2 · χ_R).
    // M[l·2 + s1', s2' · χ_R + r] = Σ_{s1, s2, b}
    //                                  G[s1'·2 + s2', s1·2 + s2] ·
    //                                  T_i[l, s1, b] · T_{i+1}[b, s2, r].
    const M = zeros(chiL * 2, 2 * chiR);
    const pre = new Float64Array(8); // 4 complex pre-gate amplitudes for one (l, r) slice
    for (let l = 0; l < chiL; l++) {
      for (let r = 0; r < chiR; r++) {
        // Contract bond index b to build pre[s1, s2].
        pre.fill(0);
        for (let s1 = 0; s1 < 2; s1++) {
          for (let s2 = 0; s2 < 2; s2++) {
            let sumR = 0, sumI = 0;
            for (let b = 0; b < chiBond; b++) {
              const ki = ((l * 2 + s1) * chiBond + b) * 2;
              const kj = ((b * 2 + s2) * chiR + r) * 2;
              const tir = Ti.data[ki]!, tii = Ti.data[ki + 1]!;
              const tjr = Tj.data[kj]!, tji = Tj.data[kj + 1]!;
              sumR += tir * tjr - tii * tji;
              sumI += tir * tji + tii * tjr;
            }
            pre[(s1 * 2 + s2) * 2]     = sumR;
            pre[(s1 * 2 + s2) * 2 + 1] = sumI;
          }
        }
        // Apply 4×4 gate: post[ip] = Σ_i G[ip, i] · pre[i].
        for (let ip = 0; ip < 4; ip++) {
          let postR = 0, postI = 0;
          for (let i = 0; i < 4; i++) {
            const kG = (ip * 4 + i) * 2;
            const gR = G.data[kG]!, gI = G.data[kG + 1]!;
            const pR = pre[i * 2]!, pI = pre[i * 2 + 1]!;
            postR += gR * pR - gI * pI;
            postI += gR * pI + gI * pR;
          }
          const s1p = ip >> 1;
          const s2p = ip & 1;
          const kM = ((l * 2 + s1p) * (2 * chiR) + s2p * chiR + r) * 2;
          M.data[kM]     = postR;
          M.data[kM + 1] = postI;
        }
      }
    }

    return { M, chiL, chiR };
  }

  /** Truncate, renormalize, and repack the SVD result back into T_q and T_{q+1}. */
  private _absorbTwoSiteSvd(
    q: number,
    chiL: number,
    chiR: number,
    result: SVDResult,
  ): void {
    const { U, sigma, Vh } = result;

    // Truncate.
    const sigma0 = sigma[0] ?? 0;
    let kKeep = Math.min(this.chiMax, sigma.length);
    while (kKeep > 0 && (sigma[kKeep - 1] ?? 0) < this.eps * sigma0) kKeep--;
    if (kKeep === 0) kKeep = 1;

    // Track truncation loss before renormalizing: 1 − (Σ kept σ²) / (Σ all σ²).
    let sumAll = 0, sumKept = 0;
    for (let i = 0; i < sigma.length; i++) sumAll += sigma[i]! * sigma[i]!;
    for (let i = 0; i < kKeep; i++) sumKept += sigma[i]! * sigma[i]!;
    if (sumAll > 0) this._truncationLoss += 1 - sumKept / sumAll;

    // Renormalize so the new state has unit norm: σ → σ / √(Σ kept σ²).
    const renorm = sumKept > 0 ? Math.sqrt(sumKept) : 1;

    // New T_i = U[:, :kKeep] — shape (chiL · 2, kKeep), already in left-grouped form.
    const Ti_new = zeros(chiL * 2, kKeep);
    for (let row = 0; row < chiL * 2; row++) {
      for (let col = 0; col < kKeep; col++) {
        const src = (row * U.cols + col) * 2;
        const dst = (row * kKeep + col) * 2;
        Ti_new.data[dst]     = U.data[src]!;
        Ti_new.data[dst + 1] = U.data[src + 1]!;
      }
    }

    // New T_{i+1}[j·2 + s2, r] = (σ[j] / renorm) · V^H[j, s2·χ_R + r].
    const Tj_new = zeros(kKeep * 2, chiR);
    for (let j = 0; j < kKeep; j++) {
      const s = sigma[j]! / renorm;
      for (let s2 = 0; s2 < 2; s2++) {
        for (let r = 0; r < chiR; r++) {
          const src = (j * Vh.cols + s2 * chiR + r) * 2;
          const dst = ((j * 2 + s2) * chiR + r) * 2;
          Tj_new.data[dst]     = s * Vh.data[src]!;
          Tj_new.data[dst + 1] = s * Vh.data[src + 1]!;
        }
      }
    }

    this.tensors[q] = Ti_new;
    this.tensors[q + 1] = Tj_new;
  }

  // ─── canonical-form maintenance ───────────────────────────
  //
  // Before every SVD we need M = T_q · T_{q+1} to satisfy |M|_F² = ‖ψ‖².
  // That holds iff T_0..T_{q−1} are left-canonical AND T_{q+2}..T_{N−1}
  // are right-canonical. Without this invariant, non-monotonic two-site
  // gate sequences (e.g. brick-wall: bonds 0, 2, then 1) measure a local
  // norm that isn't the global norm, and renormalizing σ to sum=1
  // silently distorts the state.
  //
  // We just sweep both sides before every applyTwoSite. Cost: O(N · χ³)
  // per gate. Trivial at N ≤ 20, χ ≤ 64.

  /** Push T[i] into left-canonical form; absorb remainder into T[i+1]. */
  private _leftCanonicalizeSite(i: number): void {
    const T = this.tensors[i]!;
    const { U, sigma, Vh } = svd(T); // T (chi_L*2 × chi_R) = U · diag(σ) · Vh
    const k = sigma.length;
    // T[i] := U.
    this.tensors[i] = U;

    if (i + 1 >= this.nQubits) return; // no right neighbor to absorb into

    // Build SR = diag(σ) · Vh, shape (k, T.cols)
    const SR = clone(Vh);
    for (let j = 0; j < k; j++) {
      const s = sigma[j]!;
      for (let c = 0; c < SR.cols; c++) {
        const idx = (j * SR.cols + c) * 2;
        SR.data[idx]     = SR.data[idx]!     * s;
        SR.data[idx + 1] = SR.data[idx + 1]! * s;
      }
    }

    // Absorb SR into T[i+1]. T[i+1] in left-grouped (chi_bond_old · 2, chi_next);
    // reshape to (chi_bond_old, 2·chi_next), matmul SR from left to get (k, 2·chi_next),
    // reshape back to (k · 2, chi_next). SR.cols must equal chi_bond_old.
    const Tnext = this.tensors[i + 1]!;
    const chi_bond_old = Tnext.rows / 2;
    const chi_next = Tnext.cols;
    if (SR.cols !== chi_bond_old) {
      throw new Error(`bond mismatch in leftCanonicalizeSite(${i}): SR.cols=${SR.cols}, T[${i + 1}].χL=${chi_bond_old}`);
    }
    const Tnext_flat = zeros(chi_bond_old, 2 * chi_next);
    for (let b = 0; b < chi_bond_old; b++) {
      for (let s = 0; s < 2; s++) {
        for (let r = 0; r < chi_next; r++) {
          const src = ((b * 2 + s) * chi_next + r) * 2;
          const dst = (b * (2 * chi_next) + s * chi_next + r) * 2;
          Tnext_flat.data[dst]     = Tnext.data[src]!;
          Tnext_flat.data[dst + 1] = Tnext.data[src + 1]!;
        }
      }
    }
    const product = matmul(SR, Tnext_flat); // (k, 2·chi_next)
    const Tnext_new = zeros(k * 2, chi_next);
    for (let j = 0; j < k; j++) {
      for (let s = 0; s < 2; s++) {
        for (let r = 0; r < chi_next; r++) {
          const src = (j * (2 * chi_next) + s * chi_next + r) * 2;
          const dst = ((j * 2 + s) * chi_next + r) * 2;
          Tnext_new.data[dst]     = product.data[src]!;
          Tnext_new.data[dst + 1] = product.data[src + 1]!;
        }
      }
    }
    this.tensors[i + 1] = Tnext_new;
  }

  /** Push T[i] into right-canonical form; absorb remainder into T[i−1]. */
  private _rightCanonicalizeSite(i: number): void {
    const T = this.tensors[i]!;
    const chi_L = T.rows / 2;
    const chi_R = T.cols;

    // Reshape (chi_L · 2, chi_R) → (chi_L, 2 · chi_R).
    const Tflat = zeros(chi_L, 2 * chi_R);
    for (let l = 0; l < chi_L; l++) {
      for (let s = 0; s < 2; s++) {
        for (let r = 0; r < chi_R; r++) {
          const src = ((l * 2 + s) * chi_R + r) * 2;
          const dst = (l * (2 * chi_R) + s * chi_R + r) * 2;
          Tflat.data[dst]     = T.data[src]!;
          Tflat.data[dst + 1] = T.data[src + 1]!;
        }
      }
    }

    // SVD: Tflat (chi_L × 2·chi_R) = U · diag(σ) · Vh. k = min(chi_L, 2·chi_R).
    const { U, sigma, Vh } = svd(Tflat);
    const k = sigma.length;

    // T[i] := Vh reshaped to (k · 2, chi_R): new[j·2 + s, r] = Vh[j, s·chi_R + r].
    const T_new = zeros(k * 2, chi_R);
    for (let j = 0; j < k; j++) {
      for (let s = 0; s < 2; s++) {
        for (let r = 0; r < chi_R; r++) {
          const src = (j * Vh.cols + s * chi_R + r) * 2;
          const dst = ((j * 2 + s) * chi_R + r) * 2;
          T_new.data[dst]     = Vh.data[src]!;
          T_new.data[dst + 1] = Vh.data[src + 1]!;
        }
      }
    }
    this.tensors[i] = T_new;

    if (i - 1 < 0) return; // no left neighbor

    // Absorb U · diag(σ) into T[i−1]: T[i−1] ← T[i−1] · (U · diag(σ)).
    const Tprev = this.tensors[i - 1]!;
    if (Tprev.cols !== chi_L) {
      throw new Error(`bond mismatch in rightCanonicalizeSite(${i}): T[${i - 1}].χR=${Tprev.cols}, chi_L=${chi_L}`);
    }
    // US = U · diag(σ), shape (chi_L, k). U has shape (chi_L, ?); U.cols may equal k already since svd returns m×k.
    const US = zeros(chi_L, k);
    for (let row = 0; row < chi_L; row++) {
      for (let col = 0; col < k; col++) {
        const s = sigma[col]!;
        const src = (row * U.cols + col) * 2;
        const dst = (row * k + col) * 2;
        US.data[dst]     = s * U.data[src]!;
        US.data[dst + 1] = s * U.data[src + 1]!;
      }
    }
    const Tprev_new = matmul(Tprev, US); // (chi_L_prev · 2, chi_L) · (chi_L, k) = (chi_L_prev · 2, k)
    this.tensors[i - 1] = Tprev_new;
  }

  /** Put MPS in mixed canonical form with center at bond (q, q+1): T_0..T_{q−1}
      left-canonical, T_{q+2}..T_{N−1} right-canonical. */
  private _canonicalizeBond(q: number): void {
    for (let i = 0; i < q; i++) this._leftCanonicalizeSite(i);
    for (let i = this.nQubits - 1; i > q + 1; i--) this._rightCanonicalizeSite(i);
  }

  /** Controlled-U (|c − t| = 1 only for v1). Builds the right 4×4 internally. */
  applyControlled(U: Matrix2, control: number, target: number): void {
    this.assertQubit(control);
    this.assertQubit(target);
    if (control === target) throw new Error("control === target");
    if (Math.abs(control - target) !== 1) {
      throw new Error(
        `MPS v1 supports adjacent controlled gates only (|c − t| = 1). ` +
        `Got control=${control}, target=${target}.`,
      );
    }
    const lo = Math.min(control, target);
    const controlIsLo = control < target;
    const G = buildControlledMatrix4(U, controlIsLo);
    this.applyTwoSite(G, lo);
  }

  cnot(control: number, target: number): void {
    this.applyControlled(X, control, target);
  }

  // ─── readout ──────────────────────────────────────────────
  /**
   * Full dense statevector. Interleaved [re, im], length 2 · 2^N.
   * Index convention: qubit 0 = LSB, so ψ[s_0 + 2·s_1 + … + 2^{N−1}·s_{N−1}].
   *
   * Cost: O(2^N · χ²) memory peak, O(2^N · χ² · N) time. Use only for
   * verification, never hot-path.
   */
  statevector(): Float64Array {
    if (this.nQubits > 24) {
      throw new Error(`statevector() refuses N=${this.nQubits} > 24 (would allocate ${16 * (1 << this.nQubits) / 1e9}GB)`);
    }
    // Ψ_0 = T[0] with shape (2, χ_1). 2^1 rows.
    let psi: ComplexMatrix = zeros(this.tensors[0]!.rows, this.tensors[0]!.cols);
    psi.data.set(this.tensors[0]!.data);

    for (let k = 1; k < this.nQubits; k++) {
      const Tk = this.tensors[k]!;
      const chiK = Tk.rows / 2;
      const chiNext = Tk.cols;
      if (psi.cols !== chiK) {
        throw new Error(`bond mismatch at readout site ${k}: Ψ.cols=${psi.cols}, T[${k}].χL=${chiK}`);
      }

      // Reshape Tk from left-grouped (χ_k · 2, χ_{k+1}) to (χ_k, 2·χ_{k+1}).
      //   Tk_reshape[b, s · χ_next + r] = Tk[b · 2 + s, r].
      const Tk_reshape = zeros(chiK, 2 * chiNext);
      for (let b = 0; b < chiK; b++) {
        for (let s = 0; s < 2; s++) {
          for (let r = 0; r < chiNext; r++) {
            const src = ((b * 2 + s) * chiNext + r) * 2;
            const dst = (b * (2 * chiNext) + s * chiNext + r) * 2;
            Tk_reshape.data[dst]     = Tk.data[src]!;
            Tk_reshape.data[dst + 1] = Tk.data[src + 1]!;
          }
        }
      }

      // product = Ψ · Tk_reshape of shape (2^k, 2 · χ_{k+1}).
      const product = matmul(psi, Tk_reshape);

      // Reshape to (2^{k+1}, χ_{k+1}) with qubit k as the NEWLY MOST SIGNIFICANT bit:
      //   new[row + 2^k · s, r] = product[row, s · χ_next + r].
      const prevRows = product.rows;
      const newPsi = zeros(prevRows * 2, chiNext);
      for (let row = 0; row < prevRows; row++) {
        for (let s = 0; s < 2; s++) {
          for (let r = 0; r < chiNext; r++) {
            const src = (row * (2 * chiNext) + s * chiNext + r) * 2;
            const dst = ((row + prevRows * s) * chiNext + r) * 2;
            newPsi.data[dst]     = product.data[src]!;
            newPsi.data[dst + 1] = product.data[src + 1]!;
          }
        }
      }
      psi = newPsi;
    }

    if (psi.cols !== 1) throw new Error(`final χ_N=${psi.cols}, expected 1`);
    return psi.data;
  }

  /** Σ|ψ_i|² from the full dense statevector (verification only). */
  norm(): number {
    const amps = this.statevector();
    let s = 0;
    for (let i = 0; i < amps.length; i += 2) s += amps[i]! * amps[i]! + amps[i + 1]! * amps[i + 1]!;
    return s;
  }

  // ─── helpers ──────────────────────────────────────────────
  private assertQubit(q: number): void {
    if (!Number.isInteger(q) || q < 0 || q >= this.nQubits) {
      throw new Error(`qubit ${q} out of [0, ${this.nQubits})`);
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Build a 4×4 complex matrix for controlled-U, indexed by i = s_lo · 2 + s_hi
// where lo = min(c, t), hi = max(c, t) = lo + 1.
//
//   controlIsLo = true  → control at site lo, target at site hi.
//   controlIsLo = false → control at site hi, target at site lo.
// ─────────────────────────────────────────────────────────────
function buildControlledMatrix4(U: Matrix2, controlIsLo: boolean): ComplexMatrix {
  const G = zeros(4, 4);
  const [u00, u01, u10, u11] = U;
  // Identity rows when control = 0, and action of U on target when control = 1.
  const setC = (row: number, col: number, z: Complex): void => {
    cset(G, row, col, z[0], z[1]);
  };
  const ONE: Complex = [1, 0];
  const ZERO: Complex = [0, 0];

  for (let sLo = 0; sLo < 2; sLo++) {
    for (let sHi = 0; sHi < 2; sHi++) {
      const i = sLo * 2 + sHi;
      const sControl = controlIsLo ? sLo : sHi;
      const sTarget  = controlIsLo ? sHi : sLo;

      if (sControl === 0) {
        // Identity block: output = input.
        setC(i, i, ONE);
      } else {
        // Output: Σ_{sTarget'} U[sTarget', sTarget] · |sControl, sTarget'⟩.
        for (let sTargetP = 0; sTargetP < 2; sTargetP++) {
          const uEntry =
            sTargetP === 0
              ? (sTarget === 0 ? u00 : u01)
              : (sTarget === 0 ? u10 : u11);
          const sLoP = controlIsLo ? sLo : sTargetP;
          const sHiP = controlIsLo ? sTargetP : sHi;
          const ip = sLoP * 2 + sHiP;
          setC(ip, i, uEntry);
        }
        // Make sure we haven't left stale zeros (they are zero-initialized).
        void ZERO;
      }
    }
  }
  return G;
}
