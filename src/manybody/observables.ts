// ─────────────────────────────────────────────────────────────
// observables.ts — per-site Pauli expectation values from a flat
// statevector. Used to drive the magnetization-arrow viz and the
// order-parameter sweep.
//
// For an MPS with statevector ψ ∈ ℂ^(2^N) (interleaved real/imag),
//   ⟨σ_x_q⟩ = Σ_b Re(ψ_b* · ψ_{b ⊕ bit_q})
//   ⟨σ_y_q⟩ = Σ_b sign(b)·(Im) where sign = ±1 by occupation of q
//   ⟨σ_z_q⟩ = Σ_b (1 − 2·s_q) · |ψ_b|²
//
// All three are real (Pauli matrices are Hermitian). Cost is
// O(N · 2^N), trivial for N ≤ 20 (the statevector ceiling we use
// for the viz path).
// ─────────────────────────────────────────────────────────────

export interface SiteExpectations {
  /** ⟨σ_x⟩ at each site, length N. */
  sx: Float64Array;
  /** ⟨σ_y⟩ at each site, length N. */
  sy: Float64Array;
  /** ⟨σ_z⟩ at each site, length N. */
  sz: Float64Array;
}

/** Compute the per-site Pauli expectations of an interleaved Float64 statevector. */
export function siteExpectations(psi: Float64Array, N: number): SiteExpectations {
  const dim = 1 << N;
  if (psi.length !== 2 * dim) {
    throw new Error(`siteExpectations: psi length ${psi.length} ≠ 2·2^${N}`);
  }
  const sx = new Float64Array(N);
  const sy = new Float64Array(N);
  const sz = new Float64Array(N);

  for (let q = 0; q < N; q++) {
    const bitQ = 1 << q;
    let exX = 0, exY = 0, exZ = 0;
    for (let b = 0; b < dim; b++) {
      const sq = (b >> q) & 1;
      const bp = b ^ bitQ;

      const psiR = psi[b * 2]!;
      const psiI = psi[b * 2 + 1]!;
      const ppR = psi[bp * 2]!;
      const ppI = psi[bp * 2 + 1]!;

      // σ_z: diagonal, ±1 by occupation.
      exZ += (sq === 0 ? 1 : -1) * (psiR * psiR + psiI * psiI);

      // σ_x: off-diagonal, ⟨b|σ_x|bp⟩ = 1 always.
      // ψ_b* · ψ_bp = (psiR psiPR + psiI psiPI) + i(psiR psiPI − psiI psiPR).
      // We want the real part.
      exX += psiR * ppR + psiI * ppI;

      // σ_y: ⟨0|σ_y|1⟩ = -i, ⟨1|σ_y|0⟩ = +i. Sign in front of i:
      //   sq=0 → contribution from ψ_b* · (-i · ψ_bp); real part = -(psiR·psiPI − psiI·psiPR)·(-1) = psiR·psiPI − psiI·psiPR
      //   sq=1 → +i version, real part = -(psiR·psiPI − psiI·psiPR)
      const sign = sq === 0 ? +1 : -1;
      exY += sign * (psiR * ppI - psiI * ppR);
    }
    sx[q] = exX;
    sy[q] = exY;
    sz[q] = exZ;
  }
  return { sx, sy, sz };
}

/** Z-magnetization order parameter: (1/N) Σ |⟨σ_z_i⟩|. */
export function zMagnetization(exps: SiteExpectations): number {
  let s = 0;
  for (let i = 0; i < exps.sz.length; i++) s += Math.abs(exps.sz[i]!);
  return s / exps.sz.length;
}

/** Total in-plane magnetization: (1/N) Σ √(⟨σ_x⟩² + ⟨σ_y⟩²). */
export function xyMagnetization(exps: SiteExpectations): number {
  let s = 0;
  for (let i = 0; i < exps.sx.length; i++) {
    const x = exps.sx[i]!, y = exps.sy[i]!;
    s += Math.sqrt(x * x + y * y);
  }
  return s / exps.sx.length;
}
