// ─────────────────────────────────────────────────────────────
// sector-builder.ts — build a second-quantized Hamiltonian
// directly in a single particle-number sector, never materializing
// the full 2^N × 2^N dense matrix.
//
// Why: BeH₂ in full STO-3G has 14 spin-orbitals → 2^14 = 16384-dim
// Hilbert space. The full dense H would be 16384² × 8 B = 2 GB —
// won't fit in a browser tab. But [H, N̂] = 0, so H is block-
// diagonal in particle number. The neutral BeH₂ has 6 electrons,
// living in the C(14, 6) = 3003-dim sector — only 9 M complex
// matrix entries, ~72 MB. Perfectly manageable.
//
// Algorithm:
//   1. Enumerate basis states with the requested particle number.
//   2. For each one-body term coeff · a†_p a_q:
//      • For each sector state s, compute the result state s'
//        and JW sign. Add coeff · sign to Hsec[idx[s'], idx[s]].
//   3. For each two-body term ½ coeff · a†_p a†_r a_s a_q:
//      • Apply the four operators in sequence; if all succeed,
//        bump the matrix element.
//   4. Add Vnn · I (diagonal).
//
// Output: a dense (k × k) Float64Array where k = C(2n, N_e).
// Hermitian by construction.
// ─────────────────────────────────────────────────────────────

export interface SectorH {
  /** Hsec, k × k row-major, real-symmetric. */
  readonly H: Float64Array;
  /** Sector basis state indices in the full 2^nQubits computational basis. */
  readonly sectorIdx: Int32Array;
  /** k = sector dimension = sectorIdx.length. */
  readonly k: number;
  /** nQubits = 2 · n_spatial_orbitals. */
  readonly nQubits: number;
  /** Particle number this sector projects to. */
  readonly particleNumber: number;
}

/** JW operator: returns null if the state has the wrong occupancy. */
function applyOp(state: number, qubit: number, creation: boolean): { newState: number; sign: number } | null {
  const occupied = (state >>> qubit) & 1;
  if (creation && occupied) return null;
  if (!creation && !occupied) return null;
  let sign = 1;
  let mask = 1;
  for (let k = 0; k < qubit; k++) {
    if ((state & mask) !== 0) sign = -sign;
    mask <<= 1;
  }
  return { newState: state ^ (1 << qubit), sign };
}

/** Build the index list of basis states with the given particle number. */
function enumerateSector(nQubits: number, particleNumber: number): Int32Array {
  const dim = 1 << nQubits;
  const out: number[] = [];
  for (let i = 0; i < dim; i++) {
    let n = 0;
    for (let q = 0; q < nQubits; q++) if ((i >>> q) & 1) n++;
    if (n === particleNumber) out.push(i);
  }
  return new Int32Array(out);
}

/**
 * Build the second-quantized Hamiltonian projected to a specific
 * particle-number sector.
 *
 *   H = V_nn · I + Σ_{pq, σ} h_pq · a†_{2p+σ} a_{2q+σ}
 *     + ½ Σ_{pqrs, στ} (pq|rs) · a†_{2p+σ} a†_{2r+τ} a_{2s+τ} a_{2q+σ}
 *
 * `nSpatial` = number of spatial orbitals (so nQubits = 2 · nSpatial).
 * `h_OAO` is shape (nSpatial × nSpatial), `eri_OAO` is shape n^4
 * row-major as `[((p * n + q) * n + r) * n + s]`.
 */
export function buildSectorH(
  nSpatial: number,
  h_OAO: Float64Array,
  eri_OAO: Float64Array,
  Vnn: number,
  particleNumber: number,
): SectorH {
  const nQubits = 2 * nSpatial;
  const sectorIdx = enumerateSector(nQubits, particleNumber);
  const k = sectorIdx.length;
  if (k === 0) {
    throw new Error(`buildSectorH: no basis states with N=${particleNumber} in ${nQubits}-qubit space`);
  }
  // Reverse lookup: full-index → sector-position. Use a typed array;
  // 2^14 = 16384 entries is fine.
  const stateToSector = new Int32Array(1 << nQubits).fill(-1);
  for (let a = 0; a < k; a++) stateToSector[sectorIdx[a]!] = a;

  const H = new Float64Array(k * k);
  // Vnn on the diagonal.
  for (let a = 0; a < k; a++) H[a * k + a] = Vnn;

  // ── One-body: Σ_{pq, σ} h_pq · a†_{2p+σ} a_{2q+σ} ─────────
  // Operator is defined on the full Hilbert space; we evaluate
  // its matrix elements only between sector states (preserves N).
  for (let p = 0; p < nSpatial; p++) {
    for (let q = 0; q < nSpatial; q++) {
      const hpq = h_OAO[p * nSpatial + q]!;
      if (hpq === 0) continue;
      for (let sigma = 0; sigma < 2; sigma++) {
        const qBit = 2 * q + sigma;
        const pBit = 2 * p + sigma;
        for (let a = 0; a < k; a++) {
          const s0 = sectorIdx[a]!;
          const r1 = applyOp(s0, qBit, false);
          if (!r1) continue;
          const r2 = applyOp(r1.newState, pBit, true);
          if (!r2) continue;
          const aTo = stateToSector[r2.newState]!;
          if (aTo < 0) continue;  // shouldn't happen — sanity
          const idx = aTo * k + a;
          H[idx] = H[idx]! + hpq * r1.sign * r2.sign;
        }
      }
    }
  }

  // ── Two-body: ½ Σ (pq|rs) · a†_p a†_r a_s a_q ─────────────
  // Skip near-zero terms aggressively; ERI tensor is ~80% zero
  // by chemist symmetries on most molecules.
  for (let p = 0; p < nSpatial; p++) {
    for (let q = 0; q < nSpatial; q++) {
      for (let r = 0; r < nSpatial; r++) {
        for (let s = 0; s < nSpatial; s++) {
          const pqrs = eri_OAO[((p * nSpatial + q) * nSpatial + r) * nSpatial + s]!;
          if (Math.abs(pqrs) < 1e-15) continue;
          const half = 0.5 * pqrs;
          for (let sigma = 0; sigma < 2; sigma++) {
            for (let tau = 0; tau < 2; tau++) {
              const qBit = 2 * q + sigma;
              const sBit = 2 * s + tau;
              const rBit = 2 * r + tau;
              const pBit = 2 * p + sigma;
              for (let a = 0; a < k; a++) {
                const s0 = sectorIdx[a]!;
                const r1 = applyOp(s0, qBit, false);
                if (!r1) continue;
                const r2 = applyOp(r1.newState, sBit, false);
                if (!r2) continue;
                const r3 = applyOp(r2.newState, rBit, true);
                if (!r3) continue;
                const r4 = applyOp(r3.newState, pBit, true);
                if (!r4) continue;
                const aTo = stateToSector[r4.newState]!;
                if (aTo < 0) continue;
                const idx = aTo * k + a;
                H[idx] = H[idx]! + half * r1.sign * r2.sign * r3.sign * r4.sign;
              }
            }
          }
        }
      }
    }
  }

  return { H, sectorIdx, k, nQubits, particleNumber };
}
