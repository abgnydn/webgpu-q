// ─────────────────────────────────────────────────────────────
// sector-matvec.ts — matrix-free H · x in a particle-number
// sector. Phase C v5.
//
// Why: the sector-projected dense H grows as O(k²) where
// k = C(2n, N_e). For CH₄ (n=9, N_e=10 → k=43758) that's 15 GB,
// at the edge of a 16 GB machine. For NH₃, methanol, and any
// medium-organic, k is in the 50k–500k range and the dense
// matrix doesn't fit.
//
// `applySectorH(x, y, …)` computes y = H · x by iterating every
// operator term (Vnn diagonal + one-body Σ h_pq a†_p a_q + two-
// body ½ Σ (pq|rs) a†_p a†_r a_s a_q) and applying it directly
// to x via JW. Memory cost is O(k) for the input/output vectors
// only — a 100×–1000× savings over dense storage.
//
// Cost per matvec: O(n^4 × k × n) ≈ same wall-clock as one full
// `buildSectorH` call. For Lanczos, the alternative is to
// precompute a sparse CSR Hsec ONCE (one buildSectorH-equivalent
// pass), then matvec is O(nnz) ≈ O(n^2 × k). That's the path
// `buildSparseSectorH` below takes — best of both worlds.
// ─────────────────────────────────────────────────────────────

export interface SparseSectorH {
  /** Sector dim. */
  readonly k: number;
  /** CSR row pointers, length k+1. */
  readonly rowStart: Int32Array;
  /** Column indices, length nnz. */
  readonly colIdx: Int32Array;
  /** Values, length nnz. */
  readonly values: Float64Array;
  /** Sector basis state indices (full-Hilbert encoding). */
  readonly sectorIdx: Int32Array;
  readonly nQubits: number;
  readonly particleNumber: number;
}

/** JW operator: returns null if state has the wrong occupancy. */
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
 * Build a sparse CSR sector Hamiltonian. Same FCI eigenvalue as
 * `buildSectorH` but stores only the nonzero entries — for CH₄
 * the dense H is 15 GB while the sparse equivalent is ~150 MB.
 *
 * Construction does one full pass over every operator term and
 * accumulates contributions into a per-row Map<col, value>; then
 * deduplicates and packs into CSR. Cost ≈ same as buildSectorH.
 */
export function buildSparseSectorH(
  nSpatial: number,
  h_OAO: Float64Array,
  eri_OAO: Float64Array,
  Vnn: number,
  particleNumber: number,
): SparseSectorH {
  const nQubits = 2 * nSpatial;
  const sectorIdx = enumerateSector(nQubits, particleNumber);
  const k = sectorIdx.length;
  if (k === 0) {
    throw new Error(`buildSparseSectorH: no basis states with N=${particleNumber} in ${nQubits}-qubit space`);
  }
  const stateToSector = new Int32Array(1 << nQubits).fill(-1);
  for (let a = 0; a < k; a++) stateToSector[sectorIdx[a]!] = a;

  // Per-row sparse row maps: row[i] = Map<col, value>.
  // (Map<>; pre-sized via Array.from with empty maps to avoid expand cost.)
  const rows: Map<number, number>[] = new Array(k);
  for (let i = 0; i < k; i++) rows[i] = new Map();

  // Vnn diagonal.
  for (let a = 0; a < k; a++) rows[a]!.set(a, Vnn);

  // One-body Σ_{pq, σ} h_pq a†_p a_q (per spin σ).
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
          const v = hpq * r1.sign * r2.sign;
          const row = rows[aTo]!;
          row.set(a, (row.get(a) ?? 0) + v);
        }
      }
    }
  }

  // Two-body ½ Σ (pq|rs) a†_p a†_r a_s a_q (per spin σ, τ).
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
                const v = half * r1.sign * r2.sign * r3.sign * r4.sign;
                const row = rows[aTo]!;
                row.set(a, (row.get(a) ?? 0) + v);
              }
            }
          }
        }
      }
    }
  }

  // Convert row Maps → CSR. Drop entries that ended up exactly zero
  // (e.g. exchange-cancellation between (pq|rs) terms).
  let nnz = 0;
  for (let i = 0; i < k; i++) {
    for (const v of rows[i]!.values()) {
      if (v !== 0) nnz++;
    }
  }
  const rowStart = new Int32Array(k + 1);
  const colIdx = new Int32Array(nnz);
  const values = new Float64Array(nnz);
  let cursor = 0;
  for (let i = 0; i < k; i++) {
    rowStart[i] = cursor;
    // Sort columns ascending for predictable cache behaviour during matvec.
    const entries: [number, number][] = [];
    for (const [c, v] of rows[i]!) {
      if (v !== 0) entries.push([c, v]);
    }
    entries.sort((a, b) => a[0] - b[0]);
    for (const [c, v] of entries) {
      colIdx[cursor] = c;
      values[cursor] = v;
      cursor++;
    }
    rows[i] = new Map();  // free this row's Map
  }
  rowStart[k] = cursor;
  return { k, rowStart, colIdx, values, sectorIdx, nQubits, particleNumber };
}

/** y = H · x with H stored as CSR. */
export function sparseMatvec(H: SparseSectorH, x: Float64Array, y: Float64Array): void {
  const { k, rowStart, colIdx, values } = H;
  for (let i = 0; i < k; i++) {
    let s = 0;
    const start = rowStart[i]!;
    const end = rowStart[i + 1]!;
    for (let p = start; p < end; p++) {
      s += values[p]! * x[colIdx[p]!]!;
    }
    y[i] = s;
  }
}
