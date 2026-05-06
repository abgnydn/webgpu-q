// ─────────────────────────────────────────────────────────────
// cg-molecular.ts — generic AO-integral pipeline for any
// molecule, built from CGShells (Cartesian-Gaussian shells with
// arbitrary angular momentum). Replaces the per-molecule integral
// duplication in lih-builder / beh2-builder for v3 onward.
//
// Inputs:
//   • shells: an ordered list of CGShells. Each shell is one
//     "spatial orbital" in the AO basis. For BeH₂-full STO-3G
//     this is 7 shells (Be 1s, Be 2s, Be 2p_x, 2p_y, 2p_z,
//     H_L 1s, H_R 1s).
//   • nuclei: list of (Z, position) tuples. Nuclear-attraction
//     and nuclear-nuclear repulsion are summed over every nucleus.
//
// Outputs:
//   • S_AO, h_AO: n × n row-major.
//   • eri_AO: n^4 row-major as [μν λσ].
//   • X = S^{-1/2}: Löwdin orthogonalization transform.
//   • h_OAO, eri_OAO: same shapes, in the orthogonalized basis.
//   • Vnn: nuclear-nuclear repulsion (Hartree).
// ─────────────────────────────────────────────────────────────

import { type CGShell, S_cg, T_cg, V_cg, ERI_cg } from "./integrals-cg.js";
import { eigsymmetric } from "../manybody/dense-eig.js";

export interface Nucleus {
  readonly Z: number;
  readonly pos: readonly [number, number, number];
}

export interface MolecularIntegrals {
  readonly n: number;                            // number of spatial orbitals (= shells.length)
  readonly shells: readonly CGShell[];
  readonly nuclei: readonly Nucleus[];
  readonly S_AO: Float64Array;                   // n × n
  readonly h_AO: Float64Array;                   // n × n
  readonly eri_AO: Float64Array;                 // n^4 row-major
  readonly X: Float64Array;                      // n × n  (S^{-1/2})
  readonly h_OAO: Float64Array;                  // n × n
  readonly eri_OAO: Float64Array;                // n^4
  readonly Vnn: number;                          // nuclear-nuclear repulsion
}

export interface IntegralOpts {
  /**
   * If true, transform every d-shell from the 6 Cartesian functions
   * (xx, yy, zz, xy, xz, yz) to the 5 real spherical harmonics
   * (d_z², d_x²-y², d_xy, d_xz, d_yz). Drops the redundant trace
   * combination (xx + yy + zz)/√3 — the "extra s" component that
   * makes the Cartesian basis variationally larger and produces
   * the documented ~4 mHa Cartesian-vs-spherical-d slack on H₂O
   * cc-pVDZ vs PySCF. Default false.
   */
  readonly spherical?: boolean;
}

/**
 * Compute the full atomic-orbital integral set + Löwdin orthogonalize.
 * This is the heavy step — for n shells the ERI tensor is O(n^4) and
 * each ERI is O(n_prim^4) primitives. For STO-3G (3 primitives/shell)
 * and n=7 (BeH₂ full) that's ~100k primitive ERIs. Few seconds.
 *
 * If `opts.spherical` is true, every d-shell (6 consecutive Cartesian
 * functions ordered xx, yy, zz, xy, xz, yz with identical center +
 * primitives) is converted to 5 real spherical harmonics post-AO-build,
 * matching the standard PySCF / Psi4 / ORCA convention.
 */
export function computeMolecularIntegrals(
  shells: readonly CGShell[],
  nuclei: readonly Nucleus[],
  opts: IntegralOpts = {},
): MolecularIntegrals {
  let n = shells.length;
  let S_AO: Float64Array = new Float64Array(n * n);
  let h_AO: Float64Array = new Float64Array(n * n);
  let eri_AO: Float64Array = new Float64Array(n * n * n * n);

  // ── 1-electron integrals ────────────────────────────────────
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const sij = S_cg(shells[i]!, shells[j]!);
      const tij = T_cg(shells[i]!, shells[j]!);
      let vSum = 0;
      for (const { Z, pos } of nuclei) vSum += V_cg(shells[i]!, shells[j]!, Z, pos);
      S_AO[i * n + j] = sij;
      S_AO[j * n + i] = sij;
      const h = tij + vSum;
      h_AO[i * n + j] = h;
      h_AO[j * n + i] = h;
    }
  }

  // ── 2-electron integrals ───────────────────────────────────
  // Use 8-fold symmetry to compute only unique ones, plus Schwarz
  // screening: |⟨μν|λσ⟩| ≤ √⟨μν|μν⟩ · √⟨λσ|λσ⟩, so a precomputed
  // Q[μ,ν] = √⟨μν|μν⟩ table lets us skip negligible (μν|λσ) pairs.
  // Threshold 1e-10 is well below sub-µHa precision and gives 2-5×
  // speedup on cc-pVDZ-class basis sets.
  const SCHWARZ_TOL = 1e-10;
  const Q = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu; nu < n; nu++) {
      const v = ERI_cg(shells[mu]!, shells[nu]!, shells[mu]!, shells[nu]!);
      const q = Math.sqrt(Math.abs(v));
      Q[mu * n + nu] = q;
      Q[nu * n + mu] = q;
    }
  }
  // Index encoding: (μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν).
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu; nu < n; nu++) {
      const qMuNu = Q[mu * n + nu]!;
      if (qMuNu < SCHWARZ_TOL) continue;     // whole (μν, *) row negligible
      for (let la = 0; la < n; la++) {
        for (let si = la; si < n; si++) {
          // Skip double-count via the (μν) ↔ (λσ) swap: only do
          // (μν, λσ) where the encoded pair (μ·n+ν) ≤ (λ·n+σ).
          if (mu * n + nu > la * n + si) continue;
          if (qMuNu * Q[la * n + si]! < SCHWARZ_TOL) continue;
          const v = ERI_cg(shells[mu]!, shells[nu]!, shells[la]!, shells[si]!);
          // Fill all 8 symmetric entries.
          const idx = (a: number, b: number, c: number, d: number): number =>
            ((a * n + b) * n + c) * n + d;
          eri_AO[idx(mu, nu, la, si)] = v;
          eri_AO[idx(nu, mu, la, si)] = v;
          eri_AO[idx(mu, nu, si, la)] = v;
          eri_AO[idx(nu, mu, si, la)] = v;
          eri_AO[idx(la, si, mu, nu)] = v;
          eri_AO[idx(si, la, mu, nu)] = v;
          eri_AO[idx(la, si, nu, mu)] = v;
          eri_AO[idx(si, la, nu, mu)] = v;
        }
      }
    }
  }

  // ── Optional: Cartesian d → spherical-harmonic d transform ──
  // Identifies every d-shell group (6 consecutive shells with same
  // center + primitives + angular momentum 2 in the canonical
  // xx, yy, zz, xy, xz, yz order) and replaces it with 5 real
  // spherical harmonics. Drops the redundant (xx+yy+zz)/√3 mode
  // that makes the Cartesian basis variationally larger.
  let nFinal = n;
  if (opts.spherical) {
    const T = buildSphericalDTransform(shells);
    nFinal = T.length / n;
    S_AO = transformS_2idx(S_AO, T, n, nFinal);
    h_AO = transformS_2idx(h_AO, T, n, nFinal);
    eri_AO = transformS_4idx(eri_AO, T, n, nFinal);
    n = nFinal;
  }

  // ── Löwdin orthogonalization X = S^{-1/2} ──────────────────
  const eig = eigsymmetric(S_AO, n);
  const X = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        const lam = eig.values[k]!;
        if (lam <= 0) {
          throw new Error(`computeMolecularIntegrals: AO overlap eigenvalue ${lam} ≤ 0 — basis degenerate or non-positive-definite`);
        }
        s += eig.vectors[k * n + r]! * Math.pow(lam, -0.5) * eig.vectors[k * n + c]!;
      }
      X[r * n + c] = s;
    }
  }

  // ── Transform h, ERI to OAO ─────────────────────────────────
  const h_OAO = transform2(h_AO, X, n);
  const eri_OAO = transform4(eri_AO, X, n);

  // ── Nuclear-nuclear repulsion ──────────────────────────────
  let Vnn = 0;
  for (let i = 0; i < nuclei.length; i++) {
    const a = nuclei[i]!;
    for (let j = i + 1; j < nuclei.length; j++) {
      const b = nuclei[j]!;
      const dx = a.pos[0] - b.pos[0];
      const dy = a.pos[1] - b.pos[1];
      const dz = a.pos[2] - b.pos[2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (r > 1e-10) Vnn += a.Z * b.Z / r;
    }
  }

  return { n, shells, nuclei, S_AO, h_AO, eri_AO, X, h_OAO, eri_OAO, Vnn };
}

// ── Spherical-d-shell transform ─────────────────────────────────
//
// Detects d-shell groups in the shell list and produces the
// (n_sph × n_cart) transform matrix T that maps the 6 Cartesian d
// functions to 5 real spherical harmonics for each group, keeping
// non-d shells as identity rows.
//
// Coefficients (in the unit-normalized Cartesian d basis where
// ⟨d_xx|d_xx⟩ = ⟨d_xy|d_xy⟩ = 1 but ⟨d_xx|d_yy⟩ = 1/3):
//   d_z²    = (-1/2) d_xx + (-1/2) d_yy + (1) d_zz
//   d_x²-y² = (√3/2) d_xx + (-√3/2) d_yy
//   d_xy    = d_xy
//   d_xz    = d_xz
//   d_yz    = d_yz
// All 5 spherical functions are unit-normalized and mutually
// orthogonal in the Cartesian-d metric (verified by the test in
// ccpvdz-spherical.test.ts).
//
// Expected ordering of the 6 Cartesian d-shell members within the
// shell list: angular momenta (2,0,0), (0,2,0), (0,0,2), (1,1,0),
// (1,0,1), (0,1,1) — the canonical order produced by
// `atomShellsCcPvdz` in atoms.ts. Other orderings throw.

const D_TRANSFORM: ReadonlyArray<readonly number[]> = [
  // each row is (xx, yy, zz, xy, xz, yz) coefficients
  [-1 / 2, -1 / 2, 1, 0, 0, 0],                 // d_z²
  [Math.sqrt(3) / 2, -Math.sqrt(3) / 2, 0, 0, 0, 0],  // d_x²-y²
  [0, 0, 0, 1, 0, 0],                            // d_xy
  [0, 0, 0, 0, 1, 0],                            // d_xz
  [0, 0, 0, 0, 0, 1],                            // d_yz
];

const D_CARTESIAN_ORDER: ReadonlyArray<readonly [number, number, number]> = [
  [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1],
];

/** Returns the n_sph × n_cart row-major transform matrix. */
function buildSphericalDTransform(shells: readonly CGShell[]): Float64Array {
  const nCart = shells.length;
  const dGroups: number[] = [];   // start indices of d-shell groups
  for (let i = 0; i + 6 <= nCart; ) {
    const s = shells[i]!;
    if (s.angular[0] + s.angular[1] + s.angular[2] !== 2) { i++; continue; }
    let isGroup = true;
    for (let k = 0; k < 6; k++) {
      const t = shells[i + k]!;
      const want = D_CARTESIAN_ORDER[k]!;
      if (t.angular[0] !== want[0] || t.angular[1] !== want[1] || t.angular[2] !== want[2]) {
        isGroup = false; break;
      }
      if (!sameCenterAndPrims(s, t)) { isGroup = false; break; }
    }
    if (isGroup) {
      dGroups.push(i);
      i += 6;
    } else {
      throw new Error(
        `buildSphericalDTransform: shell ${i} has angular momentum 2 but does not start a canonical (xx,yy,zz,xy,xz,yz) d-shell group at indices [${i}..${i + 5}]`,
      );
    }
  }
  const nDropped = dGroups.length;     // each group: 6 cart → 5 sph (drop 1)
  const nSph = nCart - nDropped;
  const T = new Float64Array(nSph * nCart);
  let row = 0;
  let col = 0;
  let nextDIdx = 0;
  while (col < nCart) {
    if (nextDIdx < dGroups.length && col === dGroups[nextDIdx]) {
      for (let r = 0; r < 5; r++) {
        const coeffs = D_TRANSFORM[r]!;
        for (let c = 0; c < 6; c++) {
          T[(row + r) * nCart + (col + c)] = coeffs[c]!;
        }
      }
      row += 5;
      col += 6;
      nextDIdx++;
    } else {
      T[row * nCart + col] = 1;
      row++;
      col++;
    }
  }
  return T;
}

function sameCenterAndPrims(a: CGShell, b: CGShell): boolean {
  if (a.center[0] !== b.center[0] || a.center[1] !== b.center[1] || a.center[2] !== b.center[2]) return false;
  if (a.alpha.length !== b.alpha.length || a.c.length !== b.c.length) return false;
  for (let i = 0; i < a.alpha.length; i++) {
    if (a.alpha[i] !== b.alpha[i]) return false;
    if (a.c[i] !== b.c[i]) return false;
  }
  return true;
}

/** M_sph = T M_cart T^T  for symmetric M (n_cart × n_cart). */
function transformS_2idx(M: Float64Array, T: Float64Array, nCart: number, nSph: number): Float64Array {
  // tmp[i, ν] = Σ_μ T[i, μ] M[μ, ν]
  const tmp = new Float64Array(nSph * nCart);
  for (let i = 0; i < nSph; i++) {
    for (let nu = 0; nu < nCart; nu++) {
      let s = 0;
      for (let mu = 0; mu < nCart; mu++) {
        const Ti = T[i * nCart + mu]!;
        if (Ti !== 0) s += Ti * M[mu * nCart + nu]!;
      }
      tmp[i * nCart + nu] = s;
    }
  }
  // out[i, j] = Σ_ν tmp[i, ν] T[j, ν]
  const out = new Float64Array(nSph * nSph);
  for (let i = 0; i < nSph; i++) {
    for (let j = 0; j < nSph; j++) {
      let s = 0;
      for (let nu = 0; nu < nCart; nu++) {
        const Tj = T[j * nCart + nu]!;
        if (Tj !== 0) s += tmp[i * nCart + nu]! * Tj;
      }
      out[i * nSph + j] = s;
    }
  }
  return out;
}

/** ERI_sph[p,q,r,s] = Σ_μνλσ T[p,μ] T[q,ν] T[r,λ] T[s,σ] ERI_cart[μ,ν,λ,σ]. */
function transformS_4idx(eri: Float64Array, T: Float64Array, nCart: number, nSph: number): Float64Array {
  // Pass 1: contract index 0 (μ → p)
  const t1 = new Float64Array(nSph * nCart * nCart * nCart);
  for (let p = 0; p < nSph; p++) {
    for (let mu = 0; mu < nCart; mu++) {
      const Tpμ = T[p * nCart + mu]!;
      if (Tpμ === 0) continue;
      for (let nu = 0; nu < nCart; nu++) {
        for (let la = 0; la < nCart; la++) {
          for (let si = 0; si < nCart; si++) {
            const idx = ((p * nCart + nu) * nCart + la) * nCart + si;
            t1[idx] = t1[idx]! + Tpμ * eri[((mu * nCart + nu) * nCart + la) * nCart + si]!;
          }
        }
      }
    }
  }
  // Pass 2: ν → q
  const t2 = new Float64Array(nSph * nSph * nCart * nCart);
  for (let q = 0; q < nSph; q++) {
    for (let nu = 0; nu < nCart; nu++) {
      const Tqν = T[q * nCart + nu]!;
      if (Tqν === 0) continue;
      for (let p = 0; p < nSph; p++) {
        for (let la = 0; la < nCart; la++) {
          for (let si = 0; si < nCart; si++) {
            const idx = ((p * nSph + q) * nCart + la) * nCart + si;
            t2[idx] = t2[idx]! + Tqν * t1[((p * nCart + nu) * nCart + la) * nCart + si]!;
          }
        }
      }
    }
  }
  // Pass 3: λ → r
  const t3 = new Float64Array(nSph * nSph * nSph * nCart);
  for (let r = 0; r < nSph; r++) {
    for (let la = 0; la < nCart; la++) {
      const Trλ = T[r * nCart + la]!;
      if (Trλ === 0) continue;
      for (let p = 0; p < nSph; p++) {
        for (let q = 0; q < nSph; q++) {
          for (let si = 0; si < nCart; si++) {
            const idx = ((p * nSph + q) * nSph + r) * nCart + si;
            t3[idx] = t3[idx]! + Trλ * t2[((p * nSph + q) * nCart + la) * nCart + si]!;
          }
        }
      }
    }
  }
  // Pass 4: σ → s
  const out = new Float64Array(nSph * nSph * nSph * nSph);
  for (let s = 0; s < nSph; s++) {
    for (let si = 0; si < nCart; si++) {
      const Tsσ = T[s * nCart + si]!;
      if (Tsσ === 0) continue;
      for (let p = 0; p < nSph; p++) {
        for (let q = 0; q < nSph; q++) {
          for (let r = 0; r < nSph; r++) {
            const idx = ((p * nSph + q) * nSph + r) * nSph + s;
            out[idx] = out[idx]! + Tsσ * t3[((p * nSph + q) * nSph + r) * nCart + si]!;
          }
        }
      }
    }
  }
  return out;
}

/** AO→OAO 2-index transform: h_OAO[p,q] = Σ X[μ,p] h_AO[μν] X[ν,q]. */
function transform2(M_AO: Float64Array, X: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      let s = 0;
      for (let mu = 0; mu < n; mu++) {
        for (let nu = 0; nu < n; nu++) {
          s += X[mu * n + p]! * M_AO[mu * n + nu]! * X[nu * n + q]!;
        }
      }
      out[p * n + q] = s;
    }
  }
  return out;
}

/** AO→OAO 4-index transform via four sequential 2-index passes. O(n⁵) per pass. */
function transform4(eri_AO: Float64Array, X: Float64Array, n: number): Float64Array {
  let buf1 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let nu = 0; nu < n; nu++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let mu = 0; mu < n; mu++) {
            s += X[mu * n + p]! * eri_AO[((mu * n + nu) * n + la) * n + si]!;
          }
          buf1[((p * n + nu) * n + la) * n + si] = s;
        }
      }
    }
  }
  let buf2 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let nu = 0; nu < n; nu++) {
            s += X[nu * n + q]! * buf1[((p * n + nu) * n + la) * n + si]!;
          }
          buf2[((p * n + q) * n + la) * n + si] = s;
        }
      }
    }
  }
  buf1 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let si = 0; si < n; si++) {
          let s = 0;
          for (let la = 0; la < n; la++) {
            s += X[la * n + r]! * buf2[((p * n + q) * n + la) * n + si]!;
          }
          buf1[((p * n + q) * n + r) * n + si] = s;
        }
      }
    }
  }
  buf2 = new Float64Array(n * n * n * n);
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      for (let r = 0; r < n; r++) {
        for (let s = 0; s < n; s++) {
          let acc = 0;
          for (let si = 0; si < n; si++) {
            acc += X[si * n + s]! * buf1[((p * n + q) * n + r) * n + si]!;
          }
          buf2[((p * n + q) * n + r) * n + s] = acc;
        }
      }
    }
  }
  return buf2;
}
