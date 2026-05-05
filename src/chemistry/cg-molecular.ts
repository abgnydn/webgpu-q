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

/**
 * Compute the full atomic-orbital integral set + Löwdin orthogonalize.
 * This is the heavy step — for n shells the ERI tensor is O(n^4) and
 * each ERI is O(n_prim^4) primitives. For STO-3G (3 primitives/shell)
 * and n=7 (BeH₂ full) that's ~100k primitive ERIs. Few seconds.
 */
export function computeMolecularIntegrals(
  shells: readonly CGShell[],
  nuclei: readonly Nucleus[],
): MolecularIntegrals {
  const n = shells.length;
  const S_AO = new Float64Array(n * n);
  const h_AO = new Float64Array(n * n);
  const eri_AO = new Float64Array(n * n * n * n);

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
  // Use 8-fold symmetry to compute only unique ones.
  // Index encoding: (μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν).
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu; nu < n; nu++) {
      for (let la = 0; la < n; la++) {
        for (let si = la; si < n; si++) {
          // Skip double-count via the (μν) ↔ (λσ) swap: only do
          // (μν, λσ) where the encoded pair (μ·n+ν) ≤ (λ·n+σ).
          if (mu * n + nu > la * n + si) continue;
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
