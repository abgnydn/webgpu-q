// ─────────────────────────────────────────────────────────────
// integrals.ts — Gaussian-primitive molecular integrals for H₂.
//
// H₂ in STO-3G has only s-functions on both atoms, so every
// integral here is over normalized 1s-Gaussians — no angular
// momentum bookkeeping, just exponents, centers, and the Boys
// function F_0(t) = 0.5·√(π/t)·erf(√t) for the Coulomb cores.
//
// Conventions:
//   • Atomic units throughout (Hartree, Bohr).
//   • All integrals are over CONTRACTED Gaussian shells; per-
//     primitive sums get rolled in by `contracted{S,T,V,ERI}`.
//   • Chemist notation (μν|λσ) = ∫∫ φ_μ(1)φ_ν(1) (1/r12) φ_λ(2)φ_σ(2).
//
// Reference: Helgaker, Jørgensen, Olsen — Molecular Electronic
// Structure Theory, Ch. 9 (Obara–Saika recurrence reduces to
// these closed forms for s-shells only).
// ─────────────────────────────────────────────────────────────

/** STO-3G 1s contraction for hydrogen.
 *  α: Gaussian exponents;  c: contraction coefficients (un-normalized
 *  on the *contracted* level — per-primitive normalization is rolled
 *  into the d_i below). */
export const STO3G_H_1S = {
  alpha: [3.42525091, 0.62391373, 0.16885540] as const,
  // Pople-style "d" coefficients (Hehre, Stewart, Pople 1969 Table II).
  c: [0.15432897, 0.53532814, 0.44463454] as const,
};

/** STO-3G 1s contraction for lithium. Tighter exponents than H 1s
 *  (Z = 3 vs Z = 1 → orbital pulled in toward the nucleus). */
export const STO3G_LI_1S = {
  alpha: [16.1195750, 2.9362007, 0.7946505] as const,
  c: [0.15432897, 0.53532814, 0.44463454] as const,
};

/** STO-3G 1s contraction for beryllium (Z = 4). Tighter exponents than
 *  Li 1s (Z = 3); the L-shell coefficients below carry the 2s character. */
export const STO3G_BE_1S = {
  alpha: [30.1678710, 5.4951153, 1.4871927] as const,
  c: [0.15432897, 0.53532814, 0.44463454] as const,
};

/** STO-3G 2s contraction for beryllium — the *s component* of the L-shell.
 *  Same exponent set as the 2p L-shell (added in Phase C v3); coefficients
 *  are Pople 1969 Table III. Negative leading coefficient produces the 2s
 *  radial node and enforces orthogonality with 1s post-Löwdin. */
export const STO3G_BE_2S = {
  alpha: [1.3148331, 0.3055389, 0.0993707] as const,
  c: [-0.09996723, 0.39951283, 0.70011547] as const,
};

/** STO-3G 2p contraction for beryllium — the *p component* of the L-shell.
 *  Same exponent set as STO3G_BE_2S; the only difference is the
 *  contraction coefficients (all positive for p, since the 2p radial
 *  has no node). One Cartesian-Gaussian shell per Cartesian direction
 *  (p_x, p_y, p_z) — pair this with `makeCGShell(STO3G_BE_2P, center,
 *  [1, 0, 0])` etc. from `integrals-cg.ts` to build them. */
export const STO3G_BE_2P = {
  alpha: [1.3148331, 0.3055389, 0.0993707] as const,
  c: [0.15591627, 0.60768372, 0.39195739] as const,
};

// ── Pople 1969 STO-3G for second-row atoms (B–Ne) ─────────────
// All second-row atoms share the same s-shell d-coefficients
// (radial form is universal in the STO-3G parametrization);
// only the exponent sets differ from atom to atom.
//
// L-shell (2sp): same exponents for s and p; same s-d-coeffs and
// p-d-coeffs as Be (Pople 1969 Table III).

const STO3G_L_2S_C = [-0.09996723, 0.39951283, 0.70011547] as const;
const STO3G_L_2P_C = [ 0.15591627, 0.60768372, 0.39195739] as const;
const STO3G_S_C    = [ 0.15432897, 0.53532814, 0.44463454] as const;

/** STO-3G 1s for carbon (Z = 6). Pople 1969. */
export const STO3G_C_1S = {
  alpha: [71.6168370, 13.0450960, 3.5305122] as const,
  c: STO3G_S_C,
};
export const STO3G_C_2S = {
  alpha: [2.9412494, 0.6834831, 0.2222899] as const,
  c: STO3G_L_2S_C,
};
export const STO3G_C_2P = {
  alpha: [2.9412494, 0.6834831, 0.2222899] as const,
  c: STO3G_L_2P_C,
};

/** STO-3G 1s for nitrogen (Z = 7). Pople 1969. */
export const STO3G_N_1S = {
  alpha: [99.1061690, 18.0523120, 4.8856602] as const,
  c: STO3G_S_C,
};
export const STO3G_N_2S = {
  alpha: [3.7804559, 0.8784966, 0.2857144] as const,
  c: STO3G_L_2S_C,
};
export const STO3G_N_2P = {
  alpha: [3.7804559, 0.8784966, 0.2857144] as const,
  c: STO3G_L_2P_C,
};

/** STO-3G 1s for oxygen (Z = 8). Pople 1969. */
export const STO3G_O_1S = {
  alpha: [130.7093200, 23.8088610, 6.4436083] as const,
  c: STO3G_S_C,
};
export const STO3G_O_2S = {
  alpha: [5.0331513, 1.1695961, 0.3803890] as const,
  c: STO3G_L_2S_C,
};
export const STO3G_O_2P = {
  alpha: [5.0331513, 1.1695961, 0.3803890] as const,
  c: STO3G_L_2P_C,
};

/** STO-3G 2s contraction for lithium — the *s component* of the L-shell.
 *  Coefficients can be negative to produce the 2s radial node and to
 *  enforce orthogonality with 1s after Löwdin orthogonalization.
 *  STO-3G also defines a 2p L-shell on Li with the same exponents but
 *  different (positive) coefficients; that is omitted from this v0
 *  s-only basis (Phase C scope). */
export const STO3G_LI_2S = {
  alpha: [0.6362897, 0.1478601, 0.0480887] as const,
  c: [-0.09996723, 0.39951283, 0.70011547] as const,
};

// ── Generic shell type + multi-shell integrals ───────────────
//
// A Shell is a contracted s-Gaussian: a center plus a list of
// primitive exponents and coefficients. The existing H₂ pipeline
// hard-codes H 1s; LiH (Phase C) needs to mix Li 1s, Li 2s, and
// H 1s, each with their own (alpha, c) arrays. The shell-based
// API below is the multi-element generalization.
//
// Per-primitive normalization N(α) = (2α/π)^(3/4) is applied
// inside the contractions so callers pass "raw" Pople-style
// d-coefficients (matching the constants above).

export interface Shell {
  /** Atomic center (Bohr). */
  readonly center: readonly [number, number, number];
  /** Primitive Gaussian exponents. */
  readonly alpha: readonly number[];
  /** Contraction coefficients (Pople d-style; primitive normalization rolled in here). */
  readonly c: readonly number[];
  /** Optional label, e.g. "Li:1s", "H:1s". For debugging only. */
  readonly label?: string;
}

/** Convenience: build a Shell from a basis-set entry + atom center. */
export function makeShell(
  basis: { readonly alpha: readonly number[]; readonly c: readonly number[] },
  center: readonly [number, number, number],
  label?: string,
): Shell {
  return { center, alpha: basis.alpha, c: basis.c, label };
}

/** ⟨A|B⟩ overlap between two contracted s-shells. */
export function S_shells(A: Shell, B: Shell): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ci = A.c[i]! * normS(ai);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const cj = B.c[j]! * normS(bj);
      s += ci * cj * primS(ai, A.center, bj, B.center);
    }
  }
  return s;
}

/** Kinetic ⟨A| -∇²/2 |B⟩ between two contracted s-shells. */
export function T_shells(A: Shell, B: Shell): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ci = A.c[i]! * normS(ai);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const cj = B.c[j]! * normS(bj);
      s += ci * cj * primT(ai, A.center, bj, B.center);
    }
  }
  return s;
}

/** Nuclear attraction ⟨A| -Z_C/|r-C| |B⟩ between two contracted s-shells. */
export function V_shells(
  A: Shell, B: Shell,
  Zc: number, C: readonly [number, number, number],
): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ci = A.c[i]! * normS(ai);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const cj = B.c[j]! * normS(bj);
      s += ci * cj * primV(ai, A.center, bj, B.center, Zc, C);
    }
  }
  return s;
}

/** Two-electron repulsion (chemist notation): (A B | C D). */
export function ERI_shells(A: Shell, B: Shell, Cs: Shell, Ds: Shell): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ni = normS(ai);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const nj = normS(bj);
      for (let k = 0; k < Cs.alpha.length; k++) {
        const ck = Cs.alpha[k]!;
        const nk = normS(ck);
        for (let l = 0; l < Ds.alpha.length; l++) {
          const dl = Ds.alpha[l]!;
          const nl = normS(dl);
          const coeff = A.c[i]! * B.c[j]! * Cs.c[k]! * Ds.c[l]! * ni * nj * nk * nl;
          s += coeff * primERI(ai, A.center, bj, B.center, ck, Cs.center, dl, Ds.center);
        }
      }
    }
  }
  return s;
}

const SQRT_PI = Math.sqrt(Math.PI);

/** Per-primitive normalization for an s-Gaussian: N(α) = (2α/π)^{3/4}. */
function normS(alpha: number): number {
  return Math.pow(2 * alpha / Math.PI, 0.75);
}

/** Boys function F_0(t) = 0.5·√(π/t)·erf(√t). At t=0, F_0=1. */
function boys0(t: number): number {
  if (t < 1e-12) return 1 - t / 3 + (t * t) / 10;  // small-t Taylor
  const s = Math.sqrt(t);
  return 0.5 * SQRT_PI / s * erf(s);
}

/** Abramowitz & Stegun 7.1.26 erf approximation — max error ~1.5e-7. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

// ── Primitive integrals (s-Gaussians) ────────────────────────
// All take centers (Ax, Ay, Az) and (Bx, By, Bz) and exponents
// α, β. We only ever need 1D centers in this codebase (atoms on
// the z-axis), but keep the interface 3D for clarity.

function dist2(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

function gprodCenter(
  a: readonly [number, number, number], alpha: number,
  b: readonly [number, number, number], beta: number,
): [number, number, number] {
  const p = alpha + beta;
  return [
    (alpha * a[0] + beta * b[0]) / p,
    (alpha * a[1] + beta * b[1]) / p,
    (alpha * a[2] + beta * b[2]) / p,
  ];
}

/** Primitive overlap ⟨φ_α(A)|φ_β(B)⟩ between unnormalized s-Gaussians. */
function primS(alpha: number, A: readonly [number, number, number], beta: number, B: readonly [number, number, number]): number {
  const p = alpha + beta;
  const mu = alpha * beta / p;
  const r2 = dist2(A, B);
  return Math.pow(Math.PI / p, 1.5) * Math.exp(-mu * r2);
}

/** Primitive kinetic ⟨φ_α(A)| -∇²/2 |φ_β(B)⟩, s-shell only. */
function primT(alpha: number, A: readonly [number, number, number], beta: number, B: readonly [number, number, number]): number {
  const p = alpha + beta;
  const mu = alpha * beta / p;
  const r2 = dist2(A, B);
  const overlap = primS(alpha, A, beta, B);
  return mu * (3 - 2 * mu * r2) * overlap;
}

/** Primitive nuclear attraction ⟨φ_α(A)| -Z_C/|r-C| |φ_β(B)⟩. */
function primV(
  alpha: number, A: readonly [number, number, number],
  beta: number, B: readonly [number, number, number],
  Zc: number, C: readonly [number, number, number],
): number {
  const p = alpha + beta;
  const P = gprodCenter(A, alpha, B, beta);
  const r2 = dist2(A, B);
  const PC2 = dist2(P, C);
  const mu = alpha * beta / p;
  const pre = -2 * Math.PI * Zc / p * Math.exp(-mu * r2);
  return pre * boys0(p * PC2);
}

/** Primitive two-electron repulsion (αA βB | γC δD), s-shells only. */
function primERI(
  a1: number, A: readonly [number, number, number],
  b1: number, B: readonly [number, number, number],
  g2: number, C: readonly [number, number, number],
  d2: number, D: readonly [number, number, number],
): number {
  const p = a1 + b1;
  const q = g2 + d2;
  const P = gprodCenter(A, a1, B, b1);
  const Q = gprodCenter(C, g2, D, d2);
  const muAB = a1 * b1 / p;
  const muCD = g2 * d2 / q;
  const RAB2 = dist2(A, B);
  const RCD2 = dist2(C, D);
  const RPQ2 = dist2(P, Q);
  const t = p * q / (p + q) * RPQ2;
  const pre = 2 * Math.pow(Math.PI, 2.5) /
              (p * q * Math.sqrt(p + q)) *
              Math.exp(-muAB * RAB2 - muCD * RCD2);
  return pre * boys0(t);
}

// ── Contracted-shell integrals ───────────────────────────────
// Each H 1s is a 3-primitive contraction. Below: sum-over-primitives
// with primitive normalization rolled in.

type Center = readonly [number, number, number];

const ALPHA = STO3G_H_1S.alpha;
const C = STO3G_H_1S.c;
const NPRIM = ALPHA.length;

/** ⟨1s_A | 1s_B⟩ where A, B are H atom positions (Bohr). */
export function S_AB(A: Center, B: Center): number {
  let s = 0;
  for (let i = 0; i < NPRIM; i++) {
    for (let j = 0; j < NPRIM; j++) {
      s += C[i]! * C[j]! * normS(ALPHA[i]!) * normS(ALPHA[j]!) * primS(ALPHA[i]!, A, ALPHA[j]!, B);
    }
  }
  return s;
}

/** Kinetic ⟨1s_A | -∇²/2 | 1s_B⟩. */
export function T_AB(A: Center, B: Center): number {
  let s = 0;
  for (let i = 0; i < NPRIM; i++) {
    for (let j = 0; j < NPRIM; j++) {
      s += C[i]! * C[j]! * normS(ALPHA[i]!) * normS(ALPHA[j]!) * primT(ALPHA[i]!, A, ALPHA[j]!, B);
    }
  }
  return s;
}

/** Nuclear attraction ⟨1s_A | -Z_C/|r-C| | 1s_B⟩. */
export function V_AB(A: Center, B: Center, Zc: number, Cn: Center): number {
  let s = 0;
  for (let i = 0; i < NPRIM; i++) {
    for (let j = 0; j < NPRIM; j++) {
      s += C[i]! * C[j]! * normS(ALPHA[i]!) * normS(ALPHA[j]!) * primV(ALPHA[i]!, A, ALPHA[j]!, B, Zc, Cn);
    }
  }
  return s;
}

/** Two-electron repulsion in chemist notation: (1s_A 1s_B | 1s_C 1s_D). */
export function ERI(A: Center, B: Center, Ca: Center, Da: Center): number {
  let s = 0;
  for (let i = 0; i < NPRIM; i++) {
    const ni = normS(ALPHA[i]!);
    for (let j = 0; j < NPRIM; j++) {
      const nj = normS(ALPHA[j]!);
      for (let k = 0; k < NPRIM; k++) {
        const nk = normS(ALPHA[k]!);
        for (let l = 0; l < NPRIM; l++) {
          const nl = normS(ALPHA[l]!);
          const c = C[i]! * C[j]! * C[k]! * C[l]! * ni * nj * nk * nl;
          s += c * primERI(ALPHA[i]!, A, ALPHA[j]!, B, ALPHA[k]!, Ca, ALPHA[l]!, Da);
        }
      }
    }
  }
  return s;
}
