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
