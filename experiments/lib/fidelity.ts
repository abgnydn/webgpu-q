// ─────────────────────────────────────────────────────────────
// fidelity.ts — rigorous quantum state comparison.
//
// RESEARCH.md pass bar: F = |⟨ψ_ref | ψ_test⟩|² ≥ 1 − 1e-5.
// max|Δp| alone is not sufficient — two states can have identical
// probability distributions but differ in phase, which destroys any
// downstream controlled gate.
//
// All inputs are interleaved [re, im, re, im, ...] amplitude arrays,
// Float32Array (GPU) or Float64Array (CPU reference). Compare across
// precisions by widening to Float64 internally.
// ─────────────────────────────────────────────────────────────

export interface StateMetrics {
  readonly fidelity: number;      // |⟨ψ_ref|ψ_test⟩|²
  readonly infidelity: number;    // 1 − F
  readonly maxDp: number;         // max_i |p_ref[i] − p_test[i]|
  readonly l1: number;            // Σ|p_ref − p_test| = 2 × TVD
  readonly tvd: number;           // total variation distance
  readonly l2: number;            // sqrt(Σ(p_ref − p_test)²)
  readonly normRef: number;       // Σ|ψ_ref|² (should be ≈1)
  readonly normTest: number;      // Σ|ψ_test|² (should be ≈1)
}

type AmpLike = Float32Array | Float64Array | ArrayLike<number>;

function lenAmps(psi: AmpLike): number {
  return (psi as { length: number }).length;
}

export function stateMetrics(refPsi: AmpLike, testPsi: AmpLike): StateMetrics {
  const nRef = lenAmps(refPsi);
  const nTest = lenAmps(testPsi);
  if (nRef !== nTest) {
    throw new Error(`state size mismatch: ref=${nRef} test=${nTest}`);
  }
  if (nRef % 2 !== 0) throw new Error("interleaved [re, im] must have even length");
  const nStates = nRef / 2;

  // ⟨ψ_ref | ψ_test⟩ = Σ conj(ψ_ref) · ψ_test in f64
  let innerRe = 0, innerIm = 0;
  let normRef = 0, normTest = 0;
  // Use Kahan summation for large state vectors — f32 accumulation loses
  // digits past ~2^24 amplitudes; f64 Kahan is safe to 2^50.
  let kcRe = 0, kcIm = 0, kcNR = 0, kcNT = 0;
  for (let i = 0; i < nStates; i++) {
    const ar = refPsi[i * 2] ?? 0;
    const ai = refPsi[i * 2 + 1] ?? 0;
    const br = testPsi[i * 2] ?? 0;
    const bi = testPsi[i * 2 + 1] ?? 0;
    // ⟨a|b⟩ = conj(a)·b = (ar - i·ai)(br + i·bi)
    const reTerm = ar * br + ai * bi;
    const imTerm = ar * bi - ai * br;
    ({ sum: innerRe, c: kcRe } = kahan(innerRe, kcRe, reTerm));
    ({ sum: innerIm, c: kcIm } = kahan(innerIm, kcIm, imTerm));
    ({ sum: normRef, c: kcNR } = kahan(normRef, kcNR, ar * ar + ai * ai));
    ({ sum: normTest, c: kcNT } = kahan(normTest, kcNT, br * br + bi * bi));
  }
  const fidelity = innerRe * innerRe + innerIm * innerIm;

  // Probability-space metrics.
  let maxDp = 0, l1 = 0, l2sq = 0;
  for (let i = 0; i < nStates; i++) {
    const ar = refPsi[i * 2] ?? 0;
    const ai = refPsi[i * 2 + 1] ?? 0;
    const br = testPsi[i * 2] ?? 0;
    const bi = testPsi[i * 2 + 1] ?? 0;
    const pRef = ar * ar + ai * ai;
    const pTest = br * br + bi * bi;
    const d = Math.abs(pRef - pTest);
    if (d > maxDp) maxDp = d;
    l1 += d;
    l2sq += d * d;
  }

  return {
    fidelity,
    infidelity: 1 - fidelity,
    maxDp,
    l1,
    tvd: l1 / 2,
    l2: Math.sqrt(l2sq),
    normRef,
    normTest,
  };
}

function kahan(sum: number, c: number, x: number): { sum: number; c: number } {
  const y = x - c;
  const t = sum + y;
  return { sum: t, c: (t - sum) - y };
}

/** One-line pretty print for logs / tables. */
export function fmtMetrics(m: StateMetrics): string {
  return `F=${m.fidelity.toFixed(9)} 1−F=${m.infidelity.toExponential(2)} max|Δp|=${m.maxDp.toExponential(2)} TVD=${m.tvd.toExponential(2)} ||Δp||₂=${m.l2.toExponential(2)}`;
}

/** Pass bar from RESEARCH.md: F ≥ 1 − 1e-5 for f32-amplitude GPU paths. */
export const FIDELITY_PASS_BAR = 1 - 1e-5;
export function passed(m: StateMetrics, bar = FIDELITY_PASS_BAR): boolean {
  return m.fidelity >= bar && Math.abs(m.normTest - 1) < 1e-4;
}
