// ─────────────────────────────────────────────────────────────
// order-sweep.ts — generate the canonical phase-transition curve
// for TFIM by sweeping the transverse field h, finding each
// ground state via short imaginary-time evolution, and reading
// off the order parameters.
//
// Returns: arrays of (h, m_z, m_x, max_entropy) per sample. The
// viz renders these as a small SVG line chart in the controls
// column. Total cost ~30 imag-time runs at modest χ — well under
// 5 s on N = 8 chains.
// ─────────────────────────────────────────────────────────────

import { MPS } from "../mps.js";
import { tfim, asComplex4 } from "../manybody/hamiltonian.js";
import { expmReal } from "../manybody/expm.js";
import { siteExpectations, zMagnetization, xyMagnetization } from "../manybody/observables.js";

export interface SweepPoint {
  h: number;
  mZ: number;
  mX: number;
  maxEntropy: number;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function entropyFromSigma(sigma: Float64Array): number {
  let norm2 = 0;
  for (let i = 0; i < sigma.length; i++) norm2 += sigma[i]! * sigma[i]!;
  if (norm2 <= 0) return 0;
  let S = 0;
  for (let i = 0; i < sigma.length; i++) {
    const p = (sigma[i]! * sigma[i]!) / norm2;
    if (p > 1e-18) S -= p * Math.log2(p);
  }
  return S;
}

/** Sweep h ∈ [hMin, hMax] for TFIM, returning order-parameter datapoints. */
export function sweepTFIMOrderParameter(opts: {
  N: number;
  hMin?: number;
  hMax?: number;
  steps?: number;        // h-grid resolution
  imagTimeSteps?: number;
  tau?: number;
  chiMax?: number;
  zPinning?: number;
}): SweepPoint[] {
  const N = opts.N;
  const hMin = opts.hMin ?? 0;
  const hMax = opts.hMax ?? 2;
  const steps = opts.steps ?? 25;
  const imagTimeSteps = opts.imagTimeSteps ?? 60;
  const tau = opts.tau ?? 0.05;
  const chiMax = opts.chiMax ?? 16;
  const zPin = opts.zPinning ?? 0.02;

  const points: SweepPoint[] = [];

  for (let i = 0; i < steps; i++) {
    const h = hMin + (hMax - hMin) * (i / Math.max(1, steps - 1));
    const ham = tfim(N, 1, h);

    const mps = new MPS({ nQubits: N, chiMax });
    const rng = mulberry32(0xCAFE ^ Math.round(h * 1e4));
    for (let q = 0; q < N; q++) {
      const theta = (rng() - 0.5) * 0.4;
      const c = Math.cos(theta / 2);
      const s = Math.sin(theta / 2);
      mps.apply([[c, 0], [-s, 0], [s, 0], [c, 0]], q);
    }

    const gates = ham.bonds.map((b, q) => {
      let hb = b.h;
      if (q === 0 && zPin !== 0) {
        const copy = new Float64Array(hb);
        copy[0] = (copy[0] ?? 0) - zPin;
        copy[5] = (copy[5] ?? 0) - zPin;
        copy[10] = (copy[10] ?? 0) + zPin;
        copy[15] = (copy[15] ?? 0) + zPin;
        hb = copy;
      }
      return asComplex4(expmReal(hb, 4, -tau));
    });

    for (let step = 0; step < imagTimeSteps; step++) {
      for (let q = 0; q < N - 1; q += 2) mps.applyTwoSite(gates[q]!, q);
      for (let q = 1; q < N - 1; q += 2) mps.applyTwoSite(gates[q]!, q);
    }

    const exps = siteExpectations(mps.statevector(), N);
    let maxS = 0;
    for (let cut = 0; cut < N - 1; cut++) {
      const sigma = mps.schmidtSpectrum(cut);
      maxS = Math.max(maxS, entropyFromSigma(sigma));
    }
    points.push({
      h,
      mZ: zMagnetization(exps),
      mX: xyMagnetization(exps),
      maxEntropy: maxS,
    });
  }
  return points;
}
