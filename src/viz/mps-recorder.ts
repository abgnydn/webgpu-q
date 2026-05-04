// ─────────────────────────────────────────────────────────────
// mps-recorder.ts — produce a sequence of MPSSnapshot frames
// from various sources (a brick-wall circuit, an imaginary-time
// evolution toward a Hamiltonian's ground state, …) so the viz
// time slider has a uniform thing to scrub through.
// ─────────────────────────────────────────────────────────────

import { MPS } from "../mps.js";
import { H as HGate, X, Rz, type Matrix2 } from "../gates.js";
import { tfim, heisenberg, asComplex4, type Hamiltonian1D } from "../manybody/hamiltonian.js";
import { expmReal } from "../manybody/expm.js";
import { expmComplex } from "../manybody/real-time.js";
import { siteExpectations, type SiteExpectations } from "../manybody/observables.js";

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

export interface MPSSnapshot {
  readonly bondDims: readonly number[];
  readonly entropies: readonly number[];
  readonly gateLabel: string;
  /** Optional per-site Pauli expectations. Populated by the recorders that
   *  enable `withExpectations: true` (cheap for N ≤ 20). */
  readonly expectations?: SiteExpectations;
}

export interface MPSRecording {
  readonly nQubits: number;
  readonly depth: number;
  readonly snapshots: readonly MPSSnapshot[];
  readonly maxBond: number;
  readonly maxEntropy: number;
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

function snapshotOf(mps: MPS, label: string, withExpectations = false): MPSSnapshot {
  const dims = mps.bondDimensions();
  const entropies: number[] = [];
  for (let cut = 0; cut < mps.nQubits - 1; cut++) {
    const sigma = mps.schmidtSpectrum(cut);
    entropies.push(entropyFromSigma(sigma));
  }
  if (withExpectations && mps.nQubits <= 20) {
    const exps = siteExpectations(mps.statevector(), mps.nQubits);
    return { bondDims: dims, entropies, gateLabel: label, expectations: exps };
  }
  return { bondDims: dims, entropies, gateLabel: label };
}

function summarize(snaps: readonly MPSSnapshot[]): { maxBond: number; maxEntropy: number } {
  let maxBond = 1;
  let maxS = 0;
  for (const s of snaps) {
    for (const d of s.bondDims) if (d > maxBond) maxBond = d;
    for (const e of s.entropies) if (e > maxS) maxS = e;
  }
  return { maxBond, maxEntropy: maxS };
}

/** Brick-wall random circuit: snapshot after every gate (existing behaviour). */
export function recordBrickwall(opts: {
  nQubits: number;
  depth: number;
  chiMax?: number;
  seed?: number;
  withExpectations?: boolean;
}): MPSRecording {
  const N = opts.nQubits;
  const D = opts.depth;
  const chiMax = opts.chiMax ?? 64;
  const seed = opts.seed ?? 0xBA1CA70;
  const withExp = opts.withExpectations ?? true;

  const mps = new MPS({ nQubits: N, chiMax });
  const rng = mulberry32(seed);
  const snaps: MPSSnapshot[] = [snapshotOf(mps, "|0…0⟩", withExp)];

  for (let l = 0; l < D; l++) {
    for (let q = 0; q < N; q++) {
      const pick = Math.floor(rng() * 3);
      let g: Matrix2;
      let lab: string;
      if (pick === 0) { g = HGate; lab = `H q${q}`; }
      else if (pick === 1) {
        const th = rng() * 2 * Math.PI;
        g = Rz(th); lab = `Rz(${th.toFixed(2)}) q${q}`;
      } else { g = X; lab = `X q${q}`; }
      mps.apply(g, q);
      snaps.push(snapshotOf(mps, lab, withExp));
    }
    const offset = l & 1;
    for (let q = offset; q + 1 < N; q += 2) {
      mps.cnot(q, q + 1);
      snaps.push(snapshotOf(mps, `CNOT ${q}-${q + 1}`, withExp));
    }
  }

  return { nQubits: N, depth: D, snapshots: snaps, ...summarize(snaps) };
}

/**
 * Imaginary-time evolution toward the ground state of a 1D Hamiltonian,
 * snapshotting after every Trotter step. The "depth" of the recording
 * is the total imaginary time β = Σ τ_phase × steps_phase.
 *
 * Used to drive the phase-transition demo: as you slide the TFIM h
 * field across h = J, the converged max-entropy peaks. Watch the
 * bond network *grow* its entanglement during the relaxation.
 */
export function recordImagTime(opts: {
  hamiltonian: Hamiltonian1D;
  chiMax?: number;
  seed?: number;
  /** Trotter steps to record. Each at τ. Total β = steps × τ. */
  steps?: number;
  tau?: number;
  withExpectations?: boolean;
  /** Tiny Z-pinning field on site 0 to break Z₂ symmetry so the order
   *  parameter ⟨σ_z⟩ becomes nonzero in the ordered phase. Default 0.01. */
  zPinning?: number;
}): MPSRecording {
  const chiMax = opts.chiMax ?? 32;
  const seed = opts.seed ?? 0xCAFE;
  const steps = opts.steps ?? 100;
  const tau = opts.tau ?? 0.05;
  const withExp = opts.withExpectations ?? true;
  const zPin = opts.zPinning ?? 0.01;
  const N = opts.hamiltonian.nQubits;

  const mps = new MPS({ nQubits: N, chiMax });
  const rng = mulberry32(seed);
  // Small random Ry on every site so we don't sit on a fixed point.
  for (let q = 0; q < N; q++) {
    const theta = (rng() - 0.5) * 0.4;
    const c = Math.cos(theta / 2);
    const s = Math.sin(theta / 2);
    mps.apply([[c, 0], [-s, 0], [s, 0], [c, 0]], q);
  }

  const snaps: MPSSnapshot[] = [snapshotOf(mps, "init", withExp)];

  // Pre-build the per-bond imaginary-time gate exp(-τ h_q). For the leftmost
  // bond, fold in a small Z-pinning -ε σ_z_0 so the Z₂-symmetric model can
  // converge to a symmetry-broken phase visible as ⟨σ_z⟩ ≠ 0 in the viz.
  const gates = opts.hamiltonian.bonds.map((b, q) => {
    let h = b.h;
    if (q === 0 && zPin !== 0) {
      // Add -ε · (Z ⊗ I) on the first bond. Diagonal pattern [-,-,+,+] in
      // the basis (s_q · 2 + s_{q+1}) since site q (= MSB) is the Z target.
      const copy = new Float64Array(h);
      copy[0] = (copy[0] ?? 0) + -zPin;
      copy[5] = (copy[5] ?? 0) + -zPin;
      copy[10] = (copy[10] ?? 0) + zPin;
      copy[15] = (copy[15] ?? 0) + zPin;
      h = copy;
    }
    return asComplex4(expmReal(h, 4, -tau));
  });

  for (let step = 0; step < steps; step++) {
    for (let q = 0; q < N - 1; q += 2) mps.applyTwoSite(gates[q]!, q);
    for (let q = 1; q < N - 1; q += 2) mps.applyTwoSite(gates[q]!, q);
    snaps.push(snapshotOf(mps, `β = ${((step + 1) * tau).toFixed(2)}`, withExp));
  }

  return { nQubits: N, depth: steps, snapshots: snaps, ...summarize(snaps) };
}

/** Convenience: TFIM ground-state recording at given h (J held at 1). */
export function recordTFIMGroundState(N: number, hField: number, opts: {
  chiMax?: number;
  steps?: number;
  tau?: number;
  seed?: number;
} = {}): MPSRecording {
  return recordImagTime({
    hamiltonian: tfim(N, 1, hField),
    ...opts,
  });
}

/** Convenience: Heisenberg ground-state recording at J=1. */
export function recordHeisenbergGroundState(N: number, opts: {
  chiMax?: number;
  steps?: number;
  tau?: number;
  seed?: number;
} = {}): MPSRecording {
  return recordImagTime({
    hamiltonian: heisenberg(N, 1),
    ...opts,
  });
}

/**
 * Real-time evolution exp(-i τ H) starting from a Néel-like product state
 * |↑↓↑↓…⟩. Tracks per-site ⟨σ_z⟩ and bond entropy after every Trotter
 * step so the viz can render the entanglement light cone.
 *
 * Default Hamiltonian: TFIM at h = h0 (caller picks). The classic quench
 * demo: prepare |↑↑…⟩ (Z-eigenstate), evolve under H_TFIM with h ≠ 0
 * (so Z is no longer conserved), watch entanglement spread out.
 */
export function recordQuench(opts: {
  nQubits: number;
  hamiltonian: Hamiltonian1D;
  /** "neel" = |↑↓↑↓…⟩, "all-up" = |↑↑↑…⟩. Default "all-up". */
  initialState?: "neel" | "all-up";
  chiMax?: number;
  steps?: number;
  tau?: number;
}): MPSRecording {
  const N = opts.nQubits;
  if (N !== opts.hamiltonian.nQubits) {
    throw new Error(`recordQuench: nQubits mismatch ${N} vs ${opts.hamiltonian.nQubits}`);
  }
  const chiMax = opts.chiMax ?? 32;
  const steps = opts.steps ?? 100;
  const tau = opts.tau ?? 0.05;
  const init = opts.initialState ?? "all-up";

  const mps = new MPS({ nQubits: N, chiMax });
  // Prepare initial product state. |↑⟩ is |0⟩ so we already start there;
  // for Néel we flip every other qubit.
  if (init === "neel") {
    for (let q = 1; q < N; q += 2) mps.apply(X, q);
  }

  const snaps: MPSSnapshot[] = [snapshotOf(mps, `t=0  (${init})`, true)];

  // Real-time gates exp(-i τ h_q).
  const gates = opts.hamiltonian.bonds.map((b) => expmComplex(b.h, 4, tau));

  for (let step = 0; step < steps; step++) {
    for (let q = 0; q < N - 1; q += 2) mps.applyTwoSite(gates[q]!, q);
    for (let q = 1; q < N - 1; q += 2) mps.applyTwoSite(gates[q]!, q);
    snaps.push(snapshotOf(mps, `t = ${((step + 1) * tau).toFixed(2)}`, true));
  }

  return { nQubits: N, depth: steps, snapshots: snaps, ...summarize(snaps) };
}

/** Convenience: TFIM quench from |↑↑…⟩ at given h. */
export function recordTFIMQuench(N: number, hField: number, opts: {
  chiMax?: number;
  steps?: number;
  tau?: number;
  initialState?: "neel" | "all-up";
} = {}): MPSRecording {
  return recordQuench({
    nQubits: N,
    hamiltonian: tfim(N, 1, hField),
    ...opts,
  });
}
