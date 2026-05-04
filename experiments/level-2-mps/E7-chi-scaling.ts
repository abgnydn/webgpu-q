// ─────────────────────────────────────────────────────────────
// E7 — χ scaling vs depth (entanglement-aware) for Haar-random
// 1D brick-wall circuits at N = 16.
//
// Hypothesis (protocol.md): for Haar-random nearest-neighbor
// brick-wall circuits, the χ required to hit F ≥ 0.999 satisfies
//   log₂(χ_required) ≈ depth
// because the mid-cut bipartite entropy S grows roughly linearly
// with depth in 1D until saturation (S ≤ N/2 bits at N = 16).
//
// Method: for each depth ∈ {1, 2, 3, 4, 5, 6, 8}, run `seedsPerDepth`
// seeded Haar-random circuits (single-qubit Haar U(2) + adjacent
// CNOT bricks — non-adjacent gates need SWAP ladders, not yet
// implemented in MPS v1). For each (depth, seed), sweep
// χ ∈ {2, 4, 8, 16, 32, 64, 128} and record the smallest χ hitting
// F ≥ 0.999 against CpuCircuit ground truth.
//
// Also record mid-cut bipartite entropy S measured from the
// Schmidt spectrum of the cheapest-χ MPS. When the cheapest χ is
// small, S is slightly underestimated (tail truncation) — but
// F ≥ 0.999 caps that error well below 0.01 bits, so the fit
// across depth remains valid.
//
// Pass bar: fit slope of log₂(χ_required) vs depth within ±0.2
// of 1.0, AND < 10% of cells fail to reach F ≥ 0.999 at χ = 128.
// ─────────────────────────────────────────────────────────────

import { CpuCircuit } from "../../src/cpu-reference.js";
import { MPS } from "../../src/mps.js";
import type { Matrix2 } from "../../src/gates.js";
import { stateMetrics } from "../lib/fidelity.js";
import { captureEnv } from "../lib/env.js";
import { SEEDS, mulberry32 } from "../lib/seeds.js";
import { initGPU } from "../../src/quantum.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

export interface E7Row {
  readonly depth: number;
  readonly seed: number;
  readonly cheapestChi: number | null;   // null if even χ_max fails F ≥ 0.999
  readonly log2ChiCheapest: number | null;
  readonly fidelityAtCheapest: number;
  /** NaN if χ_max was not actually tested (we passed at a cheaper χ and
   *  stopped early — the true fidelity at χ_max is ≥ fidelityAtCheapest). */
  readonly fidelityAtMaxChi: number;
  readonly entropyMidCut: number;        // bits, from cheapest-χ MPS spectrum
  readonly schmidtRankKept: number;      // number of nonzero σ in the spectrum
  readonly nQubits: number;
}

export interface E7Fit {
  readonly slope: number;            // d(log₂ χ_req) / d(depth)
  readonly intercept: number;
  readonly r2: number;
  readonly depths: readonly number[];
  readonly nPoints: number;
}

const CHI_LADDER = [2, 4, 8, 16, 32, 64, 128] as const;
const DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6, 8] as const;
const PASS_F = 0.999;
const SLOPE_TOL = 0.2;

interface GateApi {
  readonly nQubits: number;
  apply: (m: Matrix2, q: number) => void;
  cnot: (c: number, t: number) => void;
}

// ── Haar-random U(2) via normalized Gaussian quaternion ──────
// Sample four i.i.d. N(0, 1) values via Box-Muller; normalize to
// a unit quaternion (a, b, c, d) → SU(2) matrix
//   [[a + i·b,  c + i·d],
//    [-c + i·d, a − i·b]]
// The distribution on SU(2) is Haar; adding a global phase would
// give U(2) but the phase is irrelevant for entanglement growth.
function haarU2(rng: () => number): Matrix2 {
  const g = (): number => {
    let u = rng();
    const v = rng();
    if (u < 1e-300) u = 1e-300;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  let a = g(), b = g(), c = g(), d = g();
  const n = Math.sqrt(a * a + b * b + c * c + d * d) || 1;
  a /= n; b /= n; c /= n; d /= n;
  return [[a, b], [c, d], [-c, d], [a, -b]];
}

function buildHaarBrickwall(c: GateApi, depth: number, rng: () => number): void {
  const N = c.nQubits;
  for (let l = 0; l < depth; l++) {
    for (let q = 0; q < N; q++) c.apply(haarU2(rng), q);
    const offset = l & 1;
    for (let q = offset; q + 1 < N; q += 2) c.cnot(q, q + 1);
  }
}

function entropyFromSpectrum(sigma: Float64Array): { S: number; rankKept: number } {
  let norm2 = 0;
  for (let i = 0; i < sigma.length; i++) norm2 += sigma[i]! * sigma[i]!;
  if (norm2 <= 0) return { S: 0, rankKept: 0 };
  let S = 0;
  let rankKept = 0;
  for (let i = 0; i < sigma.length; i++) {
    const p = (sigma[i]! * sigma[i]!) / norm2;
    if (p > 1e-18) {
      S -= p * Math.log2(p);
      rankKept++;
    }
  }
  return { S, rankKept };
}

function fitLinear(points: readonly { x: number; y: number }[]): E7Fit {
  const n = points.length;
  if (n < 2) return { slope: NaN, intercept: NaN, r2: NaN, depths: [], nPoints: n };
  const mx = points.reduce((a, p) => a + p.x, 0) / n;
  const my = points.reduce((a, p) => a + p.y, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (const p of points) {
    sxy += (p.x - mx) * (p.y - my);
    sxx += (p.x - mx) ** 2;
    syy += (p.y - my) ** 2;
  }
  const slope = sxx === 0 ? 0 : sxy / sxx;
  const intercept = my - slope * mx;
  const r2 = syy === 0 ? 1 : (sxy * sxy) / (sxx * syy);
  const depths = [...new Set(points.map((p) => p.x))].sort((a, b) => a - b);
  return { slope, intercept, r2, depths, nPoints: n };
}

export async function runE7(
  opts: {
    nQubits?: number;
    seedsPerDepth?: number;
    depths?: readonly number[];
    chiLadder?: readonly number[];
  } = {},
): Promise<Artifact<E7Row> & { fit: E7Fit }> {
  const N = opts.nQubits ?? 16;
  // Default 5 per UI budget. Publishable numbers: pass { seedsPerDepth: 50 }.
  const seedsPerDepth = opts.seedsPerDepth ?? 5;
  const depths = opts.depths ?? DEFAULT_DEPTHS;
  const chiLadder = opts.chiLadder ?? CHI_LADDER;
  if (chiLadder.length === 0) throw new Error("E7: chiLadder cannot be empty");
  const chiMaxLadder = chiLadder[chiLadder.length - 1]!;

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E7Row[] = [];

  for (const depth of depths) {
    for (let s = 0; s < seedsPerDepth; s++) {
      const seed = (SEEDS.E7_MPS_CHI_SCALING ^ (depth * 2654435761) ^ (s * 0x9e3779b1)) >>> 0;

      // Ground truth: CPU statevector built once per seed.
      const cpu = new CpuCircuit(N);
      const rngCpu = mulberry32(seed);
      buildHaarBrickwall(cpu, depth, rngCpu);
      const cpuPsi = cpu.psi;

      let cheapestChi: number | null = null;
      let cheapestMps: MPS | null = null;
      let fAtCheapest = NaN;
      let fAtMax = NaN;

      for (const chi of chiLadder) {
        const mps = new MPS({ nQubits: N, chiMax: chi });
        const rngMps = mulberry32(seed);
        buildHaarBrickwall(mps, depth, rngMps);
        const F = stateMetrics(cpuPsi, mps.statevector()).fidelity;
        if (chi === chiMaxLadder) fAtMax = F;

        if (cheapestChi === null && F >= PASS_F) {
          cheapestChi = chi;
          fAtCheapest = F;
          cheapestMps = mps;
          break;  // higher χ in ladder would pass too — no information gained
        }
      }

      // Never passed the bar — the last build (at χ_max) was discarded.
      // Rebuild at χ_max for fAtMax and to have an MPS for entropy.
      if (cheapestMps === null) {
        const mps = new MPS({ nQubits: N, chiMax: chiMaxLadder });
        const rngMps = mulberry32(seed);
        buildHaarBrickwall(mps, depth, rngMps);
        fAtMax = stateMetrics(cpuPsi, mps.statevector()).fidelity;
        fAtCheapest = fAtMax;  // for reporting only; cheapestChi stays null
        cheapestMps = mps;
      }

      // Mid-cut entropy from the cheapest-χ MPS.
      const cut = Math.floor(N / 2) - 1;
      const sigma = cheapestMps.schmidtSpectrum(cut);
      const { S, rankKept } = entropyFromSpectrum(sigma);

      rows.push({
        depth,
        seed,
        cheapestChi,
        log2ChiCheapest: cheapestChi === null ? null : Math.log2(cheapestChi),
        fidelityAtCheapest: fAtCheapest,
        fidelityAtMaxChi: fAtMax,
        entropyMidCut: S,
        schmidtRankKept: rankKept,
        nQubits: N,
      });


      console.log(
        `[E7] d=${depth} seed=${s} `
        + `χ_req=${cheapestChi ?? `FAIL(@χ=${chiMaxLadder})`} `
        + `S=${S.toFixed(3)} F_cheap=${fAtCheapest.toFixed(5)} `
        + `F_max=${Number.isNaN(fAtMax) ? "n/a" : fAtMax.toFixed(5)}`,
      );
    }
  }

  // Fit log₂(χ_req) vs depth over passing cells.
  const fitPoints = rows
    .filter((r) => r.log2ChiCheapest !== null)
    .map((r) => ({ x: r.depth, y: r.log2ChiCheapest! }));
  const fit = fitLinear(fitPoints);

  const fracFail = rows.filter((r) => r.cheapestChi === null).length / Math.max(1, rows.length);
  const slopeOk = Number.isFinite(fit.slope) && Math.abs(fit.slope - 1) <= SLOPE_TOL;

  const status: Artifact<E7Row>["status"] =
    slopeOk && fracFail < 0.1 ? "pass" : "fail";

  const meta: ArtifactMeta = {
    protocol: "E7-chi-scaling",
    hypothesis:
      `For N=${N} Haar-random 1D brick-wall circuits, χ required for F ≥ ${PASS_F} `
      + `satisfies log₂(χ_req) ≈ depth (slope ≈ 1.0 ± ${SLOPE_TOL}).`,
    passBar:
      `fit slope ∈ [${(1 - SLOPE_TOL).toFixed(2)}, ${(1 + SLOPE_TOL).toFixed(2)}] `
      + `AND < 10% of cells fail at χ=${chiMaxLadder}`,
    seed: "E7_MPS_CHI_SCALING",
    warmup: 0,
    trials: seedsPerDepth,
  };

  const failNote = fracFail > 0
    ? ` ${(fracFail * 100).toFixed(1)}% cells failed at χ=${chiMaxLadder}.`
    : "";
  const diagnosis = slopeOk && fracFail < 0.1
    ? `slope=${fit.slope.toFixed(3)} (target 1.0 ± ${SLOPE_TOL}), intercept=${fit.intercept.toFixed(2)}, R²=${fit.r2.toFixed(3)}, fit over depths=[${fit.depths.join(",")}] on ${fit.nPoints} points.${failNote}`
    : `slope=${fit.slope.toFixed(3)} (need [${(1 - SLOPE_TOL).toFixed(2)}, ${(1 + SLOPE_TOL).toFixed(2)}]), intercept=${fit.intercept.toFixed(2)}, R²=${fit.r2.toFixed(3)}, points=${fit.nPoints}.${failNote}`;

  return { meta, env, rows, status, diagnosis, fit };
}
