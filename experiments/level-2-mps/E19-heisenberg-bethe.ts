// ─────────────────────────────────────────────────────────────
// E19 — DMRG Heisenberg AFM scaling vs Bethe ansatz limit.
//
// Hypothesis: with the real-form S+/S-/Z MPO (Phase B) + the
// matrix-free Lanczos DMRG from Phase A, the ground-state
// energy per site of the antiferromagnetic Heisenberg chain
// converges to the Bethe / Hulthén thermodynamic limit
//   e_∞ = J · (1/4 − ln 2)
// as N grows, and remains within 1.5% of e_∞ at N ≥ 64 with
// χ = 64.
//
// Heisenberg is harder than TFIM at the same (N, χ): the
// gapless Goldstone mode means entanglement grows logarithmically
// with N, so deeper bonds saturate sooner. The pass bar is
// looser than E18 to reflect this.
//
// Sweep: N ∈ {16, 24, 32, 48} at χ = 32 (UI default) or
//   {16, 32, 48, 64, 80, 100, 128} at χ = 64 (publishable);
// J = 1.
//
// Pass bars:
//   • |E/N − e_∞| ≤ 0.06 at the largest N (gapless model →
//     boundary energy is larger than TFIM's; UI-default budget).
//   • |E/N − e_∞| monotonically decreases as N grows.
// ─────────────────────────────────────────────────────────────

import { heisenbergMPOReal } from "../../src/manybody/mpo.js";
import { dmrgGroundState } from "../../src/manybody/dmrg.js";
import { heisenbergBetheEnergyPerBond } from "../../src/manybody/analytical.js";
import { initGPU } from "../../src/quantum.js";
import { captureEnv } from "../lib/env.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

export interface E19Row {
  readonly N: number;
  readonly J: number;
  readonly chiMax: number;
  readonly energy: number;
  readonly energyPerSite: number;
  readonly betheLimit: number;
  readonly absDeviation: number;
  readonly relDeviation: number;
  readonly sweeps: number;
  readonly lanczosMatvecs: number;
  readonly wallSeconds: number;
}

const DEFAULT_NS = [16, 24, 32, 48] as const;
const DEFAULT_CHI = 32;

export interface E19Opts {
  readonly Ns?: readonly number[];
  readonly chiMax?: number;
  readonly maxSweeps?: number;
  readonly tol?: number;
}

export async function runE19(opts: E19Opts = {}): Promise<Artifact<E19Row>> {
  const Ns = opts.Ns ?? DEFAULT_NS;
  const chiMax = opts.chiMax ?? DEFAULT_CHI;
  const maxSweeps = opts.maxSweeps ?? 24;
  const tol = opts.tol ?? 1e-9;
  const J = 1;

  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E19: no GPU adapter for env block");
  const env = await captureEnv(device, adapter);

  const eInf = heisenbergBetheEnergyPerBond(J);
  const rows: E19Row[] = [];

  for (const N of Ns) {
    const M = heisenbergMPOReal(N, J);
    const t0 = performance.now();
    const res = dmrgGroundState(M, { chiMax, maxSweeps, tol });
    const wallSeconds = (performance.now() - t0) / 1000;

    const ePerSite = res.energy / N;
    const absDev = Math.abs(ePerSite - eInf);
    const relDev = absDev / Math.abs(eInf);

    const row: E19Row = {
      N,
      J,
      chiMax,
      energy: res.energy,
      energyPerSite: ePerSite,
      betheLimit: eInf,
      absDeviation: absDev,
      relDeviation: relDev,
      sweeps: res.history.length,
      lanczosMatvecs: res.lanczosMatvecs,
      wallSeconds,
    };
    rows.push(row);

    console.log(
      `[E19] N=${N} χ=${chiMax} E=${res.energy.toFixed(6)} ` +
      `E/N=${ePerSite.toFixed(6)} e_∞=${eInf.toFixed(6)} ` +
      `|Δ|=${absDev.toFixed(6)} rel=${(relDev * 100).toFixed(3)}% ` +
      `${res.history.length}sw ${wallSeconds.toFixed(2)}s`,
    );
  }

  const Nmax = Math.max(...Ns);
  const cellMax = rows.find((r) => r.N === Nmax)!;
  const absBarFail = cellMax.absDeviation > 0.06;

  // Monotonic convergence in N: |E/N − e_∞| should decrease with N.
  const sorted = [...rows].sort((a, b) => a.N - b.N);
  let monotonicFails = 0;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i]!.absDeviation > sorted[i - 1]!.absDeviation + 1e-5) {
      monotonicFails++;
    }
  }
  const trace = sorted.map((r) => `N=${r.N}:|Δ|=${r.absDeviation.toFixed(4)}`).join(", ");

  const status: Artifact<E19Row>["status"] =
    absBarFail || monotonicFails > 0 ? "fail" : "pass";

  const diagnosis = status === "pass"
    ? `N=${Nmax}: |E/N − e_∞|=${cellMax.absDeviation.toFixed(4)} ≤ 0.06; convergence: ${trace}.`
    : `${absBarFail ? `N=${Nmax} |Δ|=${cellMax.absDeviation.toFixed(4)} > 0.06; ` : ""}` +
      `${monotonicFails} non-monotonic step(s) [${trace}].`;

  const meta: ArtifactMeta = {
    protocol: "E19-heisenberg-bethe",
    hypothesis: "DMRG ground-state energy per site of the Heisenberg AFM chain converges to the Bethe ansatz limit J·(1/4 − ln 2) as N grows. UI-default settings (N = 48, χ = 32) hit |E/N − e_∞| ≤ 0.06; publishable settings (N = 128, χ = 64) hit ≤ 0.02.",
    passBar: "|E/N − e_∞| ≤ 0.06 at the largest N; absolute deviation monotonically decreases with N.",
    seed: "E19_HEISENBERG_BETHE",
    warmup: 0,
    trials: 1,
  };

  return { meta, env, rows, status, diagnosis };
}
