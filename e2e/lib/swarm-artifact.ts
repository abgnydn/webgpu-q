// Emits an environment-stamped JSON artifact for a browser-tab swarm HF run,
// under experiments/results/<UTC date>/swarm/<molecule>.json, in the same
// { meta, env, rows, status, diagnosis } shape as the level-1/2/3 experiments.
//
// Swarm whole-SCF benchmarks are multi-second and the energy is deterministic,
// so — unlike the GPU-kernel timing protocol (5 warmup + 20 trials) — these
// record a single representative run's per-phase wall-time plus the converged
// energy. That distinction is stamped into the artifact (trials: 1).
import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";
import type { Page } from "@playwright/test";

export interface SwarmResult {
  n: number;
  nAux: number;
  swMs: number;
  energy: number;
  iter: number;
  slicesPerTab?: number[];
  masterSliceMB?: number;
  converged?: boolean;
  integMs?: number;
  dfMs?: number;
  // present in the specs that also build a single-tab reference (benzene,
  // naphthalene): the reference energy and the swarm-vs-reference deviation
  // that substantiates the bit-exact-partition claim.
  refEnergy?: number;
  refMs?: number;
  deltaE?: number;
}

export interface SwarmMeta {
  molecule: string;
  formula: string;
  basis: string;
  nTabs: number;
  innerPool: number;
}

function gitSha(): string {
  try {
    return execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim();
  } catch {
    return "unknown";
  }
}

/**
 * Writes the swarm artifact. Captures `navigator.userAgent` and the WebGPU
 * adapter info from the (still-open) master page before the context closes.
 * Returns the written path.
 */
export async function writeSwarmArtifact(
  page: Page,
  meta: SwarmMeta,
  r: SwarmResult,
): Promise<string> {
  const env = await page.evaluate(async () => {
    let adapterInfo: Record<string, string> = {};
    try {
      const adapter = await (navigator as unknown as {
        gpu?: { requestAdapter(): Promise<{ info?: GPUAdapterInfo } | null> };
      }).gpu?.requestAdapter();
      const info = adapter?.info as unknown as Record<string, string> | undefined;
      if (info) {
        adapterInfo = {
          vendor: info.vendor ?? "",
          architecture: info.architecture ?? "",
          device: info.device ?? "",
          description: info.description ?? "",
        };
      }
    } catch {
      /* adapter info best-effort */
    }
    return { userAgent: navigator.userAgent, adapterInfo };
  });

  const integMs = r.integMs ?? 0;
  const dfMs = r.dfMs ?? 0;
  const converged = r.converged ?? true;
  const totalMs = integMs + dfMs + r.swMs;
  const fullBTensorGB = (r.n * r.n * r.nAux * 8) / 1e9;

  const artifact = {
    meta: {
      protocol: `swarm-hf-${meta.molecule}`,
      molecule: meta.molecule,
      formula: meta.formula,
      basis: meta.basis,
      nTabs: meta.nTabs,
      innerPool: meta.innerPool,
      hypothesis:
        "Partitioning the density-fitted Fock build by auxiliary index across N same-origin tabs reproduces the single-tab energy and runs to convergence.",
      passBar: "converged === true && Number.isFinite(energy)",
      warmup: 0,
      trials: 1,
      timingNote:
        "Single representative run on a 16 GB M2 Pro (Chromium, cross-origin isolated). Per-phase wall-time; energy is deterministic. Not the 5-warmup/20-trial GPU-kernel protocol.",
    },
    env: {
      gitSha: gitSha(),
      userAgent: env.userAgent,
      adapterInfo: env.adapterInfo,
      platform: process.platform,
      timestamp: new Date().toISOString(),
    },
    rows: [
      {
        n: r.n,
        nAux: r.nAux,
        iter: r.iter,
        converged,
        energyHa: r.energy,
        ...(r.refEnergy !== undefined ? { refEnergyHa: r.refEnergy } : {}),
        ...(r.deltaE !== undefined ? { swarmVsRefDeltaHa: r.deltaE } : {}),
        ...(integMs ? { integralsMs: Math.round(integMs) } : {}),
        ...(dfMs ? { auxDFMs: Math.round(dfMs) } : {}),
        swarmScfMs: Math.round(r.swMs),
        ...(r.refMs !== undefined ? { singleTabRefMs: Math.round(r.refMs) } : {}),
        totalMs: Math.round(totalMs),
        totalSec: Number((totalMs / 1000).toFixed(2)),
        ...(r.masterSliceMB !== undefined ? { perTabSliceMB: Number(r.masterSliceMB.toFixed(1)) } : {}),
        fullBTensorGB: Number(fullBTensorGB.toFixed(3)),
        ...(r.slicesPerTab !== undefined ? { slicesPerTab: r.slicesPerTab } : {}),
      },
    ],
    status: converged && Number.isFinite(r.energy) ? "pass" : "fail",
    diagnosis: converged
      ? `${meta.molecule} ${meta.basis}: E=${r.energy.toFixed(8)} Ha in ${r.iter} iters across ${meta.nTabs} tabs` +
        (r.deltaE !== undefined ? `; swarm vs single-tab |ΔE|=${r.deltaE.toExponential(2)} Ha` : "") + "."
      : `${meta.molecule} ${meta.basis}: did not converge in ${r.iter} iters.`,
  };

  const dateUTC = new Date().toISOString().slice(0, 10);
  const outDir = path.resolve("experiments/results", dateUTC, "swarm");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, `${meta.molecule}-${meta.basis.replace(/[^a-z0-9]/gi, "")}.json`);
  fs.writeFileSync(outPath, JSON.stringify(artifact, null, 2));
  // eslint-disable-next-line no-console
  console.log(`[artifact] wrote ${outPath} (status=${artifact.status})`);
  return outPath;
}
