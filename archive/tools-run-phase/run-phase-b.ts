// Phase B publishable artifact run. NOT a test — a standalone
// script driven by vite-node, sidestepping vitest's worker IPC
// timeout on multi-minute synchronous DMRG runs.
//
// Usage:
//   npx vite-node tools/run-phase-b.ts                # default budget
//   PHASE_B_NS=16,32,48,64 npx vite-node tools/...    # custom Ns
//   PHASE_B_CHI=32 npx vite-node tools/...            # custom χ
//
// Saves two artifact JSONs to experiments/results/<UTC date>/level-2/.

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { tfimMPO, heisenbergMPOReal } from "../src/manybody/mpo.js";
import { dmrgGroundState } from "../src/manybody/dmrg.js";
import {
  tfimPfeutyEnergyPerBond,
  heisenbergBetheEnergyPerBond,
} from "../src/manybody/analytical.js";

const DEFAULT_NS = [16, 32, 48, 64, 80, 100, 128];
const NS = (process.env["PHASE_B_NS"] ?? "").length > 0
  ? process.env["PHASE_B_NS"]!.split(",").map((s) => Number.parseInt(s.trim(), 10))
  : DEFAULT_NS;
const CHI = Number.parseInt(process.env["PHASE_B_CHI"] ?? "64", 10);

interface Cell {
  N: number;
  J: number;
  h?: number;
  chiMax: number;
  energy: number;
  energyPerSite: number;
  analyticalLimit: number;
  absDeviation: number;
  relDeviation: number;
  sweeps: number;
  lanczosMatvecs: number;
  wallSeconds: number;
}

function todayUTC(): string {
  return new Date().toISOString().slice(0, 10);
}

function runOne(
  label: string,
  M: ReturnType<typeof tfimMPO>,
  expected: number,
  chi: number,
): Cell {
  const t0 = performance.now();
  const r = dmrgGroundState(M, { chiMax: chi, maxSweeps: 24, tol: 1e-9 });
  const wall = (performance.now() - t0) / 1000;
  const ePerSite = r.energy / M.nQubits;
  const absDev = Math.abs(ePerSite - expected);
  console.log(
    `[${label}] N=${M.nQubits} χ=${chi}: E=${r.energy.toFixed(6)} ` +
    `E/N=${ePerSite.toFixed(6)} e_∞=${expected.toFixed(6)} ` +
    `|Δ|=${absDev.toFixed(5)} sweeps=${r.history.length} ${wall.toFixed(1)}s`,
  );
  return {
    N: M.nQubits,
    J: 1,
    chiMax: chi,
    energy: r.energy,
    energyPerSite: ePerSite,
    analyticalLimit: expected,
    absDeviation: absDev,
    relDeviation: absDev / Math.abs(expected),
    sweeps: r.history.length,
    lanczosMatvecs: r.lanczosMatvecs,
    wallSeconds: wall,
  };
}

function saveArtifact(
  protocol: string,
  hypothesis: string,
  passBar: string,
  rows: Cell[],
  status: "pass" | "fail",
  diagnosis: string,
): string {
  const outDir = resolve(process.cwd(), "experiments", "results", todayUTC(), "level-2");
  mkdirSync(outDir, { recursive: true });
  const path = resolve(outDir, `${protocol}.json`);
  const artifact = {
    meta: {
      protocol,
      hypothesis,
      passBar,
      seed: protocol.toUpperCase().replace(/-/g, "_"),
      warmup: 0,
      trials: 1,
    },
    env: {
      runtime: `node ${process.version}`,
      platform: process.platform,
      arch: process.arch,
      cpus: typeof process.cpuUsage === "function" ? "host CPU" : "unknown",
      timestamp: new Date().toISOString(),
      sweepNs: NS,
      chiMax: CHI,
    },
    rows,
    status,
    diagnosis,
  };
  writeFileSync(path, JSON.stringify(artifact, null, 2), "utf-8");
  console.log(`→ ${path}`);
  return path;
}

console.log(`Phase B publishable run — Ns=[${NS.join(",")}] χ=${CHI}`);
console.log("=".repeat(72));

// ── E18: TFIM at h = 1 (critical) ─────────────────────────────
console.log("\n── E18: TFIM h=1 (critical, gapped → 0 at criticality) ──");
const eInfTfim = tfimPfeutyEnergyPerBond(1, 1);
const tfimRows: Cell[] = [];
for (const N of NS) {
  tfimRows.push({ ...runOne("TFIM h=1", tfimMPO(N, 1, 1), eInfTfim, CHI), h: 1 });
}
const tfimNmax = Math.max(...NS);
const tfimMax = tfimRows.find((r) => r.N === tfimNmax)!;
const tfimStatus: "pass" | "fail" = tfimMax.absDeviation <= 0.01 ? "pass" : "fail";
const tfimTrace = tfimRows.map((r) => `N=${r.N}:|Δ|=${r.absDeviation.toFixed(5)}`).join(", ");
const tfimDiag = tfimStatus === "pass"
  ? `TFIM (h=1) at N=${tfimNmax}, χ=${CHI} hits |E/N − e_∞|=${tfimMax.absDeviation.toFixed(5)} ≤ 0.01. ${tfimTrace}.`
  : `N=${tfimNmax} |Δ|=${tfimMax.absDeviation.toFixed(5)} > 0.01. ${tfimTrace}.`;
saveArtifact(
  "E18-tfim-pfeuty-publishable",
  "DMRG ground-state energy per site of TFIM (h=1, critical) at χ=64 converges to Pfeuty's −4/π within 1% by N=128.",
  `|E/N − e_∞(λ=1)| ≤ 0.01 at N = ${tfimNmax}, χ = ${CHI}.`,
  tfimRows, tfimStatus, tfimDiag,
);

// ── E19: Heisenberg AFM at J = 1 ──────────────────────────────
console.log("\n── E19: Heisenberg AFM J=1 (gapless → log) ──");
const eInfHeis = heisenbergBetheEnergyPerBond(1);
const heisRows: Cell[] = [];
for (const N of NS) {
  heisRows.push(runOne("Heis", heisenbergMPOReal(N, 1), eInfHeis, CHI));
}
const heisNmax = Math.max(...NS);
const heisMax = heisRows.find((r) => r.N === heisNmax)!;
const heisStatus: "pass" | "fail" = heisMax.absDeviation <= 0.02 ? "pass" : "fail";
const heisTrace = heisRows.map((r) => `N=${r.N}:|Δ|=${r.absDeviation.toFixed(5)}`).join(", ");
const heisDiag = heisStatus === "pass"
  ? `Heisenberg AFM at N=${heisNmax}, χ=${CHI} hits |E/N − e_∞|=${heisMax.absDeviation.toFixed(5)} ≤ 0.02. ${heisTrace}.`
  : `N=${heisNmax} |Δ|=${heisMax.absDeviation.toFixed(5)} > 0.02. ${heisTrace}.`;
saveArtifact(
  "E19-heisenberg-bethe-publishable",
  "DMRG ground-state energy per site of the Heisenberg AFM chain at χ=64 converges to the Bethe ansatz limit J·(1/4 − ln 2) within 2% by N=128.",
  `|E/N − e_∞| ≤ 0.02 at N = ${heisNmax}, χ = ${CHI}.`,
  heisRows, heisStatus, heisDiag,
);

console.log("\nDone.");
