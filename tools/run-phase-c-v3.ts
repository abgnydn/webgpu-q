// Phase C v3 publishable artifact run — full STO-3G BeH₂ FCI
// dissociation curve. Writes a JSON artifact mirroring the
// Phase B / Phase C v1 / Phase C v2 publishable runs.
//
// Usage:
//   npx vite-node tools/run-phase-c-v3.ts

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { buildBeH2Full } from "../src/chemistry/beh2-full-builder.js";
import { lanczosGroundState } from "../src/manybody/lanczos.js";

const BONDS = [0.8, 1.0, 1.2, 1.34, 1.5, 1.7, 2.0, 2.5, 3.0];
const PYSCF_REF_R134 = -15.5949;       // PySCF BeH₂ STO-3G FCI at R=1.34 Å
const SONLY_REF_R134 = -15.349162;     // Phase C v2 result for comparison

function fciEnergy(H: Float64Array, k: number): { energy: number; iter: number } {
  const matvec = (x: Float64Array, out: Float64Array) => {
    for (let i = 0; i < k; i++) {
      let s = 0;
      const row = i * k;
      for (let j = 0; j < k; j++) s += H[row + j]! * x[j]!;
      out[i] = s;
    }
  };
  const x0 = new Float64Array(k);
  for (let i = 0; i < k; i++) x0[i] = 1;
  const r = lanczosGroundState(matvec, x0, { maxIter: 100, tol: 1e-10 });
  return { energy: r.energy, iter: r.iter };
}

function todayUTC(): string {
  return new Date().toISOString().slice(0, 10);
}

interface Cell {
  R_angstrom: number;
  R_bohr: number;
  Vnn: number;
  fciHartree: number;
  sOnlyDeltaMHa: number;        // E_full − E_sOnly (in mHa) — Be 2p contribution
  pyscfDeltaMHa: number;        // E_ours − E_PySCF reference (R=1.34 only)
  lanczosIter: number;
  buildSeconds: number;
  fciSeconds: number;
}

console.log(`Phase C v3 — full STO-3G BeH₂ FCI (Be 1s+2s+2p + 2 H1s = 7 spatial / 14 spin-orbitals)`);
console.log(`Sector: N=6, dim = C(14, 6) = 3003`);
console.log("=".repeat(72));

const rows: Cell[] = [];
let bestR = 0, bestE = Infinity;

for (const R of BONDS) {
  const t0 = performance.now();
  const data = buildBeH2Full(R);
  const tBuild = (performance.now() - t0) / 1000;

  const t1 = performance.now();
  const r = fciEnergy(data.sector.H, data.sector.k);
  const tFci = (performance.now() - t1) / 1000;

  const sOnlyDelta = (r.energy - SONLY_REF_R134) * 1000;
  const pyscfDelta = Math.abs(R - 1.34) < 1e-9 ? (r.energy - PYSCF_REF_R134) * 1000 : NaN;

  rows.push({
    R_angstrom: R,
    R_bohr: data.R_Bohr,
    Vnn: data.integrals.Vnn,
    fciHartree: r.energy,
    sOnlyDeltaMHa: sOnlyDelta,
    pyscfDeltaMHa: pyscfDelta,
    lanczosIter: r.iter,
    buildSeconds: tBuild,
    fciSeconds: tFci,
  });

  if (r.energy < bestE) { bestE = r.energy; bestR = R; }

  console.log(
    `  R=${R.toFixed(2).padStart(5)} Å:  E_FCI = ${r.energy.toFixed(6)} Ha  ` +
    `Δ vs s-only = ${sOnlyDelta.toFixed(2).padStart(8)} mHa  ` +
    `[build ${tBuild.toFixed(2)}s, lanczos ${tFci.toFixed(2)}s, ${r.iter} iter]`,
  );
}

const eqRow = rows.find((r) => Math.abs(r.R_angstrom - 1.34) < 1e-9)!;
const passedPySCF = Math.abs(eqRow.pyscfDeltaMHa) <= 1.0;
const passedSOnlyGain = Math.abs(eqRow.sOnlyDeltaMHa) >= 200;
const passedMinAtBond = Math.abs(bestR - 1.34) < 0.01;
const status: "pass" | "fail" =
  passedPySCF && passedSOnlyGain && passedMinAtBond ? "pass" : "fail";

const diagnosis = status === "pass"
  ? `Full STO-3G BeH₂ FCI matches PySCF (${eqRow.pyscfDeltaMHa.toFixed(3)} mHa at R=1.34 Å); ` +
    `Be 2p adds ${(-eqRow.sOnlyDeltaMHa).toFixed(1)} mHa correlation; minimum at R=${bestR}.`
  : `Pass bar miss — PySCF Δ=${eqRow.pyscfDeltaMHa.toFixed(3)} mHa, sOnly gain=${(-eqRow.sOnlyDeltaMHa).toFixed(1)} mHa, min at R=${bestR}.`;

const artifact = {
  meta: {
    protocol: "E22-beh2-full-fci-publishable",
    hypothesis:
      "Full STO-3G BeH₂ FCI (7 spatial / 14 spin-orbitals, sector-projected to N=6) reproduces the PySCF reference at R = 1.34 Å within 1 mHa, recovers ≥ 200 mHa of correlation vs the s-only basis, and shows the dissociation-curve minimum at the experimental Be-H bond.",
    passBar:
      "|E_FCI − E_PySCF| ≤ 1 mHa at R=1.34 Å AND ≥ 200 mHa correlation gain vs s-only AND min at R=1.34 Å",
    seed: "E22_BEH2_FULL_FCI",
    warmup: 0, trials: 1,
  },
  env: {
    runtime: `node ${process.version}`,
    platform: process.platform, arch: process.arch,
    timestamp: new Date().toISOString(),
    bonds: BONDS,
    sectorDim: 3003,
    nQubits: 14,
    nSpatial: 7,
    particleNumber: 6,
    fciMethod: "matrix-free Lanczos",
    integrals: "McMurchie-Davidson Hermite-Gaussian (Phase C v3 stage 1)",
  },
  rows, status, diagnosis,
};

const outDir = resolve(process.cwd(), "experiments", "results", todayUTC(), "level-6");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "E22-beh2-full-fci-publishable.json");
writeFileSync(outPath, JSON.stringify(artifact, null, 2), "utf-8");
console.log(`\n→ ${outPath}`);
console.log(`Status: ${status}`);
console.log(diagnosis);
