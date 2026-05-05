// Phase C v4 publishable artifacts: H₂O symmetric-stretch curve.
// CH₄ is also computable via the same pipeline but materializes a
// 43758² ≈ 15 GB Hsec — opt in with PHASE_C4_CH4=1 to include it.
//
// Usage:
//   npx vite-node tools/run-phase-c-v4.ts                   # H₂O only
//   PHASE_C4_CH4=1 npx vite-node tools/run-phase-c-v4.ts    # + CH₄ (~5 min, ~15 GB RAM)

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { buildMoleculeFCI } from "../src/chemistry/molecule-builder.js";

const PYSCF_H2O_REF = -75.0124;     // PySCF STO-3G FCI at R_OH=0.9572, ∠HOH=104.52°
const PYSCF_CH4_REF = -39.8068;     // PySCF STO-3G FCI at R_CH=1.09 Å tetrahedral

const RUN_CH4 = !!process.env["PHASE_C4_CH4"];

function todayUTC(): string {
  return new Date().toISOString().slice(0, 10);
}

// ── H₂O: symmetric stretch at fixed angle ──────────────────
console.log("================================");
console.log("  H₂O symmetric stretch (∠HOH = 104.52° fixed)");
console.log("================================");

const ANGLE_DEG = 104.52;
const halfAngle = (ANGLE_DEG / 2) * Math.PI / 180;
const R_OHs = [0.7, 0.85, 0.9572, 1.05, 1.20, 1.50, 2.0];

interface H2ORow {
  R_OH_angstrom: number;
  angle_degrees: number;
  Vnn: number;
  energyHartree: number;
  pyscfDeltaMHa: number | null;     // populated only at the experimental geometry
  lanczosIter: number;
  buildSeconds: number;
  fciSeconds: number;
}

const h2oRows: H2ORow[] = [];
let bestR = 0, bestE = Infinity;

for (const R of R_OHs) {
  const x = R * Math.sin(halfAngle);
  const z = R * Math.cos(halfAngle);
  const t0 = performance.now();
  const data = buildMoleculeFCI([
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ]);
  const tBuild = (performance.now() - t0) / 1000;
  const t1 = performance.now();
  const r = data.fci();
  const tFci = (performance.now() - t1) / 1000;
  const isEq = Math.abs(R - 0.9572) < 1e-9;
  const pyscfDelta = isEq ? (r.energy - PYSCF_H2O_REF) * 1000 : null;
  h2oRows.push({
    R_OH_angstrom: R,
    angle_degrees: ANGLE_DEG,
    Vnn: data.integrals.Vnn,
    energyHartree: r.energy,
    pyscfDeltaMHa: pyscfDelta,
    lanczosIter: r.iter,
    buildSeconds: tBuild,
    fciSeconds: tFci,
  });
  if (r.energy < bestE) { bestE = r.energy; bestR = R; }
  const eqMark = isEq ? "  ←  experimental" : "";
  const deltaStr = pyscfDelta !== null ? `, Δ vs PySCF = ${pyscfDelta.toFixed(3)} mHa` : "";
  console.log(
    `  R_OH=${R.toFixed(4).padStart(6)} Å:  E_FCI = ${r.energy.toFixed(6)} Ha  ` +
    `[${tBuild.toFixed(2)}s + ${tFci.toFixed(2)}s, ${r.iter} iter]${deltaStr}${eqMark}`,
  );
}

const eqRow = h2oRows.find((r) => Math.abs(r.R_OH_angstrom - 0.9572) < 1e-9)!;
const h2oPass =
  Math.abs(eqRow.pyscfDeltaMHa!) <= 1.0 && Math.abs(bestR - 0.9572) < 0.1;

console.log(`\nH₂O dissociation min at R_OH ≈ ${bestR} Å (experimental: 0.9572 Å)`);
console.log(`H₂O FCI at eq − PySCF = ${eqRow.pyscfDeltaMHa!.toFixed(4)} mHa`);
console.log(`H₂O status: ${h2oPass ? "pass" : "fail"}`);

const h2oArtifact = {
  meta: {
    protocol: "E23-h2o-full-fci-publishable",
    hypothesis:
      "Full STO-3G H₂O FCI in a browser tab matches PySCF at the experimental geometry (R_OH = 0.9572 Å, ∠HOH = 104.52°) within 1 mHa, and the symmetric-stretch dissociation minimum lands at R_OH ≈ 0.9572 Å.",
    passBar: "|E_FCI − E_PySCF| ≤ 1 mHa at experimental geometry AND min within 0.1 Å of 0.9572 Å",
    seed: "E23_H2O_FULL_FCI",
    warmup: 0, trials: 1,
  },
  env: {
    runtime: `node ${process.version}`,
    platform: process.platform, arch: process.arch,
    timestamp: new Date().toISOString(),
    nSpatial: 7,
    nQubits: 14,
    nElectrons: 10,
    sectorDim: 1001,
    fciMethod: "matrix-free Lanczos",
    integrals: "McMurchie-Davidson Hermite-Gaussian (Phase C v3 stage 1)",
  },
  rows: h2oRows,
  status: h2oPass ? "pass" : "fail",
  diagnosis: h2oPass
    ? `H₂O FCI matches PySCF to ${Math.abs(eqRow.pyscfDeltaMHa!).toFixed(4)} mHa at R_OH=0.9572 Å; dissociation min at R_OH=${bestR} Å.`
    : `H₂O eq Δ=${eqRow.pyscfDeltaMHa!.toFixed(4)} mHa or min at R=${bestR} Å outside spec.`,
};

const outDir = resolve(process.cwd(), "experiments", "results", todayUTC(), "level-6");
mkdirSync(outDir, { recursive: true });
writeFileSync(resolve(outDir, "E23-h2o-full-fci-publishable.json"), JSON.stringify(h2oArtifact, null, 2));
console.log(`→ ${resolve(outDir, "E23-h2o-full-fci-publishable.json")}`);

// ── CH₄: opt-in single-point FCI ───────────────────────────
if (RUN_CH4) {
  console.log("\n================================");
  console.log("  CH₄ tetrahedral, R_CH = 1.09 Å (opt-in; ~5 min, ~15 GB RAM)");
  console.log("================================");
  const R_CH = 1.09;
  const r3 = R_CH / Math.sqrt(3);
  const t0 = performance.now();
  const data = buildMoleculeFCI([
    { symbol: "C", pos: [0, 0, 0] },
    { symbol: "H", pos: [ r3,  r3,  r3] },
    { symbol: "H", pos: [ r3, -r3, -r3] },
    { symbol: "H", pos: [-r3,  r3, -r3] },
    { symbol: "H", pos: [-r3, -r3,  r3] },
  ]);
  const tBuild = (performance.now() - t0) / 1000;
  const t1 = performance.now();
  const r = data.fci({ maxIter: 200 });
  const tFci = (performance.now() - t1) / 1000;
  const delta = (r.energy - PYSCF_CH4_REF) * 1000;
  console.log(`E_FCI = ${r.energy.toFixed(6)} Ha  [${tBuild.toFixed(2)}s + ${tFci.toFixed(2)}s, ${r.iter} iter]`);
  console.log(`Δ vs PySCF (-39.8068 Ha) = ${delta.toFixed(3)} mHa`);

  const ch4Pass = Math.abs(delta) <= 1.6;
  const ch4Artifact = {
    meta: {
      protocol: "E24-ch4-full-fci-publishable",
      hypothesis: "Full STO-3G CH₄ FCI in a browser tab (9 spatial / 18 spin-orbitals / N=10 sector dim 43758) matches PySCF within 1.6 mHa at R_CH = 1.09 Å.",
      passBar: "|E_FCI − E_PySCF| ≤ 1.6 mHa (chemical accuracy) at R_CH=1.09 Å",
      seed: "E24_CH4_FULL_FCI",
      warmup: 0, trials: 1,
    },
    env: {
      runtime: `node ${process.version}`,
      platform: process.platform, arch: process.arch,
      timestamp: new Date().toISOString(),
      nSpatial: 9, nQubits: 18, nElectrons: 10, sectorDim: 43758,
      hsecMemoryGB: (43758 * 43758 * 8) / 1e9,
      fciMethod: "matrix-free Lanczos",
      integrals: "McMurchie-Davidson (Phase C v3 stage 1)",
    },
    row: {
      R_CH_angstrom: R_CH,
      Vnn: data.integrals.Vnn,
      energyHartree: r.energy,
      pyscfDeltaMHa: delta,
      lanczosIter: r.iter,
      buildSeconds: tBuild,
      fciSeconds: tFci,
    },
    status: ch4Pass ? "pass" : "fail",
    diagnosis: ch4Pass
      ? `CH₄ FCI = ${r.energy.toFixed(6)} Ha, |Δ vs PySCF| = ${Math.abs(delta).toFixed(3)} mHa (chemical accuracy).`
      : `|Δ vs PySCF| = ${Math.abs(delta).toFixed(3)} mHa exceeds 1.6 mHa.`,
  };
  writeFileSync(resolve(outDir, "E24-ch4-full-fci-publishable.json"), JSON.stringify(ch4Artifact, null, 2));
  console.log(`→ ${resolve(outDir, "E24-ch4-full-fci-publishable.json")}`);
  console.log(`CH₄ status: ${ch4Pass ? "pass" : "fail"}`);
} else {
  console.log("\n(skipping CH₄ — set PHASE_C4_CH4=1 to include)");
}
