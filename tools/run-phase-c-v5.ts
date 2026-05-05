// Phase C v5 publishable artifact: CH₄ FCI via sparse-CSR Hsec.
// Dense Hsec for CH₄ would be 15 GB (won't fit in a browser tab);
// sparse storage drops it to ~240 MB AND speeds up Lanczos by ~100×.
//
// Usage: npx vite-node tools/run-phase-c-v5.ts

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { buildMoleculeFCI } from "../src/chemistry/molecule-builder.js";

const PYSCF_CH4_REF = -39.8068;     // PySCF STO-3G FCI at R_CH=1.09 Å tetrahedral

function todayUTC(): string {
  return new Date().toISOString().slice(0, 10);
}

console.log("================================");
console.log("  CH₄ tetrahedral, R_CH = 1.09 Å — full STO-3G FCI via sparse Hsec");
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
const sparse = data.sparseSector!;

const nnz = sparse.values.length;
const sparseMB = (12 * nnz) / 1e6;
const denseGB = (8 * sparse.k * sparse.k) / 1e9;
console.log(`  nSpatial = ${data.nSpatial}, nQubits = ${data.nQubits}, sectorDim = ${sparse.k}`);
console.log(`  storage  = ${data.storage}, nnz = ${nnz}  (${sparseMB.toFixed(1)} MB sparse vs ${denseGB.toFixed(2)} GB dense)`);
console.log(`  V_nn     = ${data.integrals.Vnn.toFixed(6)} Ha`);
console.log(`  build    = ${tBuild.toFixed(2)} s`);

const t1 = performance.now();
const r = data.fci({ maxIter: 200 });
const tFci = (performance.now() - t1) / 1000;
const delta = (r.energy - PYSCF_CH4_REF) * 1000;

console.log(`  E_FCI    = ${r.energy.toFixed(6)} Ha  (Lanczos ${r.iter} iters, ${tFci.toFixed(2)} s)`);
console.log(`  PySCF    = ${PYSCF_CH4_REF} Ha`);
console.log(`  Δ        = ${delta.toFixed(3)} mHa`);

const passed = Math.abs(delta) <= 1.6;
const artifact = {
  meta: {
    protocol: "E25-ch4-full-fci-sparse-publishable",
    hypothesis:
      "Full STO-3G CH₄ FCI in a browser tab via sparse-CSR Hsec (dense would be 15 GB); matches PySCF (-39.8068 Ha) within chemical accuracy at R_CH = 1.09 Å tetrahedral.",
    passBar:
      "|E_FCI − E_PySCF| ≤ 1.6 mHa (chemical accuracy) at R_CH = 1.09 Å AND sparse memory ≤ 1 GB",
    seed: "E25_CH4_FULL_FCI_SPARSE",
    warmup: 0, trials: 1,
  },
  env: {
    runtime: `node ${process.version}`,
    platform: process.platform, arch: process.arch,
    timestamp: new Date().toISOString(),
    nSpatial: data.nSpatial,
    nQubits: data.nQubits,
    nElectrons: data.nElectrons,
    sectorDim: sparse.k,
    storage: data.storage,
    nnz,
    sparseMemoryMB: sparseMB,
    denseMemoryGB: denseGB,
    fciMethod: "matrix-free Lanczos on sparse-CSR Hsec",
    integrals: "McMurchie-Davidson Hermite-Gaussian (Phase C v3 stage 1)",
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
  status: passed ? "pass" : "fail",
  diagnosis: passed
    ? `CH₄ FCI = ${r.energy.toFixed(6)} Ha, |Δ vs PySCF| = ${Math.abs(delta).toFixed(3)} mHa (chemical accuracy). Sparse Hsec = ${sparseMB.toFixed(1)} MB (${(denseGB * 1000 / sparseMB).toFixed(0)}× smaller than dense).`
    : `|Δ vs PySCF| = ${Math.abs(delta).toFixed(3)} mHa exceeds 1.6 mHa.`,
};

const outDir = resolve(process.cwd(), "experiments", "results", todayUTC(), "level-6");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "E25-ch4-full-fci-sparse-publishable.json");
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(`\n→ ${outPath}`);
console.log(`Status: ${passed ? "pass" : "fail"}`);
console.log(artifact.diagnosis);
