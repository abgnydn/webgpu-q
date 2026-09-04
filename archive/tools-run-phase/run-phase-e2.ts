// Phase E stage 2 publishable: H₂O in cc-pVDZ — first non-minimal
// basis set ("real chemistry territory"). Compare HF / MP2 in
// STO-3G vs cc-pVDZ, side-by-side, to show the correlation captured
// scales dramatically with basis size.

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { computeMolecularIntegrals } from "../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type BasisName } from "../src/chemistry/atoms.js";
import { runRHFSCF } from "../src/chemistry/hf-scf.js";
import { runMP2 } from "../src/chemistry/mp2.js";

const half = (104.52 / 2) * Math.PI / 180;
const x = 0.9572 * Math.sin(half);
const z = 0.9572 * Math.cos(half);
const atoms: Atom[] = [
  { symbol: "O", pos: [0, 0, 0] },
  { symbol: "H", pos: [ x, 0, z] },
  { symbol: "H", pos: [-x, 0, z] },
];

interface Row {
  basis: BasisName;
  nShells: number;
  nElectrons: number;
  E_HF_Ha: number;
  E_MP2_Ha: number;
  MP2_corr_mHa: number;
  HF_iters: number;
  total_seconds: number;
  pyscf_HF: number;
  pyscf_MP2: number;
}

const PYSCF_REFS: Record<BasisName, { hf: number; mp2: number }> = {
  "sto-3g":  { hf: -74.9629, mp2: -74.9989 },
  "cc-pvdz": { hf: -76.0267, mp2: -76.230 },
};

console.log("Phase E stage 2 — H₂O in STO-3G vs cc-pVDZ (full HF + MP2 stack)");
console.log("=".repeat(96));
console.log(
  `${"basis".padEnd(10)}  ${"shells".padStart(7)}  ${"E_HF".padStart(11)}  ${"E_MP2".padStart(11)}  ${"corr (mHa)".padStart(10)}  ${"vs PySCF".padStart(12)}`,
);

const rows: Row[] = [];
for (const basis of ["sto-3g", "cc-pvdz"] as const) {
  const t0 = performance.now();
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons, {
    maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
  });
  const mp2 = runMP2(hf, integrals);
  const wall = (performance.now() - t0) / 1000;
  const refs = PYSCF_REFS[basis];
  const dHF = (hf.energy - refs.hf) * 1000;
  const dMP2 = (mp2.totalEnergy - refs.mp2) * 1000;
  rows.push({
    basis, nShells: shells.length, nElectrons,
    E_HF_Ha: hf.energy,
    E_MP2_Ha: mp2.totalEnergy,
    MP2_corr_mHa: mp2.correlationEnergy * 1000,
    HF_iters: hf.iter,
    total_seconds: wall,
    pyscf_HF: refs.hf,
    pyscf_MP2: refs.mp2,
  });
  console.log(
    `${basis.padEnd(10)}  ${String(shells.length).padStart(7)}  ${hf.energy.toFixed(6).padStart(11)}  ${mp2.totalEnergy.toFixed(6).padStart(11)}  ${(mp2.correlationEnergy * 1000).toFixed(2).padStart(10)}  Δ_HF=${dHF.toFixed(2)} Δ_MP2=${dMP2.toFixed(2)}  [${wall.toFixed(2)}s, ${hf.iter} HF iters]`,
  );
}

const status: "pass" | "fail" =
  rows.every((r) => Math.abs(r.E_HF_Ha - r.pyscf_HF) < 1e-3) ? "pass" : "fail";

const artifact = {
  meta: {
    protocol: "E28-h2o-ccpvdz-publishable",
    hypothesis:
      "H₂O in cc-pVDZ HF + MP2 in a browser tab matches PySCF cc-pVDZ HF to <1 mHa, and MP2 captures ~4× more correlation (~205 mHa) than STO-3G's ~50 mHa, demonstrating that the chemistry pipeline scales transparently from minimal to standard basis sets.",
    passBar: "|E_HF − E_PySCF_HF| ≤ 1 mHa for both basis sets",
    seed: "E28_H2O_CCPVDZ",
    warmup: 0, trials: 1,
  },
  env: {
    runtime: `node ${process.version}`,
    platform: process.platform, arch: process.arch,
    timestamp: new Date().toISOString(),
    note: "cc-pVDZ uses 6 Cartesian d-functions (vs PySCF default 5 spherical); the extra (d_xx + d_yy + d_zz)/√3 redundant linear combination is variationally absorbed by Löwdin orthogonalization.",
  },
  rows,
  status,
  diagnosis: status === "pass"
    ? `H₂O in cc-pVDZ HF matches PySCF to ${Math.abs(rows[1]!.E_HF_Ha - rows[1]!.pyscf_HF) * 1000} mHa; MP2 captures ${(-rows[1]!.MP2_corr_mHa).toFixed(1)} mHa correlation (vs ${(-rows[0]!.MP2_corr_mHa).toFixed(1)} mHa in STO-3G — ~${((rows[1]!.MP2_corr_mHa) / (rows[0]!.MP2_corr_mHa)).toFixed(1)}× more).`
    : "HF energy out of PySCF tolerance for one or both basis sets.",
};

const outDir = resolve(process.cwd(), "experiments", "results", new Date().toISOString().slice(0, 10), "level-6");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "E28-h2o-ccpvdz-publishable.json");
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(`\n→ ${outPath}`);
console.log(`Status: ${status}`);
console.log(artifact.diagnosis);
