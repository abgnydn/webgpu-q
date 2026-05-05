// Phase D publishable artifact: HF SCF energies for the standard
// molecule set, compared to PySCF references and to FCI to surface
// the correlation energy that post-HF methods need to capture.

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { computeMolecularIntegrals } from "../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../src/chemistry/atoms.js";
import { runRHFSCF } from "../src/chemistry/hf-scf.js";
import { buildMoleculeFCI } from "../src/chemistry/molecule-builder.js";

interface Mol {
  name: string;
  atoms: Atom[];
  pyscf_HF: number;
  pyscf_FCI: number;
}

const mols: Mol[] = [
  {
    name: "H2",
    atoms: [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ],
    pyscf_HF: -1.116694,
    pyscf_FCI: -1.137270,
  },
  {
    name: "LiH (s-only basis)",
    atoms: [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ],
    pyscf_HF: -7.8042,
    pyscf_FCI: -7.8434,
  },
  {
    name: "BeH2 (full STO-3G)",
    atoms: [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -1.34] },
      { symbol: "H",  pos: [0, 0,  1.34] },
    ],
    pyscf_HF: -15.5594,
    pyscf_FCI: -15.5949,
  },
  {
    name: "H2O",
    atoms: (() => {
      const half = (104.52 / 2) * Math.PI / 180;
      const x = 0.9572 * Math.sin(half);
      const z = 0.9572 * Math.cos(half);
      return [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ x, 0, z] },
        { symbol: "H", pos: [-x, 0, z] },
      ] as Atom[];
    })(),
    pyscf_HF: -74.9629,
    pyscf_FCI: -75.0124,
  },
  {
    name: "CH4",
    atoms: (() => {
      const r3 = 1.09 / Math.sqrt(3);
      return [
        { symbol: "C", pos: [0, 0, 0] },
        { symbol: "H", pos: [ r3,  r3,  r3] },
        { symbol: "H", pos: [ r3, -r3, -r3] },
        { symbol: "H", pos: [-r3,  r3, -r3] },
        { symbol: "H", pos: [-r3, -r3,  r3] },
      ] as Atom[];
    })(),
    pyscf_HF: -39.7267,
    pyscf_FCI: -39.8068,
  },
];

interface Row {
  molecule: string;
  nElectrons: number;
  nSpatial: number;
  E_HF_Ha: number;
  E_HF_PySCF_Ha: number;
  HF_delta_mHa: number;
  HF_iterations: number;
  HF_converged: boolean;
  E_FCI_Ha: number;
  E_FCI_PySCF_Ha: number;
  FCI_delta_mHa: number;
  correlation_mHa: number;        // E_HF − E_FCI
  HF_seconds: number;
  FCI_seconds: number;
}

console.log("Phase D — Hartree-Fock SCF + correlation table");
console.log("=".repeat(82));
console.log(
  `${"molecule".padEnd(22)}  ${"E_HF".padStart(11)}  ${"PySCF".padStart(11)}  ` +
  `${"Δ_HF".padStart(7)}  ${"E_FCI".padStart(11)}  ${"corr".padStart(7)}`,
);

const rows: Row[] = [];
for (const m of mols) {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);

  const t0 = performance.now();
  const hf = runRHFSCF(integrals, nElectrons, {
    maxIter: 500, damping: 0.3, energyTol: 1e-10, densityTol: 1e-8,
  });
  const tHF = (performance.now() - t0) / 1000;

  const t1 = performance.now();
  const mol = buildMoleculeFCI(m.atoms);
  const fci = mol.fci({ maxIter: 200 });
  const tFCI = (performance.now() - t1) / 1000;

  const dHF_mHa = (hf.energy - m.pyscf_HF) * 1000;
  const dFCI_mHa = (fci.energy - m.pyscf_FCI) * 1000;
  const corr_mHa = (hf.energy - fci.energy) * 1000;

  rows.push({
    molecule: m.name,
    nElectrons,
    nSpatial: shells.length,
    E_HF_Ha: hf.energy,
    E_HF_PySCF_Ha: m.pyscf_HF,
    HF_delta_mHa: dHF_mHa,
    HF_iterations: hf.iter,
    HF_converged: hf.converged,
    E_FCI_Ha: fci.energy,
    E_FCI_PySCF_Ha: m.pyscf_FCI,
    FCI_delta_mHa: dFCI_mHa,
    correlation_mHa: corr_mHa,
    HF_seconds: tHF,
    FCI_seconds: tFCI,
  });

  console.log(
    `${m.name.padEnd(22)}  ${hf.energy.toFixed(6).padStart(11)}  ${m.pyscf_HF.toFixed(6).padStart(11)}  ` +
    `${dHF_mHa.toFixed(3).padStart(7)}  ${fci.energy.toFixed(6).padStart(11)}  ${corr_mHa.toFixed(2).padStart(7)} mHa`,
  );
}

const allHFOK = rows.every((r) => Math.abs(r.HF_delta_mHa) <= 1.0);
const allFCIOK = rows.every((r) => Math.abs(r.FCI_delta_mHa) <= 1.0);
const status: "pass" | "fail" = allHFOK && allFCIOK ? "pass" : "fail";

const artifact = {
  meta: {
    protocol: "E26-hf-scf-publishable",
    hypothesis:
      "Closed-shell RHF SCF on the standard molecule set (H₂, LiH s-only, BeH₂ full, H₂O, CH₄) matches PySCF STO-3G HF energies to ≤ 1 mHa, and the correlation gap E_HF − E_FCI matches canonical literature values (~20-80 mHa).",
    passBar: "|E_HF − E_PySCF_HF| ≤ 1 mHa AND |E_FCI − E_PySCF_FCI| ≤ 1 mHa for every molecule",
    seed: "E26_HF_SCF",
    warmup: 0, trials: 1,
  },
  env: {
    runtime: `node ${process.version}`,
    platform: process.platform, arch: process.arch,
    timestamp: new Date().toISOString(),
    method: "RHF Roothaan-Hall iteration with α=0.3 damping",
    convergence: { energyTol: 1e-10, densityTol: 1e-8, maxIter: 500 },
  },
  rows,
  status,
  diagnosis: status === "pass"
    ? `All ${rows.length} molecules: HF matches PySCF to <1 mHa, FCI matches PySCF to <1 mHa. Correlation energies (mHa): ${rows.map(r => `${r.molecule}=${r.correlation_mHa.toFixed(1)}`).join(", ")}.`
    : `Out-of-spec: HF [${rows.filter(r => Math.abs(r.HF_delta_mHa) > 1).map(r => r.molecule).join(",")}], FCI [${rows.filter(r => Math.abs(r.FCI_delta_mHa) > 1).map(r => r.molecule).join(",")}].`,
};

const outDir = resolve(process.cwd(), "experiments", "results", new Date().toISOString().slice(0, 10), "level-6");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "E26-hf-scf-publishable.json");
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(`\n→ ${outPath}`);
console.log(`Status: ${status}`);
