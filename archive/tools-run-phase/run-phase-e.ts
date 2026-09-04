// Phase E stage 1 publishable artifact: HF / MP2 / FCI table
// for the standard molecule set, showing the correlation energy
// captured by MP2 vs the gold-standard FCI.

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { computeMolecularIntegrals } from "../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../src/chemistry/atoms.js";
import { runRHFSCF } from "../src/chemistry/hf-scf.js";
import { runMP2 } from "../src/chemistry/mp2.js";
import { buildMoleculeFCI } from "../src/chemistry/molecule-builder.js";

interface Mol {
  name: string;
  atoms: Atom[];
  pyscf_HF: number;
  // E_FCI from PySCF references; for CH₄ we use the value from
  // Phase C v5 (matches PySCF to 0.76 mHa).
  pyscf_FCI: number;
  // Optional cached FCI when rerunning would be too slow (CH₄).
  cached_FCI?: number;
}

const mols: Mol[] = [
  {
    name: "H2 (R=0.7414Å)",
    atoms: [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ],
    pyscf_HF: -1.116694,
    pyscf_FCI: -1.137270,
  },
  {
    name: "LiH s-only (R=1.595Å)",
    atoms: [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ],
    pyscf_HF: -7.8042,
    pyscf_FCI: -7.8434,
  },
  {
    name: "BeH2 full (R=1.34Å)",
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
    // Cached from Phase C v5 sparse-FCI run; rebuilding here would take 80+ s.
    cached_FCI: -39.806036,
  },
];

interface Row {
  molecule: string;
  E_HF_Ha: number;
  E_MP2_Ha: number;
  E_FCI_Ha: number;
  pyscf_FCI_Ha: number;
  HF_correlation_mHa: number;       // E_FCI − E_HF (negative)
  MP2_correlation_mHa: number;      // E_MP2 − E_HF (negative)
  capture_pct: number;              // MP2_correlation / HF_correlation × 100
  HF_seconds: number;
  MP2_seconds: number;
  FCI_seconds: number | null;
}

console.log("Phase E stage 1 — HF / MP2 / FCI table (STO-3G)");
console.log("=".repeat(96));
console.log(
  `${"molecule".padEnd(22)}  ${"E_HF".padStart(11)}  ${"E_MP2".padStart(11)}  ${"E_FCI".padStart(11)}  ${"capture".padStart(8)}`,
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
  const mp2 = runMP2(hf, integrals);
  const tMP2 = (performance.now() - t1) / 1000;

  let E_FCI: number;
  let tFCI: number | null;
  if (m.cached_FCI !== undefined) {
    E_FCI = m.cached_FCI;
    tFCI = null;
  } else {
    const t2 = performance.now();
    const mol = buildMoleculeFCI(m.atoms);
    const fci = mol.fci({ maxIter: 200 });
    tFCI = (performance.now() - t2) / 1000;
    E_FCI = fci.energy;
  }

  const HF_corr = (E_FCI - hf.energy) * 1000;
  const MP2_corr = mp2.correlationEnergy * 1000;
  const capture = (MP2_corr / HF_corr) * 100;

  rows.push({
    molecule: m.name,
    E_HF_Ha: hf.energy,
    E_MP2_Ha: mp2.totalEnergy,
    E_FCI_Ha: E_FCI,
    pyscf_FCI_Ha: m.pyscf_FCI,
    HF_correlation_mHa: HF_corr,
    MP2_correlation_mHa: MP2_corr,
    capture_pct: capture,
    HF_seconds: tHF,
    MP2_seconds: tMP2,
    FCI_seconds: tFCI,
  });

  console.log(
    `${m.name.padEnd(22)}  ${hf.energy.toFixed(6).padStart(11)}  ${mp2.totalEnergy.toFixed(6).padStart(11)}  ${E_FCI.toFixed(6).padStart(11)}  ${capture.toFixed(1).padStart(6)}%`,
  );
}

const allOK = rows.every((r) =>
  r.HF_correlation_mHa < 0 &&
  r.MP2_correlation_mHa < 0 &&
  r.MP2_correlation_mHa > r.HF_correlation_mHa &&  // MP2 doesn't over-correct
  r.capture_pct > 50 && r.capture_pct < 100,
);
const status: "pass" | "fail" = allOK ? "pass" : "fail";

const artifact = {
  meta: {
    protocol: "E27-mp2-publishable",
    hypothesis:
      "Closed-shell MP2 on top of RHF reproduces canonical correlation-capture properties: variational ordering HF > HF+MP2 > FCI for every molecule, and MP2 captures 50-80% of the FCI correlation gap.",
    passBar:
      "for every molecule: variational ordering holds AND MP2 captures 50% < ratio < 100% of the HF→FCI correlation gap",
    seed: "E27_MP2",
    warmup: 0, trials: 1,
  },
  env: {
    runtime: `node ${process.version}`,
    platform: process.platform, arch: process.arch,
    timestamp: new Date().toISOString(),
    method: "Closed-shell MP2 on RHF MOs",
    erIntegralTransform: "AO→MO 4-pass O(n^5)",
  },
  rows,
  status,
  diagnosis: status === "pass"
    ? `MP2 captures ${Math.min(...rows.map(r => r.capture_pct)).toFixed(1)}-${Math.max(...rows.map(r => r.capture_pct)).toFixed(1)}% of FCI correlation across the standard set; variational ordering HF > MP2 > FCI holds for every molecule.`
    : `Variational or capture-ratio violation in: ${rows.filter(r => !(r.HF_correlation_mHa < 0 && r.MP2_correlation_mHa < 0 && r.MP2_correlation_mHa > r.HF_correlation_mHa && r.capture_pct > 50 && r.capture_pct < 100)).map(r => r.molecule).join(", ")}.`,
};

const outDir = resolve(process.cwd(), "experiments", "results", new Date().toISOString().slice(0, 10), "level-6");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "E27-mp2-publishable.json");
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(`\n→ ${outPath}`);
console.log(`Status: ${status}`);
console.log(artifact.diagnosis);
