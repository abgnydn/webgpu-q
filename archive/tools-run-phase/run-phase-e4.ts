// Phase E stage 4 publishable artifact: HF / MP2 / CCSD / CCSD(T) /
// FCI table for the standard molecule set. Demonstrates that the
// perturbative-triples correction closes most of the residual
// CCSD−FCI gap, bringing every closed-shell molecule near
// equilibrium to within 0.5 mHa of FCI in STO-3G.
//
// (T) is structurally zero on H₂ and LiH s-only (NOCC < 3 or
// NVIRT < 3 — no triple excitations possible). For the larger
// systems, (T) is sub-mHa in STO-3G but already captures the
// dominant missing dynamic correlation.

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { computeMolecularIntegrals } from "../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../src/chemistry/atoms.js";
import { runRHFSCF } from "../src/chemistry/hf-scf.js";
import { runMP2 } from "../src/chemistry/mp2.js";
import { runCCSD } from "../src/chemistry/ccsd.js";
import { runCCSDT } from "../src/chemistry/ccsd-t.js";
import { buildMoleculeFCI } from "../src/chemistry/molecule-builder.js";

interface Mol {
  name: string;
  atoms: Atom[];
  pyscf_FCI: number;
  cached_FCI?: number;
}

const mols: Mol[] = [
  {
    name: "H2 (R=0.7414Å)",
    atoms: [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ],
    pyscf_FCI: -1.137270,
  },
  {
    name: "LiH s-only (R=1.595Å)",
    atoms: [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ],
    pyscf_FCI: -7.843394,
  },
  {
    name: "BeH2 full (R=1.34Å)",
    atoms: [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -1.34] },
      { symbol: "H",  pos: [0, 0,  1.34] },
    ],
    pyscf_FCI: -15.594861,
  },
  {
    name: "H2O (R=0.9572Å, ∠=104.52°)",
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
    pyscf_FCI: -75.012403,
  },
  {
    name: "CH4 (R=1.09Å, tetrahedral)",
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
    pyscf_FCI: -39.806036,
    cached_FCI: -39.806036,
  },
];

interface Row {
  molecule: string;
  basis: string;
  E_HF_Ha: number;
  E_MP2_Ha: number;
  E_CCSD_Ha: number;
  E_CCSDT_Ha: number;
  E_FCI_Ha: number;
  pyscf_FCI_Ha: number;
  total_correlation_mHa: number;        // E_FCI − E_HF
  CCSD_capture_pct: number;
  CCSDT_capture_pct: number;
  T_correction_mHa: number;             // E_CCSD(T) − E_CCSD
  CCSDT_minus_FCI_mHa: number;
  CCSD_iter: number;
  HF_seconds: number;
  MP2_seconds: number;
  CCSD_seconds: number;
  CCSDT_seconds: number;
  FCI_seconds: number | null;
}

console.log("Phase E stage 4 — HF / MP2 / CCSD / CCSD(T) / FCI table (STO-3G)");
console.log("=".repeat(124));
console.log(
  `${"molecule".padEnd(28)}  ${"E_HF".padStart(11)}  ${"E_MP2".padStart(11)}  ${"E_CCSD".padStart(11)}  ${"E_CCSD(T)".padStart(11)}  ${"E_FCI".padStart(11)}  ${"(T) mHa".padStart(8)}  ${"Δvs FCI".padStart(9)}`,
);
console.log("-".repeat(124));

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

  const t2 = performance.now();
  const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-10 });
  const tCCSD = (performance.now() - t2) / 1000;

  const t3 = performance.now();
  const ccsdt = runCCSDT(ccsd, hf, integrals);
  const tCCSDT = (performance.now() - t3) / 1000;

  let E_FCI: number;
  let tFCI: number | null = null;
  if (m.cached_FCI !== undefined) {
    E_FCI = m.cached_FCI;
  } else {
    const t4 = performance.now();
    const mol = buildMoleculeFCI(m.atoms);
    const fci = mol.fci({ maxIter: 200 });
    tFCI = (performance.now() - t4) / 1000;
    E_FCI = fci.energy;
  }

  const totalCorr_mHa = (E_FCI - hf.energy) * 1000;
  const ccsdCapture = (ccsd.correlationEnergy / (E_FCI - hf.energy)) * 100;
  const ccsdtCapture = ((ccsdt.totalEnergy - hf.energy) / (E_FCI - hf.energy)) * 100;
  const tCorr_mHa = ccsdt.tripleCorrection * 1000;
  const ccsdtMinusFCI = (ccsdt.totalEnergy - E_FCI) * 1000;

  rows.push({
    molecule: m.name,
    basis: "sto-3g",
    E_HF_Ha: hf.energy,
    E_MP2_Ha: mp2.totalEnergy,
    E_CCSD_Ha: ccsd.totalEnergy,
    E_CCSDT_Ha: ccsdt.totalEnergy,
    E_FCI_Ha: E_FCI,
    pyscf_FCI_Ha: m.pyscf_FCI,
    total_correlation_mHa: totalCorr_mHa,
    CCSD_capture_pct: ccsdCapture,
    CCSDT_capture_pct: ccsdtCapture,
    T_correction_mHa: tCorr_mHa,
    CCSDT_minus_FCI_mHa: ccsdtMinusFCI,
    CCSD_iter: ccsd.iter,
    HF_seconds: tHF,
    MP2_seconds: tMP2,
    CCSD_seconds: tCCSD,
    CCSDT_seconds: ccsdt.seconds,
    FCI_seconds: tFCI,
  });

  console.log(
    `${m.name.padEnd(28)}  ${hf.energy.toFixed(6).padStart(11)}  ${mp2.totalEnergy.toFixed(6).padStart(11)}  ${ccsd.totalEnergy.toFixed(6).padStart(11)}  ${ccsdt.totalEnergy.toFixed(6).padStart(11)}  ${E_FCI.toFixed(6).padStart(11)}  ${tCorr_mHa.toFixed(3).padStart(8)}  ${ccsdtMinusFCI.toFixed(4).padStart(9)}`,
  );
  void tCCSDT;
}

// Pass criteria:
//   - Variational ordering HF > MP2 > CCSD > CCSD(T) (≈ FCI), with
//     CCSD(T) allowed to dip slightly below FCI (non-variational).
//   - For molecules where (T) is non-zero (NOCC≥3 AND NVIRT≥3),
//     CCSD(T) is within 0.5 mHa of FCI.
//   - For (T)=0 molecules (H₂, LiH s-only), CCSD already equals FCI
//     so CCSD(T) trivially within 0.5 mHa.
const allOK = rows.every((r) => {
  const variational = r.E_HF_Ha > r.E_MP2_Ha &&
                      r.E_MP2_Ha > r.E_CCSD_Ha &&
                      r.E_CCSDT_Ha <= r.E_CCSD_Ha + 1e-12;
  const closure = Math.abs(r.CCSDT_minus_FCI_mHa) < 0.5;
  return variational && closure;
});
const status: "pass" | "fail" = allOK ? "pass" : "fail";

const artifact = {
  meta: {
    protocol: "E30-ccsdt-publishable",
    hypothesis:
      "Perturbative-triples CCSD(T) on top of converged CCSD closes the residual CCSD−FCI gap to within 0.5 mHa for every closed-shell molecule near equilibrium in STO-3G — chemistry-grade precision via spin-orbital Stanton-Bartlett (T) formula in a browser tab.",
    passBar:
      "every CCSD(T) run produces variationally-ordered HF > MP2 > CCSD ≥ CCSD(T) AND |CCSD(T) − FCI| < 0.5 mHa.",
    seed: "E30_CCSDT",
    warmup: 0, trials: 1,
  },
  env: {
    runtime: `node ${process.version}`,
    platform: process.platform, arch: process.arch,
    timestamp: new Date().toISOString(),
    method: "Spin-orbital CCSD(T): connected (W) + disconnected (V) terms via P(i/jk)P(a/bc), prefactors 1/36 + 1/4. One-shot non-iterative correction on converged CCSD T1, T2.",
    triplesCost: "O(N_o³ N_v³ · 9 · (N_o + N_v)) per molecule.",
  },
  rows,
  status,
  diagnosis: status === "pass"
    ? `CCSD(T) within ${Math.max(...rows.map(r => Math.abs(r.CCSDT_minus_FCI_mHa))).toFixed(3)} mHa of FCI for every molecule. (T) corrections range ${Math.min(...rows.map(r => r.T_correction_mHa)).toFixed(3)} to ${Math.max(...rows.map(r => r.T_correction_mHa)).toFixed(3)} mHa — sub-mHa in STO-3G as expected for closed-shell at equilibrium.`
    : `Pass criteria failed.`,
};

const outDir = resolve(process.cwd(), "experiments", "results", new Date().toISOString().slice(0, 10), "level-6");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "E30-ccsdt-publishable.json");
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(`\n→ ${outPath}`);
console.log(`Status: ${status}`);
console.log(artifact.diagnosis);
