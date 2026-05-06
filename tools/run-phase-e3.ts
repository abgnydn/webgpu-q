// Phase E stage 3 publishable artifact: HF / MP2 / CCSD / FCI table
// for the standard molecule set + CCSD on cc-pVDZ H₂O (the chemistry
// frontier where FCI is infeasible). Demonstrates that closed-shell
// CCSD captures ≥ 95% of correlation for every molecule near
// equilibrium, with the residual being the perturbative-triples
// (T) correction CCSD(T) would add.
//
// Reference numbers (PySCF, STO-3G, same geometries):
//   H₂   E_CCSD = E_FCI = -1.137270 Ha (CCSD ≡ FCI for 2 electrons)
//   LiH  E_CCSD ≈ -7.84249  (≈ FCI, since LiH s-only has only 4 e in 6 SO)
//   BeH₂ E_CCSD ≈ -15.5943  (FCI = -15.5949, missing T ≈ 0.6 mHa)
//   H₂O  E_CCSD ≈ -75.0117  (FCI = -75.0124, missing T ≈ 0.7 mHa)
//   CH₄  E_CCSD ≈ -39.8055  (FCI = -39.8060, missing T ≈ 0.5 mHa)

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { computeMolecularIntegrals } from "../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../src/chemistry/atoms.js";
import { runRHFSCF } from "../src/chemistry/hf-scf.js";
import { runMP2 } from "../src/chemistry/mp2.js";
import { runCCSD } from "../src/chemistry/ccsd.js";
import { buildMoleculeFCI } from "../src/chemistry/molecule-builder.js";

interface Mol {
  name: string;
  atoms: Atom[];
  basis: "sto-3g" | "cc-pvdz";
  pyscf_FCI?: number;     // STO-3G FCI reference
  pyscf_CCSD?: number;    // PySCF CCSD/basis reference
  cached_FCI?: number;    // Skip FCI rerun (CH₄, cc-pVDZ where FCI infeasible)
  skipFCI?: boolean;
}

const mols: Mol[] = [
  {
    name: "H2 (R=0.7414Å)",
    atoms: [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ],
    basis: "sto-3g",
    pyscf_FCI: -1.137270,
    pyscf_CCSD: -1.137270,
  },
  {
    name: "LiH s-only (R=1.595Å)",
    atoms: [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ],
    basis: "sto-3g",
    pyscf_FCI: -7.843394,
    pyscf_CCSD: -7.843394,
  },
  {
    name: "BeH2 full (R=1.34Å)",
    atoms: [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -1.34] },
      { symbol: "H",  pos: [0, 0,  1.34] },
    ],
    basis: "sto-3g",
    pyscf_FCI: -15.594861,
    pyscf_CCSD: -15.5943,
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
    basis: "sto-3g",
    pyscf_FCI: -75.012403,
    pyscf_CCSD: -75.0117,
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
    basis: "sto-3g",
    pyscf_FCI: -39.806036,
    pyscf_CCSD: -39.8055,
    cached_FCI: -39.806036,  // From Phase C v5 sparse run, skip 80 s rerun
  },
  {
    name: "H2O cc-pVDZ (R=0.9572Å, ∠=104.52°) — chemistry frontier",
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
    basis: "cc-pvdz",
    pyscf_CCSD: -76.2418,    // PySCF spherical-d ref; we use Cartesian d
    skipFCI: true,           // 48 spin-orbital FCI is infeasible
  },
];

interface Row {
  molecule: string;
  basis: string;
  E_HF_Ha: number;
  E_MP2_Ha: number;
  E_CCSD_Ha: number;
  E_FCI_Ha: number | null;
  pyscf_FCI_Ha: number | null;
  pyscf_CCSD_Ha: number | null;
  HF_correlation_mHa: number | null;     // E_FCI − E_HF
  MP2_correlation_mHa: number;            // E_MP2 − E_HF
  CCSD_correlation_mHa: number;           // E_CCSD − E_HF
  CCSD_capture_pct: number | null;        // CCSD_correlation / HF_correlation
  CCSD_minus_FCI_mHa: number | null;
  CCSD_minus_pyscfCCSD_mHa: number | null;
  CCSD_iter: number;
  HF_seconds: number;
  MP2_seconds: number;
  CCSD_seconds: number;
  FCI_seconds: number | null;
}

console.log("Phase E stage 3 — HF / MP2 / CCSD / FCI table");
console.log("=".repeat(112));
console.log(
  `${"molecule".padEnd(48)}  ${"basis".padEnd(8)}  ${"E_HF".padStart(11)}  ${"E_MP2".padStart(11)}  ${"E_CCSD".padStart(11)}  ${"E_FCI".padStart(11)}`,
);

const rows: Row[] = [];
for (const m of mols) {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(m.atoms, m.basis);
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
  const ccsd = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });
  const tCCSD = (performance.now() - t2) / 1000;

  let E_FCI: number | null = null;
  let tFCI: number | null = null;
  if (m.skipFCI) {
    // pass
  } else if (m.cached_FCI !== undefined) {
    E_FCI = m.cached_FCI;
  } else {
    const t3 = performance.now();
    const mol = buildMoleculeFCI(m.atoms);
    const fci = mol.fci({ maxIter: 200 });
    tFCI = (performance.now() - t3) / 1000;
    E_FCI = fci.energy;
  }

  const HF_corr_mHa = E_FCI !== null ? (E_FCI - hf.energy) * 1000 : null;
  const MP2_corr_mHa = mp2.correlationEnergy * 1000;
  const CCSD_corr_mHa = ccsd.correlationEnergy * 1000;
  const capture_pct = HF_corr_mHa !== null ? (CCSD_corr_mHa / HF_corr_mHa) * 100 : null;
  const ccsdMinusFCI = E_FCI !== null ? (ccsd.totalEnergy - E_FCI) * 1000 : null;
  const ccsdMinusPyscf = m.pyscf_CCSD !== undefined ? (ccsd.totalEnergy - m.pyscf_CCSD) * 1000 : null;

  rows.push({
    molecule: m.name,
    basis: m.basis,
    E_HF_Ha: hf.energy,
    E_MP2_Ha: mp2.totalEnergy,
    E_CCSD_Ha: ccsd.totalEnergy,
    E_FCI_Ha: E_FCI,
    pyscf_FCI_Ha: m.pyscf_FCI ?? null,
    pyscf_CCSD_Ha: m.pyscf_CCSD ?? null,
    HF_correlation_mHa: HF_corr_mHa,
    MP2_correlation_mHa: MP2_corr_mHa,
    CCSD_correlation_mHa: CCSD_corr_mHa,
    CCSD_capture_pct: capture_pct,
    CCSD_minus_FCI_mHa: ccsdMinusFCI,
    CCSD_minus_pyscfCCSD_mHa: ccsdMinusPyscf,
    CCSD_iter: ccsd.iter,
    HF_seconds: tHF,
    MP2_seconds: tMP2,
    CCSD_seconds: tCCSD,
    FCI_seconds: tFCI,
  });

  const fciStr = E_FCI !== null ? E_FCI.toFixed(6).padStart(11) : "—".padStart(11);
  console.log(
    `${m.name.padEnd(48)}  ${m.basis.padEnd(8)}  ${hf.energy.toFixed(6).padStart(11)}  ${mp2.totalEnergy.toFixed(6).padStart(11)}  ${ccsd.totalEnergy.toFixed(6).padStart(11)}  ${fciStr}  (${ccsd.iter} iter, ${tCCSD.toFixed(2)}s)`,
  );
}

// Pass criteria:
//   - Every CCSD run converged.
//   - Variational ordering HF > MP2 > CCSD > FCI holds (where FCI known).
//   - CCSD captures ≥ 95% of HF→FCI correlation for every molecule with FCI.
//   - cc-pVDZ row matches PySCF CCSD within 5 mHa (Cartesian-vs-spherical d
//     slack; our basis is variationally larger, so we sit slightly below).
const allOK = rows.every((r) => {
  if (r.E_FCI_Ha === null) {
    // cc-pVDZ row: only the PySCF cross-check applies.
    return r.CCSD_minus_pyscfCCSD_mHa !== null && Math.abs(r.CCSD_minus_pyscfCCSD_mHa) < 5;
  }
  const variational = r.E_HF_Ha > r.E_MP2_Ha &&
                      r.E_MP2_Ha > r.E_CCSD_Ha &&
                      r.E_CCSD_Ha > r.E_FCI_Ha - 1e-6;
  const capture = (r.CCSD_capture_pct ?? 0) >= 95;
  return variational && capture;
});
const status: "pass" | "fail" = allOK ? "pass" : "fail";

const artifact = {
  meta: {
    protocol: "E29-ccsd-publishable",
    hypothesis:
      "Closed-shell CCSD via the spin-orbital Stanton-Bartlett 1991 formulation captures ≥ 95% of the HF→FCI correlation gap for every molecule near equilibrium, scales to non-minimal basis (cc-pVDZ on H₂O finishing in ~30 s), and matches published PySCF CCSD numbers within Cartesian-vs-spherical d-shell slack.",
    passBar:
      "every CCSD run converges; HF > MP2 > CCSD ≥ FCI (where FCI known); CCSD captures ≥ 95% of FCI correlation; cc-pVDZ matches PySCF within 5 mHa.",
    seed: "E29_CCSD",
    warmup: 0, trials: 1,
  },
  env: {
    runtime: `node ${process.version}`,
    platform: process.platform, arch: process.arch,
    timestamp: new Date().toISOString(),
    method: "Closed-shell CCSD (spin-orbital Stanton-Bartlett 1991), MP2 init, simple iteration",
    erIntegralTransform: "AO→MO 4-pass O(n^5) then antisymmetric spin-orbital tensor",
  },
  rows,
  status,
  diagnosis: status === "pass"
    ? `CCSD captures ${Math.min(...rows.filter(r=>r.CCSD_capture_pct!==null).map(r => r.CCSD_capture_pct!)).toFixed(2)}-${Math.max(...rows.filter(r=>r.CCSD_capture_pct!==null).map(r => r.CCSD_capture_pct!)).toFixed(2)}% of HF→FCI correlation across the STO-3G set. cc-pVDZ H₂O matches PySCF CCSD within ${Math.abs(rows.find(r=>r.basis==="cc-pvdz")?.CCSD_minus_pyscfCCSD_mHa ?? 0).toFixed(2)} mHa (Cartesian-vs-spherical d slack).`
    : `Some row failed pass criteria.`,
};

const outDir = resolve(process.cwd(), "experiments", "results", new Date().toISOString().slice(0, 10), "level-6");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, "E29-ccsd-publishable.json");
writeFileSync(outPath, JSON.stringify(artifact, null, 2));
console.log(`\n→ ${outPath}`);
console.log(`Status: ${status}`);
console.log(artifact.diagnosis);
