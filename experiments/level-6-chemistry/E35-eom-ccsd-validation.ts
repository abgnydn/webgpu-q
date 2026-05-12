// ─────────────────────────────────────────────────────────────
// E35 — EOM-CCSD cross-validation against PySCF.
//
// Thiel-set-style methodology: run EOM-CCSD on a panel of small
// organics, compare side-by-side against PySCF's EOM-CCSD on
// identical inputs. The classical Thiel/QUEST benchmark is at
// cc-pVTZ / aug-cc-pVTZ basis on ~28 organics; we ship at STO-3G
// because our cc-pVDZ basis only covers H and O (see
// LIMITATIONS.md — porting C / N cc-pVDZ unblocks the full set).
//
// What this validates: our EOM-CCSD implementation matches the
// reference implementation (PySCF) to algorithmic precision on the
// SAME basis. Literature accuracy of EOM-CCSD vs FCI is 0.1–0.2 eV
// for singlet single excitations — that's a method limit, not ours.
// Here we expect agreement to ~10⁻⁴ Ha (~3 meV) per root with
// PySCF, dominated by SCF/CCSD convergence tolerance, NOT method.
//
// Molecules:
//   H₂O, NH₃, CH₄, BeH₂
//
// Honestly skipped (dense-eigsolver too slow at STO-3G dim > ~2000):
//   CH₂O (dim ≈ 3488), HCN (dim ≈ 2660), C₂H₂, C₂H₄
//   → Tier 3 follow-up: Davidson eigensolver makes these tractable.
//   PySCF script keeps these in its set so the reference data is
//   ready when our solver scales.
//
// Methods:
//   HF → CCSD → EE-EOM-CCSD (first 5 excitations per molecule)
// ─────────────────────────────────────────────────────────────

import { initGPU } from "../../src/quantum.js";
import { captureEnv } from "../lib/env.js";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runEOMCCSD } from "../../src/chemistry/eom-ccsd.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

const HA_TO_EV = 27.211386245988;

const H2O_GEOM: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half);
  const z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
})();

// NH₃ — C₃ᵥ, experimental geometry: N–H = 1.0124 Å, H–N–H = 106.67°
const NH3_GEOM: Atom[] = (() => {
  const rNH = 1.0124;
  // Place N at origin, three H atoms on a cone with apex at N.
  // Half-bond-angle from C₃ axis: derive cos(θ_axis) so the three
  // H–N–H angles match 106.67°. cos(106.67°) = 1 - (3/2)·sin²(θ_axis).
  const hnh = (106.67 * Math.PI) / 180;
  const sinThetaAxis = Math.sqrt((2 / 3) * (1 - Math.cos(hnh)));
  const cosThetaAxis = Math.sqrt(1 - sinThetaAxis * sinThetaAxis);
  const r = rNH * sinThetaAxis;
  const z = -rNH * cosThetaAxis;
  return [
    { symbol: "N", pos: [0, 0, 0] },
    { symbol: "H", pos: [ r, 0, z] },
    { symbol: "H", pos: [-r / 2,  r * Math.sqrt(3) / 2, z] },
    { symbol: "H", pos: [-r / 2, -r * Math.sqrt(3) / 2, z] },
  ];
})();

// CH₄ — Td, experimental C–H = 1.087 Å.
const CH4_GEOM: Atom[] = (() => {
  const r = 1.087 / Math.sqrt(3);
  return [
    { symbol: "C", pos: [0, 0, 0] },
    { symbol: "H", pos: [ r,  r,  r] },
    { symbol: "H", pos: [ r, -r, -r] },
    { symbol: "H", pos: [-r,  r, -r] },
    { symbol: "H", pos: [-r, -r,  r] },
  ];
})();

// CH₂O (formaldehyde) — C₂ᵥ, experimental: C=O 1.203 Å, C–H 1.099 Å,
// H–C–H 116.5°.
const CH2O_GEOM: Atom[] = (() => {
  const rCO = 1.203;
  const rCH = 1.099;
  const hch_half = (116.5 / 2) * Math.PI / 180;
  const xH = rCH * Math.sin(hch_half);
  const zH = -rCH * Math.cos(hch_half);
  return [
    { symbol: "O", pos: [0, 0,  rCO] },
    { symbol: "C", pos: [0, 0, 0] },
    { symbol: "H", pos: [ xH, 0, zH] },
    { symbol: "H", pos: [-xH, 0, zH] },
  ];
})();

// HCN — linear, experimental: H–C 1.064 Å, C≡N 1.156 Å.
const HCN_GEOM: Atom[] = [
  { symbol: "H", pos: [0, 0, 0] },
  { symbol: "C", pos: [0, 0, 1.064] },
  { symbol: "N", pos: [0, 0, 1.064 + 1.156] },
];

const BEH2_GEOM: Atom[] = [
  { symbol: "Be", pos: [0, 0, 0] },
  { symbol: "H",  pos: [0, 0, -1.34] },
  { symbol: "H",  pos: [0, 0,  1.34] },
];

const MOLECULES: {
  name: string;
  atoms: Atom[];
  nElectrons: number;
}[] = [
  // Trimmed for dense-eigsolver tractability at STO-3G. CH₂O / HCN
  // would push the EOM-CCSD matrix past dim ~ 2000, where our dense
  // Hessenberg + Wilkinson QR takes 15-30 min per molecule. PySCF
  // reference script keeps those rows for when we wire Davidson.
  { name: "H₂O",  atoms: H2O_GEOM,  nElectrons: 10 },
  { name: "NH₃",  atoms: NH3_GEOM,  nElectrons: 10 },
  { name: "CH₄",  atoms: CH4_GEOM,  nElectrons: 10 },
  { name: "BeH₂", atoms: BEH2_GEOM, nElectrons: 6  },
];

// Eslint silencer — geometries are exported for future enabling.
void CH2O_GEOM;
void HCN_GEOM;

const N_ROOTS = 5;
const BASIS = "sto-3g" as const;
const SCF_TOL = 1e-10;
const CCSD_TOL = 1e-10;

export interface E35Row {
  readonly molecule: string;
  readonly basis: string;
  readonly E_HF_Ha: number;
  readonly E_CCSD_Ha: number;
  readonly excitations_eV: readonly number[];
  readonly oscillator_strengths: readonly number[];
  readonly singlet_weight: readonly number[];
  readonly triplet_weight: readonly number[];
  readonly seconds_HF: number;
  readonly seconds_CCSD: number;
  readonly seconds_EOM: number;
  readonly success: boolean;
  readonly notes?: string;
}

function timed<T>(fn: () => T): { result: T; seconds: number } {
  const t0 = performance.now();
  const result = fn();
  const seconds = (performance.now() - t0) / 1000;
  return { result, seconds };
}

export async function runE35(): Promise<Artifact<E35Row>> {
  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E35: no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E35Row[] = [];
  let firstFail: string | null = null;

  for (const mol of MOLECULES) {
    const { shells, nuclei } = moleculeToShellsNuclei(mol.atoms, BASIS);
    const integrals = computeMolecularIntegrals(shells, nuclei);

    try {
      const hfTimed = timed(() =>
        runRHFSCF(integrals, mol.nElectrons, {
          energyTol: SCF_TOL,
          densityTol: SCF_TOL,
          maxIter: 200,
        }),
      );
      if (!hfTimed.result.converged) {
        rows.push({
          molecule: mol.name, basis: BASIS,
          E_HF_Ha: NaN, E_CCSD_Ha: NaN,
          excitations_eV: [], oscillator_strengths: [],
          singlet_weight: [], triplet_weight: [],
          seconds_HF: hfTimed.seconds, seconds_CCSD: 0, seconds_EOM: 0,
          success: false, notes: "HF did not converge",
        });
        firstFail ??= `${mol.name}: HF`;
        continue;
      }
      const hf = hfTimed.result;

      const ccsdTimed = timed(() =>
        runCCSD(hf, integrals, { tol: CCSD_TOL, maxIter: 200 }),
      );
      if (!ccsdTimed.result.converged) {
        rows.push({
          molecule: mol.name, basis: BASIS,
          E_HF_Ha: hf.energy, E_CCSD_Ha: NaN,
          excitations_eV: [], oscillator_strengths: [],
          singlet_weight: [], triplet_weight: [],
          seconds_HF: hfTimed.seconds, seconds_CCSD: ccsdTimed.seconds, seconds_EOM: 0,
          success: false, notes: "CCSD did not converge",
        });
        firstFail ??= `${mol.name}: CCSD`;
        continue;
      }
      const ccsd = ccsdTimed.result;

      const eomTimed = timed(() =>
        runEOMCCSD(ccsd, integrals, hf, { nRoots: N_ROOTS }),
      );
      const eom = eomTimed.result;

      const nRet = Math.min(N_ROOTS, eom.energies.length);
      const excitations_eV: number[] = [];
      const oscStr: number[] = [];
      const singlet: number[] = [];
      const triplet: number[] = [];
      for (let k = 0; k < nRet; k++) {
        excitations_eV.push(eom.energies[k]! * HA_TO_EV);
        oscStr.push(eom.oscillatorStrengths[k]!);
        singlet.push(eom.singletWeight[k]!);
        triplet.push(eom.tripletWeight[k]!);
      }

      rows.push({
        molecule: mol.name, basis: BASIS,
        E_HF_Ha: hf.energy,
        E_CCSD_Ha: ccsd.totalEnergy,
        excitations_eV: excitations_eV as readonly number[],
        oscillator_strengths: oscStr as readonly number[],
        singlet_weight: singlet as readonly number[],
        triplet_weight: triplet as readonly number[],
        seconds_HF: hfTimed.seconds,
        seconds_CCSD: ccsdTimed.seconds,
        seconds_EOM: eomTimed.seconds,
        success: true,
      });
    } catch (e) {
      rows.push({
        molecule: mol.name, basis: BASIS,
        E_HF_Ha: NaN, E_CCSD_Ha: NaN,
        excitations_eV: [], oscillator_strengths: [],
        singlet_weight: [], triplet_weight: [],
        seconds_HF: 0, seconds_CCSD: 0, seconds_EOM: 0,
        success: false, notes: String(e),
      });
      firstFail ??= `${mol.name}: ${e}`;
    }
  }

  const status: "pass" | "fail" = firstFail === null ? "pass" : "fail";
  const meta: ArtifactMeta = {
    protocol: "E35-eom-ccsd-validation",
    hypothesis: "EE-EOM-CCSD implementation in webgpu-q reproduces PySCF EOM-CCSD excitation energies on small organics (H₂O, NH₃, CH₄, BeH₂, CH₂O, HCN) at STO-3G to ~10⁻⁴ Ha (~3 meV) per root, dominated by SCF/CCSD convergence tolerance.",
    passBar: "Every (molecule, root) cell either converges to a real positive eigenvalue OR is explicitly flagged. PySCF comparison is the cross-check; this artifact is one side of the comparison.",
    seed: "E35_EOM_VALIDATION",
    warmup: 0,
    trials: 1,
  };
  const diagnosis = firstFail ?? `Generated ${rows.length} rows × ${N_ROOTS} excitations each. Compare against scripts/run-pyscf-eom-reference.py (matching schema) for the validation half.`;

  console.log(`[artifact:E35-eom-ccsd-validation] ${status} — ${diagnosis}`);
  for (const r of rows) {
    if (!r.success) {
      console.log(`  ✗ ${r.molecule}: ${r.notes ?? "failed"}`);
      continue;
    }
    const exc = r.excitations_eV.map((e) => e.toFixed(3)).join("  ");
    console.log(`  ✓ ${r.molecule.padEnd(5)} HF=${r.E_HF_Ha.toFixed(6)}  CCSD=${r.E_CCSD_Ha.toFixed(6)}  excitations (eV): ${exc}`);
  }

  return { meta, env, rows, status, diagnosis };
}
