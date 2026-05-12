// ─────────────────────────────────────────────────────────────
// E32 — CCSD(T) on GPU vs CPU cross-check.
//
// Validates the WebGPU port of the perturbative-triples kernel
// (src/shaders/ccsd-t.wgsl + src/chemistry/ccsd-t-gpu.ts) against
// the CPU reference (src/chemistry/ccsd-t.ts) on closed-shell
// molecules where (T) is non-trivial (NOCC ≥ 3, NVIRT ≥ 3 spin
// orbitals).
//
// Pass bar: |E_(T)_GPU − E_(T)_CPU| ≤ 5e-6 Ha per molecule.
// The bar accounts for f32 storage on GPU (CPU uses f64). Per-
// (i,j,k) partial sums are bounded in magnitude (µHa-mHa range),
// so f32→f64 reduction error sits at ~10⁻⁹ Ha for STO-3G systems;
// 5 µHa is a comfortable upper bound that catches real bugs while
// tolerating GPU floating-point ordering effects.
// ─────────────────────────────────────────────────────────────

import { initGPU } from "../../src/quantum.js";
import { captureEnv } from "../lib/env.js";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runCCSDT } from "../../src/chemistry/ccsd-t.js";
import { runCCSDT_GPU } from "../../src/chemistry/ccsd-t-gpu.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

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

const MOLECULES: {
  name: string;
  atoms: Atom[];
  nElectrons: number;
  basis?: "sto-3g" | "cc-pvdz";
}[] = [
  {
    name: "LiH/STO-3G",
    atoms: [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ],
    nElectrons: 4,
  },
  {
    name: "BeH₂/STO-3G",
    atoms: [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -1.34] },
      { symbol: "H",  pos: [0, 0,  1.34] },
    ],
    nElectrons: 6,
  },
  {
    name: "H₂O/STO-3G",
    atoms: H2O_GEOM,
    nElectrons: 10,
  },
  {
    name: "H₂O/cc-pVDZ",
    atoms: H2O_GEOM,
    nElectrons: 10,
    basis: "cc-pvdz",
  },
];

export interface E32Row {
  readonly molecule: string;
  readonly NOCC: number;
  readonly NVIRT: number;
  readonly cpuET: number;
  readonly gpuET: number;
  readonly deltaHartree: number;
  readonly cpuSeconds: number;
  readonly gpuSeconds: number;
  readonly gpuTotalSeconds: number;
  readonly speedup: number;
  readonly pass: boolean;
}

export async function runE32(): Promise<Artifact<E32Row>> {
  const PASS_BAR_HARTREE = 5e-6;
  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E32: no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E32Row[] = [];
  let firstFail: string | null = null;

  for (const mol of MOLECULES) {
    const { shells, nuclei } = moleculeToShellsNuclei(mol.atoms, mol.basis ?? "sto-3g");
    const integrals = computeMolecularIntegrals(shells, nuclei, {
      spherical: mol.basis === "cc-pvdz",
    });
    const hf = runRHFSCF(integrals, mol.nElectrons);
    if (!hf.converged) {
      rows.push({
        molecule: mol.name, NOCC: 0, NVIRT: 0,
        cpuET: NaN, gpuET: NaN, deltaHartree: NaN,
        cpuSeconds: 0, gpuSeconds: 0, gpuTotalSeconds: 0,
        speedup: 0, pass: false,
      });
      firstFail ??= `${mol.name}: HF did not converge`;
      continue;
    }
    const ccsd = runCCSD(hf, integrals, { tol: 1e-10, maxIter: 100 });
    if (!ccsd.converged) {
      rows.push({
        molecule: mol.name, NOCC: 2 * hf.nOccupied,
        NVIRT: 2 * integrals.n - 2 * hf.nOccupied,
        cpuET: NaN, gpuET: NaN, deltaHartree: NaN,
        cpuSeconds: 0, gpuSeconds: 0, gpuTotalSeconds: 0,
        speedup: 0, pass: false,
      });
      firstFail ??= `${mol.name}: CCSD did not converge`;
      continue;
    }

    const tCpuStart = performance.now();
    const cpuRes = runCCSDT(ccsd, hf, integrals);
    const cpuSeconds = (performance.now() - tCpuStart) / 1000;

    const gpuRes = await runCCSDT_GPU(ccsd, hf, integrals, device);
    const delta = Math.abs(gpuRes.tripleCorrection - cpuRes.tripleCorrection);
    const pass = delta <= PASS_BAR_HARTREE;
    const speedup = gpuRes.gpuSeconds > 0 ? cpuSeconds / gpuRes.gpuSeconds : 0;
    rows.push({
      molecule: mol.name,
      NOCC: 2 * hf.nOccupied,
      NVIRT: 2 * integrals.n - 2 * hf.nOccupied,
      cpuET: cpuRes.tripleCorrection,
      gpuET: gpuRes.tripleCorrection,
      deltaHartree: delta,
      cpuSeconds,
      gpuSeconds: gpuRes.gpuSeconds,
      gpuTotalSeconds: gpuRes.seconds,
      speedup,
      pass,
    });
    if (!pass && firstFail === null) {
      firstFail = `${mol.name}: |Δ| = ${delta.toExponential(2)} Ha > ${PASS_BAR_HARTREE.toExponential(0)} Ha`;
    }
  }

  const status: "pass" | "fail" = firstFail === null ? "pass" : "fail";
  const meta: ArtifactMeta = {
    protocol: "E32-ccsdt-gpu-vs-cpu",
    hypothesis: "WGSL port of (T) reproduces CPU CCSD(T) within f32 reduction noise.",
    passBar: `|E_(T)_GPU − E_(T)_CPU| ≤ ${PASS_BAR_HARTREE.toExponential(0)} Ha per molecule`,
    seed: "E32_CCSDT_GPU",
    warmup: 0,
    trials: 1,
  };
  const diagnosis = firstFail ?? "GPU CCSD(T) matches CPU on every molecule within the pass bar.";

  console.log(`[artifact:E32-ccsdt-gpu-vs-cpu] ${status} — ${diagnosis}`);
  for (const r of rows) {
    console.log(`  ${r.molecule}: CPU ${r.cpuET.toExponential(4)}, GPU ${r.gpuET.toExponential(4)}, |Δ| ${r.deltaHartree.toExponential(2)} Ha, ${r.cpuSeconds.toFixed(3)}s vs ${r.gpuSeconds.toFixed(3)}s (${r.speedup.toFixed(1)}×)`);
  }

  return { meta, env, rows, status, diagnosis };
}
