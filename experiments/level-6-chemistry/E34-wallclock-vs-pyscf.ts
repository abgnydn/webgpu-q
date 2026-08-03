// ─────────────────────────────────────────────────────────────
// E34 — Wall-clock comparison against PySCF on identical inputs.
//
// Generates the webgpu-q side of an apples-to-apples performance
// benchmark vs PySCF / gpu4pyscf. Same molecules, same basis sets,
// same convergence thresholds, same algorithmic steps (HF → MP2 →
// CCSD → CCSD(T)).
//
// Mirror PySCF reference: scripts/run-pyscf-reference.py emits a
// matching JSON schema. The combined comparison lives in
// experiments/results/<date>/level-6/E34-wallclock-comparison.json
// (merged offline since PySCF runs outside the browser).
//
// What this experiment answers — the reviewer's #1 question:
//   "Is webgpu-q actually fast compared to PySCF on the same input?"
//
// Honest expected outcome:
//   - PySCF wins on CPU because BLAS / NumPy is highly optimized.
//   - gpu4pyscf wins on big systems with cuBLAS / mature CUDA.
//   - We win on: zero install, zero startup, no Python interpreter
//     overhead, and the cc-pVDZ CCSD(T) WGSL kernel.
//
// Pass bar: no failures (all energies converge). Performance is
// informational, not a pass/fail gate — wall-clock differences are
// honest data points, not bugs.
// ─────────────────────────────────────────────────────────────

import { initGPU } from "../../src/quantum.js";
import { captureEnv } from "../lib/env.js";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runMP2 } from "../../src/chemistry/mp2.js";
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
}[] = [
  {
    name: "H₂",
    atoms: [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ],
    nElectrons: 2,
  },
  {
    name: "LiH",
    atoms: [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ],
    nElectrons: 4,
  },
  {
    name: "BeH₂",
    atoms: [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -1.34] },
      { symbol: "H",  pos: [0, 0,  1.34] },
    ],
    nElectrons: 6,
  },
  {
    name: "H₂O",
    atoms: H2O_GEOM,
    nElectrons: 10,
  },
];

const BASES: Array<"sto-3g" | "cc-pvdz"> = ["sto-3g", "cc-pvdz"];
const SCF_TOL = 1e-10;
const CCSD_TOL = 1e-10;

export interface E34Row {
  readonly molecule: string;
  readonly basis: string;
  readonly method: string;
  readonly seconds: number;
  readonly energy_Ha: number;
  readonly success: boolean;
  readonly notes?: string;
}

function timed<T>(fn: () => T): { result: T; seconds: number } {
  const t0 = performance.now();
  const result = fn();
  const seconds = (performance.now() - t0) / 1000;
  return { result, seconds };
}

async function timedAsync<T>(fn: () => Promise<T>): Promise<{ result: T; seconds: number }> {
  const t0 = performance.now();
  const result = await fn();
  const seconds = (performance.now() - t0) / 1000;
  return { result, seconds };
}

export async function runE34(): Promise<Artifact<E34Row>> {
  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E34: no adapter for env block");
  const env = await captureEnv(device, adapter);

  const rows: E34Row[] = [];
  let firstFail: string | null = null;

  for (const mol of MOLECULES) {
    for (const basis of BASES) {
      const label = `${mol.name}/${basis}`;

      // cc-pVDZ is wired for H, He, Li, Be, C, N, O and F (see the cc-pvdz
      // branch of atoms.ts). This gate used to skip anything that was not H or
      // O — a belief that went stale once the basis tables were completed — and
      // silently dropped 10 rows (LiH and BeH₂ cc-pVDZ × 5 methods) from the
      // artifact behind the README's headline comparison table.
      const CCPVDZ_ELEMENTS = new Set(["H", "He", "Li", "Be", "C", "N", "O", "F"]);
      const symbols = new Set(mol.atoms.map((a) => a.symbol));
      const ccPvdzMissing = basis === "cc-pvdz"
        && [...symbols].some((s) => !CCPVDZ_ELEMENTS.has(s));
      if (ccPvdzMissing) {
        for (const method of ["HF", "MP2", "CCSD", "CCSD(T)-CPU", "CCSD(T)-GPU"]) {
          rows.push({
            molecule: mol.name, basis, method,
            seconds: 0, energy_Ha: NaN, success: true,
            notes: "skipped — cc-pVDZ basis only wired for H and O (see LIMITATIONS.md)",
          });
        }
        continue;
      }

      const { shells, nuclei } = moleculeToShellsNuclei(mol.atoms, basis);
      const integrals = computeMolecularIntegrals(shells, nuclei, {
        spherical: basis === "cc-pvdz",
      });

      // HF
      let hf;
      try {
        const t = timed(() =>
          runRHFSCF(integrals, mol.nElectrons, {
            energyTol: SCF_TOL,
            densityTol: SCF_TOL,
            maxIter: 200,
          }),
        );
        hf = t.result;
        rows.push({
          molecule: mol.name, basis, method: "HF",
          seconds: t.seconds,
          energy_Ha: hf.energy,
          success: hf.converged,
          notes: hf.converged ? undefined : "SCF did not converge",
        });
        if (!hf.converged) { firstFail ??= `${label} HF`; continue; }
      } catch (e) {
        rows.push({ molecule: mol.name, basis, method: "HF", seconds: 0, energy_Ha: NaN, success: false, notes: String(e) });
        firstFail ??= `${label} HF threw`;
        continue;
      }

      // MP2
      try {
        const t = timed(() => runMP2(hf!, integrals));
        rows.push({
          molecule: mol.name, basis, method: "MP2",
          seconds: t.seconds,
          energy_Ha: t.result.totalEnergy,
          success: true,
        });
      } catch (e) {
        rows.push({ molecule: mol.name, basis, method: "MP2", seconds: 0, energy_Ha: NaN, success: false, notes: String(e) });
        firstFail ??= `${label} MP2 threw`;
      }

      // CCSD
      let ccsd;
      try {
        const t = timed(() => runCCSD(hf!, integrals, { tol: CCSD_TOL, maxIter: 200 }));
        ccsd = t.result;
        rows.push({
          molecule: mol.name, basis, method: "CCSD",
          seconds: t.seconds,
          energy_Ha: ccsd.totalEnergy,
          success: ccsd.converged,
          notes: ccsd.converged ? undefined : "CCSD did not converge",
        });
        if (!ccsd.converged) { firstFail ??= `${label} CCSD`; continue; }
      } catch (e) {
        rows.push({ molecule: mol.name, basis, method: "CCSD", seconds: 0, energy_Ha: NaN, success: false, notes: String(e) });
        firstFail ??= `${label} CCSD threw`;
        continue;
      }

      // CCSD(T) CPU — skip for cc-pVDZ unless it's H2 (cheap)
      // because CPU (T) at cc-pVDZ is minutes per molecule.
      const runCpuT = basis === "sto-3g" || mol.name === "H₂";
      if (runCpuT) {
        try {
          const t = timed(() => runCCSDT(ccsd!, hf!, integrals));
          rows.push({
            molecule: mol.name, basis, method: "CCSD(T)-CPU",
            seconds: t.seconds,
            energy_Ha: t.result.totalEnergy,
            success: true,
          });
        } catch (e) {
          rows.push({ molecule: mol.name, basis, method: "CCSD(T)-CPU", seconds: 0, energy_Ha: NaN, success: false, notes: String(e) });
          firstFail ??= `${label} (T)-CPU threw`;
        }
      } else {
        rows.push({
          molecule: mol.name, basis, method: "CCSD(T)-CPU",
          seconds: 0, energy_Ha: NaN, success: true,
          notes: "skipped — CPU (T) at cc-pVDZ is minutes; GPU only",
        });
      }

      // CCSD(T) GPU
      try {
        const t = await timedAsync(() => runCCSDT_GPU(ccsd!, hf!, integrals, device));
        rows.push({
          molecule: mol.name, basis, method: "CCSD(T)-GPU",
          seconds: t.seconds,
          energy_Ha: t.result.totalEnergy,
          success: true,
        });
      } catch (e) {
        rows.push({ molecule: mol.name, basis, method: "CCSD(T)-GPU", seconds: 0, energy_Ha: NaN, success: false, notes: String(e) });
        firstFail ??= `${label} (T)-GPU threw`;
      }
    }
  }

  const status: "pass" | "fail" = firstFail === null ? "pass" : "fail";
  const meta: ArtifactMeta = {
    protocol: "E34-wallclock-vs-pyscf",
    hypothesis: "Generates webgpu-q side of an apples-to-apples wall-clock and energy comparison vs PySCF on 4 small molecules × 2 basis sets × {HF, MP2, CCSD, CCSD(T) CPU, CCSD(T) GPU}. PySCF reference produced separately via scripts/run-pyscf-reference.py with matching schema.",
    passBar: "Every method on every (molecule, basis) cell either converges OR is explicitly skipped. Performance numbers are informational, not pass/fail.",
    // Disclosure, using the same field the swarm artifacts already carry. This
    // experiment does NOT follow the repo's 5-warmup/20-trial protocol: `timed()`
    // is a single un-warmed performance.now() bracket per cell. On the fastest
    // cells (H₂/STO-3G HF at 300 µs) the browser clock's 100 µs quantization is
    // ~±17% before any run-to-run variance, so the derived speedups are
    // order-of-magnitude indicative, not precise. The slow cells that carry the
    // loss rows (CCSD H₂O cc-pVDZ, 41.6 s) dominate their own quantization.
    timingNote: "Single representative run, no warmup, 1 trial, 100 µs timer floor — indicative, NOT the 5-warmup/20-trial protocol used for the GPU-kernel experiments.",
    seed: "E34_WALLCLOCK",
    warmup: 0,
    trials: 1,
  };
  const diagnosis = firstFail ?? `Generated ${rows.length} timing/energy rows. To complete the comparison, run scripts/run-pyscf-reference.py with PySCF 2.13 and merge the resulting JSON with this artifact (matching schema).`;

  console.log(`[artifact:E34-wallclock-vs-pyscf] ${status} — ${diagnosis}`);
  for (const r of rows) {
    const flag = r.success ? "✓" : "✗";
    const energy = isNaN(r.energy_Ha) ? "       (skip)" : r.energy_Ha.toFixed(8);
    console.log(`  ${flag} ${r.molecule.padEnd(5)} ${r.basis.padEnd(8)} ${r.method.padEnd(14)} ${r.seconds.toFixed(4).padStart(10)} s   E = ${energy} Ha${r.notes ? "   " + r.notes : ""}`);
  }

  return { meta, env, rows, status, diagnosis };
}
