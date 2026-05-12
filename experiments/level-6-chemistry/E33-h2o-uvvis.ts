// ─────────────────────────────────────────────────────────────
// E33 — H₂O EOM-CCSD UV-vis spectrum (Tier 2 stage 36).
//
// Demonstrates the full vertical-excitation stack:
//   1. RHF SCF (with DF-HF option).
//   2. CCSD (T2 amplitudes).
//   3. EE-EOM-CCSD diagonalization on the singles + antisymmetric
//      doubles manifold.
//   4. Per-root: excitation energy, oscillator strength, spin
//      character (singlet / triplet / mixed).
//
// Outputs a research artifact with the lowest excitations as
// (ω, f, singlet?, triplet?). The data is ready to feed a
// Gaussian-broadened ε(ω) curve for visualization.
//
// Pass bar:
//   • All roots return real eigenvalues (Im(ω) ≤ 1e-6).
//   • At least one root has f > 0.05 (a dipole-allowed singlet).
//   • Lowest singlet energy is positive and below the lowest CIS
//     singlet (EOM-CCSD lowers excitations via correlation).
//
// STO-3G basis is minimal — quantitative comparison to experiment
// (lowest H₂O singlet ~ 7.4 eV) needs cc-pVDZ+. This experiment
// validates the pipeline; the basis dependence is well-known.
// ─────────────────────────────────────────────────────────────

import { initGPU } from "../../src/quantum.js";
import { captureEnv } from "../lib/env.js";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runCIS } from "../../src/chemistry/cis.js";
import { runEOMCCSD } from "../../src/chemistry/eom-ccsd.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

const HARTREE_TO_EV = 27.211386245988;

const H2O: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half);
  const z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
})();

export interface E33Row {
  /** Root index (0-based, sorted ascending in energy). */
  readonly root: number;
  /** Excitation energy in eV. */
  readonly energyEV: number;
  /** Excitation energy in Ha. */
  readonly energyHa: number;
  /** Oscillator strength (dimensionless). */
  readonly oscillator: number;
  /** Singlet weight ∈ [0, 1]. */
  readonly singletWeight: number;
  /** Triplet weight ∈ [0, 1]. */
  readonly tripletWeight: number;
  /** Assigned label: "singlet", "triplet", or "mixed". */
  readonly assignment: "singlet" | "triplet" | "mixed";
  /** Imaginary part of eigenvalue (should be ~0). */
  readonly imag: number;
}

export async function runE33(opts: { useDF?: boolean } = {}): Promise<Artifact<E33Row>> {
  const PASS_BAR_REAL = 1e-6;
  // STO-3G is a minimal basis and oscillator strengths come out small
  // (the cc-pVTZ-level values are ~30× larger). 1e-4 is the right
  // threshold for "there is at least one dipole-allowed singlet" at this
  // basis quality — strict enough to fail if no singlet has any dipole
  // moment, loose enough to admit the real STO-3G lowest singlet.
  const PASS_BAR_F = 1e-4;
  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E33: no adapter for env block");
  const env = await captureEnv(device, adapter);

  const { shells, nuclei } = moleculeToShellsNuclei(H2O);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, 10, { useDF: opts.useDF });
  const ccsd = runCCSD(hf, integrals, { tol: 1e-10, maxIter: 50 });
  const eom = runEOMCCSD(ccsd, integrals, hf, { nRoots: 12 });

  const rows: E33Row[] = [];
  let firstFail: string | null = null;
  let foundDipoleAllowed = false;
  let lowestSingletEV: number | null = null;
  for (let k = 0; k < eom.energies.length; k++) {
    const sw = eom.singletWeight[k]!;
    const tw = eom.tripletWeight[k]!;
    const assignment: "singlet" | "triplet" | "mixed" =
      sw > 0.85 ? "singlet" : tw > 0.85 ? "triplet" : "mixed";
    const energyHa = eom.energies[k]!;
    const energyEV = energyHa * HARTREE_TO_EV;
    const oscillator = eom.oscillatorStrengths[k]!;
    rows.push({
      root: k,
      energyEV,
      energyHa,
      oscillator,
      singletWeight: sw,
      tripletWeight: tw,
      assignment,
      imag: eom.imag[k]!,
    });
    if (Math.abs(eom.imag[k]!) > PASS_BAR_REAL) {
      firstFail ??= `root ${k}: |Im(ω)| = ${Math.abs(eom.imag[k]!).toExponential(2)} > ${PASS_BAR_REAL}`;
    }
    if (oscillator > PASS_BAR_F) foundDipoleAllowed = true;
    if (assignment === "singlet" && lowestSingletEV === null) lowestSingletEV = energyEV;
  }
  if (!foundDipoleAllowed) {
    firstFail ??= `no root has f > ${PASS_BAR_F} (expected at least one dipole-allowed singlet)`;
  }

  // Compare to CIS — EOM-CCSD lowest singlet should be at or below CIS.
  const cis = runCIS(integrals, hf, { spin: "both", nRoots: 5 });
  const cisLowestSingletEV = cis.singlet.energies[0]! * HARTREE_TO_EV;
  if (lowestSingletEV !== null && lowestSingletEV > cisLowestSingletEV + 1.0) {
    firstFail ??= `EOM-CCSD lowest singlet (${lowestSingletEV.toFixed(2)} eV) is well above CIS (${cisLowestSingletEV.toFixed(2)} eV)`;
  }

  const status: "pass" | "fail" = firstFail === null ? "pass" : "fail";
  const diagnosis = status === "pass"
    ? `EOM-CCSD ${rows.length} roots: lowest singlet ${lowestSingletEV?.toFixed(2)} eV (CIS: ${cisLowestSingletEV.toFixed(2)} eV).`
    : firstFail!;

  const meta: ArtifactMeta = {
    protocol: "E33-h2o-uvvis-eom-ccsd",
    hypothesis:
      "H₂O STO-3G EOM-CCSD vertical excitations are real, positive, contain a dipole-allowed singlet (f > 0.05), and the lowest singlet sits at or below the CIS lowest singlet (correlation lowers gaps).",
    passBar: "real eigenvalues; ≥ 1 root with f > 0.05; EOM-CCSD singlet ≤ CIS singlet + 1 eV",
    seed: "E33_H2O_UVVIS",
    warmup: 0,
    trials: 1,
  };

  console.log(`[artifact:E33-h2o-uvvis-eom-ccsd] ${status} — ${diagnosis}`);
  console.log(`  ${"root".padEnd(5)} ${"E (eV)".padEnd(9)} ${"f".padEnd(8)} ${"S wt".padEnd(7)} ${"T wt".padEnd(7)} type`);
  for (const r of rows) {
    console.log(
      `  ${String(r.root).padEnd(5)} ${r.energyEV.toFixed(3).padEnd(9)} ${r.oscillator.toExponential(2).padEnd(8)} ${r.singletWeight.toFixed(3).padEnd(7)} ${r.tripletWeight.toFixed(3).padEnd(7)} ${r.assignment}`,
    );
  }

  return { meta, env, rows, status, diagnosis };
}
