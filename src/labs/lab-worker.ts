// ─────────────────────────────────────────────────────────────
// labs/lab-worker.ts — off-main-thread compute for the three
// teaching labs. Each lab streams partial results back as they
// land so the page fills in row by row rather than freezing and
// then dumping everything at the end.
//
// A non-converged calculation is reported as an ERROR, never as a
// number. This is the whole pedagogical point of Lab 1: CCSD stops
// converging once the HOMO–LUMO gap closes, and a lab that silently
// printed those amplitudes as data would teach the opposite of what
// it should.
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "../chemistry/atoms.js";
import { moleculeToShellsNuclei } from "../chemistry/atoms.js";
import { computeMolecularIntegrals } from "../chemistry/cg-molecular.js";
import { runRHFSCF } from "../chemistry/hf-scf.js";
import { runMP2 } from "../chemistry/mp2.js";
import { runCCSD } from "../chemistry/ccsd.js";
import { runCCSDT } from "../chemistry/ccsd-t.js";
import { fciState } from "../viz/h2-fci-state.js";

export type LabRequest =
  | { kind: "dissociation"; seq: number }
  | { kind: "basis"; seq: number }
  | { kind: "ladder"; basis: BasisName; seq: number };

export type LabReply =
  | { kind: "row"; seq: number; lab: string; row: Record<string, number | string | boolean> }
  | { kind: "done"; seq: number; lab: string; seconds: number }
  | { kind: "error"; seq: number; lab: string; message: string };

const post = (m: LabReply) => (self as unknown as Worker).postMessage(m);

const SCF = { useDIIS: true, maxIter: 300, energyTol: 1e-11, densityTol: 1e-9 } as const;

function water(): Atom[] {
  const h = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(h), z = 0.9572 * Math.cos(h);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
}

function rhf(atoms: Atom[], basis: BasisName) {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons, SCF);
  if (!hf.converged) throw new Error(`RHF did not converge (${basis})`);
  return { hf, integrals, n: integrals.n };
}

// ── Lab 1: H₂ dissociation, RHF vs exact ─────────────────────
// The exact curve comes from `fciState`, a 16-dimensional full CI in
// the H₂ STO-3G minimal basis. That is H₂-SPECIFIC (it builds a dense
// 4-qubit Hamiltonian); it is not a general FCI solver and this lab
// does not generalise to N₂ or anything else.
function labDissociation(seq: number) {
  const t0 = performance.now();
  const Rs = [0.4, 0.5, 0.6, 0.7414, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0];
  for (const R of Rs) {
    const exact = fciState(R);
    let eRHF = Number.NaN;
    let rhfOk = false;
    try {
      const { hf } = rhf([
        { symbol: "H", pos: [0, 0, 0] },
        { symbol: "H", pos: [0, 0, R] },
      ], "sto-3g");
      eRHF = hf.energy;
      rhfOk = true;
    } catch { /* recorded as non-converged, not as a number */ }
    post({
      kind: "row", seq, lab: "dissociation",
      row: {
        R, eExact: exact.energy, eRHF, rhfOk,
        cG: exact.cG, cU: exact.cU,
        errMHa: rhfOk ? (eRHF - exact.energy) * 1000 : Number.NaN,
      },
    });
  }
  post({ kind: "done", seq, lab: "dissociation", seconds: (performance.now() - t0) / 1000 });
}

// ── Lab 2: basis-set convergence on H₂O ──────────────────────
function labBasis(seq: number) {
  const t0 = performance.now();
  const atoms = water();
  let prev: number | null = null;
  for (const basis of ["sto-3g", "cc-pvdz", "aug-cc-pvdz"] as const) {
    const { hf, n } = rhf(atoms, basis);
    post({
      kind: "row", seq, lab: "basis",
      row: {
        basis, n, energy: hf.energy,
        gainMHa: prev === null ? Number.NaN : (hf.energy - prev) * 1000,
      },
    });
    prev = hf.energy;
  }
  post({ kind: "done", seq, lab: "basis", seconds: (performance.now() - t0) / 1000 });
}

// ── Lab 3: the correlation ladder on H₂O ─────────────────────
function labLadder(basis: BasisName, seq: number) {
  const t0 = performance.now();
  const atoms = water();
  const { hf, integrals, n } = rhf(atoms, basis);
  post({ kind: "row", seq, lab: "ladder", row: { method: "RHF", energy: hf.energy, corrMHa: 0, pct: 0, n } });

  const mp2 = runMP2(hf, integrals);
  const ccsd = runCCSD(hf, integrals);
  if (!ccsd.converged) throw new Error("CCSD amplitudes did not converge");
  const t = runCCSDT(ccsd, hf, integrals);

  // CCSD(T) is the reference for "100% of recovered correlation" here —
  // it is the best number this ladder produces, not the exact answer.
  const total = t.totalEnergy - hf.energy;
  const rows: [string, number][] = [
    ["MP2", mp2.totalEnergy], ["CCSD", ccsd.totalEnergy], ["CCSD(T)", t.totalEnergy],
  ];
  for (const [method, energy] of rows) {
    const corr = energy - hf.energy;
    post({
      kind: "row", seq, lab: "ladder",
      row: { method, energy, corrMHa: corr * 1000, pct: (corr / total) * 100, n },
    });
  }
  post({ kind: "done", seq, lab: "ladder", seconds: (performance.now() - t0) / 1000 });
}

self.addEventListener("message", (ev: MessageEvent<LabRequest>) => {
  const req = ev.data;
  try {
    if (req.kind === "dissociation") labDissociation(req.seq);
    else if (req.kind === "basis") labBasis(req.seq);
    else labLadder(req.basis, req.seq);
  } catch (e) {
    post({
      kind: "error", seq: req.seq, lab: req.kind,
      message: e instanceof Error ? e.message : String(e),
    });
  }
});
