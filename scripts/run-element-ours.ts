// Element-validation — webgpu-q side.
//
// Reads the geometry + reference table emitted by
// scripts/run-element-reference.py and recomputes every cell with our own
// engine, under the MATCHING d-convention (Gate 0.1 of
// docs/RUN-PLAN-24H-ELEMENTS.md):
//
//   PySCF spherical (default, 5d) <-> computeMolecularIntegrals(..., {spherical:true})
//   PySCF cartesian (cart=True)   <-> computeMolecularIntegrals(...)   [default path]
//
// A non-converged SCF is a FAILURE, never a datum.
//
// Usage:  npx --yes tsx scripts/run-element-ours.ts <ref.json> <out.json>
import { readFileSync, writeFileSync } from "node:fs";
import { computeMolecularIntegrals } from "../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../src/chemistry/atoms.js";
import { runRHFSCF } from "../src/chemistry/hf-scf.js";

interface RefRow {
  element: string; molecule: string; basis: string; convention: string;
  atoms: { symbol: string; pos: number[] }[];
  ok: boolean; E_HF?: number; nao?: number; note?: string;
}

const [, , refPath, outPath] = process.argv;
if (!refPath || !outPath) {
  console.error("usage: tsx scripts/run-element-ours.ts <ref.json> <out.json>");
  process.exit(2);
}

const ref = JSON.parse(readFileSync(refPath, "utf8")) as { rows: RefRow[] };
const rows: Record<string, unknown>[] = [];

for (const r of ref.rows) {
  const atoms: Atom[] = r.atoms.map((a) => ({
    symbol: a.symbol as Atom["symbol"],
    pos: [a.pos[0]!, a.pos[1]!, a.pos[2]!],
  }));
  const spherical = r.convention === "spherical";
  const out: Record<string, unknown> = {
    element: r.element, molecule: r.molecule, basis: r.basis,
    convention: r.convention, ref_ok: r.ok, E_ref: r.E_HF ?? null,
  };
  try {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, r.basis);
    const integrals = computeMolecularIntegrals(shells, nuclei, { spherical });
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 300, energyTol: 1e-11, densityTol: 1e-9,
    });
    if (!hf.converged) {
      out["ok"] = false;
      out["note"] = "our SCF did not converge";
    } else {
      out["ok"] = true;
      out["E_ours"] = hf.energy;
      out["n"] = integrals.n;
      if (r.ok && r.E_HF !== undefined) {
        out["delta_Ha"] = hf.energy - r.E_HF;
        out["delta_mHa"] = (hf.energy - r.E_HF) * 1000;
        out["nao_match"] = r.nao === integrals.n;
      }
    }
  } catch (e) {
    out["ok"] = false;
    out["note"] = e instanceof Error ? e.message : String(e);
  }
  rows.push(out);

  const d = out["delta_mHa"];
  const tag = out["ok"] === true
    ? (typeof d === "number" ? `${d >= 0 ? "+" : ""}${d.toFixed(6)} mHa` : "no ref")
    : `FAIL: ${String(out["note"]).slice(0, 60)}`;
  const nm = out["nao_match"] === false ? "  [nao MISMATCH]" : "";
  console.log(
    `${r.molecule.padEnd(5)} ${r.basis.padEnd(12)} ${r.convention.padEnd(9)} ${tag}${nm}`,
  );
}

writeFileSync(outPath, JSON.stringify({ rows }, null, 2));
console.log(`\nwrote ${outPath} (${rows.length} rows)`);
