// ─────────────────────────────────────────────────────────────
// learn/compute-worker.ts — off-main-thread SCF for the learn
// page. quickReport (a real RHF SCF) is ~tens of ms at STO-3G
// but ~1 s+ at cc-pVDZ; running it here keeps the drag at 60 fps
// no matter which basis is selected.
//
// Protocol (plain module worker, no SharedArrayBuffer):
//   in : { kind: "report", atoms, basis, seq }
//   out: { kind: "report", report, seq }  on success
//        { kind: "report", error,  seq }  when the SCF refuses to converge
//   in : { kind: "sweep", rBohrLen, basis, seq }   — μ(angle) reference curve
//   out: { kind: "sweep", points, seq }            — [{angleDeg, muDebye}]
// seq is echoed verbatim so the main thread can drop stale
// replies (latest-wins for reports; sweeps are cached by key).
// MolecularReport is structured-clone safe (numbers + Float64Arrays).
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "../chemistry/atoms.js";
import { moleculeToShellsNuclei } from "../chemistry/atoms.js";
import { computeMolecularIntegrals } from "../chemistry/cg-molecular.js";
import { runRHFSCF } from "../chemistry/hf-scf.js";
import { evalMOAtPoint } from "../chemistry/cube.js";
import { quickReport } from "../chemistry/quick-report.js";
import type { MolecularReport } from "../chemistry/molecular-report.js";

const ANGSTROM_TO_BOHR = 1.8897259886;

export interface ReportRequest {
  readonly kind: "report";
  readonly atoms: readonly Atom[];
  readonly basis: BasisName;
  readonly seq: number;
}

/** μ(angle) sweep at fixed O–H bond length — the Know-depth reference curve.
 *  Every point is a full RHF SCF; nothing is interpolated or hand-drawn. */
export interface SweepRequest {
  readonly kind: "sweep";
  /** O–H bond length in Å (both bonds). */
  readonly rAngstrom: number;
  readonly basis: BasisName;
  readonly seq: number;
}

/** Real MO ψ(x,y,0) sampled on a 2D grid over the molecular plane — the
 *  Know-depth orbital overlay. Runs its own fast RHF/STO-3G to get C_MO
 *  (quickReport doesn't expose coefficients), then evaluates the chosen MO
 *  with the same primitive the Gaussian-cube exporter uses. */
export interface OrbitalRequest {
  readonly kind: "orbital";
  readonly atoms: readonly Atom[];
  /** MO index counted DOWN from HOMO: 0 = HOMO, 1 = HOMO−1, … */
  readonly belowHomo: number;
  /** Grid window in Å (molecule frame) + resolution. */
  readonly x0: number; readonly y0: number; readonly x1: number; readonly y1: number;
  readonly nx: number; readonly ny: number;
  readonly seq: number;
}

export type ComputeRequest = ReportRequest | SweepRequest | OrbitalRequest;

export interface SweepPoint { readonly angleDeg: number; readonly muDebye: number }

export interface OrbitalGrid {
  readonly nx: number; readonly ny: number;
  readonly x0: number; readonly y0: number; readonly x1: number; readonly y1: number;
  /** ψ values, row-major [iy*nx + ix], y ascending. */
  readonly values: Float32Array;
  readonly moIndex: number;
  readonly label: string;
}

export type ComputeReply =
  | { readonly kind: "report"; readonly seq: number; readonly report: MolecularReport; readonly error?: undefined }
  | { readonly kind: "report"; readonly seq: number; readonly error: string; readonly report?: undefined }
  | { readonly kind: "sweep"; readonly seq: number; readonly points: readonly SweepPoint[] }
  | { readonly kind: "orbital"; readonly seq: number; readonly grid: OrbitalGrid | null; readonly error?: string };

function waterAtAngle(angleDeg: number, r: number): Atom[] {
  const h = (angleDeg / 2) * Math.PI / 180;
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [-r * Math.sin(h), -r * Math.cos(h), 0] },
    { symbol: "H", pos: [r * Math.sin(h), -r * Math.cos(h), 0] },
  ];
}

function computeOrbitalGrid(req: OrbitalRequest): OrbitalGrid {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(req.atoms, "sto-3g");
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons, { useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8 });
  const n = integrals.n;
  const moIndex = hf.nOccupied - 1 - req.belowHomo;
  if (moIndex < 0) throw new Error(`orbital: HOMO−${req.belowHomo} does not exist (nOcc=${hf.nOccupied})`);
  const coeffs = new Float64Array(n);
  for (let mu = 0; mu < n; mu++) coeffs[mu] = hf.C_MO[mu * n + moIndex]!;
  const values = new Float32Array(req.nx * req.ny);
  for (let iy = 0; iy < req.ny; iy++) {
    const y = (req.y0 + ((req.y1 - req.y0) * iy) / (req.ny - 1)) * ANGSTROM_TO_BOHR;
    for (let ix = 0; ix < req.nx; ix++) {
      const x = (req.x0 + ((req.x1 - req.x0) * ix) / (req.nx - 1)) * ANGSTROM_TO_BOHR;
      values[iy * req.nx + ix] = evalMOAtPoint(coeffs, shells, [x, y, 0]);
    }
  }
  const label = req.belowHomo === 0 ? "HOMO" : `HOMO−${req.belowHomo}`;
  return { nx: req.nx, ny: req.ny, x0: req.x0, y0: req.y0, x1: req.x1, y1: req.y1, values, moIndex, label };
}

self.addEventListener("message", (ev: MessageEvent<ComputeRequest>) => {
  const req = ev.data;
  let reply: ComputeReply;
  if (req.kind === "orbital") {
    try {
      reply = { kind: "orbital", seq: req.seq, grid: computeOrbitalGrid(req) };
    } catch (e) {
      reply = { kind: "orbital", seq: req.seq, grid: null, error: e instanceof Error ? e.message : String(e) };
    }
  } else if (req.kind === "sweep") {
    const points: SweepPoint[] = [];
    for (let a = 70; a <= 180; a += 5) {
      try {
        const rep = quickReport(waterAtAngle(a, req.rAngstrom), { basis: req.basis });
        points.push({ angleDeg: a, muDebye: rep.dipoleMagnitudeDebye });
      } catch {
        // a non-converging point is skipped, not faked — the curve shows
        // only geometries the SCF actually solved
      }
    }
    reply = { kind: "sweep", seq: req.seq, points };
  } else {
    try {
      reply = { kind: "report", seq: req.seq, report: quickReport(req.atoms, { basis: req.basis }) };
    } catch (e) {
      reply = { kind: "report", seq: req.seq, error: e instanceof Error ? e.message : String(e) };
    }
  }
  (self as unknown as Worker).postMessage(reply);
});
