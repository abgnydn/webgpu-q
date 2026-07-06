// ─────────────────────────────────────────────────────────────
// learn/compute-worker.ts — off-main-thread SCF for the learn
// page. quickReport (a real RHF SCF) is ~tens of ms at STO-3G
// but ~1 s+ at cc-pVDZ; running it here keeps the drag at 60 fps
// no matter which basis is selected.
//
// Protocol (plain module worker, no SharedArrayBuffer):
//   in : { atoms, basis, seq }
//   out: { report, seq }  on success
//        { error,  seq }  when the SCF refuses to converge
// seq is echoed verbatim so the main thread can drop stale
// replies (latest-wins). MolecularReport is structured-clone
// safe (numbers + Float64Arrays only).
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "../chemistry/atoms.js";
import { quickReport } from "../chemistry/quick-report.js";
import type { MolecularReport } from "../chemistry/molecular-report.js";

export interface ComputeRequest {
  readonly atoms: readonly Atom[];
  readonly basis: BasisName;
  readonly seq: number;
}

export type ComputeReply =
  | { readonly seq: number; readonly report: MolecularReport; readonly error?: undefined }
  | { readonly seq: number; readonly error: string; readonly report?: undefined };

self.addEventListener("message", (ev: MessageEvent<ComputeRequest>) => {
  const { atoms, basis, seq } = ev.data;
  let reply: ComputeReply;
  try {
    reply = { seq, report: quickReport(atoms, { basis }) };
  } catch (e) {
    reply = { seq, error: e instanceof Error ? e.message : String(e) };
  }
  (self as unknown as Worker).postMessage(reply);
});
