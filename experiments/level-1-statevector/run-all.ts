// ─────────────────────────────────────────────────────────────
// run-all.ts — Level 1 orchestrator.
//
// Exposes:
//   • runLevel1(opts) — programmatic runner, returns all four
//     artifacts + an aggregate status.
//   • wireRunAllButton() — browser glue for experiments/index.html.
//     Adds handlers that run each experiment, render its row in the
//     results table, and offer the JSON as a download.
// ─────────────────────────────────────────────────────────────

import { runE1 } from "./E1-gate-fidelity.js";
import { runE2 } from "./E2-bandwidth-roofline.js";
import { runE3 } from "./E3-scaling-law.js";
import { runE4 } from "./E4-dispatch-overhead.js";
import { downloadArtifact, emitArtifact } from "../lib/runner.js";

export async function runLevel1(): Promise<{
  status: "pass" | "fail" | "noisy";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  artifacts: any[];
}> {
   
  console.log("── Level 1 — Statevector ──");
   
  console.log("Running E1 (fidelity) …");
  const e1 = await runE1();
  emitArtifact(e1);

   
  console.log("Running E2 (bandwidth) …");
  const e2 = await runE2();
  emitArtifact(e2);

   
  console.log("Running E3 (scaling) …");
  const e3 = await runE3();
  emitArtifact(e3);

   
  console.log("Running E4 (dispatch overhead) …");
  const e4 = await runE4();
  emitArtifact(e4);

  const all = [e1, e2, e3, e4];
  const status: "pass" | "fail" | "noisy" =
    all.some((a) => a.status === "fail") ? "fail" :
    all.some((a) => a.status === "noisy") ? "noisy" : "pass";

  return { status, artifacts: all };
}

// ── browser glue ─────────────────────────────────────────────

type Row = { id: string; label: string; status: string; diagnosis: string; artifact: unknown };

function renderRow(tableBody: HTMLTableSectionElement, r: Row): void {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${escape(r.id)}</td>
    <td>${escape(r.label)}</td>
    <td class="status-${escape(r.status)}">${escape(r.status)}</td>
    <td>${escape(r.diagnosis)}</td>
    <td><button data-id="${escape(r.id)}">download JSON</button></td>
  `;
  tableBody.appendChild(tr);
  tr.querySelector("button")?.addEventListener("click", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    downloadArtifact(r.artifact as any, `${r.id}.json`);
  });
}

function escape(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  }[c]!));
}

export function wireRunAllButton(): void {
  const btn = document.getElementById("run-level-1");
  const tableBody = document.querySelector<HTMLTableSectionElement>("#level-1-results tbody");
  const banner = document.getElementById("level-1-status");
  if (!btn || !tableBody || !banner) return;

  btn.addEventListener("click", async () => {
    (btn as HTMLButtonElement).disabled = true;
    tableBody.innerHTML = "";
    banner.textContent = "running…";
    banner.className = "";

    const peakInput = document.getElementById("peak-gbps") as HTMLInputElement | null;
    const peakRaw = peakInput ? Number(peakInput.value) : NaN;
    const peakGbps = Number.isFinite(peakRaw) && peakRaw > 0 ? peakRaw : undefined;

    const tasks: Array<[string, string, () => Promise<unknown>]> = [
      ["E1", "gate fidelity vs CPU reference",    async () => runE1()],
      ["E2", "bandwidth vs roofline",              async () => runE2(peakGbps ? { peakGbps } : {})],
      ["E3", "runtime scaling T(N)",               async () => runE3()],
      ["E4", "dispatch overhead α + β·2^N",        async () => runE4()],
    ];

    const artifacts: Array<{ status: string; [k: string]: unknown }> = [];
    for (const [id, label, fn] of tasks) {
      banner.textContent = `running ${id}…`;
      let art: { status: string; diagnosis: string } & Record<string, unknown>;
      try {
        art = (await fn()) as typeof art;
      } catch (err) {
        art = { status: "fail", diagnosis: String(err) } as typeof art;
      }
      artifacts.push(art);
      renderRow(tableBody, {
        id, label,
        status: String(art.status),
        diagnosis: String(art.diagnosis),
        artifact: art,
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      emitArtifact(art as any);
    }

    const agg: "pass" | "fail" | "noisy" =
      artifacts.some((a) => a.status === "fail") ? "fail" :
      artifacts.some((a) => a.status === "noisy") ? "noisy" : "pass";
    banner.textContent = `Level 1 — ${agg.toUpperCase()}`;
    banner.className = `status-${agg}`;
    (btn as HTMLButtonElement).disabled = false;
  });
}
