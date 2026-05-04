// ─────────────────────────────────────────────────────────────
// run-all.ts — Level 2 orchestrator.
//
// Exposes:
//   • runLevel2(opts)            — programmatic runner for E5/E6/E7.
//   • wireRunLevel2Button()      — browser glue for experiments/index.html.
//
// Defaults match a single-browser-tab UI budget (smaller sweeps than
// the "publishable" settings in the protocol). To reproduce the full
// research-grade numbers, invoke `runLevel2({ … })` directly from
// devtools with the larger parameter values documented in each file.
// ─────────────────────────────────────────────────────────────

import { runE5 } from "./E5-mps-correctness.js";
import { runE6 } from "./E6-qubit-ceiling.js";
import { runE7 } from "./E7-chi-scaling.js";
import { downloadArtifact, emitArtifact } from "../lib/runner.js";

export async function runLevel2(): Promise<{
  status: "pass" | "fail" | "noisy";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  artifacts: any[];
}> {

  console.log("── Level 2 — MPS ──");


  console.log("Running E5 (correctness) …");
  const e5 = await runE5();
  emitArtifact(e5);


  console.log("Running E6 (qubit ceiling) …");
  const e6 = await runE6();
  emitArtifact(e6);


  console.log("Running E7 (χ scaling) …");
  const e7 = await runE7();
  emitArtifact(e7);

  const all = [e5, e6, e7];
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

export function wireRunLevel2Button(): void {
  const btn = document.getElementById("run-level-2");
  const tableBody = document.querySelector<HTMLTableSectionElement>("#level-2-results tbody");
  const banner = document.getElementById("level-2-status");
  if (!btn || !tableBody || !banner) return;

  btn.addEventListener("click", async () => {
    (btn as HTMLButtonElement).disabled = true;
    tableBody.innerHTML = "";
    banner.textContent = "running…";
    banner.className = "";

    const tasks: Array<[string, string, () => Promise<unknown>]> = [
      ["E5", "MPS correctness vs CPU statevector (N ≤ 20)", async () => runE5()],
      ["E6", "qubit-count ceiling (χ=32, depth=4)",          async () => runE6()],
      ["E7", "χ scaling vs depth (Haar brick-wall, N=16)",    async () => runE7()],
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
    banner.textContent = `Level 2 — ${agg.toUpperCase()}`;
    banner.className = `status-${agg}`;
    (btn as HTMLButtonElement).disabled = false;
  });
}
