// ─────────────────────────────────────────────────────────────
// run-all.ts — Level 3 orchestrator (kernel fusion).
//
// Exposes:
//   • runLevel3(opts)            — programmatic runner for E8/E9/E10.
//   • wireRunLevel3Button()      — browser glue.
// ─────────────────────────────────────────────────────────────

import { runE8 } from "./E8-fusion-correctness.js";
import { runE9 } from "./E9-dispatch-collapse.js";
import { runE10 } from "./E10-throughput.js";
import { runE11 } from "./E11-brickwall-fusion.js";
import { runE12 } from "./E12-tier-c-fusion.js";
import { runE13 } from "./E13-tier-d-fusion.js";
import { downloadArtifact, emitArtifact } from "../lib/runner.js";

export async function runLevel3(): Promise<{
  status: "pass" | "fail" | "noisy";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  artifacts: any[];
}> {

  console.log("── Level 3 — Kernel fusion ──");


  console.log("Running E8 (fusion correctness) …");
  const e8 = await runE8();
  emitArtifact(e8);


  console.log("Running E9 (dispatch-cost collapse) …");
  const e9 = await runE9();
  emitArtifact(e9);


  console.log("Running E10 (throughput) …");
  const e10 = await runE10();
  emitArtifact(e10);


  console.log("Running E11 (brick-wall layer fusion) …");
  const e11 = await runE11();
  emitArtifact(e11);


  console.log("Running E12 (Tier C 3-qubit cascade fusion) …");
  const e12 = await runE12();
  emitArtifact(e12);


  console.log("Running E13 (Tier D 4-qubit cascade fusion) …");
  const e13 = await runE13();
  emitArtifact(e13);

  const all = [e8, e9, e10, e11, e12, e13];
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

export function wireRunLevel3Button(): void {
  const btn = document.getElementById("run-level-3");
  const tableBody = document.querySelector<HTMLTableSectionElement>("#level-3-results tbody");
  const banner = document.getElementById("level-3-status");
  if (!btn || !tableBody || !banner) return;

  btn.addEventListener("click", async () => {
    (btn as HTMLButtonElement).disabled = true;
    tableBody.innerHTML = "";
    banner.textContent = "running…";
    banner.className = "";

    const tasks: Array<[string, string, () => Promise<unknown>]> = [
      ["E8",  "fusion correctness vs unfused reference",            async () => runE8()],
      ["E9",  "dispatch-cost collapse α_eff(k)",                    async () => runE9()],
      ["E10", "throughput fused vs unfused (same-qubit chain)",     async () => runE10()],
      ["E11", "brick-wall layer fusion (single+single+CNOT → 4×4)", async () => runE11()],
      ["E12", "Tier C 3-qubit cascade fusion (5 ops → 8×8)",         async () => runE12()],
      ["E13", "Tier D 4-qubit cascade fusion (7 ops → 16×16)",       async () => runE13()],
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
    banner.textContent = `Level 3 — ${agg.toUpperCase()}`;
    banner.className = `status-${agg}`;
    (btn as HTMLButtonElement).disabled = false;
  });
}
