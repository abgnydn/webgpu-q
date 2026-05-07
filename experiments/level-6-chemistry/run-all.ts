// ─────────────────────────────────────────────────────────────
// run-all.ts — Level 6 orchestrator (chemistry).
//
// Exposes:
//   • runLevel6(opts)        — programmatic runner for E16 (VQE on H₂).
//   • wireRunLevel6Button()  — browser glue.
//
// E17 (cross-sections vs Itikawa-Mason 2005 / Geant4-DNA reference)
// shipped 2026-05-07 as an honest negative — Bethe-Born from a
// finite TDA-DFT spectrum can't represent the genuine ionization
// continuum, so σ_ion undershoots reference values. σ_exc lands
// closer. See E17-cross-sections.ts for the full diagnosis.
// ─────────────────────────────────────────────────────────────

import { runE16 } from "./E16-h2-vqe.js";
import { runE17 } from "./E17-cross-sections.js";
import { runE20 } from "./E20-lih-vqe.js";
import { runE21 } from "./E21-beh2-vqe.js";
import { downloadArtifact, emitArtifact } from "../lib/runner.js";

export async function runLevel6(): Promise<{
  status: "pass" | "fail" | "noisy";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  artifacts: any[];
}> {

  console.log("── Level 6 — Chemistry ──");

  console.log("Running E16 (H₂ VQE) …");
  const e16 = await runE16();
  emitArtifact(e16);


  console.log("Running E20 (LiH VQE, Phase C) …");
  const e20 = await runE20();
  emitArtifact(e20);


  console.log("Running E21 (BeH₂ VQE, Phase C v2) …");
  const e21 = await runE21();
  emitArtifact(e21);

  console.log("Running E17 (e⁻ + H₂O cross sections) …");
  const e17 = await runE17();
  emitArtifact(e17);

  const all = [e16, e20, e21, e17];
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

export function wireRunLevel6Button(): void {
  const btn = document.getElementById("run-level-6");
  const tableBody = document.querySelector<HTMLTableSectionElement>("#level-6-results tbody");
  const banner = document.getElementById("level-6-status");
  if (!btn || !tableBody || !banner) return;

  btn.addEventListener("click", async () => {
    (btn as HTMLButtonElement).disabled = true;
    tableBody.innerHTML = "";
    banner.textContent = "running…";
    banner.className = "";

    const tasks: Array<[string, string, () => Promise<unknown>]> = [
      ["E16", "VQE H₂ STO-3G — 10 random inits at R=0.7414 Å",            async () => runE16()],
      ["E20", "VQE LiH STO-3G s-only — 5 inits, 5 R values (Phase C v1)", async () => runE20()],
      ["E21", "VQE BeH₂ STO-3G s-only — 5 inits, 5 R values (Phase C v2)", async () => runE21()],
      ["E17", "Bethe-Born σ(T) for e⁻+H₂O vs Itikawa-Mason 2005",         async () => runE17()],
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
    banner.textContent = `Level 6 — ${agg.toUpperCase()}`;
    banner.className = `status-${agg}`;
    (btn as HTMLButtonElement).disabled = false;
  });
}
