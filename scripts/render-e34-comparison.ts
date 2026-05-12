// ─────────────────────────────────────────────────────────────
// scripts/render-e34-comparison.ts
//
// Merges the webgpu-q and PySCF E34 artifacts into a single side-by-
// side markdown comparison table.
//
// Usage:
//   npx tsx scripts/render-e34-comparison.ts \
//     experiments/results/2026-05-12/level-6/E34-wallclock-vs-pyscf.json \
//     experiments/results/2026-05-12/level-6/E34-pyscf.json \
//     > experiments/results/2026-05-12/level-6/E34-comparison.md
// ─────────────────────────────────────────────────────────────

import * as fs from "node:fs";

interface Row {
  molecule: string;
  basis: string;
  method: string;
  seconds: number;
  // webgpu-q emits NaN as the f64 NaN (preserved through stringify),
  // PySCF (now) emits null. Normalise to number in `normalise()`.
  energy_Ha: number | null;
  success: boolean;
  notes?: string;
}

interface Artifact {
  meta: { protocol: string };
  env: Record<string, unknown>;
  rows: Row[];
  status: string;
}

function loadArtifact(path: string): Artifact {
  // Tolerate bare NaN tokens that the webgpu-q artifact emits via
  // JSON.stringify on f64 NaN, which strict JSON disallows.
  const raw = fs.readFileSync(path, "utf-8").replace(/\bNaN\b/g, "null");
  return JSON.parse(raw) as Artifact;
}

function energyOf(r: Row): number {
  return r.energy_Ha === null ? Number.NaN : r.energy_Ha;
}

function key(r: Row): string {
  return `${r.molecule}/${r.basis}/${r.method}`;
}

function fmt(n: number, digits: number, pad: number): string {
  if (!Number.isFinite(n)) return "—".padStart(pad);
  return n.toFixed(digits).padStart(pad);
}

function fmtSec(n: number): string {
  if (!Number.isFinite(n) || n === 0) return "—";
  if (n < 1e-3) return `${(n * 1e6).toFixed(0)} µs`;
  if (n < 1) return `${(n * 1e3).toFixed(2)} ms`;
  return `${n.toFixed(3)} s`;
}

function ratioStr(usSec: number, themSec: number): string {
  if (!Number.isFinite(usSec) || !Number.isFinite(themSec)) return "—";
  if (usSec <= 0 || themSec <= 0) return "—";
  const r = themSec / usSec;
  if (r > 1) return `**${r.toFixed(2)}× faster**`;
  return `${(1 / r).toFixed(2)}× slower`;
}

function main(): void {
  const [wgpuPath, pyscfPath] = process.argv.slice(2);
  if (!wgpuPath || !pyscfPath) {
    console.error("Usage: render-e34-comparison.ts <webgpu-q.json> <pyscf.json>");
    process.exit(1);
  }

  const wgpu = loadArtifact(wgpuPath);
  const pyscf = loadArtifact(pyscfPath);

  const pyscfByKey = new Map<string, Row>(pyscf.rows.map((r) => [key(r), r]));

  console.log("# E34 — Wall-clock + energy comparison vs PySCF");
  console.log();
  console.log("Apples-to-apples on identical geometries, identical basis,");
  console.log("identical convergence thresholds (SCF 1e-10, CCSD 1e-10).");
  console.log("STO-3G Li in PySCF uses the matched s-only basis (1s + 2s, no 2p)");
  console.log("to match webgpu-q's current scope; see LIMITATIONS.md.");
  console.log();
  console.log(`- webgpu-q artifact: \`${wgpuPath}\``);
  console.log(`- PySCF artifact:    \`${pyscfPath}\``);
  console.log();

  // Header.
  console.log("| molecule | basis | method | webgpu-q (s) | PySCF (s) | speedup | E_wgpu (Ha) | E_pyscf (Ha) | \\|ΔE\\| (Ha) |");
  console.log("|---|---|---|---:|---:|---:|---:|---:|---:|");

  for (const w of wgpu.rows) {
    const p = pyscfByKey.get(key(w));
    if (!p) continue;
    const isSkipped = w.notes?.startsWith("skipped") || p.notes?.startsWith("skipped") || p.notes?.startsWith("not run");
    const energyDelta = (Number.isFinite(energyOf(w)) && Number.isFinite(energyOf(p)))
      ? Math.abs(energyOf(w) - energyOf(p))
      : NaN;

    const speedup = isSkipped ? "—" : ratioStr(w.seconds, p.seconds);

    console.log(
      `| ${w.molecule} | ${w.basis} | ${w.method} | ${fmtSec(w.seconds)} | ${fmtSec(p.seconds)} | ${speedup} | ${fmt(energyOf(w), 8, 1)} | ${fmt(energyOf(p), 8, 1)} | ${Number.isFinite(energyDelta) ? energyDelta.toExponential(2) : "—"} |`,
    );
  }

  // Summary
  console.log();
  console.log("## Summary");
  console.log();

  const comparable = wgpu.rows
    .map((w) => {
      const p = pyscfByKey.get(key(w));
      if (!p) return null;
      if (!Number.isFinite(energyOf(w)) || !Number.isFinite(energyOf(p))) return null;
      if (w.seconds <= 0 || p.seconds <= 0) return null;
      return { w, p, dE: Math.abs(energyOf(w) - energyOf(p)), ratio: p.seconds / w.seconds };
    })
    .filter((x): x is NonNullable<typeof x> => x !== null);

  const maxDE = Math.max(...comparable.map((c) => c.dE));
  const meanDE = comparable.reduce((a, c) => a + c.dE, 0) / comparable.length;
  const fasterCount = comparable.filter((c) => c.ratio > 1).length;
  const slowerCount = comparable.filter((c) => c.ratio <= 1).length;
  const maxSpeedup = Math.max(...comparable.map((c) => c.ratio));
  const minSpeedup = Math.min(...comparable.map((c) => c.ratio));

  console.log(`- **Energy agreement**: ${comparable.length} cells compared, max |ΔE| = ${maxDE.toExponential(2)} Ha, mean |ΔE| = ${meanDE.toExponential(2)} Ha.`);
  console.log(`- **Wall-clock**: webgpu-q is faster on ${fasterCount} / ${comparable.length} comparable cells. Speedups range from ${minSpeedup.toFixed(2)}× (slowest of us vs PySCF) to ${maxSpeedup.toFixed(2)}× (best of us vs PySCF).`);
  console.log(`- **PySCF on CPU only**: no gpu4pyscf run here. CCSD(T)-GPU comparison left for a follow-up with gpu4pyscf installed.`);
}

main();
