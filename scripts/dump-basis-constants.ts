// Dump every contracted-shell constant exported by integrals.ts as JSON.
//
// Consumed by scripts/check-basis-vs-pyscf.py (Gate 0.2 of
// docs/RUN-PLAN-24H-ELEMENTS.md): the basis-table generator is only
// trusted for new elements once PySCF's own tables are shown to
// reproduce the eight hand-transcribed elements already in the repo.
//
// Usage:  npx --yes tsx scripts/dump-basis-constants.ts
import * as integrals from "../src/chemistry/integrals.js";

interface ShellConst { alpha: readonly number[]; c: readonly number[] }

function isShellConst(v: unknown): v is ShellConst {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return Array.isArray(o["alpha"]) && Array.isArray(o["c"])
    && (o["alpha"] as unknown[]).every((x) => typeof x === "number")
    && (o["c"] as unknown[]).every((x) => typeof x === "number");
}

const out: Record<string, { alpha: readonly number[]; c: readonly number[] }> = {};
for (const [name, value] of Object.entries(integrals)) {
  if (isShellConst(value)) out[name] = { alpha: value.alpha, c: value.c };
}

process.stdout.write(JSON.stringify(out, null, 2) + "\n");
