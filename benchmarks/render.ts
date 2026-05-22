// ─────────────────────────────────────────────────────────────
// benchmarks/render.ts — generate a human-readable BENCHMARKS.md
// from the empirically-verified `numbers.ts` source of truth.
//
// Run with `npm run benchmarks:render` after editing claims.
// CI will fail (via numbers.test.ts) if any claim drifts from
// the running code. The .md is just a static dump for humans.
//
// Format:
//   • One headline summary table (claims tagged `headline: true`)
//   • Per-category sections with the full table for each
//   • References block listing every experimental / literature
//     comparison
//
// The output file is checked in so anyone landing on the repo
// can read it without running the build. Re-run after every
// commit that changes numbers.ts.
// ─────────────────────────────────────────────────────────────
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { CLAIMS, claimsByCategory, type Claim, type Category } from "./numbers.js";

const CATEGORY_LABELS: Record<Category, string> = {
  "test-surface": "Test surface",
  "energy":       "Energies",
  "geometry":     "Geometry",
  "spectroscopy": "Spectroscopy (vibrational + electronic)",
  "thermo":       "Thermochemistry",
  "field":        "Field response (μ · α · β · charges)",
  "redox":        "Ionization potentials",
  "openshell":    "Open-shell SCF (UHF)",
  "symmetry":     "Symmetry-forced exact validations",
};

function fmt(value: number, units: string): string {
  if (units === "tests") return value.toFixed(0);
  if (Math.abs(value) >= 1000)   return value.toFixed(1);
  if (Math.abs(value) >= 100)    return value.toFixed(2);
  if (Math.abs(value) >= 1)      return value.toFixed(4);
  if (Math.abs(value) >= 0.001)  return value.toFixed(5);
  if (Math.abs(value) >= 1e-6)   return value.toExponential(3);
  return value.toExponential(2);
}

function renderClaim(c: Claim): string {
  const ref = c.reference
    ? `${fmt(c.reference.value, c.units)} ${c.units}<br/><sub>${c.reference.source}</sub>`
    : "—";
  return `| \`${c.id}\` | ${c.label} | **${fmt(c.value, c.units)}** ${c.units} | ${ref} |`;
}

function renderHeadline(): string {
  const headline = CLAIMS.filter((c) => c.headline);
  const lines = [
    "## Headline numbers",
    "",
    "| id | claim | value | reference |",
    "| :-- | :-- | --: | :-- |",
    ...headline.map(renderClaim),
  ];
  return lines.join("\n");
}

function renderByCategory(): string {
  const lines: string[] = [];
  for (const [cat, claims] of claimsByCategory()) {
    lines.push("");
    lines.push(`## ${CATEGORY_LABELS[cat]}`);
    lines.push("");
    lines.push("| id | claim | value | reference |");
    lines.push("| :-- | :-- | --: | :-- |");
    for (const c of claims) lines.push(renderClaim(c));
  }
  return lines.join("\n");
}

function renderHeader(): string {
  const isoDate = new Date().toISOString().slice(0, 10);
  return [
    "# Benchmarks — empirically reproducible",
    "",
    "Every number this project quotes — in `README.md`, `CHANGELOG.md`,",
    "`/index.html`, `/molecule.html`, the GitHub release notes — is",
    "checked in [`benchmarks/numbers.ts`](./benchmarks/numbers.ts) with a",
    "`verify()` function that re-derives it from running source code.",
    "",
    "The unit-test [`tests/numbers.test.ts`](./tests/numbers.test.ts) runs",
    "every `verify()` and asserts the result is within tolerance of the",
    "claimed value. **CI fails if any number drifts.** That's the entire",
    "reproducibility guarantee.",
    "",
    `> _Auto-generated from \`benchmarks/numbers.ts\`. Last regenerated:_`,
    `> _${isoDate}._ Run \`npm run benchmarks:render\` to refresh.`,
    "",
    "**How to read the table.** The **value** column is what the docs",
    "quote — verified at HEAD by `numbers.test.ts`. The **reference**",
    "column is the experimental or literature value with its source.",
    "Where the reference has a tolerance set in `numbers.ts`, the test",
    "also asserts agreement with experiment within that bound, so we",
    "catch silent regressions in *external* match too.",
    "",
  ].join("\n");
}

const md = [
  renderHeader(),
  renderHeadline(),
  renderByCategory(),
  "",
].join("\n");

const outPath = resolve(process.cwd(), "BENCHMARKS.md");
writeFileSync(outPath, md);
// eslint-disable-next-line no-console
console.log(`wrote ${outPath} (${md.length} bytes, ${CLAIMS.length} claims)`);
