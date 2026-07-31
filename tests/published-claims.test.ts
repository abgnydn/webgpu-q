// Guard for the drift class that `benchmarks/numbers.ts` cannot catch.
//
// numbers.ts pins values that can be RE-DERIVED by running code (each Claim has
// a verify()). That leaves everything else unprotected: GPU speedups need a GPU,
// and prose lives in CITATION.cff / index.html / CLAUDE.md / the papers — which
// is precisely where the 2026-07 audit found retired numbers still published,
// including in the Zenodo-deposited abstract that cannot be retracted.
//
// This file is a text-level backstop. It does not check that a number is right;
// it checks that numbers the project has explicitly RETIRED, and CI claims the
// workflows contradict, have not crept back into a published surface.
import { describe, expect, test } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const read = (p: string): string => (existsSync(resolve(ROOT, p)) ? readFileSync(resolve(ROOT, p), "utf-8") : "");

/** Surfaces a reader/reviewer/search-engine actually sees. Deliberately EXCLUDES
 *  CHANGELOG.md and LIMITATIONS.md, whose job is to record superseded numbers,
 *  and README.md's key-numbers table, which names the retired value to retire it. */
const PUBLISHED = [
  "CITATION.cff",
  "index.html",
  "CLAUDE.md",
  "BENCHMARKS.md",
  "paper/main.tex",
  "paper/main-fusion.tex",
  "deploy/hf-space/README.md",
  "public/readme-numbers.svg",
  "public/readme-perf.svg",
];

describe("published claims — retired numbers must not reappear", () => {
  // Each entry: the retired token, and what superseded it (for the failure message).
  const RETIRED: ReadonlyArray<{ pattern: RegExp; was: string; now: string }> = [
    { pattern: /39\.3\s*[×x]/, was: "39.3× CCSD(T) GPU", now: "13.8× median / 28.4× p10 (LIMITATIONS.md:168-179)" },
    { pattern: /39\s*[×x]\s*(best|on H)/i, was: "39× best run", now: "28.4× p10, best of 20 trials" },
    { pattern: /4\.18\s*[×x]\s*(headline|at the)/i, was: "4.18× fusion headline", now: "4.22× (E12 artifact bestSpeedup 4.2171)" },
    { pattern: /plateaus at 3\.14\s*[×x]/i, was: "Tier D 3.14×", now: "3.78× (E13 2026-05-05)" },
    { pattern: /~?199\s*s\s*CPU/i, was: "~199 s CPU baseline", now: "116.4 s (unsourced figure — existed in no artifact)" },
  ];

  for (const file of PUBLISHED) {
    const body = read(file);
    if (body === "") continue;
    for (const { pattern, was, now } of RETIRED) {
      test(`${file} does not republish "${was}"`, () => {
        const hit = body.match(pattern);
        expect(
          hit,
          `${file} contains the retired claim "${was}" (matched: ${hit?.[0]}). Superseded by ${now}.`,
        ).toBeNull();
      });
    }
  }
});

describe("published claims — CI scope must be described truthfully", () => {
  // .github/workflows/ci.yml runs lint / tsc / vitest / build and deliberately
  // does NOT run the Playwright e2e suite (hosted runners expose no WebGPU).
  // Saying otherwise misleads reviewers about what is actually gated.
  const FALSE_CI_CLAIMS: ReadonlyArray<RegExp> = [
    /Every PR runs unit \+ e2e/i,
    /runs in CI automatically/i,
    /on every CI pass/i,
    /Playwright e2e · CI green/i,
  ];

  test("ci.yml genuinely does not run the e2e suite (premise of this guard)", () => {
    const ci = read(".github/workflows/ci.yml");
    expect(ci).not.toBe("");
    // Strip comments first: ci.yml legitimately NAMES `npm run test:e2e` in the
    // comment explaining why it is excluded. Only executable steps count.
    const steps = ci
      .split("\n")
      .filter((l) => !l.trim().startsWith("#"))
      .join("\n");
    expect(
      /npm run test:e2e|playwright test/.test(steps),
      "ci.yml now runs e2e — good! Update the docs this guard protects, then relax it.",
    ).toBe(false);
  });

  for (const file of ["README.md", "RESEARCH_STANDARDS.md", "BENCHMARKS.md", "paper/main.tex"]) {
    const body = read(file);
    if (body === "") continue;
    for (const claim of FALSE_CI_CLAIMS) {
      test(`${file} does not claim e2e runs in CI (${claim.source})`, () => {
        expect(claim.test(body), `${file} matches ${claim} — but ci.yml skips e2e.`).toBe(false);
      });
    }
  }
});

describe("shipped HTML — no bare directory URLs", () => {
  // HF Spaces' static server does not resolve directory URLs: /experiments/ 302s
  // off-site to huggingface.co/Experiments and /experiments/gpu-mps/ returns 401.
  // Vercel tolerates them, so this breaks on exactly one of the two deploys.
  const PAGES = [
    "index.html", "learn.html", "demo.html", "molecule.html",
    "screening.html", "swarm.html", "viz.html",
    "experiments/index.html", "experiments/gpu-mps/index.html",
  ];
  for (const page of PAGES) {
    const body = read(page);
    if (body === "") continue;
    test(`${page} links no bare directories`, () => {
      const bare = [...body.matchAll(/href="(\/[^"?#]*\/)"/g)].map((m) => m[1]);
      expect(bare, `${page} has bare directory href(s) — use an explicit index.html`).toEqual([]);
    });
  }
});
