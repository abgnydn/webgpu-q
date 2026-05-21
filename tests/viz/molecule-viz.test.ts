// Smoke tests for the molecule + property overlay visualizations.

import { describe, expect, test } from "vitest";
import { dispersionD2Chart } from "../../src/viz/dispersion-d2-chart.js";
import {
  moleculeGraph2D, moleculeStructure,
} from "../../src/viz/molecular-graph-2d.js";
import { mullikenChart } from "../../src/viz/mulliken-chart.js";

const h2o = [
  { symbol: "O", pos: [0.000,  0.117, 0.000] as const },
  { symbol: "H", pos: [0.757, -0.467, 0.000] as const },
  { symbol: "H", pos: [-0.757, -0.467, 0.000] as const },
];

describe("Molecule visualization modules", () => {
  test("dispersionD2Chart renders sorted pairwise bars", () => {
    const pairs = [
      { indexA: 0, indexB: 1, distanceBohr: 1.8, contributionHartree: -0.000095 },
      { indexA: 0, indexB: 2, distanceBohr: 1.8, contributionHartree: -0.000095 },
      { indexA: 1, indexB: 2, distanceBohr: 2.86, contributionHartree: -0.000018 },
    ];
    const svg = dispersionD2Chart(pairs, {
      atomLabels: ["O", "H", "H"],
      totalHartree: -0.000208,
    });
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("Grimme D2");
    expect(svg).toContain("O-H");
    expect(svg).toContain("mHa");
    expect(svg).toContain("total:");
  });

  test("dispersionD2Chart handles empty pair list", () => {
    const svg = dispersionD2Chart([], {});
    expect(svg).toContain("no D2 pairs");
  });

  test("dispersionD2Chart respects maxBars cap", () => {
    const pairs = Array.from({ length: 30 }, (_, i) => ({
      indexA: 0, indexB: i + 1, distanceBohr: 2 + i * 0.1,
      contributionHartree: -1e-4 / (i + 1),
    }));
    const svg = dispersionD2Chart(pairs, { maxBars: 5 });
    // Largest contribution is i=0 (1e-4). Should appear.
    expect(svg).toContain("0-1");
    // Lowest-magnitude (i=29) should not.
    expect(svg.includes("0-30")).toBe(false);
  });

  test("moleculeStructure renders atoms + bonds", () => {
    const svg = moleculeStructure(h2o, { title: "H₂O" });
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("H₂O");
    expect(svg).toContain("3 atoms");
    // Three atoms = three circle fills (plus possibly bg). Count.
    const circleCount = (svg.match(/<circle/g) ?? []).length;
    expect(circleCount).toBeGreaterThanOrEqual(3);
    // Two O-H bonds → at least 2 lines (plus axis-style lines).
    const lineCount = (svg.match(/<line/g) ?? []).length;
    expect(lineCount).toBeGreaterThanOrEqual(2);
  });

  test("moleculeGraph2D with Mulliken charges shows rings + numerics", () => {
    const charges = [-0.7, 0.35, 0.35];
    const svg = moleculeGraph2D(h2o, {
      charges, atomLabels: ["O", "H", "H"], title: "H₂O charges",
    });
    expect(svg).toContain("δ+ Mulliken");
    expect(svg).toContain("δ−");
    expect(svg).toContain("+0.35"); // H charge label
    expect(svg).toContain("-0.7"); // O charge label
  });

  test("moleculeGraph2D supports xz projection", () => {
    const svg = moleculeGraph2D(h2o, { projection: "xz" });
    expect(svg).toContain("xz projection");
  });

  test("moleculeGraph2D with spin populations colors numerics", () => {
    const svg = moleculeGraph2D(
      [{ symbol: "H", pos: [0, 0, 0] as const }, { symbol: "H", pos: [0.74, 0, 0] as const }],
      { spinPopulations: [0.5, -0.5] },
    );
    expect(svg).toContain("0.5");
    expect(svg).toContain("-0.5");
  });

  test("moleculeGraph2D handles empty molecule", () => {
    const svg = moleculeGraph2D([], {});
    expect(svg).toContain("empty molecule");
  });

  test("mullikenChart renders symmetric ±q bars", () => {
    const svg = mullikenChart([-0.7, 0.35, 0.35], {
      atomLabels: ["O", "H1", "H2"],
    });
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("Mulliken");
    expect(svg).toContain("H1");
    expect(svg).toContain("Σq");
    expect(svg).toContain("+0.35");
    expect(svg).toContain("-0.7");
    // 3 atoms = 3 bars.
    const rectCount = (svg.match(/<rect[^>]*fill-opacity="0.85"/g) ?? []).length;
    expect(rectCount).toBe(3);
  });

  test("mullikenChart handles empty input", () => {
    const svg = mullikenChart([], {});
    expect(svg).toContain("no atoms");
  });
});
