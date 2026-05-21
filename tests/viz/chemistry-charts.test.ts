// Smoke tests for the chemistry visualization SVG modules.
// Each chart produces a valid SVG string with the expected sections.

import { describe, expect, test } from "vitest";
import { alphaOmegaChart } from "../../src/viz/alpha-omega-chart.js";
import { noonChart } from "../../src/viz/noon-chart.js";
import { uvVisChart } from "../../src/viz/uv-vis-chart.js";
import {
  energyDecompChart, decompositionToComponents,
} from "../../src/viz/energy-decomp-chart.js";
import { c6MatrixChart } from "../../src/viz/c6-matrix-chart.js";
import { rotationalCard } from "../../src/viz/rotational-card.js";

describe("Chemistry chart SVG modules", () => {
  test("alphaOmegaChart produces valid SVG with series legend", () => {
    const svg = alphaOmegaChart(
      [
        { name: "RHF", data: [[0, 5.0], [0.1, 5.2], [0.2, 5.8], [0.3, 7.0]] },
        { name: "B3LYP", data: [[0, 6.5], [0.1, 6.8], [0.2, 7.5], [0.3, 9.0]] },
      ],
      { title: "Test α(ω)" },
    );
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg.endsWith("</svg>")).toBe(true);
    expect(svg).toContain("Test α(ω)");
    expect(svg).toContain("RHF");
    expect(svg).toContain("B3LYP");
    expect(svg).toContain("ω (Ha)");
  });

  test("alphaOmegaChart handles empty data without crashing", () => {
    const svg = alphaOmegaChart([], {});
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("no data");
  });

  test("noonChart renders bars with multireference badge", () => {
    const occ = new Float64Array([2.0, 2.0, 2.0, 0.0, 0.0]);
    const svg = noonChart(occ, {
      nOccupied: 3, severity: "single-reference",
    });
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("HOMO");
    expect(svg).toContain("single-reference");
    // Count rect bars (5 occupations).
    const barCount = (svg.match(/<rect[^>]*fill-opacity="0.85"/g) ?? []).length;
    expect(barCount).toBe(5);
  });

  test("noonChart with multireference verdict shows red badge", () => {
    const occ = new Float64Array([2.0, 1.5, 0.5, 0.0]);
    const svg = noonChart(occ, { severity: "multi-reference" });
    expect(svg).toContain("multi-reference");
  });

  test("uvVisChart renders broadened spectrum + peaks", () => {
    const spectrum = Array.from({ length: 50 }, (_, i) => ({
      energy: i * 0.5,
      absorption: Math.exp(-Math.pow((i * 0.5 - 8), 2) / 2),
    }));
    const peaks = [{ energy: 8.0, absorption: 1.0 }];
    const svg = uvVisChart(spectrum, peaks);
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("UV-vis");
    expect(svg).toContain("ω (eV)");
    // Peak label "8.00" should appear.
    expect(svg).toMatch(/8\.0[0-9]/);
  });

  test("energyDecompChart with positive and negative components", () => {
    const decomp = {
      oneElectron: -120,
      coulomb: 45,
      exchange: -8,
      nuclearRepulsion: 8,
    };
    const components = decompositionToComponents(decomp, { correlation: -0.5 });
    expect(components.length).toBe(5);
    const svg = energyDecompChart(components, { total: -75.5 });
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("Energy decomposition");
    expect(svg).toContain("total: -75.5");
  });

  test("c6MatrixChart renders N×N labeled heatmap", () => {
    const labels = ["H₂", "He", "BeH₂"];
    const matrix = [
      [0.7, 0.2, 1.5],
      [0.2, 0.1, 0.5],
      [1.5, 0.5, 3.0],
    ];
    const svg = c6MatrixChart(labels, matrix);
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("C₆");
    expect(svg).toContain("H₂");
    expect(svg).toContain("BeH₂");
    // Should have 9 cells.
    const cellCount = (svg.match(/<rect[^>]*stroke="#020617"/g) ?? []).length;
    expect(cellCount).toBe(9);
  });

  test("rotationalCard renders A/B/C values in cm⁻¹ and GHz", () => {
    const svg = rotationalCard({
      A_cm1: 27.88, B_cm1: 14.52, C_cm1: 9.28,
      A_GHz: 27.88 * 29.9792458,
      B_GHz: 14.52 * 29.9792458,
      C_GHz: 9.28 * 29.9792458,
    });
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("Rotational");
    expect(svg).toContain("27.88");
    expect(svg).toContain("cm⁻¹");
    expect(svg).toContain("GHz");
  });

  test("rotationalCard handles linear molecule (NaN constants)", () => {
    const svg = rotationalCard({
      A_cm1: NaN, B_cm1: 5.0, C_cm1: 5.0,
      A_GHz: NaN, B_GHz: 150, C_GHz: 150,
    });
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("linear");
  });
});
