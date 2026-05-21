// Smoke tests for the 2D-field + iterative + basis viz modules.

import { describe, expect, test } from "vitest";
import { contour2DChart } from "../../src/viz/contour-2d-chart.js";
import {
  localizedOrbitalGallery, lmoCaption,
} from "../../src/viz/localized-orbital-gallery.js";
import {
  convergenceTrace,
} from "../../src/viz/convergence-trace.js";
import {
  basisCoverageChart, basisSummary,
} from "../../src/viz/basis-coverage-chart.js";

describe("Field + convergence viz modules", () => {
  function makeGaussianField(nx: number, ny: number): Float64Array {
    const out = new Float64Array(nx * ny);
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const x = (ix / nx - 0.5) * 4;
        const y = (iy / ny - 0.5) * 4;
        // Signed two-lobe pattern (looks like a p-orbital).
        out[ix + iy * nx] = x * Math.exp(-(x * x + y * y));
      }
    }
    return out;
  }

  test("contour2DChart renders divergent heatmap", () => {
    const f = makeGaussianField(16, 16);
    const svg = contour2DChart(f, {
      xMin: -2, xMax: 2, yMin: -2, yMax: 2,
      nx: 16, ny: 16, colormap: "divergent",
      atoms: [{ symbol: "H", x: 0, y: 0 }],
      title: "Test p-orbital",
      fieldLabel: "ψ",
    });
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("Test p-orbital");
    expect(svg).toContain("ψ");
    // 16x16 = 256 cells.
    const cellCount = (svg.match(/<rect[^>]*fill="rgb/g) ?? []).length;
    expect(cellCount).toBe(256);
    // Atom overlay → atom symbol rendered as <text>…H…</text>.
    expect(svg).toMatch(/>H</);
  });

  test("contour2DChart sequential colormap for positive density", () => {
    const nx = 8, ny = 8;
    const f = new Float64Array(nx * ny).map(() => Math.random());
    const svg = contour2DChart(f, {
      xMin: 0, xMax: 1, yMin: 0, yMax: 1,
      nx, ny, colormap: "sequential",
      title: "Density",
    });
    expect(svg).toContain("Density");
    const cellCount = (svg.match(/<rect[^>]*fill="rgb/g) ?? []).length;
    expect(cellCount).toBe(64);
  });

  test("contour2DChart detects grid mismatch", () => {
    const svg = contour2DChart(new Float64Array(10), {
      xMin: 0, xMax: 1, yMin: 0, yMax: 1, nx: 4, ny: 4,
    });
    expect(svg).toContain("grid mismatch");
  });

  test("contour2DChart with contourCount draws contour markers", () => {
    const f = makeGaussianField(12, 12);
    const svg = contour2DChart(f, {
      xMin: -2, xMax: 2, yMin: -2, yMax: 2, nx: 12, ny: 12,
      contourCount: 3,
    });
    // Contour markers are small circles with r="0.7".
    expect(svg).toMatch(/r="0\.7"/);
  });

  test("localizedOrbitalGallery composes multi-panel SVG", () => {
    const slices = [0, 1, 2, 3].map((i) => ({
      title: `LMO ${i}`,
      values: makeGaussianField(10, 10),
      nx: 10, ny: 10,
      xMin: -2, xMax: 2, yMin: -2, yMax: 2,
      caption: lmoCaption([0, 0, 0], 0.5),
    }));
    const svg = localizedOrbitalGallery(slices, { columns: 2 });
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("LMO 0");
    expect(svg).toContain("LMO 3");
    expect(svg).toContain("σ = 0.5");
  });

  test("localizedOrbitalGallery handles empty list", () => {
    const svg = localizedOrbitalGallery([]);
    expect(svg).toContain("no orbitals");
  });

  test("convergenceTrace renders log-scale series with threshold", () => {
    const energies = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8];
    const densities = [5e-2, 5e-3, 1e-4, 1e-5, 1e-7];
    const svg = convergenceTrace(
      [
        { name: "|ΔE|", values: energies },
        { name: "|ΔD|", values: densities },
      ],
      { title: "SCF convergence", threshold: 1e-7 },
    );
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("SCF convergence");
    expect(svg).toContain("|ΔE|");
    expect(svg).toContain("|ΔD|");
    expect(svg).toContain("τ =");
    expect(svg).toContain("log₁₀ scale");
  });

  test("convergenceTrace handles empty series", () => {
    const svg = convergenceTrace([]);
    expect(svg).toContain("no series");
  });

  test("convergenceTrace clamps non-positive metric to 1e-16", () => {
    const svg = convergenceTrace([{ name: "x", values: [1, 0, -0.5, 1e-8] }]);
    // Should not crash; should produce an SVG.
    expect(svg.startsWith("<svg")).toBe(true);
  });

  test("basisCoverageChart renders stacked shells per atom", () => {
    const svg = basisCoverageChart(
      [
        { symbol: "O", counts: [3, 2, 1] }, // 3s 2p 1d
        { symbol: "H", counts: [2] },
        { symbol: "H", counts: [2] },
      ],
      { title: "H₂O · cc-pVDZ", subtitle: "spherical-d" },
    );
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("H₂O");
    expect(svg).toContain("spherical-d");
    expect(svg).toContain("O #0");
    expect(svg).toContain("3 atoms");
  });

  test("basisSummary builds a compact label", () => {
    expect(basisSummary([3, 2, 1])).toBe("3s 2p 1d");
    expect(basisSummary([2])).toBe("2s");
    expect(basisSummary([0, 0, 0])).toBe("—");
  });

  test("basisCoverageChart handles empty input", () => {
    const svg = basisCoverageChart([]);
    expect(svg).toContain("no atoms");
  });
});
