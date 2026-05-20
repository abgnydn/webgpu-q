// Spectrum convolution tests.

import { describe, expect, test } from "vitest";
import { broadenSpectrum, findSpectrumPeaks } from "../../src/chemistry/spectrum.js";

describe("Spectrum broadening + peak finding", () => {
  test("Single excitation produces a Gaussian centered at ω", () => {
    const exc = [{ energy: 5.0, oscillatorStrength: 1.0 }];
    const s = broadenSpectrum(exc, { eMin: 4, eMax: 6, step: 0.01, fwhm: 0.2 });
    // Peak should be at ω = 5.0.
    let maxIdx = 0;
    for (let i = 1; i < s.length; i++) {
      if (s[i]!.absorption > s[maxIdx]!.absorption) maxIdx = i;
    }
    expect(s[maxIdx]!.energy).toBeCloseTo(5.0, 1);
    // Peak height = 1/(σ√(2π)) where σ = 0.2/2.3548.
    const sigma = 0.2 / (2 * Math.sqrt(2 * Math.log(2)));
    expect(s[maxIdx]!.absorption).toBeCloseTo(1 / (sigma * Math.sqrt(2 * Math.PI)), 4);
  });

  test("Two well-separated excitations: two peaks", () => {
    const exc = [
      { energy: 3.0, oscillatorStrength: 0.5 },
      { energy: 6.0, oscillatorStrength: 1.0 },
    ];
    const s = broadenSpectrum(exc, { eMin: 0, eMax: 8, step: 0.01, fwhm: 0.3 });
    const peaks = findSpectrumPeaks(s, 0.01);
    expect(peaks.length).toBe(2);
    expect(peaks[0]!.energy).toBeCloseTo(3.0, 1);
    expect(peaks[1]!.energy).toBeCloseTo(6.0, 1);
    // Second peak should be roughly 2× higher (since f₂ = 2·f₁).
    expect(peaks[1]!.absorption / peaks[0]!.absorption).toBeCloseTo(2, 1);
  });

  test("Empty excitations → empty spectrum", () => {
    expect(broadenSpectrum([]).length).toBe(0);
  });

  test("Integrated spectrum ≈ total oscillator strength", () => {
    const exc = [
      { energy: 5.0, oscillatorStrength: 0.5 },
      { energy: 6.0, oscillatorStrength: 0.3 },
      { energy: 7.0, oscillatorStrength: 0.2 },
    ];
    const s = broadenSpectrum(exc, { eMin: 0, eMax: 12, step: 0.001, fwhm: 0.2 });
    // Σ A(e) · step ≈ Σ f_n (Gaussians integrate to 1).
    let integral = 0;
    for (const p of s) integral += p.absorption;
    integral *= 0.001;
    const totalF = 0.5 + 0.3 + 0.2;
    expect(integral).toBeCloseTo(totalF, 2);
  });

  test("Peak finder: threshold filters out small peaks", () => {
    const exc = [
      { energy: 3.0, oscillatorStrength: 0.01 },
      { energy: 6.0, oscillatorStrength: 1.0 },
    ];
    const s = broadenSpectrum(exc, { eMin: 0, eMax: 8, step: 0.01, fwhm: 0.3 });
    const allPeaks = findSpectrumPeaks(s, 0);
    const bigPeaks = findSpectrumPeaks(s, 0.5);
    expect(allPeaks.length).toBe(2);
    expect(bigPeaks.length).toBe(1);
    expect(bigPeaks[0]!.energy).toBeCloseTo(6, 1);
  });
});
