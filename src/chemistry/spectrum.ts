// ─────────────────────────────────────────────────────────────
// spectrum.ts — broaden discrete excitations (from TDDFT / EOM-CCSD /
// CIS) into a continuous absorption spectrum for comparison with
// experiment.
//
// Standard formula (Gaussian broadening at FWHM σ_FWHM):
//
//   σ_abs(ω) = Σ_n f_n · g(ω − ω_n; σ_gauss)
//   g(x; σ) = (1 / σ√(2π)) · exp(−x² / (2σ²))
//   σ_gauss = σ_FWHM / (2·√(2·ln 2)) ≈ σ_FWHM / 2.3548
//
// Output: array of (energy, absorption) pairs on a regular grid.
// Energy and broadening can be in any consistent unit (Hartree, eV,
// cm⁻¹, nm via the `energyUnit` discriminator); we use the user's
// chosen unit throughout.
//
// Typical broadening:
//   σ_FWHM = 0.3-0.5 eV   (UV-vis comparison with solution-phase
//                          experiment)
//   σ_FWHM = 30-50 cm⁻¹    (high-resolution gas-phase comparison)
// ─────────────────────────────────────────────────────────────

export interface SpectrumOpts {
  /** Energy grid spacing (same units as excitations).
   *  Default 0.01 eV equivalent (0.01 in whatever unit user uses).  */
  readonly step?: number;
  /** Energy grid lower bound (defaults to 0). */
  readonly eMin?: number;
  /** Energy grid upper bound (defaults to 1.2 × max(excitations)). */
  readonly eMax?: number;
  /** Gaussian FWHM broadening (same units as excitations). */
  readonly fwhm?: number;
}

export interface SpectrumPoint {
  /** Photon energy (input units). */
  readonly energy: number;
  /** Absorption value (sum of Gaussian-broadened oscillator strengths). */
  readonly absorption: number;
}

/**
 * Gaussian-broaden a set of discrete (energy, oscillator-strength)
 * excitations into a continuous absorption spectrum. The result is
 * an array of (energy, absorption) pairs on a regular grid.
 *
 * Useful for: comparing computed UV-vis to experimental absorption
 * spectra, generating publication-ready plots, identifying which
 * computed excitation corresponds to which experimental peak.
 */
export function broadenSpectrum(
  excitations: readonly { readonly energy: number; readonly oscillatorStrength: number }[],
  opts: SpectrumOpts = {},
): SpectrumPoint[] {
  if (excitations.length === 0) return [];
  const fwhm = opts.fwhm ?? 0.3;     // default 0.3 eV — typical UV-vis broadening
  // Convert FWHM to Gaussian σ.
  const sigma = fwhm / (2 * Math.sqrt(2 * Math.log(2)));
  const step = opts.step ?? 0.01;
  const maxE = excitations.reduce((m, e) => Math.max(m, e.energy), 0);
  const eMin = opts.eMin ?? 0;
  const eMax = opts.eMax ?? Math.max(maxE * 1.2, eMin + 1);

  const norm = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const result: SpectrumPoint[] = [];
  for (let e = eMin; e <= eMax + 1e-9; e += step) {
    let absorption = 0;
    for (const exc of excitations) {
      const dx = e - exc.energy;
      absorption += exc.oscillatorStrength * norm * Math.exp(-(dx * dx) / (2 * sigma * sigma));
    }
    result.push({ energy: e, absorption });
  }
  return result;
}

/**
 * Locate the absorption peaks (local maxima) in a Gaussian-broadened
 * spectrum. Each peak is a (energy, absorption) pair where the
 * absorption is greater than both neighbors. Useful for assigning
 * computed bands to experimental peaks.
 */
export function findSpectrumPeaks(
  spectrum: readonly SpectrumPoint[],
  threshold: number = 0,
): SpectrumPoint[] {
  const peaks: SpectrumPoint[] = [];
  for (let i = 1; i < spectrum.length - 1; i++) {
    const a = spectrum[i - 1]!.absorption;
    const b = spectrum[i]!.absorption;
    const c = spectrum[i + 1]!.absorption;
    if (b > a && b > c && b > threshold) {
      peaks.push({ energy: spectrum[i]!.energy, absorption: b });
    }
  }
  return peaks;
}
