// ─────────────────────────────────────────────────────────────
// units.ts — physical unit conversions for quantum chemistry.
//
// All conversion factors from CODATA 2018 / NIST 2024. Sources cited
// in trailing comments. The convention in this codebase is: internal
// calculations always in atomic units (Hartree energies, Bohr lengths,
// e charges, atomic units of dipole = e·Bohr), conversions done at
// I/O boundaries.
// ─────────────────────────────────────────────────────────────

// ─── Energy ──────────────────────────────────────────────────
export const HARTREE_TO_EV          = 27.211386245988;        // NIST 2018
export const HARTREE_TO_KCAL_PER_MOL = 627.5094740631;
export const HARTREE_TO_KJ_PER_MOL  = 2625.4996394798;
export const HARTREE_TO_WAVENUMBER  = 219474.6313632;          // cm⁻¹
export const HARTREE_TO_HZ          = 6.5796839204999e15;      // s⁻¹
export const HARTREE_TO_KELVIN      = 315775.02480407;         // 1 Ha = k_B · T
export const HARTREE_TO_JOULE       = 4.3597447222071e-18;     // SI

export const EV_TO_HARTREE          = 1 / HARTREE_TO_EV;
export const KCAL_PER_MOL_TO_HARTREE = 1 / HARTREE_TO_KCAL_PER_MOL;
export const KJ_PER_MOL_TO_HARTREE  = 1 / HARTREE_TO_KJ_PER_MOL;

// ─── Length ──────────────────────────────────────────────────
export const BOHR_TO_ANGSTROM       = 0.529177210903;
export const ANGSTROM_TO_BOHR       = 1 / BOHR_TO_ANGSTROM;
export const BOHR_TO_PICOMETRE      = BOHR_TO_ANGSTROM * 100;

// ─── Dipole moment ───────────────────────────────────────────
export const AU_DIPOLE_TO_DEBYE     = 2.541746473;             // e·Bohr → Debye
export const DEBYE_TO_AU_DIPOLE     = 1 / AU_DIPOLE_TO_DEBYE;

// ─── Polarizability (e²·a₀²/E_h) ────────────────────────────
export const AU_POL_TO_CUBIC_ANGSTROM = 0.14818471;            // α in a.u. → Å³
export const AU_POL_TO_CUBIC_BOHR     = 1;                     // a.u. polarizability is already Bohr³

// ─── Mass ────────────────────────────────────────────────────
export const AMU_TO_KG              = 1.66053906660e-27;
export const ELECTRON_MASS_AMU      = 5.48579909065e-4;        // m_e in amu

// ─── Spectroscopy ────────────────────────────────────────────
export const WAVENUMBER_TO_GHZ      = 29.9792458;              // 1 cm⁻¹ → 29.98 GHz
export const NM_TO_WAVENUMBER       = (lambda_nm: number) => 1e7 / lambda_nm;
export const WAVENUMBER_TO_NM       = (nu_cm1: number) => 1e7 / nu_cm1;
export const HARTREE_TO_NM          = (E_Ha: number) => 1e7 / (E_Ha * HARTREE_TO_WAVENUMBER);

// ─── Convenience converters with type-safe signatures ────────

/** Hartree → eV (default), kcal/mol, kJ/mol, or cm⁻¹. */
export function fromHartree(E_Ha: number, unit: "eV" | "kcal/mol" | "kJ/mol" | "cm-1" | "K"): number {
  switch (unit) {
    case "eV":       return E_Ha * HARTREE_TO_EV;
    case "kcal/mol": return E_Ha * HARTREE_TO_KCAL_PER_MOL;
    case "kJ/mol":   return E_Ha * HARTREE_TO_KJ_PER_MOL;
    case "cm-1":     return E_Ha * HARTREE_TO_WAVENUMBER;
    case "K":        return E_Ha * HARTREE_TO_KELVIN;
  }
}

/** eV / kcal/mol / kJ/mol / cm⁻¹ → Hartree. */
export function toHartree(value: number, unit: "eV" | "kcal/mol" | "kJ/mol" | "cm-1" | "K"): number {
  switch (unit) {
    case "eV":       return value / HARTREE_TO_EV;
    case "kcal/mol": return value / HARTREE_TO_KCAL_PER_MOL;
    case "kJ/mol":   return value / HARTREE_TO_KJ_PER_MOL;
    case "cm-1":     return value / HARTREE_TO_WAVENUMBER;
    case "K":        return value / HARTREE_TO_KELVIN;
  }
}
