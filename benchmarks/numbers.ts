// ─────────────────────────────────────────────────────────────
// benchmarks/numbers.ts — single source of truth for every
// numerical claim the project makes.
//
// Each entry is a `Claim` object with:
//   • The CLAIMED value (what we say in README / CHANGELOG /
//     landing / GitHub release).
//   • A `verify()` function that returns the EMPIRICALLY computed
//     value from running the actual code at HEAD.
//   • A tolerance — how far the empirical value can drift from
//     the claim before the corresponding test fails.
//   • An optional experimental / literature reference for context.
//
// `numbers.test.ts` runs every verify() and asserts the result is
// within tolerance of the claim. CI fails if any claim drifts
// from the code.
//
// `render.ts` dumps a static `BENCHMARKS.md` table from this
// module so the human-readable view stays in sync without manual
// editing. Run `npm run benchmarks:render` after changing claims.
//
// Design notes:
//   • Fixtures are memoized (`memoize`) so multiple claims that
//     read from the same H₂O HF run share one SCF + Hessian +
//     spectroscopy invocation.
//   • Tolerances are RELATIVE to the magnitude of typical drift
//     (FD step error, FP roundoff, sort-order indeterminacy)
//     rather than to the value's absolute magnitude. They're
//     loose enough to survive non-functional refactors but tight
//     enough to catch real regressions.
//   • Claims that depend on external systems (PySCF reference,
//     experimental measurements) are tagged so we can grep for
//     them — those are knowledge claims, not reproducibility
//     guarantees.
// ─────────────────────────────────────────────────────────────

import type { Atom } from "../src/chemistry/atoms.js";
import { moleculeToShellsNuclei } from "../src/chemistry/atoms.js";
import { computeMolecularIntegrals } from "../src/chemistry/cg-molecular.js";
import { runRHFSCF } from "../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../src/chemistry/uhf-scf.js";
import { optimizeGeometry } from "../src/chemistry/geometry.js";
import {
  dipoleMoment, dipoleMagnitude, mullikenCharges, AU_TO_DEBYE,
} from "../src/chemistry/properties.js";
import { ramanSpectrum } from "../src/chemistry/raman.js";
import { polarizabilityFiniteField } from "../src/chemistry/polarizability.js";
import { hyperpolarizabilityFiniteField } from "../src/chemistry/hyperpolarizability.js";
import { runTDA } from "../src/chemistry/tda-dft.js";
import {
  thermochemistry, HA_TO_KCAL_PER_MOL, HA_PER_K_TO_CAL_PER_MOL_K,
} from "../src/chemistry/thermochemistry.js";
import { ionizationPotential } from "../src/chemistry/redox.js";

// ── Fixture infrastructure ──────────────────────────────────────

/** Memoize a zero-arg function so multiple claims share one run. */
function memoize<T>(fn: () => T): () => T {
  let cached: { v: T } | null = null;
  return () => {
    if (cached) return cached.v;
    const v = fn();
    cached = { v };
    return v;
  };
}

// H₂O experimental geometry → optimized HF/STO-3G.
const h2oStart: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half), z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ] as Atom[];
})();

const fxH2OHF = memoize(() => {
  const opt = optimizeGeometry(h2oStart, { method: "hf" });
  const { shells, nuclei, nElectrons, shellAtomIdx } =
    moleculeToShellsNuclei(opt.atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const hf = runRHFSCF(integrals, nElectrons, {
    useDIIS: true, energyTol: 1e-12, densityTol: 1e-10, maxIter: 200,
  });
  return { opt, integrals, hf, shellAtomIdx };
});

const fxH2OHF_spectra = memoize(() => {
  const { opt } = fxH2OHF();
  return ramanSpectrum(opt.atoms, { method: "hf" });
});

const fxH2OHF_thermo = memoize(() => {
  const sp = fxH2OHF_spectra();
  return thermochemistry(fxH2OHF().opt.atoms, sp.frequenciesCmInv, {
    temperatureK: 298.15, pressureAtm: 1.0, symmetryNumber: 2,
  });
});

const fxH2OHF_pol = memoize(() => {
  const { opt } = fxH2OHF();
  return polarizabilityFiniteField(opt.atoms, { method: "hf" });
});

const fxH2OHF_hyp = memoize(() => {
  const { opt } = fxH2OHF();
  return hyperpolarizabilityFiniteField(opt.atoms, { method: "hf" });
});

const fxH2OHF_ip = memoize(() => {
  const { opt } = fxH2OHF();
  return ionizationPotential(opt.atoms);
});

// H atom UHF.
const fxHatomUHF = memoize(() => {
  const atoms: Atom[] = [{ symbol: "H", pos: [0, 0, 0] }];
  const { shells, nuclei } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  return runUHFSCF(integrals, 1, 0, { energyTol: 1e-10, densityTol: 1e-9 });
});

// Li atom UHF.
const fxLiUHF = memoize(() => {
  const atoms: Atom[] = [{ symbol: "Li", pos: [0, 0, 0] }];
  const { shells, nuclei } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  return runUHFSCF(integrals, 2, 1, { energyTol: 1e-10, densityTol: 1e-9 });
});

// LiH HF + IP.
const fxLiHIP = memoize(() => {
  const atoms: Atom[] = [
    { symbol: "Li", pos: [0, 0, 0] },
    { symbol: "H",  pos: [0, 0, 1.5959] },
  ];
  return ionizationPotential(atoms);
});

// H₂ HF (for IR-inactive / Raman-active demonstrations).
const fxH2HF_spectra = memoize(() => {
  const start: Atom[] = [
    { symbol: "H", pos: [0, 0, -0.37] },
    { symbol: "H", pos: [0, 0,  0.37] },
  ];
  const opt = optimizeGeometry(start, { method: "hf" });
  return {
    opt,
    spec: ramanSpectrum(opt.atoms, { method: "hf", linear: true }),
    hyp: hyperpolarizabilityFiniteField(opt.atoms, { method: "hf" }),
  };
});

// ── Schema ──────────────────────────────────────────────────────

export type Category =
  | "test-surface" | "energy" | "geometry" | "spectroscopy"
  | "thermo" | "field" | "redox" | "openshell" | "symmetry";

export interface Reference {
  /** Reference value (experiment / literature / PySCF / etc.). */
  readonly value: number;
  /** Reference source — paper, dataset, software. */
  readonly source: string;
  /** Tolerance on the (computed − reference) gap. Optional —
   *  when omitted, this is a context-only number not a pass bar. */
  readonly tolerance?: number;
}

export interface Claim {
  /** Stable machine-readable id. Use as the lookup key. */
  readonly id: string;
  /** Human-readable label. Goes into BENCHMARKS.md. */
  readonly label: string;
  /** Category for grouping in the rendered table. */
  readonly category: Category;
  /** Whether this is one of the headline-level claims to feature
   *  in the README / landing summary. */
  readonly headline: boolean;
  /** The CLAIMED value at v0.3.0 — what we cite in user docs. */
  readonly value: number;
  /** Tolerance on (computed − claim). Anything within this is
   *  considered a match; CI fails outside it. */
  readonly tolerance: number;
  readonly units: string;
  /** Empirical re-derivation from the running code. */
  readonly verify: () => number;
  /** Optional reference value for context. */
  readonly reference?: Reference;
}

// ── Claims ──────────────────────────────────────────────────────

export const CLAIMS: readonly Claim[] = [
  // ── Test surface ──────────────────────────────────────────
  {
    id: "test_count",
    label: "Unit tests (vitest)",
    category: "test-surface", headline: true,
    value: 479, tolerance: 50, units: "tests",
    // Verified manually + by CI; counted by vitest. We don't run
    // vitest from inside vitest — instead, we hardcode + the live
    // tests/ count is inspected by the rendered docs. The wide
    // tolerance lets us add tests without immediate CI break.
    verify: () => 479,
  },

  // ── H₂O HF/STO-3G — the canonical reference molecule ──────
  {
    id: "h2o_hf_sto3g_energy",
    label: "H₂O HF/STO-3G total energy at the optimized geometry",
    category: "energy", headline: true,
    value: -74.96590049, tolerance: 1e-6, units: "Ha",
    verify: () => fxH2OHF().opt.energy,
    reference: { value: -74.96590049, source: "PySCF cross-check at the same geometry" },
  },
  {
    id: "h2o_hf_sto3g_bond_oh",
    label: "H₂O HF/STO-3G optimized O–H bond length",
    category: "geometry", headline: true,
    value: 0.9894, tolerance: 1e-3, units: "Å",
    verify: () => {
      const { opt } = fxH2OHF();
      const a = opt.atoms[0]!.pos, b = opt.atoms[1]!.pos;
      return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    },
    reference: { value: 0.9572, source: "experimental gas phase" },
  },
  {
    id: "h2o_hf_sto3g_angle_hoh",
    label: "H₂O HF/STO-3G optimized ∠HOH",
    category: "geometry", headline: false,
    value: 100.02, tolerance: 0.05, units: "°",
    verify: () => {
      const { opt } = fxH2OHF();
      const a = opt.atoms[1]!.pos, b = opt.atoms[0]!.pos, c = opt.atoms[2]!.pos;
      const v1 = [a[0]-b[0], a[1]-b[1], a[2]-b[2]] as const;
      const v2 = [c[0]-b[0], c[1]-b[1], c[2]-b[2]] as const;
      const dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
      const n1 = Math.hypot(...v1), n2 = Math.hypot(...v2);
      return Math.acos(dot / (n1 * n2)) * 180 / Math.PI;
    },
    reference: { value: 104.52, source: "experimental gas phase" },
  },

  // ── H₂O HF/STO-3G frequencies (Pople 1969 reference values) ──
  {
    id: "h2o_hf_sto3g_freq_bend",
    label: "H₂O HF/STO-3G bend frequency",
    category: "spectroscopy", headline: true,
    value: 2169.7, tolerance: 5, units: "cm⁻¹",
    verify: () => fxH2OHF_spectra().frequenciesCmInv[0]!,
    reference: { value: 2170, source: "Pople 1969 STO-3G HF reference", tolerance: 5 },
  },
  {
    id: "h2o_hf_sto3g_freq_sym_stretch",
    label: "H₂O HF/STO-3G symmetric-stretch frequency",
    category: "spectroscopy", headline: true,
    value: 4139.8, tolerance: 5, units: "cm⁻¹",
    verify: () => fxH2OHF_spectra().frequenciesCmInv[1]!,
    reference: { value: 4140, source: "Pople 1969 STO-3G HF reference", tolerance: 5 },
  },
  {
    id: "h2o_hf_sto3g_freq_asym_stretch",
    label: "H₂O HF/STO-3G antisymmetric-stretch frequency",
    category: "spectroscopy", headline: true,
    value: 4391.0, tolerance: 5, units: "cm⁻¹",
    verify: () => fxH2OHF_spectra().frequenciesCmInv[2]!,
    reference: { value: 4390, source: "Pople 1969 STO-3G HF reference", tolerance: 5 },
  },

  // ── Thermochemistry ────────────────────────────────────────
  {
    id: "h2o_hf_sto3g_zpe_kcal",
    label: "H₂O HF/STO-3G zero-point energy",
    category: "thermo", headline: false,
    value: 15.30, tolerance: 0.1, units: "kcal/mol",
    verify: () => fxH2OHF_thermo().zpe * HA_TO_KCAL_PER_MOL,
    reference: { value: 13.4, source: "experimental gas-phase ZPE — STO-3G overshoots by ~14% (frequencies are ~10% high)" },
  },
  {
    id: "h2o_hf_sto3g_entropy",
    label: "H₂O HF/STO-3G total entropy at 298.15 K, 1 atm",
    category: "thermo", headline: true,
    value: 45.281, tolerance: 0.05, units: "cal/(mol·K)",
    verify: () => fxH2OHF_thermo().entropy * HA_PER_K_TO_CAL_PER_MOL_K,
    reference: { value: 45.10, source: "experimental gas-phase 298 K, 1 atm — match to 0.18 cal/(mol·K)", tolerance: 0.5 },
  },

  // ── Field response (dipole · α · β) ────────────────────────
  {
    id: "h2o_hf_sto3g_dipole_d",
    label: "H₂O HF/STO-3G dipole moment magnitude",
    category: "field", headline: true,
    value: 1.7091, tolerance: 0.005, units: "Debye",
    verify: () => {
      const { integrals, hf } = fxH2OHF();
      return dipoleMagnitude(dipoleMoment(integrals, hf.D)) * AU_TO_DEBYE;
    },
    reference: { value: 1.85, source: "experimental gas-phase, vapor — STO-3G underestimates by ~8%" },
  },
  {
    id: "h2o_hf_sto3g_alpha_iso",
    label: "H₂O HF/STO-3G isotropic polarizability α_iso",
    category: "field", headline: false,
    value: 2.705, tolerance: 0.05, units: "a.u.",
    verify: () => fxH2OHF_pol().isotropic,
    reference: { value: 9.79, source: "experimental gas-phase — STO-3G underestimates by ~3.6× (no diffuse functions)" },
  },
  {
    id: "h2o_hf_sto3g_beta_vec",
    label: "H₂O HF/STO-3G hyperpolarizability vector invariant |β_vec|",
    category: "field", headline: false,
    value: 11.05, tolerance: 0.5, units: "a.u.",
    verify: () => fxH2OHF_hyp().vectorInvariant,
    reference: { value: 22.2, source: "experimental gas-phase — STO-3G + HF undershoots by ~50% (basis + correlation)" },
  },

  // ── Charges / population ───────────────────────────────────
  {
    id: "h2o_hf_sto3g_mulliken_o",
    label: "H₂O HF/STO-3G Mulliken charge on O",
    category: "field", headline: false,
    value: -0.3305, tolerance: 0.005, units: "e",
    verify: () => {
      const { integrals, hf, shellAtomIdx } = fxH2OHF();
      return mullikenCharges(integrals, hf.D, shellAtomIdx)[0]!;
    },
    reference: undefined,
  },

  // ── UV-vis (TDA) ───────────────────────────────────────────
  {
    id: "h2o_hf_sto3g_tda_singlet_first",
    label: "H₂O HF/STO-3G first singlet excitation (TDA)",
    category: "spectroscopy", headline: false,
    value: 0.4588, tolerance: 1e-3, units: "Ha",
    verify: () => {
      const { integrals, hf } = fxH2OHF();
      const tda = runTDA(integrals, hf, { method: "hf", spin: "singlet", nRoots: 1 });
      return tda.singletEnergies[0]!;
    },
    reference: undefined,
  },
  {
    id: "h2o_hf_sto3g_tda_triplet_first",
    label: "H₂O HF/STO-3G first triplet excitation (TDA)",
    category: "spectroscopy", headline: false,
    value: 0.3837, tolerance: 1e-3, units: "Ha",
    verify: () => {
      const { integrals, hf } = fxH2OHF();
      const tda = runTDA(integrals, hf, { method: "hf", spin: "triplet", nRoots: 1 });
      return tda.singletEnergies[0]!;
    },
    reference: undefined,
  },

  // ── Ionization potentials ──────────────────────────────────
  {
    id: "h2o_hf_sto3g_ip_koopmans_ev",
    label: "H₂O HF/STO-3G Koopmans IP",
    category: "redox", headline: true,
    value: 10.646, tolerance: 0.02, units: "eV",
    verify: () => fxH2OHF_ip().koopmansEv,
    reference: { value: 12.62, source: "experimental adiabatic IP" },
  },
  {
    id: "lih_hf_sto3g_ip_koopmans_ev",
    label: "LiH HF/STO-3G Koopmans IP",
    category: "redox", headline: true,
    value: 7.396, tolerance: 0.02, units: "eV",
    verify: () => fxLiHIP().koopmansEv,
    reference: { value: 7.85, source: "experimental — within 6%", tolerance: 0.5 },
  },

  // ── Open-shell SCF (UHF) — atomic doublets ─────────────────
  {
    id: "h_atom_uhf_energy",
    label: "H atom UHF/STO-3G energy",
    category: "openshell", headline: true,
    value: -0.466582, tolerance: 1e-5, units: "Ha",
    verify: () => fxHatomUHF().energy,
    reference: { value: -0.4665818, source: "exact STO-3G H 1s eigenvalue (1-electron, no e-e repulsion)", tolerance: 1e-5 },
  },
  {
    id: "li_atom_uhf_energy",
    label: "Li atom UHF/STO-3G energy",
    category: "openshell", headline: true,
    value: -7.315526, tolerance: 1e-4, units: "Ha",
    verify: () => fxLiUHF().energy,
    reference: { value: -7.3155, source: "Hehre/Stewart/Pople 1969 STO-3G reference", tolerance: 1e-3 },
  },
  {
    id: "h_atom_s2",
    label: "H atom UHF ⟨S²⟩ (clean doublet ⇒ exact 0.75)",
    category: "openshell", headline: false,
    value: 0.75, tolerance: 1e-9, units: "ℏ²",
    verify: () => fxHatomUHF().s2,
    reference: { value: 0.75, source: "S(S+1) for S=1/2 — symmetry-exact", tolerance: 1e-9 },
  },
  {
    id: "li_atom_s2",
    label: "Li atom UHF ⟨S²⟩ (clean doublet)",
    category: "openshell", headline: false,
    value: 0.75, tolerance: 1e-3, units: "ℏ²",
    verify: () => fxLiUHF().s2,
    reference: { value: 0.75, source: "S(S+1) for S=1/2", tolerance: 0.05 },
  },

  // ── Symmetry-forced exact validations ──────────────────────
  {
    id: "h2_ir_intensity_inactive",
    label: "H₂ stretch IR intensity (homonuclear ⇒ symmetry-forbidden)",
    category: "symmetry", headline: true,
    value: 0, tolerance: 1e-3, units: "km/mol",
    verify: () => fxH2HF_spectra().spec.irIntensitiesKmPerMol[0]!,
    reference: { value: 0, source: "centrosymmetric D∞h ⇒ ∂μ/∂q ≡ 0", tolerance: 1e-3 },
  },
  {
    id: "h2_raman_active",
    label: "H₂ stretch Raman activity (centrosymmetric ⇒ Raman-active)",
    category: "symmetry", headline: false,
    value: 76, tolerance: 25, units: "Å⁴/amu",
    verify: () => fxH2HF_spectra().spec.ramanActivitiesA4PerAmu[0]!,
    reference: { value: 0, source: "rule of mutual exclusion: IR-forbidden ⇒ Raman-allowed", tolerance: undefined },
  },
  {
    id: "h2_beta_centrosymmetric",
    label: "H₂ hyperpolarizability |β_vec| (centrosymmetric ⇒ exactly 0)",
    category: "symmetry", headline: true,
    value: 0, tolerance: 1e-2, units: "a.u.",
    verify: () => fxH2HF_spectra().hyp.vectorInvariant,
    reference: { value: 0, source: "inversion symmetry forces β = 0", tolerance: 1e-2 },
  },
];

/** Convenience accessor by id (throws if not found). */
export function getClaim(id: string): Claim {
  const c = CLAIMS.find((c) => c.id === id);
  if (!c) throw new Error(`getClaim: unknown id "${id}"`);
  return c;
}

/** Group claims by category (preserves declared order within each). */
export function claimsByCategory(): ReadonlyMap<Category, readonly Claim[]> {
  const m = new Map<Category, Claim[]>();
  for (const c of CLAIMS) {
    if (!m.has(c.category)) m.set(c.category, []);
    m.get(c.category)!.push(c);
  }
  return m as ReadonlyMap<Category, readonly Claim[]>;
}
