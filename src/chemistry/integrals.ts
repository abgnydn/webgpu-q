// ─────────────────────────────────────────────────────────────
// integrals.ts — Gaussian-primitive molecular integrals for H₂.
//
// H₂ in STO-3G has only s-functions on both atoms, so every
// integral here is over normalized 1s-Gaussians — no angular
// momentum bookkeeping, just exponents, centers, and the Boys
// function F_0(t) = 0.5·√(π/t)·erf(√t) for the Coulomb cores.
//
// Conventions:
//   • Atomic units throughout (Hartree, Bohr).
//   • All integrals are over CONTRACTED Gaussian shells; per-
//     primitive sums get rolled in by `contracted{S,T,V,ERI}`.
//   • Chemist notation (μν|λσ) = ∫∫ φ_μ(1)φ_ν(1) (1/r12) φ_λ(2)φ_σ(2).
//
// Reference: Helgaker, Jørgensen, Olsen — Molecular Electronic
// Structure Theory, Ch. 9 (Obara–Saika recurrence reduces to
// these closed forms for s-shells only).
// ─────────────────────────────────────────────────────────────

/** STO-3G 1s contraction for hydrogen.
 *  α: Gaussian exponents;  c: contraction coefficients (un-normalized
 *  on the *contracted* level — per-primitive normalization is rolled
 *  into the d_i below). */
export const STO3G_H_1S = {
  alpha: [3.42525091, 0.62391373, 0.16885540] as const,
  // Pople-style "d" coefficients (Hehre, Stewart, Pople 1969 Table II).
  c: [0.15432897, 0.53532814, 0.44463454] as const,
};

/** STO-3G 1s contraction for helium. From EMSL Basis Set Exchange
 *  (Hehre, Stewart, Pople 1969). Closed-shell Z = 2; orbital tighter
 *  than H 1s by roughly factor 2 in exponent. */
export const STO3G_HE_1S = {
  alpha: [6.36242139, 1.15892300, 0.31364979] as const,
  c: [0.15432897, 0.53532814, 0.44463454] as const,
};

/** STO-3G 1s contraction for lithium. Tighter exponents than H 1s
 *  (Z = 3 vs Z = 1 → orbital pulled in toward the nucleus). */
export const STO3G_LI_1S = {
  alpha: [16.1195750, 2.9362007, 0.7946505] as const,
  c: [0.15432897, 0.53532814, 0.44463454] as const,
};

/** STO-3G 1s contraction for beryllium (Z = 4). Tighter exponents than
 *  Li 1s (Z = 3); the L-shell coefficients below carry the 2s character. */
export const STO3G_BE_1S = {
  alpha: [30.1678710, 5.4951153, 1.4871927] as const,
  c: [0.15432897, 0.53532814, 0.44463454] as const,
};

/** STO-3G 2s contraction for beryllium — the *s component* of the L-shell.
 *  Same exponent set as the 2p L-shell (added in Phase C v3); coefficients
 *  are Pople 1969 Table III. Negative leading coefficient produces the 2s
 *  radial node and enforces orthogonality with 1s post-Löwdin. */
export const STO3G_BE_2S = {
  alpha: [1.3148331, 0.3055389, 0.0993707] as const,
  c: [-0.09996723, 0.39951283, 0.70011547] as const,
};

/** STO-3G 2p contraction for beryllium — the *p component* of the L-shell.
 *  Same exponent set as STO3G_BE_2S; the only difference is the
 *  contraction coefficients (all positive for p, since the 2p radial
 *  has no node). One Cartesian-Gaussian shell per Cartesian direction
 *  (p_x, p_y, p_z) — pair this with `makeCGShell(STO3G_BE_2P, center,
 *  [1, 0, 0])` etc. from `integrals-cg.ts` to build them. */
export const STO3G_BE_2P = {
  alpha: [1.3148331, 0.3055389, 0.0993707] as const,
  c: [0.15591627, 0.60768372, 0.39195739] as const,
};

// ── Pople 1969 STO-3G for second-row atoms (B–Ne) ─────────────
// All second-row atoms share the same s-shell d-coefficients
// (radial form is universal in the STO-3G parametrization);
// only the exponent sets differ from atom to atom.
//
// L-shell (2sp): same exponents for s and p; same s-d-coeffs and
// p-d-coeffs as Be (Pople 1969 Table III).

const STO3G_L_2S_C = [-0.09996723, 0.39951283, 0.70011547] as const;
const STO3G_L_2P_C = [ 0.15591627, 0.60768372, 0.39195739] as const;
const STO3G_S_C    = [ 0.15432897, 0.53532814, 0.44463454] as const;

/** STO-3G 1s for carbon (Z = 6). Pople 1969. */
export const STO3G_C_1S = {
  alpha: [71.6168370, 13.0450960, 3.5305122] as const,
  c: STO3G_S_C,
};
export const STO3G_C_2S = {
  alpha: [2.9412494, 0.6834831, 0.2222899] as const,
  c: STO3G_L_2S_C,
};
export const STO3G_C_2P = {
  alpha: [2.9412494, 0.6834831, 0.2222899] as const,
  c: STO3G_L_2P_C,
};

/** STO-3G 1s for nitrogen (Z = 7). Pople 1969. */
export const STO3G_N_1S = {
  alpha: [99.1061690, 18.0523120, 4.8856602] as const,
  c: STO3G_S_C,
};
export const STO3G_N_2S = {
  alpha: [3.7804559, 0.8784966, 0.2857144] as const,
  c: STO3G_L_2S_C,
};
export const STO3G_N_2P = {
  alpha: [3.7804559, 0.8784966, 0.2857144] as const,
  c: STO3G_L_2P_C,
};

/** STO-3G 1s for oxygen (Z = 8). Pople 1969. */
export const STO3G_O_1S = {
  alpha: [130.7093200, 23.8088610, 6.4436083] as const,
  c: STO3G_S_C,
};
export const STO3G_O_2S = {
  alpha: [5.0331513, 1.1695961, 0.3803890] as const,
  c: STO3G_L_2S_C,
};
export const STO3G_O_2P = {
  alpha: [5.0331513, 1.1695961, 0.3803890] as const,
  c: STO3G_L_2P_C,
};

// ── cc-pVDZ basis sets (Dunning, T.H. 1989) ───────────────────
// "Correlation-consistent polarized valence double-zeta" — the
// smallest "real" basis set chemistry codes use. STO-3G is a toy;
// cc-pVDZ is what 90% of academic chemistry papers rely on for
// preliminary calculations. Pharma uses cc-pVTZ or cc-pVQZ for
// final numbers, but cc-pVDZ is the entry-level standard.
//
// For each atom cc-pVDZ has split valence + polarization shells.
// Adopted convention (matches PySCF / EMSL Basis Set Exchange):
//   H:   2s + 1p           (5 basis functions)
//   C:   3s + 2p + 1d      (14 basis functions)
//   N/O: 3s + 2p + 1d      (14 basis functions)
//
// Ordering of shell exports: contracted shells first, uncontracted
// (single-primitive) split-valence shells next, polarization last.
//
// All coefficients from EMSL Basis Set Exchange, retrieved
// 2026-05-05.

// ── Hydrogen cc-pVDZ ─────────────────────────────────────────
// 2 s-shells (one contracted, one uncontracted) + 1 p-shell.
// Corrected 2026-08-10 against pyscf.gto.basis.load("cc-pvdz", "H").
// The exponents carried extra non-canonical digits (13.0107010 /
// 1.9622572 / 0.4445298 / 0.1219496) against Dunning's published
// 13.01 / 1.962 / 0.4446 / 0.122. Coefficients were already correct.
// Small (~2-5 uHa) but hydrogen is in nearly every validation molecule,
// so it put a permanent floor under every cc-pVDZ vs-PySCF comparison.
export const CCPVDZ_H_1S = {
  alpha: [13.0100000, 1.9620000, 0.4446000] as const,
  c: [0.0196850, 0.1379770, 0.4781480] as const,
};
export const CCPVDZ_H_2S = {
  alpha: [0.1220000] as const,
  c: [1.0] as const,
};
export const CCPVDZ_H_2P = {
  alpha: [0.7270000] as const,
  c: [1.0] as const,
};

// ── Helium cc-pVDZ ───────────────────────────────────────────
// 2 s-shells + 1 p-shell (EMSL Basis Set Exchange — Dunning 1989).
// He cc-pVDZ has 5 basis functions: 1s, 2s, 2p_{x,y,z}.
export const CCPVDZ_HE_1S = {
  alpha: [38.3600000, 5.7700000, 1.2400000] as const,
  c: [0.0238090, 0.1548910, 0.4699870] as const,
};
export const CCPVDZ_HE_2S = {
  alpha: [0.2976000] as const,
  c: [1.0] as const,
};
export const CCPVDZ_HE_2P = {
  alpha: [1.2750000] as const,
  c: [1.0] as const,
};

// ── Helium aug-cc-pVDZ diffuse ───────────────────────────────
// One diffuse s + one diffuse p (EMSL).
// Corrected 2026-08-10 against pyscf.gto.basis.load("aug-cc-pvdz", "He")
// — was 0.0713 / 0.3300.
export const AUG_CCPVDZ_HE_DIFFUSE_S = {
  alpha: [0.0725500] as const, c: [1.0] as const,
};
export const AUG_CCPVDZ_HE_DIFFUSE_P = {
  alpha: [0.2473000] as const, c: [1.0] as const,
};

// ── Oxygen cc-pVDZ ────────────────────────────────────────────
// 3 s-shells + 2 p-shells + 1 d-shell. Coefficients are the
// "general contraction" form: a single block of primitives with
// per-shell contraction vectors. Here we expand to per-shell
// contractions matching the integrals-cg.ts CGShell convention.
//
// EMSL ccPVDZ for O — radial primitives split into:
//   1s (deep core, 8 prims), 2s (split valence, 8 prims contracted
//   differently), 2s' (uncontracted), 2p, 2p', 3d.
export const CCPVDZ_O_1S = {
  alpha: [
    11720.0000, 1759.0000, 400.8000, 113.7000, 37.0300,
    13.2700, 5.0250, 1.0130,
  ] as const,
  c: [
    0.000710, 0.005470, 0.027837, 0.104800, 0.283062,
    0.448719, 0.270952, 0.015458,
  ] as const,
};
// Inner valence "2s" — same primitive set as 1s, different coefficients.
export const CCPVDZ_O_2S = {
  alpha: [
    11720.0000, 1759.0000, 400.8000, 113.7000, 37.0300,
    13.2700, 5.0250, 1.0130,
  ] as const,
  c: [
    -0.000160, -0.001263, -0.006267, -0.025716, -0.070924,
    -0.165411, -0.116955, 0.557368,
  ] as const,
};
// Outer valence (uncontracted single primitive).
export const CCPVDZ_O_2S_P = {
  alpha: [0.3023] as const,
  c: [1.0] as const,
};
export const CCPVDZ_O_2P = {
  alpha: [17.7000, 3.8540, 1.0460] as const,
  c: [0.043018, 0.228913, 0.508728] as const,
};
export const CCPVDZ_O_2P_P = {
  alpha: [0.2753] as const,
  c: [1.0] as const,
};
export const CCPVDZ_O_3D = {
  alpha: [1.1850] as const,
  c: [1.0] as const,
};

// ── aug-cc-pVDZ diffuse functions ────────────────────────────
// Dunning's augmented basis: cc-pVDZ + one diffuse primitive per
// angular momentum class. Diffuse exponents are 1-2 orders of
// magnitude smaller than the smallest valence exponent, so the
// resulting basis covers the long tail of electron density that
// minimal/double-zeta basis sets miss. Required for any anion,
// excited state, or van der Waals interaction; for neutral
// closed-shell ground states it shifts HF by 1-3 mHa.
//
// Exponents from EMSL Basis Set Exchange aug-cc-pVDZ (retrieved
// 2026-05-06). Other atoms beyond H and O can be added the same
// way when needed.
export const AUG_CCPVDZ_H_DIFFUSE_S = {
  alpha: [0.0297400] as const,
  c: [1.0] as const,
};
export const AUG_CCPVDZ_H_DIFFUSE_P = {
  alpha: [0.1410000] as const,
  c: [1.0] as const,
};
// Corrected 2026-08-10 against pyscf.gto.basis.load("aug-cc-pvdz", "O")
// — was 0.0845800 / 0.0856000. The d exponent was already correct.
export const AUG_CCPVDZ_O_DIFFUSE_S = {
  alpha: [0.0789600] as const,
  c: [1.0] as const,
};
export const AUG_CCPVDZ_O_DIFFUSE_P = {
  alpha: [0.0685600] as const,
  c: [1.0] as const,
};
export const AUG_CCPVDZ_O_DIFFUSE_D = {
  alpha: [0.3320000] as const,
  c: [1.0] as const,
};

// aug-cc-pVDZ diffuse functions for Li, Be, C, N, F.
// One diffuse primitive per angular momentum class (s, p, d).
// EMSL Basis Set Exchange, retrieved 2026-05-18.
// Corrected 2026-08-10 against pyscf.gto.basis.load("aug-cc-pvdz", ...).
// Of the eight elements, only H and C were right; He/Li/Be/N/O/F all
// carried wrong diffuse exponents. Previous values, for the record:
//   Li  S 0.0072930  P 0.0074000  D 0.0950000
//   Be  S 0.0207000  P 0.0142000  D 0.0722000
//   N   S 0.0576000  P 0.0491000  (D was correct)
//   F   S 0.1076000  P 0.0832000  D 0.5000000
export const AUG_CCPVDZ_LI_DIFFUSE_S = { alpha: [0.0086400] as const, c: [1.0] as const };
export const AUG_CCPVDZ_LI_DIFFUSE_P = { alpha: [0.0057900] as const, c: [1.0] as const };
export const AUG_CCPVDZ_LI_DIFFUSE_D = { alpha: [0.0725000] as const, c: [1.0] as const };
export const AUG_CCPVDZ_BE_DIFFUSE_S = { alpha: [0.0187700] as const, c: [1.0] as const };
export const AUG_CCPVDZ_BE_DIFFUSE_P = { alpha: [0.0085000] as const, c: [1.0] as const };
export const AUG_CCPVDZ_BE_DIFFUSE_D = { alpha: [0.0740000] as const, c: [1.0] as const };
export const AUG_CCPVDZ_B_DIFFUSE_S  = { alpha: [0.0310500] as const, c: [1.0] as const };
export const AUG_CCPVDZ_B_DIFFUSE_P  = { alpha: [0.0237800] as const, c: [1.0] as const };
export const AUG_CCPVDZ_B_DIFFUSE_D  = { alpha: [0.0904000] as const, c: [1.0] as const };
export const AUG_CCPVDZ_NE_DIFFUSE_S = { alpha: [0.1230000] as const, c: [1.0] as const };
export const AUG_CCPVDZ_NE_DIFFUSE_P = { alpha: [0.1064000] as const, c: [1.0] as const };
export const AUG_CCPVDZ_NE_DIFFUSE_D = { alpha: [0.6310000] as const, c: [1.0] as const };
export const AUG_CCPVDZ_C_DIFFUSE_S  = { alpha: [0.0469000] as const, c: [1.0] as const };
export const AUG_CCPVDZ_C_DIFFUSE_P  = { alpha: [0.0404100] as const, c: [1.0] as const };
export const AUG_CCPVDZ_C_DIFFUSE_D  = { alpha: [0.1510000] as const, c: [1.0] as const };
export const AUG_CCPVDZ_N_DIFFUSE_S  = { alpha: [0.0612400] as const, c: [1.0] as const };
export const AUG_CCPVDZ_N_DIFFUSE_P  = { alpha: [0.0561100] as const, c: [1.0] as const };
export const AUG_CCPVDZ_N_DIFFUSE_D  = { alpha: [0.2300000] as const, c: [1.0] as const };
export const AUG_CCPVDZ_F_DIFFUSE_S  = { alpha: [0.0986300] as const, c: [1.0] as const };
export const AUG_CCPVDZ_F_DIFFUSE_P  = { alpha: [0.0850200] as const, c: [1.0] as const };
export const AUG_CCPVDZ_F_DIFFUSE_D  = { alpha: [0.4640000] as const, c: [1.0] as const };

/** STO-3G 2s contraction for lithium — the *s component* of the L-shell.
 *  Coefficients can be negative to produce the 2s radial node and to
 *  enforce orthogonality with 1s after Löwdin orthogonalization.
 *  STO-3G also defines a 2p L-shell on Li with the same exponents but
 *  different (positive) coefficients; that is omitted from this v0
 *  s-only basis (Phase C scope). */
export const STO3G_LI_2S = {
  alpha: [0.6362897, 0.1478601, 0.0480887] as const,
  c: [-0.09996723, 0.39951283, 0.70011547] as const,
};

// ── STO-3G for Fluorine ───────────────────────────────────────
// EMSL Basis Set Exchange STO-3G, retrieved 2026-05-18.
// Same L-shell pattern as C/N/O (3 primitives for 1s, 3 shared
// primitives for the 2s/2p L-shell).
export const STO3G_F_1S = {
  alpha: [166.6791340, 30.3608120, 8.2168207] as const,
  c: [0.15432897, 0.53532814, 0.44463454] as const,
};
export const STO3G_F_2S = {
  alpha: [6.4648032, 1.5022812, 0.4885885] as const,
  c: [-0.09996723, 0.39951283, 0.70011547] as const,
};
export const STO3G_F_2P = {
  alpha: [6.4648032, 1.5022812, 0.4885885] as const,
  c: [0.15591627, 0.60768372, 0.39195739] as const,
};

// ── STO-3G for Boron and Neon (2026-08-10) ───────────────────
// Generated by scripts/gen-basis-tables.py from PySCF, completing
// row 2. Structurally identical to C/N/O/F: 1s + an L-shell whose
// s and p components share exponents.
export const STO3G_B_1S = {
  alpha: [48.7911130, 8.8873622, 2.4052670] as const,
  c: STO3G_S_C,
};
export const STO3G_B_2S = {
  alpha: [2.2369561, 0.5198205, 0.1690618] as const,
  c: STO3G_L_2S_C,
};
export const STO3G_B_2P = {
  alpha: [2.2369561, 0.5198205, 0.1690618] as const,
  c: STO3G_L_2P_C,
};
export const STO3G_NE_1S = {
  alpha: [207.0156100, 37.7081510, 10.2052970] as const,
  c: STO3G_S_C,
};
export const STO3G_NE_2S = {
  alpha: [8.2463151, 1.9162662, 0.6232293] as const,
  c: STO3G_L_2S_C,
};
export const STO3G_NE_2P = {
  alpha: [8.2463151, 1.9162662, 0.6232293] as const,
  c: STO3G_L_2P_C,
};

// ── cc-pVDZ for Lithium ──────────────────────────────────────
// EMSL ccPVDZ for Li (9s,4p,1d) → [3s,2p,1d]. Retrieved 2026-05-18.
export const CCPVDZ_LI_1S = {
  alpha: [
    1469.0000, 220.5000, 50.2600, 14.2400,
    4.5810, 1.5800, 0.5640, 0.0734500,
  ] as const,
  c: [
    0.000766, 0.005892, 0.029671, 0.109180,
    0.282789, 0.453123, 0.274774, 0.009751,
  ] as const,
};
export const CCPVDZ_LI_2S = {
  alpha: [
    1469.0000, 220.5000, 50.2600, 14.2400,
    4.5810, 1.5800, 0.5640, 0.0734500,
  ] as const,
  c: [
    -0.000120, -0.000923, -0.004689, -0.017682,
    -0.048902, -0.096009, -0.136380, 0.575102,
  ] as const,
};
// Corrected 2026-08-10 against pyscf.gto.basis.load("cc-pvdz", "Li").
// The uncontracted s exponent read 0.0285000 (digit transposition of
// 0.02805) and the 2p contraction coefficients were wholly wrong —
// together worth 1.29 mHa on LiH, which is 2.6x the repo's own
// <= 0.5 mHa HF-vs-PySCF gate. See tests/chemistry/elements/.
export const CCPVDZ_LI_2S_P = {
  alpha: [0.0280500] as const,
  c: [1.0] as const,
};
export const CCPVDZ_LI_2P = {
  alpha: [1.5340, 0.2749, 0.073620] as const,
  c: [0.022784, 0.139107, 0.500375] as const,
};
export const CCPVDZ_LI_2P_P = {
  alpha: [0.024030] as const,
  c: [1.0] as const,
};
export const CCPVDZ_LI_3D = {
  alpha: [0.1239] as const,
  c: [1.0] as const,
};

// ── cc-pVDZ for Beryllium ─────────────────────────────────────
// EMSL ccPVDZ for Be (9s,4p,1d) → [3s,2p,1d]. Retrieved 2026-05-18.
export const CCPVDZ_BE_1S = {
  alpha: [
    2940.0000, 441.2000, 100.5000, 28.4300,
    9.1690, 3.1960, 1.1590, 0.1811,
  ] as const,
  // Corrected 2026-08-10 against pyscf.gto.basis.load("cc-pvdz", "Be"):
  // the tail was transcribed at reduced precision (0.4514 vs 0.451469,
  // 0.2950 vs 0.295074, 0.012580 vs 0.012587).
  c: [
    0.000680, 0.005236, 0.026606, 0.099993,
    0.269702, 0.451469, 0.295074, 0.012587,
  ] as const,
};
export const CCPVDZ_BE_2S = {
  alpha: [
    2940.0000, 441.2000, 100.5000, 28.4300,
    9.1690, 3.1960, 1.1590, 0.1811,
  ] as const,
  // Corrected 2026-08-10 against PySCF. Elements 4-8 were not merely
  // rounded but substantively different (-0.05403 vs -0.05328,
  // -0.1133 vs -0.120723, -0.1462 vs -0.133435, 0.5392 vs 0.530767),
  // i.e. sourced from a different/incorrect table.
  c: [
    -0.000123, -0.000966, -0.004831, -0.019314,
    -0.053280, -0.120723, -0.133435, 0.530767,
  ] as const,
};
export const CCPVDZ_BE_2S_P = {
  alpha: [0.0589000] as const,
  c: [1.0] as const,
};
export const CCPVDZ_BE_2P = {
  alpha: [3.6190, 0.7110, 0.1951] as const,
  // Corrected 2026-08-10 against PySCF (was 0.169650 / 0.487810 — the
  // last coefficient was off by 5%).
  c: [0.029111, 0.169365, 0.513458] as const,
};
export const CCPVDZ_BE_2P_P = {
  alpha: [0.060180] as const,
  c: [1.0] as const,
};
export const CCPVDZ_BE_3D = {
  alpha: [0.2380] as const,
  c: [1.0] as const,
};

// ── cc-pVDZ for Carbon ────────────────────────────────────────
// EMSL ccPVDZ for C (9s,4p,1d) → [3s,2p,1d]. Retrieved 2026-05-18.
export const CCPVDZ_C_1S = {
  alpha: [
    6665.0000, 1000.0000, 228.0000, 64.7100,
    21.0600, 7.4950, 2.7970, 0.5215,
  ] as const,
  c: [
    0.000692, 0.005329, 0.027077, 0.101718,
    0.274740, 0.448564, 0.285074, 0.015204,
  ] as const,
};
export const CCPVDZ_C_2S = {
  alpha: [
    6665.0000, 1000.0000, 228.0000, 64.7100,
    21.0600, 7.4950, 2.7970, 0.5215,
  ] as const,
  c: [
    -0.000146, -0.001154, -0.005725, -0.023312,
    -0.063955, -0.149981, -0.127262, 0.544529,
  ] as const,
};
export const CCPVDZ_C_2S_P = {
  alpha: [0.1596] as const,
  c: [1.0] as const,
};
export const CCPVDZ_C_2P = {
  alpha: [9.4390, 2.0020, 0.5456] as const,
  c: [0.038109, 0.209480, 0.508557] as const,
};
export const CCPVDZ_C_2P_P = {
  alpha: [0.1517] as const,
  c: [1.0] as const,
};
export const CCPVDZ_C_3D = {
  alpha: [0.5500] as const,
  c: [1.0] as const,
};

// ── cc-pVDZ for Nitrogen ──────────────────────────────────────
// EMSL ccPVDZ for N (9s,4p,1d) → [3s,2p,1d]. Retrieved 2026-05-18.
export const CCPVDZ_N_1S = {
  alpha: [
    9046.0000, 1357.0000, 309.3000, 87.7300,
    28.5600, 10.2100, 3.8380, 0.7466,
  ] as const,
  // Corrected 2026-08-10 against PySCF (was 0.278722 / 0.448581).
  c: [
    0.000700, 0.005389, 0.027406, 0.103207,
    0.278723, 0.448540, 0.278238, 0.015440,
  ] as const,
};
export const CCPVDZ_N_2S = {
  alpha: [
    9046.0000, 1357.0000, 309.3000, 87.7300,
    28.5600, 10.2100, 3.8380, 0.7466,
  ] as const,
  c: [
    -0.000153, -0.001208, -0.005992, -0.024544,
    -0.067459, -0.158078, -0.121831, 0.549003,
  ] as const,
};
export const CCPVDZ_N_2S_P = {
  alpha: [0.2248] as const,
  c: [1.0] as const,
};
export const CCPVDZ_N_2P = {
  alpha: [13.5500, 2.9170, 0.7973] as const,
  c: [0.039919, 0.217169, 0.510319] as const,
};
export const CCPVDZ_N_2P_P = {
  alpha: [0.2185] as const,
  c: [1.0] as const,
};
export const CCPVDZ_N_3D = {
  alpha: [0.8170] as const,
  c: [1.0] as const,
};

// ── cc-pVDZ for Fluorine ──────────────────────────────────────
// EMSL ccPVDZ for F (9s,4p,1d) → [3s,2p,1d]. Retrieved 2026-05-18.
export const CCPVDZ_F_1S = {
  alpha: [
    14710.0000, 2207.0000, 502.8000, 142.6000,
    46.4700, 16.7000, 6.3560, 1.3160,
  ] as const,
  c: [
    0.000721, 0.005553, 0.028267, 0.106444,
    0.286814, 0.448641, 0.264761, 0.015333,
  ] as const,
};
export const CCPVDZ_F_2S = {
  alpha: [
    14710.0000, 2207.0000, 502.8000, 142.6000,
    46.4700, 16.7000, 6.3560, 1.3160,
  ] as const,
  c: [
    -0.000165, -0.001308, -0.006495, -0.026691,
    -0.073690, -0.170776, -0.112327, 0.562814,
  ] as const,
};
export const CCPVDZ_F_2S_P = {
  alpha: [0.3897] as const,
  c: [1.0] as const,
};
export const CCPVDZ_F_2P = {
  alpha: [22.6700, 4.9770, 1.3470] as const,
  c: [0.044878, 0.235718, 0.508521] as const,
};
export const CCPVDZ_F_2P_P = {
  alpha: [0.3471] as const,
  c: [1.0] as const,
};
export const CCPVDZ_F_3D = {
  alpha: [1.6400] as const,
  c: [1.0] as const,
};

// ── cc-pVDZ for Boron and Neon (2026-08-10) ──────────────────
// Generated by scripts/gen-basis-tables.py from PySCF. Same
// [3s,2p,1d] shape as C/N/O/F, so `heavyShells` builds them
// unchanged: 15 Cartesian / 14 spherical functions per atom.
export const CCPVDZ_B_1S = {
  alpha: [
    4570.0000, 685.9000, 156.5000, 44.4700,
    14.4800, 5.1310, 1.8980, 0.3329,
  ] as const,
  c: [
    0.000696, 0.005353, 0.027134, 0.101380,
    0.272055, 0.448403, 0.290123, 0.014322,
  ] as const,
};
export const CCPVDZ_B_2S = {
  alpha: [
    4570.0000, 685.9000, 156.5000, 44.4700,
    14.4800, 5.1310, 1.8980, 0.3329,
  ] as const,
  c: [
    -0.000139, -0.001097, -0.005444, -0.021916,
    -0.059751, -0.138732, -0.131482, 0.539526,
  ] as const,
};
export const CCPVDZ_B_2S_P = {
  alpha: [0.1043000] as const,
  c: [1.0] as const,
};
export const CCPVDZ_B_2P = {
  alpha: [6.0010, 1.2410, 0.3364] as const,
  c: [0.035481, 0.198072, 0.505230] as const,
};
export const CCPVDZ_B_2P_P = {
  alpha: [0.0953800] as const,
  c: [1.0] as const,
};
export const CCPVDZ_B_3D = {
  alpha: [0.3430] as const,
  c: [1.0] as const,
};
export const CCPVDZ_NE_1S = {
  alpha: [
    17880.0000, 2683.0000, 611.5000, 173.5000,
    56.6400, 20.4200, 7.8100, 1.6530,
  ] as const,
  c: [
    0.000738, 0.005677, 0.028883, 0.108540,
    0.290907, 0.448324, 0.258026, 0.015063,
  ] as const,
};
export const CCPVDZ_NE_2S = {
  alpha: [
    17880.0000, 2683.0000, 611.5000, 173.5000,
    56.6400, 20.4200, 7.8100, 1.6530,
  ] as const,
  c: [
    -0.000172, -0.001357, -0.006737, -0.027663,
    -0.076208, -0.175227, -0.107038, 0.567050,
  ] as const,
};
export const CCPVDZ_NE_2S_P = {
  alpha: [0.4869000] as const,
  c: [1.0] as const,
};
export const CCPVDZ_NE_2P = {
  alpha: [28.3900, 6.2700, 1.6950] as const,
  c: [0.046087, 0.240181, 0.508744] as const,
};
export const CCPVDZ_NE_2P_P = {
  alpha: [0.4317000] as const,
  c: [1.0] as const,
};
export const CCPVDZ_NE_3D = {
  alpha: [2.2020] as const,
  c: [1.0] as const,
};

// ── Generic shell type + multi-shell integrals ───────────────
//
// A Shell is a contracted s-Gaussian: a center plus a list of
// primitive exponents and coefficients. The existing H₂ pipeline
// hard-codes H 1s; LiH (Phase C) needs to mix Li 1s, Li 2s, and
// H 1s, each with their own (alpha, c) arrays. The shell-based
// API below is the multi-element generalization.
//
// Per-primitive normalization N(α) = (2α/π)^(3/4) is applied
// inside the contractions so callers pass "raw" Pople-style
// d-coefficients (matching the constants above).

export interface Shell {
  /** Atomic center (Bohr). */
  readonly center: readonly [number, number, number];
  /** Primitive Gaussian exponents. */
  readonly alpha: readonly number[];
  /** Contraction coefficients (Pople d-style; primitive normalization rolled in here). */
  readonly c: readonly number[];
  /** Optional label, e.g. "Li:1s", "H:1s". For debugging only. */
  readonly label?: string;
}

/** Convenience: build a Shell from a basis-set entry + atom center. */
export function makeShell(
  basis: { readonly alpha: readonly number[]; readonly c: readonly number[] },
  center: readonly [number, number, number],
  label?: string,
): Shell {
  return { center, alpha: basis.alpha, c: basis.c, label };
}

/** ⟨A|B⟩ overlap between two contracted s-shells. */
export function S_shells(A: Shell, B: Shell): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ci = A.c[i]! * normS(ai);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const cj = B.c[j]! * normS(bj);
      s += ci * cj * primS(ai, A.center, bj, B.center);
    }
  }
  return s;
}

/** Kinetic ⟨A| -∇²/2 |B⟩ between two contracted s-shells. */
export function T_shells(A: Shell, B: Shell): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ci = A.c[i]! * normS(ai);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const cj = B.c[j]! * normS(bj);
      s += ci * cj * primT(ai, A.center, bj, B.center);
    }
  }
  return s;
}

/** Nuclear attraction ⟨A| -Z_C/|r-C| |B⟩ between two contracted s-shells. */
export function V_shells(
  A: Shell, B: Shell,
  Zc: number, C: readonly [number, number, number],
): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ci = A.c[i]! * normS(ai);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const cj = B.c[j]! * normS(bj);
      s += ci * cj * primV(ai, A.center, bj, B.center, Zc, C);
    }
  }
  return s;
}

/** Two-electron repulsion (chemist notation): (A B | C D). */
export function ERI_shells(A: Shell, B: Shell, Cs: Shell, Ds: Shell): number {
  let s = 0;
  for (let i = 0; i < A.alpha.length; i++) {
    const ai = A.alpha[i]!;
    const ni = normS(ai);
    for (let j = 0; j < B.alpha.length; j++) {
      const bj = B.alpha[j]!;
      const nj = normS(bj);
      for (let k = 0; k < Cs.alpha.length; k++) {
        const ck = Cs.alpha[k]!;
        const nk = normS(ck);
        for (let l = 0; l < Ds.alpha.length; l++) {
          const dl = Ds.alpha[l]!;
          const nl = normS(dl);
          const coeff = A.c[i]! * B.c[j]! * Cs.c[k]! * Ds.c[l]! * ni * nj * nk * nl;
          s += coeff * primERI(ai, A.center, bj, B.center, ck, Cs.center, dl, Ds.center);
        }
      }
    }
  }
  return s;
}

const SQRT_PI = Math.sqrt(Math.PI);

/** Per-primitive normalization for an s-Gaussian: N(α) = (2α/π)^{3/4}. */
function normS(alpha: number): number {
  return Math.pow(2 * alpha / Math.PI, 0.75);
}

/** Boys function F_0(t) = 0.5·√(π/t)·erf(√t). At t=0, F_0=1. */
function boys0(t: number): number {
  if (t < 1e-12) return 1 - t / 3 + (t * t) / 10;  // small-t Taylor
  const s = Math.sqrt(t);
  return 0.5 * SQRT_PI / s * erf(s);
}

/** Abramowitz & Stegun 7.1.26 erf approximation — max error ~1.5e-7. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

// ── Primitive integrals (s-Gaussians) ────────────────────────
// All take centers (Ax, Ay, Az) and (Bx, By, Bz) and exponents
// α, β. We only ever need 1D centers in this codebase (atoms on
// the z-axis), but keep the interface 3D for clarity.

function dist2(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

function gprodCenter(
  a: readonly [number, number, number], alpha: number,
  b: readonly [number, number, number], beta: number,
): [number, number, number] {
  const p = alpha + beta;
  return [
    (alpha * a[0] + beta * b[0]) / p,
    (alpha * a[1] + beta * b[1]) / p,
    (alpha * a[2] + beta * b[2]) / p,
  ];
}

/** Primitive overlap ⟨φ_α(A)|φ_β(B)⟩ between unnormalized s-Gaussians. */
function primS(alpha: number, A: readonly [number, number, number], beta: number, B: readonly [number, number, number]): number {
  const p = alpha + beta;
  const mu = alpha * beta / p;
  const r2 = dist2(A, B);
  return Math.pow(Math.PI / p, 1.5) * Math.exp(-mu * r2);
}

/** Primitive kinetic ⟨φ_α(A)| -∇²/2 |φ_β(B)⟩, s-shell only. */
function primT(alpha: number, A: readonly [number, number, number], beta: number, B: readonly [number, number, number]): number {
  const p = alpha + beta;
  const mu = alpha * beta / p;
  const r2 = dist2(A, B);
  const overlap = primS(alpha, A, beta, B);
  return mu * (3 - 2 * mu * r2) * overlap;
}

/** Primitive nuclear attraction ⟨φ_α(A)| -Z_C/|r-C| |φ_β(B)⟩. */
function primV(
  alpha: number, A: readonly [number, number, number],
  beta: number, B: readonly [number, number, number],
  Zc: number, C: readonly [number, number, number],
): number {
  const p = alpha + beta;
  const P = gprodCenter(A, alpha, B, beta);
  const r2 = dist2(A, B);
  const PC2 = dist2(P, C);
  const mu = alpha * beta / p;
  const pre = -2 * Math.PI * Zc / p * Math.exp(-mu * r2);
  return pre * boys0(p * PC2);
}

/** Primitive two-electron repulsion (αA βB | γC δD), s-shells only. */
function primERI(
  a1: number, A: readonly [number, number, number],
  b1: number, B: readonly [number, number, number],
  g2: number, C: readonly [number, number, number],
  d2: number, D: readonly [number, number, number],
): number {
  const p = a1 + b1;
  const q = g2 + d2;
  const P = gprodCenter(A, a1, B, b1);
  const Q = gprodCenter(C, g2, D, d2);
  const muAB = a1 * b1 / p;
  const muCD = g2 * d2 / q;
  const RAB2 = dist2(A, B);
  const RCD2 = dist2(C, D);
  const RPQ2 = dist2(P, Q);
  const t = p * q / (p + q) * RPQ2;
  const pre = 2 * Math.pow(Math.PI, 2.5) /
              (p * q * Math.sqrt(p + q)) *
              Math.exp(-muAB * RAB2 - muCD * RCD2);
  return pre * boys0(t);
}

// ── Contracted-shell integrals ───────────────────────────────
// Each H 1s is a 3-primitive contraction. Below: sum-over-primitives
// with primitive normalization rolled in.

type Center = readonly [number, number, number];

const ALPHA = STO3G_H_1S.alpha;
const C = STO3G_H_1S.c;
const NPRIM = ALPHA.length;

/** ⟨1s_A | 1s_B⟩ where A, B are H atom positions (Bohr). */
export function S_AB(A: Center, B: Center): number {
  let s = 0;
  for (let i = 0; i < NPRIM; i++) {
    for (let j = 0; j < NPRIM; j++) {
      s += C[i]! * C[j]! * normS(ALPHA[i]!) * normS(ALPHA[j]!) * primS(ALPHA[i]!, A, ALPHA[j]!, B);
    }
  }
  return s;
}

/** Kinetic ⟨1s_A | -∇²/2 | 1s_B⟩. */
export function T_AB(A: Center, B: Center): number {
  let s = 0;
  for (let i = 0; i < NPRIM; i++) {
    for (let j = 0; j < NPRIM; j++) {
      s += C[i]! * C[j]! * normS(ALPHA[i]!) * normS(ALPHA[j]!) * primT(ALPHA[i]!, A, ALPHA[j]!, B);
    }
  }
  return s;
}

/** Nuclear attraction ⟨1s_A | -Z_C/|r-C| | 1s_B⟩. */
export function V_AB(A: Center, B: Center, Zc: number, Cn: Center): number {
  let s = 0;
  for (let i = 0; i < NPRIM; i++) {
    for (let j = 0; j < NPRIM; j++) {
      s += C[i]! * C[j]! * normS(ALPHA[i]!) * normS(ALPHA[j]!) * primV(ALPHA[i]!, A, ALPHA[j]!, B, Zc, Cn);
    }
  }
  return s;
}

/** Two-electron repulsion in chemist notation: (1s_A 1s_B | 1s_C 1s_D). */
export function ERI(A: Center, B: Center, Ca: Center, Da: Center): number {
  let s = 0;
  for (let i = 0; i < NPRIM; i++) {
    const ni = normS(ALPHA[i]!);
    for (let j = 0; j < NPRIM; j++) {
      const nj = normS(ALPHA[j]!);
      for (let k = 0; k < NPRIM; k++) {
        const nk = normS(ALPHA[k]!);
        for (let l = 0; l < NPRIM; l++) {
          const nl = normS(ALPHA[l]!);
          const c = C[i]! * C[j]! * C[k]! * C[l]! * ni * nj * nk * nl;
          s += c * primERI(ALPHA[i]!, A, ALPHA[j]!, B, ALPHA[k]!, Ca, ALPHA[l]!, Da);
        }
      }
    }
  }
  return s;
}
