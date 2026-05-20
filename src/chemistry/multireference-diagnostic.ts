// ─────────────────────────────────────────────────────────────
// multireference-diagnostic.ts — aggregate "is this calculation
// multi-reference?" indicator from multiple individual diagnostics.
//
// Single-reference methods (HF, MP2, CCSD, CCSD(T)) assume the
// wavefunction is dominated by a single Slater determinant. When
// the actual wavefunction has significant multi-reference character
// (multiple competing electronic configurations), these methods
// become unreliable — energies wrong by 10s of kcal/mol, wrong
// equilibrium geometries, qualitatively wrong predictions.
//
// Common multi-reference cases:
//   - Stretched / dissociating bonds (H₂ at large R)
//   - Open-shell singlets (di-radicals: oxygen O₂, ortho-benzyne)
//   - Transition-metal complexes with near-degenerate d/f orbitals
//   - Photochemistry near conical intersections
//   - Strongly correlated systems (Cr₂, Mn complexes)
//
// We collect four diagnostics, each from a different angle:
//
//   t1: Lee-Taylor T1 diagnostic = ‖T1‖_F / √(N_e)
//      Cutoffs (Lee-Taylor 1989, Crawford-Schaefer 2000):
//      < 0.02 → single-reference (safe to use CCSD)
//      0.02–0.04 → moderate (CCSD still useful, watch for issues)
//      > 0.04 → multi-reference (CCSD unreliable, need CASSCF)
//
//   d1: Janssen-Nielsen D1 diagnostic = σ_max(T1)
//      Cutoffs (Janssen-Nielsen 1998):
//      < 0.05 → single-reference
//      0.05–0.10 → moderate
//      > 0.10 → multi-reference
//
//   ⟨S²⟩ contamination = |s² − s²_exact| / s²_exact (for open-shell)
//      < 0.05 → low contamination (UHF / UCCSD reliable)
//      0.05–0.20 → moderate
//      > 0.20 → severe (broken-symmetry artifact)
//
//   NOON deviation = max(|n_i − 2|_occupied, |n_a − 0|_virtual)
//      < 0.05 → essentially single-determinant
//      0.05–0.30 → moderate correlation
//      > 0.30 → strong multi-reference character (active orbitals at
//             ~1 electron each, "HEHE" half-electron-half-electron)
//
// Aggregate severity: worst (across-diagnostic) wins. Single-
// reference if all four are in the safe range; borderline if any
// is in the moderate range but none is in the multi-reference
// range; multi-reference if any diagnostic is in that range.
//
// Output flags array describes which specific diagnostics triggered
// (e.g., ["T1 = 0.045 (multi-reference threshold)", "⟨S²⟩ contamination = 0.18 (moderate)"]).
// ─────────────────────────────────────────────────────────────

export type MRSeverity = "single-reference" | "borderline" | "multi-reference";

export interface MRDiagnosticInput {
  /** T1 diagnostic from CCSD (Lee-Taylor 1989). Optional. */
  readonly t1?: number;
  /** D1 diagnostic from CCSD (Janssen-Nielsen 1998). Optional. */
  readonly d1?: number;
  /** UHF / UKS ⟨S²⟩ value. */
  readonly s2?: number;
  /** Exact ⟨S²⟩ = S(S+1). */
  readonly s2Exact?: number;
  /** NOON values (sorted descending). */
  readonly noon?: Float64Array;
  /** Number of doubly-occupied MOs (used to threshold NOON). */
  readonly nOccupied?: number;
}

export interface MRDiagnosticResult {
  readonly severity: MRSeverity;
  readonly flags: string[];
  readonly t1: number | undefined;
  readonly d1: number | undefined;
  readonly s2Contamination: number | undefined;
  readonly noonDeviation: number | undefined;
}

const T1_MR = 0.04;
const T1_BORDERLINE = 0.02;
const D1_MR = 0.10;
const D1_BORDERLINE = 0.05;
const S2_MR = 0.20;
const S2_BORDERLINE = 0.05;
const NOON_MR = 0.30;
const NOON_BORDERLINE = 0.05;

export function multireferenceDiagnostic(
  input: MRDiagnosticInput,
): MRDiagnosticResult {
  const flags: string[] = [];
  let severity: MRSeverity = "single-reference";

  function escalate(level: MRSeverity): void {
    if (level === "multi-reference") severity = "multi-reference";
    else if (level === "borderline" && severity !== "multi-reference") {
      severity = "borderline";
    }
  }

  // T1 diagnostic.
  if (input.t1 !== undefined) {
    const t1 = input.t1;
    if (t1 > T1_MR) {
      flags.push(`T1 = ${t1.toFixed(4)} > ${T1_MR} (multi-reference cutoff, Lee-Taylor)`);
      escalate("multi-reference");
    } else if (t1 > T1_BORDERLINE) {
      flags.push(`T1 = ${t1.toFixed(4)} between ${T1_BORDERLINE} and ${T1_MR} (borderline)`);
      escalate("borderline");
    }
  }

  // D1 diagnostic.
  if (input.d1 !== undefined) {
    const d1 = input.d1;
    if (d1 > D1_MR) {
      flags.push(`D1 = ${d1.toFixed(4)} > ${D1_MR} (multi-reference cutoff, Janssen-Nielsen)`);
      escalate("multi-reference");
    } else if (d1 > D1_BORDERLINE) {
      flags.push(`D1 = ${d1.toFixed(4)} between ${D1_BORDERLINE} and ${D1_MR} (borderline)`);
      escalate("borderline");
    }
  }

  // ⟨S²⟩ contamination.
  let s2Contamination: number | undefined;
  if (input.s2 !== undefined && input.s2Exact !== undefined) {
    const s2 = input.s2, s2Exact = input.s2Exact;
    // Use absolute contamination if s2Exact = 0 (closed-shell singlet);
    // relative otherwise.
    const denom = s2Exact > 1e-6 ? s2Exact : 1;
    s2Contamination = Math.abs(s2 - s2Exact) / denom;
    if (s2Contamination > S2_MR) {
      flags.push(`⟨S²⟩ contamination = ${s2Contamination.toFixed(4)} > ${S2_MR} (severe broken-symmetry)`);
      escalate("multi-reference");
    } else if (s2Contamination > S2_BORDERLINE) {
      flags.push(`⟨S²⟩ contamination = ${s2Contamination.toFixed(4)} between ${S2_BORDERLINE} and ${S2_MR} (moderate)`);
      escalate("borderline");
    }
  }

  // NOON deviation.
  let noonDeviation: number | undefined;
  if (input.noon && input.nOccupied !== undefined) {
    const noon = input.noon;
    const nOcc = input.nOccupied;
    let maxDev = 0;
    for (let p = 0; p < noon.length; p++) {
      const target = p < nOcc ? 2 : 0;
      const dev = Math.abs(noon[p]! - target);
      if (dev > maxDev) maxDev = dev;
    }
    noonDeviation = maxDev;
    if (maxDev > NOON_MR) {
      flags.push(`NOON deviation = ${maxDev.toFixed(4)} > ${NOON_MR} (active orbital populations far from 0/2)`);
      escalate("multi-reference");
    } else if (maxDev > NOON_BORDERLINE) {
      flags.push(`NOON deviation = ${maxDev.toFixed(4)} between ${NOON_BORDERLINE} and ${NOON_MR} (moderate correlation)`);
      escalate("borderline");
    }
  }

  return {
    severity,
    flags,
    t1: input.t1,
    d1: input.d1,
    s2Contamination,
    noonDeviation,
  };
}
