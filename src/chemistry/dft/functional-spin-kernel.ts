// ─────────────────────────────────────────────────────────────
// functional-spin-kernel.ts — LSDA XC kernel components for UKS
// linear response (CPKS / TDDFT). Computes the spin-resolved second
// derivatives
//
//   f^σσ'(r) = ∂²(ρ ε_xc) / (∂ρ_σ · ∂ρ_σ')        per volume
//
// at every grid point, via central finite-differencing of the
// per-spin first derivatives `v_ρ↑` / `v_ρ↓` returned by
// `evalXC_LSDA`. Returns three arrays:
//
//   fUU = ∂v_↑/∂ρ_↑    (αα channel)
//   fUD = ∂v_↑/∂ρ_↓ = ∂v_↓/∂ρ_↑    (cross-spin, symmetric)
//   fDD = ∂v_↓/∂ρ_↓    (ββ channel)
//
// Cost: 4 finite-difference function calls (two pairs for ρ_↑±h and
// ρ_↓±h), each evaluating evalXC_LSDA over the full grid. For
// H₂O Lebedev-110 (~16k points) this is sub-millisecond.
//
// Used by `uksCphfPolarizability` (uks-cphf.ts) to build the UKS
// (A+B) orbital Hessian for static linear-response polarizability.
// ─────────────────────────────────────────────────────────────

import { evalXC_LSDA } from "./functional-spin.js";
import { FD_REL_STEP, FD_STEP_FLOOR } from "../numerical-tolerances.js";

const EPS_RHO = 1e-15;

export interface XCKernelLSDASpin {
  /** fαα = ∂²(ρε)/∂ρ_α² (per volume), at each grid point. */
  readonly fUU: Float64Array;
  /** fαβ = ∂²(ρε)/(∂ρ_α ∂ρ_β) (per volume). Symmetric: ∂v_↑/∂ρ_↓ = ∂v_↓/∂ρ_↑. */
  readonly fUD: Float64Array;
  /** fββ = ∂²(ρε)/∂ρ_β² (per volume). */
  readonly fDD: Float64Array;
}

export function evalXCKernelLSDA_spin(
  rhoUp: Float64Array,
  rhoDn: Float64Array,
): XCKernelLSDASpin {
  const n = rhoUp.length;
  if (rhoDn.length !== n) {
    throw new Error(`evalXCKernelLSDA_spin: rhoUp / rhoDn length mismatch (${n} vs ${rhoDn.length})`);
  }
  const fUU = new Float64Array(n);
  const fUD = new Float64Array(n);
  const fDD = new Float64Array(n);

  // Step sizes scaled per-point.
  const hUp = new Float64Array(n);
  const hDn = new Float64Array(n);
  for (let p = 0; p < n; p++) {
    hUp[p] = Math.max(rhoUp[p]! * FD_REL_STEP, FD_STEP_FLOOR);
    hDn[p] = Math.max(rhoDn[p]! * FD_REL_STEP, FD_STEP_FLOOR);
  }

  // Scratch arrays for ρ_↑ ± h_↑.
  const rhoUp_p = new Float64Array(n);
  const rhoUp_m = new Float64Array(n);
  const rhoDn_p = new Float64Array(n);
  const rhoDn_m = new Float64Array(n);
  for (let p = 0; p < n; p++) {
    rhoUp_p[p] = rhoUp[p]! + hUp[p]!;
    rhoUp_m[p] = Math.max(rhoUp[p]! - hUp[p]!, EPS_RHO);
    rhoDn_p[p] = rhoDn[p]! + hDn[p]!;
    rhoDn_m[p] = Math.max(rhoDn[p]! - hDn[p]!, EPS_RHO);
  }

  // FD on ρ_↑ varying — gives ∂v/∂ρ_↑ for both v_↑ and v_↓.
  const epsTmp = new Float64Array(n);
  const vUpAtUp_p = new Float64Array(n);
  const vDnAtUp_p = new Float64Array(n);
  const vUpAtUp_m = new Float64Array(n);
  const vDnAtUp_m = new Float64Array(n);
  evalXC_LSDA(rhoUp_p, rhoDn, epsTmp, vUpAtUp_p, vDnAtUp_p);
  evalXC_LSDA(rhoUp_m, rhoDn, epsTmp, vUpAtUp_m, vDnAtUp_m);

  // FD on ρ_↓ varying — gives ∂v/∂ρ_↓ for both v_↑ and v_↓.
  const vUpAtDn_p = new Float64Array(n);
  const vDnAtDn_p = new Float64Array(n);
  const vUpAtDn_m = new Float64Array(n);
  const vDnAtDn_m = new Float64Array(n);
  evalXC_LSDA(rhoUp, rhoDn_p, epsTmp, vUpAtDn_p, vDnAtDn_p);
  evalXC_LSDA(rhoUp, rhoDn_m, epsTmp, vUpAtDn_m, vDnAtDn_m);

  for (let p = 0; p < n; p++) {
    const inv2hU = 1 / (2 * hUp[p]!);
    const inv2hD = 1 / (2 * hDn[p]!);
    // f^αα = ∂v_↑/∂ρ_↑
    fUU[p] = (vUpAtUp_p[p]! - vUpAtUp_m[p]!) * inv2hU;
    // f^ββ = ∂v_↓/∂ρ_↓
    fDD[p] = (vDnAtDn_p[p]! - vDnAtDn_m[p]!) * inv2hD;
    // f^αβ symmetric: ½ · (∂v_↑/∂ρ_↓ + ∂v_↓/∂ρ_↑) — average of the two
    // FD estimates for numerical stability.
    const dvUpDrhoDn = (vUpAtDn_p[p]! - vUpAtDn_m[p]!) * inv2hD;
    const dvDnDrhoUp = (vDnAtUp_p[p]! - vDnAtUp_m[p]!) * inv2hU;
    fUD[p] = 0.5 * (dvUpDrhoDn + dvDnDrhoUp);
  }

  return { fUU, fUD, fDD };
}
