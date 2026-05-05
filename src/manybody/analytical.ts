// ─────────────────────────────────────────────────────────────
// analytical.ts — exact thermodynamic-limit ground-state
// energies-per-bond for the textbook 1D models. Used by Phase B
// experiments to validate MPS / DMRG runs at N ≥ 64 against
// the analytical reference, since exact diagonalization is
// out of reach above N ≈ 14.
//
// References:
//   • TFIM:        Pfeuty 1970, Ann. Phys. 57, 79.
//   • Heisenberg:  Bethe 1931 / Hulthén 1938 ground-state energy.
//
// Energies returned are *per bond* (= per site in the
// thermodynamic limit), so the comparison at finite N is
// E_DMRG / (N − 1)  vs.  e_inf(model). The (N − 1) divisor is
// because we use open boundary conditions throughout the
// project — N sites give N − 1 bonds. The per-site limit is
// the same up to O(1/N) corrections (boundary energy).
// ─────────────────────────────────────────────────────────────

/**
 * Pfeuty thermodynamic-limit ground-state energy per bond for
 * the transverse-field Ising chain
 *   H = -J Σ Z_iZ_{i+1} - h Σ X_i
 * in the periodic-boundary, infinite-N limit:
 *
 *   e_∞(λ) = -(J/π) ∫_0^π √(1 + λ² - 2λ cos k) dk,    λ = h/J
 *
 * Computed via Simpson's rule with a fine quadrature grid;
 * absolute error ≤ 1e-12 over λ ∈ [0, 5]. Self-validating limit
 * checks: e_∞(0) = -J (pure Ising), e_∞(1) = -4J/π (critical).
 */
export function tfimPfeutyEnergyPerBond(J: number, h: number): number {
  const lambda = h / J;
  // Simpson's rule on [0, π] with 2^14 + 1 points. Each integrand
  // call is one sqrt, so the cost is negligible (microseconds).
  const N = 1 << 14;
  const a = 0;
  const b = Math.PI;
  const dx = (b - a) / N;
  let sum = 0;
  // Endpoints with weight 1.
  sum += integrand(a, lambda);
  sum += integrand(b, lambda);
  for (let i = 1; i < N; i++) {
    const x = a + i * dx;
    const w = (i & 1) ? 4 : 2;
    sum += w * integrand(x, lambda);
  }
  const integral = sum * dx / 3;
  return -(J / Math.PI) * integral;
}

function integrand(k: number, lambda: number): number {
  return Math.sqrt(1 + lambda * lambda - 2 * lambda * Math.cos(k));
}

/**
 * Bethe / Hulthén thermodynamic-limit ground-state energy per
 * bond for the antiferromagnetic Heisenberg chain
 *   H = J Σ S_i · S_{i+1},    S = σ/2
 * (which equals (J/4) Σ (X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1})):
 *
 *   e_∞ = J · (1/4 − ln 2) ≈ -0.4431471805599453 · J
 *
 * The closed form goes back to Hulthén 1938 evaluating Bethe's
 * 1931 quantization integral.
 */
export function heisenbergBetheEnergyPerBond(J = 1): number {
  return J * (0.25 - Math.LN2);
}

/**
 * XX model ground-state energy per bond (free-fermion limit of
 * XXZ at Δ = 0). Closed form via JW + half-filled Fermi sea:
 *
 *   H = J Σ (X_iX_{i+1} + Y_iY_{i+1}),   PBC, N → ∞
 *   e_∞ = -2J / π
 *
 * Sanity-check value: ≈ -0.6366 · J. More negative than the
 * Heisenberg point (Δ = 1), as expected for the unfrustrated
 * XY chain.
 */
export function xxBetheEnergyPerBond(J = 1): number {
  return -2 * J / Math.PI;
}
