// ─────────────────────────────────────────────────────────────
// h2-fci-state.ts — extract the FCI ground-state coefficients
// (c_g, c_u) for the closed-shell H₂ singlet at bond length R.
//
// Spin-orbital order for our Jordan-Wigner mapping:
//   qubit 0 = σ_g↑   qubit 1 = σ_g↓
//   qubit 2 = σ_u↑   qubit 3 = σ_u↓
//
// The closed-shell singlet ground state is a 2-component
// superposition in occupation-number basis:
//
//   |ψ⟩ = c_g · |σ_g σ_g⟩  +  c_u · |σ_u σ_u⟩
//       = c_g · |1 1 0 0⟩  +  c_u · |0 0 1 1⟩
//
// In our index convention (qubit 0 = LSB) those are basis
// indices 3 and 12 respectively.
//
// We diagonalize the 16×16 dense Hamiltonian, take the lowest
// eigenvector, and read off c_g, c_u. All other components of
// the eigenvector are zero by symmetry (closed-shell singlet);
// we sanity-check that and bail loudly if not.
// ─────────────────────────────────────────────────────────────

import { buildH2Dense } from "../chemistry/h2-builder.js";
import { smallestEigenpair } from "../chemistry/eigensolver.js";

export interface H2FCIState {
  R_angstrom: number;
  energy: number;
  /** Coefficient on |σ_g σ_g⟩ — bonding determinant, dominant near equilibrium. */
  cG: number;
  /** Coefficient on |σ_u σ_u⟩ — anti-bonding double excitation, grows with R. */
  cU: number;
}

const HF_INDEX = 3;        // |1 1 0 0⟩ = qubits 0,1 occupied
const DOUBLE_INDEX = 12;   // |0 0 1 1⟩ = qubits 2,3 occupied

export function fciState(R_angstrom: number): H2FCIState {
  const { H } = buildH2Dense(R_angstrom);
  const { value, vector } = smallestEigenpair(H, 16);

  let cG = vector[HF_INDEX]!;
  let cU = vector[DOUBLE_INDEX]!;

  // Sanity: every other component should be ~0 within sweep tolerance.
  let leak = 0;
  for (let i = 0; i < 16; i++) {
    if (i === HF_INDEX || i === DOUBLE_INDEX) continue;
    leak += vector[i]! * vector[i]!;
  }
  if (leak > 1e-8) {
    throw new Error(`fciState: ground state has unexpected support outside (|HF⟩, |double⟩) sector — leak² = ${leak}`);
  }

  // Renormalize the 2-component projection (the leak should already be ~0 but
  // cleanup stops slow round-off from showing up in the conditional density).
  const n = Math.sqrt(cG * cG + cU * cU);
  cG /= n; cU /= n;

  // Sign convention: c_g ≥ 0. Flip both if needed.
  if (cG < 0) { cG = -cG; cU = -cU; }

  return { R_angstrom, energy: value, cG, cU };
}
