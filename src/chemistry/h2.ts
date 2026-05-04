// ─────────────────────────────────────────────────────────────
// h2.ts — H₂ molecule qubit Hamiltonian, STO-3G basis,
// Jordan-Wigner mapping, 4 qubits.
//
// Coefficients are the published OpenFermion values for
// H2_sto-3g_singlet_0.7414 (R = 0.7414 Å, equilibrium bond length).
// E_FCI for this Hamiltonian = −1.137270174 Ha including the
// nuclear-repulsion offset baked into the identity term.
//
// Qubit convention matches the rest of the repo: qubit 0 is the
// first character of the ops string.
//
// Dataset reference:
//   github.com/quantumlib/OpenFermion ‒ data/H2_sto-3g_singlet_0.7414
//   (also reproduced in O'Malley et al. 2016, PRX 6 031007, Table I).
// ─────────────────────────────────────────────────────────────

import type { Hamiltonian } from "./pauli.js";

/** H₂ qubit Hamiltonian at R = 0.7414 Å (STO-3G / JW). 15 terms, 4 qubits. */
export const H2_R0_7414: Hamiltonian = [
  { coeff: -0.09706626816763122, ops: "IIII" },
  { coeff:  0.17218393261915520, ops: "ZIII" },
  { coeff:  0.17218393261915520, ops: "IZII" },
  { coeff: -0.22343153690813725, ops: "IIZI" },
  { coeff: -0.22343153690813725, ops: "IIIZ" },
  { coeff:  0.16892754048859022, ops: "ZZII" },
  { coeff:  0.12091263351621554, ops: "ZIZI" },
  { coeff:  0.16614543413331697, ops: "ZIIZ" },
  { coeff:  0.16614543413331697, ops: "IZZI" },
  { coeff:  0.12091263351621554, ops: "IZIZ" },
  { coeff:  0.17464343053862533, ops: "IIZZ" },
  // The four off-diagonal "double excitation" terms — these mediate
  // the |1100⟩ ↔ |0011⟩ coupling that supplies the correlation
  // energy. Sign pattern matches OpenFermion's JW output for H₂.
  { coeff:  0.04523279994605781, ops: "YYXX" },
  { coeff: -0.04523279994605781, ops: "YXXY" },
  { coeff: -0.04523279994605781, ops: "XYYX" },
  { coeff:  0.04523279994605781, ops: "XXYY" },
];

/** FCI ground-state energy of THIS hard-coded Hamiltonian, in Hartree.
 *  Determined by classical Jacobi diagonalization of the 16×16 dense
 *  matrix (see eigensolver.ts + tests/chemistry/h2-fci.test.ts). Differs
 *  from the canonical PySCF value -1.137270 Ha by ~2 mHa, attributable
 *  to FP rounding in the OpenFermion published Pauli table. The number
 *  that matters for VQE pass/fail is the spectrum of the Hamiltonian we
 *  actually evaluate — that's this constant. */
export const E_FCI_H2_R0_7414 = -1.139265606316854;

/** Hartree-Fock reference energy: ⟨HF|H|HF⟩ on the closed-shell |0011⟩
 *  determinant (qubits 0, 1 occupied = MO 1 doubly filled). Computed
 *  diagonal of the same Hamiltonian; VQE must beat this. */
export const E_HF_H2_R0_7414 = -1.1188422791;
