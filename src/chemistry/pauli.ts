// ─────────────────────────────────────────────────────────────
// pauli.ts — Pauli strings and expectation values against an
// (interleaved Float64) statevector.
//
// A Pauli string is one operator per qubit, drawn from {I, X, Y, Z}.
// We store it as a string of length n (e.g. "IXYZ" for q0=I, q1=X,
// q2=Y, q3=Z under the convention that index 0 of the string is
// qubit 0, matching the rest of this codebase).
//
// Expectation: ⟨ψ|P|ψ⟩ = Σ_i conj(ψ_i) (Pψ)_i. For a tensor product
// of Paulis, (Pψ)_i is computed cheaply by toggling the basis index
// for X/Y components and applying ±1 / ±i phases, no full matrix.
// ─────────────────────────────────────────────────────────────

export type PauliOp = "I" | "X" | "Y" | "Z";

export interface PauliTerm {
  /** Real coefficient (qubit Hamiltonians from JW are Hermitian — coefs are real). */
  readonly coeff: number;
  /** One char per qubit, length = nQubits. Index 0 of string = qubit 0. */
  readonly ops: string;
}

export type Hamiltonian = readonly PauliTerm[];

function validateOps(ops: string): void {
  for (let i = 0; i < ops.length; i++) {
    const c = ops[i]!;
    if (c !== "I" && c !== "X" && c !== "Y" && c !== "Z") {
      throw new Error(`Pauli op '${c}' at qubit ${i} not in {I,X,Y,Z}`);
    }
  }
}

export function pauli(coeff: number, ops: string): PauliTerm {
  validateOps(ops);
  return { coeff, ops };
}

/**
 * Expectation ⟨ψ|P|ψ⟩ for a single Pauli string against an
 * interleaved Float64 statevector of length 2·2^n.
 *
 * Apply-and-accumulate in one pass:
 *   index j is the basis index after X/Y bit-flips on every X/Y site;
 *   per-amplitude phase is i^(#Y) · (−1)^(#Z·bit_q · 1).
 */
export function expectationPauli(psi: Float64Array, ops: string): number {
  const n = ops.length;
  const N = 1 << n;
  if (psi.length !== 2 * N) {
    throw new Error(`expectationPauli: psi length ${psi.length} ≠ 2·2^${n}`);
  }

  // Pre-compute bitmasks per op so the inner loop is branch-light.
  let xyMask = 0;          // bits where X or Y act (basis flip)
  let yMask = 0;           // bits where Y acts (extra ±i factor)
  let zMask = 0;           // bits where Z acts (sign by occupancy)
  for (let q = 0; q < n; q++) {
    const c = ops[q]!;
    if (c === "X" || c === "Y") xyMask |= 1 << q;
    if (c === "Y") yMask |= 1 << q;
    if (c === "Z") zMask |= 1 << q;
  }

  // Per-amplitude (Pψ)_j = phase · ψ_{j ⊕ xyMask}, where
  //   phase = (i)^(#Y) · (−1)^(parity(j & yMask))   ← the Y operator
  //                                                    contributes a
  //                                                    sign from the
  //                                                    after-flip bit;
  //                                                    expanding gives
  //                                                    (i)^(#Y) above.
  //         · (−1)^(parity(j & zMask))              ← Z signs.
  //
  // We compute ⟨ψ|P|ψ⟩ = Σ_j conj(ψ_j) · (Pψ)_j.
  // Rather than building (Pψ) explicitly, accumulate directly.

  // i^k cycles in {1, i, -1, -i}. We store as (re, im).
  const yCount = popcount(yMask);
  // Global phase from the i^(#Y) factor:
  //   yCount mod 4 = 0 → (1, 0)
  //   yCount mod 4 = 1 → (0, 1)
  //   yCount mod 4 = 2 → (−1, 0)
  //   yCount mod 4 = 3 → (0, −1)
  const yPhaseRe = [1, 0, -1, 0][yCount & 3]!;
  const yPhaseIm = [0, 1, 0, -1][yCount & 3]!;

  let accRe = 0;
  let accIm = 0;
  for (let j = 0; j < N; j++) {
    // Sign from Z operators (parity of occupied Z bits).
    const zSign = (popcount(j & zMask) & 1) ? -1 : 1;
    // Sign from each Y: yields (i)·(σ_y_bit_j) per Y, where σ_y on |b⟩
    // is i·|1−b⟩ for b=0 and −i·|1−b⟩ for b=1. Above we factored out
    // i^(#Y) globally; the residual sign per Y is +1 if b=0, −1 if b=1
    // — i.e. (−1)^(parity(j & yMask)) on the AFTER-flip basis. But the
    // amplitude we read is ψ_{j ⊕ xyMask}, and for basis bookkeeping
    // we want the sign tied to the OUTPUT basis j. The two are equal
    // bit-for-bit on the Y mask up to the X part of XY, which doesn't
    // affect Y's sign:
    //   parity(j & yMask) before the X flip = parity(j_xor_xyMask & yMask)
    //   after — same value because Y bits are inside XY mask too.
    const ySign = (popcount(j & yMask) & 1) ? -1 : 1;

    const sign = zSign * ySign;

    // ψ_j (output index) and ψ_{j ⊕ xyMask} (read index).
    const jj = j * 2;
    const jr = j ^ xyMask;
    const jjr = jr * 2;

    const psiJre = psi[jj]!;
    const psiJim = psi[jj + 1]!;
    const psiKre = psi[jjr]!;
    const psiKim = psi[jjr + 1]!;

    // (Pψ)_j = sign · (yPhaseRe + i·yPhaseIm) · ψ_K
    const pPsiRe = sign * (yPhaseRe * psiKre - yPhaseIm * psiKim);
    const pPsiIm = sign * (yPhaseRe * psiKim + yPhaseIm * psiKre);

    // conj(ψ_j) · (Pψ)_j  →  (psiJre − i·psiJim)·(pPsiRe + i·pPsiIm)
    accRe += psiJre * pPsiRe + psiJim * pPsiIm;
    accIm += psiJre * pPsiIm - psiJim * pPsiRe;
  }

  // ⟨ψ|H|ψ⟩ for Hermitian P is real to roundoff. We return the real part;
  // the imaginary part is a useful diagnostic but not exposed.
  void accIm;
  return accRe;
}

/** ⟨ψ|H|ψ⟩ = Σ_t coeff_t · ⟨ψ|P_t|ψ⟩ for a Pauli sum H. */
export function expectationHamiltonian(psi: Float64Array, H: Hamiltonian): number {
  let s = 0;
  for (const term of H) {
    s += term.coeff * expectationPauli(psi, term.ops);
  }
  return s;
}

function popcount(x: number): number {
  let n = 0;
  while (x !== 0) { n += x & 1; x >>>= 1; }
  return n;
}
