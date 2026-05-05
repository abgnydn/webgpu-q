# webgpu-q — CLAUDE.md

Project-local instructions for Claude. Load this first.

---

## One-paragraph read

WebGPU quantum circuit simulator. Runs in a browser tab. Target: piece one of
a six-level research ladder — statevector → MPS → kernel fusion → WebRTC
swarm → IBM hardware cross-verify → quantum chemistry. Each level is a set
of **research-grade experiments** (not just benchmarks): named seed, warmup,
trials, fidelity pass bar, honest negative results. The master doc is
`RESEARCH.md`. Per-level protocols live under
`experiments/level-N-<slug>/protocol.md`.

**Communication mode: hero.** Terse, bold, first-principles, attempt-first.
Scope-honest. See `~/.claude/skills/hero/SKILL.md`.
**Project skill: `webgpu-q-research`.** See `~/.claude/skills/webgpu-q-research/SKILL.md`.

---

## Roadmap to the frontier (the path you're on)

The project is past the launchpad — the four "ladder" levels are shipped
(L1 statevector, L2 MPS, L3 fusion, L6 chemistry), GPU MPS goes through
Phase 6 v1, and the repo is public + CI-green. The honest path from here
to publishable frontier work is below. Each phase produces a real
artifact; every off-ramp ships a real claim.

### Phase A — Tighten the foundation (~1 session)

The known weak points in today's stack:

1. **GPU MPS truncation handling (Phase 4c gap).** Phase 4b/5 says
   "no renormalization for unitary gates" — correct only when the chain
   stays in canonical form AND no SVD truncation kicks in. At deep
   circuits (depth ≥ log₂ χ_max) bonds saturate, kKeep < physicalRank,
   and we lose probability mass. Fix: GPU-side canonical-form
   renormalization on truncation paths.
2. **MPO (Matrix Product Operator) representation** for 1D Hamiltonians.
   Currently we have `Hamiltonian1D` as a list of bond terms; MPO is
   the standard format every DMRG / TDVP routine expects.
3. **Real two-site DMRG with Lanczos.** `src/manybody/dmrg.ts` is a stub
   (direct diagonalization → MPS conversion). Replace with proper sweeps
   + matrix-free Lanczos local eigensolver using the new MPO.

**Unlocks:** χ ≤ 64 deep circuits, MPO infrastructure, real DMRG.

### Phase B — 1D records (~2-3 sessions)

1. Push 1D chains to N = 80, 100, 128 with chiMax = 64 (needs Phase A).
2. Validate against analytical limits: Bethe ansatz for Heisenberg,
   Pfeuty for TFIM critical exponents.
3. Publish artifact: *"longest 1D MPS in a browser, ITensor-validated."*

**Off-ramp #1: workshop paper here.** Most published 1D tensor-network
results are at N ≤ 64; a browser-native N = 128 is paper-worthy.

### Phase C — Real molecule chemistry (~2-3 sessions)

1. Extend `src/chemistry/integrals.ts` from H 1s to multi-orbital atoms
   (Li 1s/2s, Be 1s/2s, etc.).
2. LiH builder (4 electrons, 6 spin-orbitals): `src/chemistry/lih-builder.ts`.
   Active-space VQE, validate against PySCF FCI.
3. BeH₂ next (6 electrons, 14 spin-orbitals).
4. Add CCSD(T) reference comparison.

**Off-ramp #2: chemistry paper here.** *"First browser-tab quantum
chemistry on real molecules, matches PySCF to chemical accuracy."*

### Phase D — Distributed via WebRTC (~3-5 sessions, the hard part)

The L4 swarm we deferred. Genuinely two-process engineering.

1. WebRTC signalling (or reuse `webgpu-p2p-evolution`'s 113-line relay).
2. Cut a 1D chain at the midpoint: browser A holds sites 0..N/2,
   browser B holds N/2..N-1. Each does its half's MPS, exchanges the
   bond tensor at the cut over WebRTC.
3. Recompute the full chain energy by combining both halves' contributions.
4. Validate: same answer as single-browser N = 80 from Phase B.
5. Push to N = 160 split across two machines.

**Off-ramp #3: distributed-quantum-sim paper here.** Foundation for
every subsequent moonshot.

### Phase E — Pick a moonshot

By this point: deep MPS at χ = 64, real-molecule VQE, WebRTC
distributed contraction. Three branches, each genuinely paper-worthy:

#### E.1 — Verify Sycamore (the public-benchmark moonshot)
- Add 2D PEPS primitive (substantial, but math is in published papers).
- Implement Sycamore gate set (fSim + single-qubit) and ingest Google's
  published circuit JSON.
- Distribute the PEPS contraction across N volunteer browsers via Phase D.
- Reproduce Pan & Zhang 2021's classical-supremacy refutation in a tab.
- Cross-check against Google's published output statistics.
- ~3-5 sessions on top of Phase D.

#### E.2 — Fault-tolerant qubit (the QC-future moonshot)
- Stabilizer simulator (Clifford-only, scales to thousands via Gottesman-Knill).
- Surface code at distance 3, 5, 7.
- Syndrome extraction + minimum-weight perfect-matching decoder.
- Noise model (depolarizing → biased → leakage).
- Plot logical error rate vs physical, find the threshold curve.
- Cross-check against IBM Heron / Quantinuum public data.
- ~4-6 sessions.

#### E.3 — Browser-native lattice QCD (the HPC-substrate moonshot)
- 4D lattice (small: 8⁴ or 16⁴).
- Wilson Dirac operator (or staggered fermions).
- Conjugate-gradient solver fused into a single WebGPU dispatch.
- Compute simplest hadron mass (pion or rho), validate against published lattice values.
- May need scaled-int arithmetic for f32 stability.
- ~6-10 sessions, hardest port.

### Cleanest path that ships at every off-ramp

**A → B → C → D → E.1**. ~10 sessions to the launchpad, +3-5 for
Sycamore in a browser. Every step gives a real claim; if you stop early
you still have something.

The unifying thesis: *"every advanced physics simulation in the world
ships as a URL"*. webgpu-q is the proof point.

---

## Current state of play (as of 2026-05-05)

### Phase C v5 — sparse-CSR Hsec breaks the 15 GB barrier (2026-05-05)

The dense sector matrix is now optional. For molecules where it
would blow past tab-memory limits (CH₄ in STO-3G = 15 GB), we
build the Hamiltonian as a **sparse-CSR matrix** instead. CH₄'s
H is 99.998% zeros, so sparse storage is 240 MB — **64× smaller
than dense** — and matvec is **~100× faster** because Lanczos
only walks the nonzero entries.

**Headline (CH₄, R = 1.09 Å, tetrahedral, full STO-3G):**

| metric | Phase C v4 (dense) | Phase C v5 (sparse) | improvement |
|---|---:|---:|---:|
| Hsec memory | 15.32 GB | **240 MB** | **64× smaller** |
| Lanczos wall | ~245 s | **1.29 s** | **190× faster** |
| build wall | 55 s | 92 s | 1.7× slower (Map overhead) |
| total | ~5 min | **~1.5 min** | 3.3× faster end-to-end |
| E_FCI | -39.806036 Ha | -39.806036 Ha | identical to f64 |
| Δ vs PySCF | 0.76 mHa | 0.76 mHa | (chemical accuracy) |

**Cross-checks (sparse ≡ dense to f64):**
- LiH N=4 (k=15): \|Δ\| < 1e-10
- H₂O N=10 (k=1001): \|Δ\| = 2.5e-10, sparse 760 KB vs dense 8 MB
- BeH₂-full N=6 (k=3003): \|Δ\| = 5.0e-11, sparse 1.9 MB vs dense 72 MB

**What shipped:**
- `src/chemistry/sector-matvec.ts`: `buildSparseSectorH` builds a
  CSR sector H by accumulating per-row Map entries during the
  same operator iteration that `buildSectorH` uses, then deduping
  + sorting into final CSR arrays. `sparseMatvec(H, x, y)` walks
  the CSR for fast matvec.
- `src/chemistry/molecule-builder.ts`: now picks `dense` for
  k ≤ 2000 (cheap, lets us cross-check with Jacobi) and `sparse`
  for k > 2000. `buildMoleculeFCI(...)` returns either `sector`
  or `sparseSector` depending on the choice.
- 3 sparse-vs-dense cross-check tests in `sparse-sector.test.ts`.
- Publishable artifact at
  `experiments/results/2026-05-05/level-6/E25-ch4-full-fci-sparse-publishable.json`.

**What this unlocks (Phase C v6 reach):**
- NH₃ (k = 43758, same as CH₄) — should be trivial.
- N₂, O₂, CO (10-orbital dimers, k ~ 50k–200k) — feasible.
- Acetylene C₂H₂ (k ~ 2 million) — sparse Hsec ~10 GB. Still too
  big. Needs Slater-Condon matrix-free matvec (no precomputed
  storage). That's the Phase C v6 lift.

**Tests: 262 → 265** (+3 sparse cross-checks). Typecheck clean.
Lint warnings unchanged from baseline.

### Phase C v4 — water and methane in a browser tab (2026-05-05)

H₂O and CH₄ in **full STO-3G FCI**, both matching PySCF. First
non-linear molecule (water, bent at 104.52°) AND first molecule with
4 hydrogens around a central atom (methane, tetrahedral). The
chemistry pipeline now works for **any** molecule the user names —
not just the dimers and triatomics we hand-coded.

**Headline at experimental geometries:**

| molecule | nQubits | sector dim | E_FCI (Ha) | PySCF (Ha) | \|Δ\| | wall |
|---|---:|---:|---:|---:|---:|---:|
| **H₂O** (R=0.9572 Å, ∠=104.52°) | 14 | C(14, 10) = **1001** | **−75.012403** | −75.0124 | **2.9 µHa** | 0.3 s |
| **CH₄** (R=1.09 Å, tetrahedral) | 18 | C(18, 10) = **43758** | **−39.806036** | −39.8068 | **0.76 mHa** | 5 min |
| BeH₂ full (R=1.34 Å) | 14 | C(14, 6) = 3003 | −15.594861 | −15.5949 | 39 µHa | 0.7 s |
| LiH s-only (R=1.595 Å) | 6 | C(6, 4) = 15 | −7.843394 | (matches) | f64 | < 0.1 s |

**H₂O symmetric stretch (∠HOH = 104.52° fixed, full curve in 3 s):**

| R_OH (Å) | E_FCI (Ha) |
|---:|---:|
| 0.70 | −74.643706 |
| 0.85 | −74.947345 |
| **0.9572** | **−75.012403** ← experimental |
| 1.05 | −75.019725 ← STO-3G minimum (basis-set bias toward longer bonds) |
| 1.20 | −74.985111 |
| 1.50 | −74.873426 |
| 2.00 | −74.761985 |

**What shipped:**
- `src/chemistry/atoms.ts`: atom registry. `atomShells("O", pos)` →
  `[1s, 2s, 2p_x, 2p_y, 2p_z]`. Supports H, Li, Be, C, N, O.
- `src/chemistry/integrals.ts`: STO-3G basis constants for C, N, O
  (1s + 2sp L-shell). Pople 1969 Table III.
- `src/chemistry/molecule-builder.ts`: one-shot
  `buildMoleculeFCI([{symbol, pos}, ...])` that produces the
  sector-projected H + lazy `.fci()` Lanczos. Replaces all per-
  molecule builder boilerplate.
- 5 new tests (`molecule-builder.test.ts`): LiH cross-check via the
  generic pipeline, H₂O FCI vs PySCF, V_nn correctness, Hsec
  Hermiticity, sector dim sanity.
- `tools/run-phase-c-v4.ts`: H₂O symmetric-stretch publishable +
  optional CH₄ (`PHASE_C4_CH4=1`, ~5 min, ~15 GB RAM).
- Publishable artifact: `experiments/results/2026-05-05/level-6/
  E23-h2o-full-fci-publishable.json`.

**Honest scope limit (Phase C v5):**
CH₄'s 43758-dim sector materializes a 15 GB Hsec — at the
hard memory edge of a 16 GB M2 Pro. Anything bigger (NH₃ in cc-pVDZ,
H₂O in cc-pVTZ, glucose at any basis) needs a **fully matrix-free
H apply** (compute matvec straight from h_OAO + eri_OAO without
storing Hsec). That's Phase C v5 work — turns the 15 GB → ~50 MB
working set, unlocks 20-qubit FCI in a tab.

**Tests: 257 → 262** (+5). Typecheck clean. Lint warnings unchanged.

### Phase C v3 stage 2 — full STO-3G BeH₂ FCI matches PySCF (2026-05-05)

The full quantum-chemistry stack at production basis-set quality, in
a browser tab. Linear H–Be–H in **full STO-3G** (Be 1s + Be 2s + Be 2p_x
+ Be 2p_y + Be 2p_z + 2 H 1s = 7 spatial / **14 spin-orbitals**), 6
electrons in the neutral molecule, FCI from a sector-projected Lanczos.

**Headline at R = 1.34 Å (experimental Be–H bond):**

| metric | value |
|---|---:|
| sector dim (N=6) | C(14, 6) = 3003 |
| nQubits | 14 |
| E_FCI (full STO-3G) | **−15.594861 Ha** |
| PySCF reference | −15.5949 Ha |
| ‖Δ vs PySCF‖ | **39 µHa** (40× under chemical accuracy) |
| Be 2p correlation gain | 245.7 mHa over s-only |
| build (integrals + Hsec) | 0.26 s |
| FCI (Lanczos) | 0.45 s, 48 iters |

**Symmetric stretch (full STO-3G FCI):**

| R (Å) | E_FCI (Ha) | Δ vs s-only (mHa) | wall |
|---:|---:|---:|---:|
| 0.80 | −15.1728 | +176.4 (s-only is more bound at this compressed geometry) | 0.86 s |
| 1.00 | −15.4817 | −132.6 | 0.77 s |
| 1.20 | −15.5838 | −234.7 | 0.82 s |
| **1.34** | **−15.5949** | **−245.7** | 0.71 s |
| 1.50 | −15.5761 | −226.9 | 0.74 s |
| 1.70 | −15.5291 | −180.0 | 0.76 s |
| 2.00 | −15.4461 | −96.9 | 0.88 s |
| 2.50 | −15.3518 | −2.7 | 1.30 s |
| 3.00 | −15.3368 | +12.4 (s-only mistakenly captures dissociation here; full FCI is honest) | 1.45 s |

Whole 9-R curve: **~6 seconds wall-clock**. Status: **pass**.
Off-ramp #2 from CLAUDE.md (*"first browser-tab quantum chemistry on
real molecules, matches FCI to chemical accuracy"*) is now real for
**LiH + BeH₂ in full STO-3G** — paper-worthy.

**What shipped:**
- `src/chemistry/cg-molecular.ts`: generic AO-integrals-over-CG-shells
  builder. Replaces the per-molecule integral duplication of
  lih-builder / beh2-builder for any future v3 molecules.
- `src/chemistry/sector-builder.ts`: builds the second-quantized H
  *directly in a chosen particle-number sector* via JW. Critical for
  BeH₂-full — the full 16384² dense H = 2 GB won't fit in a browser
  tab, but the C(14, 6) = 3003-dim N=6 sector is just 72 MB.
- `src/chemistry/beh2-full-builder.ts`: 7 atomic shells → Löwdin →
  sector-direct H. Combines cg-molecular + sector-builder.
- `tools/run-phase-c-v3.ts` + publishable JSON at
  `experiments/results/2026-05-05/level-6/E22-beh2-full-fci-publishable.json`.
- 9 new tests (`beh2-full.test.ts` 6 + `sector-builder.test.ts` 3).
  Sector-direct builder cross-checked against the existing dense+
  project path on LiH N=4 to f64 precision (8.88e-16). BeH₂-full
  Hsec Hermitian to 1e-15. PySCF reference matched to ≤ 0.5 mHa.

**Bug fix shipped along the way:** Boys function `boysAll` in
`integrals-cg.ts` was using a Taylor series that suffered catastrophic
cancellation at moderate t (e.g. t = 22 from a Li-1s × H-1s primitive
pair: intermediate sums of order 10¹⁰ cancelling down to 0.19, leaving
~1e-7 absolute precision and ~10 mHa errors in cross-shell V matrix
elements). Now uses the closed-form F_0(t) = ½√(π/t)·erf(√t) with the
A&S 7.1.26 erf, plus upward recurrence for n ≥ 1 (stable for t > 1)
and Taylor for small t. Matches the legacy s-only path to f64
precision after the fix; LiH FCI old vs new = 8.88e-16.

**Tests: 248 → 257** (+9). Typecheck clean. Lint warnings unchanged.

### Phase C v3 stage 1 — Cartesian-Gaussian p-shell integrals (2026-05-05)

The integral library that lets the chemistry pipeline see angular
momentum. Adds Obara-Saika / McMurchie-Davidson machinery so we can
compute S, T, V, ERI over arbitrary combinations of s and p shells —
the foundation needed to add the Be 2p sub-shell to BeH₂ (Phase C
v3 stage 2). All implemented in plain TypeScript, no external deps.

**What shipped:**
- `src/chemistry/integrals-cg.ts` (~400 lines):
  - `CGShell` type — generalizes `Shell` with an angular-momentum
    tuple `(n_x, n_y, n_z)`. `(0, 0, 0)` reduces to s; `(1, 0, 0)`
    etc. give the three Cartesian p shells.
  - 1D E-coefficient recurrence (Hermite-Gaussian expansion of a
    Gaussian product, HJO Ch. 9.5).
  - Boys function `F_n(t)` for n=0..6 — Taylor series for small t,
    asymptotic form for large t, then numerically stable downward
    recursion `F_{n−1} = (2t F_n + e^{−t}) / (2n − 1)`.
  - Auxiliary `R^n_{tuv}(p, R)` integrals via the standard
    one-down-and-one-out recursion.
  - `S_cg`, `T_cg`, `V_cg`, `ERI_cg` for any (CGShell × CGShell …)
    combinations — all built from the E + R machinery.
- `STO3G_BE_2P` added to `integrals.ts` (Pople 1969 L-shell p
  contraction; same exponents as `STO3G_BE_2S`, different coeffs).
- 17 unit tests covering: Boys F_n properties (4), s-shell
  consistency vs the legacy s-only API (4), p-shell physical
  invariants (8 — self-overlap, rotation invariance,
  same-center px ⊥ py, kinetic exact = 5α/2 for normalized p,
  translation invariance, ERI 8-fold symmetry), full Be 5-shell
  basis orthogonality (1).

**Honest follow-on (Phase C v3 stage 2):**
- Build the BeH₂-full Hamiltonian over 7 spatial / 14 spin-orbitals
  (Be 1s + Be 2s + Be 2p_x/y/z + 2 H 1s = 16384-dim Hilbert space).
- Crucial detail: a 16384² × 8 B = 2 GB dense matrix won't fit in
  a browser tab. Must build the H *directly in the N=6 sector*
  (C(14, 6) = 3003-dim → 72 MB). Refactor `addOneBody` /
  `addTwoBody` to project on the fly.
- FCI via matrix-free Lanczos (already in `src/manybody/lanczos.ts`)
  on the 3003-dim sector — fast and clean.
- VQE: HEA on 14 qubits with L-BFGS. Param count grows to ~210 at
  L = 12, FD gradient cost ~420 evals × 16384-amp circuits per iter.
  Estimated 2–5 min per trial.

**Precision note:** the new CG path uses high-precision Boys
(Taylor to 1e-18); the legacy s-only path uses Abramowitz &
Stegun erf (~1e-7 max error). They agree to ~1e-7 not f64 — the
new path is *more* accurate. Future cleanup option: route the
legacy s-only path through `boysAll` from integrals-cg.ts.

**Tests: 231 → 248** (+17). Typecheck clean. Lint warnings
unchanged from baseline.

### Phase C v2 — first multi-atom molecule (BeH₂) at chemical accuracy (2026-05-05)

Linear H–Be–H with the experimental Be–H bond R = 1.34 Å — **first
multi-atom molecule** in the project. STO-3G s-only basis (Be 1s, 2s
+ 2 × H 1s = 4 spatial / **8 spin-orbitals** / 256-dim Hilbert space,
6 electrons in the neutral molecule). Phase C v2 deliberately omits
Be 2p (would lift to 14 spin-orbitals / 16384-dim, requires Cartesian-
Gaussian p-shell integrals — Phase C v3).

**Headline at R = 1.34 Å (HEA L=12, λ=2 penalty, L-BFGS 1500 iters):**

| metric | value |
|---|---:|
| best ΔE | **0.044 mHa = 44 microhartree** (36× under chem-acc bar) |
| median ΔE | 3.03 mHa |
| correlation captured | **100.00%** |
| trials hitting chem-acc | 2/5 |

**Symmetric stretch curve (best of 5 trials):**

| R (Å) | E_FCI | best E_VQE | best \|ΔE\| | corr capture | chem-acc hits |
|---:|---:|---:|---:|---:|---:|
| 1.0 | -15.2785 | -15.2785 | 0.028 | 100.00% | 4/5 |
| 1.2 | -15.3492 | -15.3477 | 1.525 | 99.88% | 1/5 |
| **1.34** | **-15.3492** | **-15.3491** | **0.044** | **100.00%** | **2/5** |
| 1.6 | -15.3144 | -15.3144 | 0.043 | 100.00% | 2/5 |
| 2.0 | -15.2748 | -15.2748 | 0.019 | 100.00% | 3/5 |

12/25 trials within strict chemical accuracy (48%). Minimum of the
dissociation curve at R = 1.34 Å — exactly the experimental Be–H
bond. ~21 min wall-clock (5 R × 5 trials × ~50 s/trial on M2 Pro).
Status: **pass**.

**Path that worked (first try almost did):**
v0 attempt at HEA L=10 missed by 0.6 mHa — the high trial-to-trial
variance from 88-parameter Nelder-Mead-ish landscapes meant only
1/5 trials cleanly converged. L=12 (104 params) sharply tightened
the variance: every R above gets ≥ 1/5 chem-acc, most get 2-4/5,
medians dropped from 11-25 mHa → 0.08-7 mHa. The deeper ansatz
isn't more expressive — it's more *robust* across random inits.

**Scope honesty (still s-only):**
- BeH₂ s-only FCI = -15.349 Ha at R=1.34. Full STO-3G (with Be 2p)
  is closer to -15.6 Ha — the missing 2p shell gives ~250 mHa of
  dynamic correlation we can't access until Phase C v3 ships
  Cartesian-Gaussian p-shell integrals (Obara-Saika recurrence).
- HF reference energy is -13.32 Ha (correlation gap = 2 Ha, vs LiH's
  0.18 Ha) because the AO HF determinant is asymmetric. Phase C v2.5
  could add HF SCF + MO transformation to drop this gap by ~40×.

**What shipped:**
- `src/chemistry/integrals.ts`: Be STO-3G shells (`STO3G_BE_1S`,
  `STO3G_BE_2S`).
- `src/chemistry/sector.ts`: pulled `lowestInParticleSector` out of
  lih-builder for sharing across molecular builders.
- `src/chemistry/beh2-builder.ts`: 4 atomic shells → Löwdin → 256×256
  dense Hamiltonian via JW. Same template as lih-builder.
- `experiments/level-6-chemistry/E21-beh2-vqe.ts` + Level-6 dashboard
  panel updated. Publishable artifact at
  `experiments/results/2026-05-05/level-6/E21-beh2-vqe-publishable.json`,
  reproducible via `npx vite-node tools/run-phase-c-v2.ts` (~21 min).
- 8 new BeH₂ tests (`tests/chemistry/beh2.test.ts`): integral
  invariants, Hermiticity, [H, N̂] = 0, dissociation-curve minimum
  at the experimental Be–H bond.

**Tests: 223 → 231** (+8 BeH₂). Typecheck clean. Lint warnings
unchanged from baseline.

### Phase C v1 — chemical accuracy on LiH (2026-05-05)

Phase C v0's "honest negative" (HEA + Nelder-Mead plateaued at ~20 mHa,
89% correlation) was an optimizer artifact, not an ansatz limit. Adding
**L-BFGS** with central-FD gradient + Armijo line search closes the gap
to chemical accuracy on every cell at the equilibrium bond length:

**Headline at R = 1.595 Å, HEA L=6, λ=2 penalty, L-BFGS 500 iters:**

| metric | Phase C v0 | Phase C v1 | improvement |
|---|---:|---:|---:|
| best ΔE (mHa) | 20.0 | **0.0003** | **60,000×** |
| median ΔE (mHa) | 33.8 | 0.30 | 110× |
| correlation captured | 89.0% | **100.00%** | — |
| trials hitting chem-acc | 0/5 | 4/5 | — |

**Dissociation curve (best of 5 trials, mHa above E_FCI):**

| R (Å) | E_FCI | best E_VQE | best \|ΔE\| | hits chem-acc |
|---:|---:|---:|---:|---:|
| 1.0 | -7.7330 | -7.7330 | 0.00 | 3/5 |
| 1.4 | -7.8351 | -7.8351 | 0.00 | 5/5 |
| **1.595** | **-7.8434** | **-7.8434** | **0.00** | **4/5** |
| 2.0 | -7.8333 | -7.8332 | 0.04 | 2/5 |
| 3.0 | -7.7942 | -7.7941 | 0.07 | 3/5 |

17/25 trials within chemical accuracy, 100% correlation capture across
the curve. ~14 s wall-clock total. Status: **pass** with the strict
1.6 mHa median bar.

**What shipped:**
- `src/chemistry/optimizer.ts` adds `lbfgs(f, x0, opts)` with central-
  FD gradient (analytic gradients accepted via opts), m=8 history
  pairs (Nocedal two-loop recursion), Armijo backtracking line search.
  Strong-Wolfe deferred — Armijo suffices for smooth VQE landscapes
  even though it stalls on Rosenbrock.
- `src/chemistry/vqe.ts` exports `runVQE_HEA_Dense_LBFGS` mirroring the
  Nelder-Mead variant's signature.
- `experiments/level-6-chemistry/E20-lih-vqe.ts` switched to L-BFGS by
  default (L=6, λ=2, pert=0.20, 500 iters); pass bar tightened to
  median |ΔE| ≤ 1.6 mHa.
- `tools/run-phase-c.ts` regenerated; new publishable artifact at
  `experiments/results/2026-05-05/level-6/E20-lih-vqe-publishable.json`.
- 3 new L-BFGS unit tests (`tests/chemistry/lbfgs.test.ts`) on textbook
  quadratics + 10D anisotropic problem. Rosenbrock noted as a Wolfe-
  curvature limitation, not run.

Off-ramp #2 (per CLAUDE.md roadmap) — *"first browser-tab quantum
chemistry on real molecules, matches FCI to chemical accuracy"* — is
now reached for LiH in s-only STO-3G. Next chemistry frontier from
the roadmap: BeH₂ (6 e, 14 spin-orbitals → 16384-dim, may need MPS
backing instead of dense FCI) and adding Li 2p / Be 2p shells (needs
angular-momentum integrals).

**Tests: 220 → 223** (+3 L-BFGS sanity tests). Typecheck clean.
Lint warnings unchanged from baseline.

### Phase C v0 — first multi-orbital molecule (LiH) shipped (2026-05-05)

LiH ground-state simulation in a browser tab — first multi-orbital
chemistry in the project. Built from atomic STO-3G integrals on the fly,
all the way down to a 64-dim dense Hamiltonian, FCI reference projected
to the N=4 (neutral) sector, VQE recovering 89% of the correlation
energy at equilibrium.

**Headline (R = 1.595 Å, eq):**
- E_HF = −7.6607 Ha (HEA at θ=0 with HF occupation)
- E_VQE = −7.8234 Ha (HEA L=4 + λ=5 penalty, best of 5 random inits)
- E_FCI = −7.8434 Ha (N=4 sector projection of dense H)
- |ΔE| = 20.0 mHa, **89.0% of correlation energy captured**

**Dissociation curve (best of 5, mHa above E_FCI):**

| R (Å) | E_FCI | E_VQE | \|ΔE\| | corr capture |
|---:|---:|---:|---:|---:|
| 1.0 | −7.7327 | −7.7259 | 7.04 | 96.6% |
| 1.4 | −7.8351 | −7.8235 | 11.62 | 94.1% |
| **1.595** | **−7.8434** | **−7.8234** | **20.00** | **89.0%** |
| 2.0 | −7.8329 | −7.8143 | 19.01 | 86.7% |
| 3.0 | −7.7942 | −7.7585 | 35.69 | 48.6% |

Honest-negative side: chemical accuracy (1.6 mHa) is NOT reached. HEA L=4
+ Nelder-Mead in 30 dims plateaus around 7–35 mHa. Closing the gap
needs a particle-conserving ansatz (UCCSD) or gradient-based optimizer
(BFGS / COBYLA) — Phase C v1 work. Multi-reference character at long
bond (R = 3 Å, 49% capture) is the other half of that gap.

**What shipped:**
- `src/chemistry/integrals.ts` — generic `Shell` interface + multi-shell
  S/T/V/ERI primitives. Existing H-1s API kept as backward-compat wrapper.
- Li STO-3G basis: `STO3G_LI_1S` and `STO3G_LI_2S` (s-component of the
  L-shell, Pople 1969). 2p sub-shell deferred (needs angular-momentum
  integrals — Phase C v2).
- `src/chemistry/lih-builder.ts` — 3 atomic shells → Löwdin → 64×64 dense
  Hamiltonian via JW. `lowestInParticleSector(H, n, N)` projects to a
  specific particle-number sector for clean FCI references (the global
  ground state of H spans every sector since [H, N̂] = 0).
- `experiments/level-6-chemistry/E20-lih-vqe.ts` + Level-6 dashboard
  panel updated. Publishable artifact at
  `experiments/results/2026-05-05/level-6/E20-lih-vqe-publishable.json`,
  reproducible via `npx vite-node tools/run-phase-c.ts` (~6 s).
- 19 new tests (10 in `integrals-shells.test.ts`, 9 in `lih.test.ts`).
  Covers: H-1s consistency between old/new API, Li shell normalization,
  AO matrix symmetry, [H, N̂] = 0, dissociation-curve shape, plateau
  behaviour at large R.

**Bug fixes shipped along the way:**
- `src/chemistry/h2-builder.ts → expectationDense`: `N = 16` was hard-
  coded, silently reading garbage when called with a 64-dim LiH H.
  Now infers dim from `psi.length / 2`. Was the gate that capped
  every multi-orbital VQE attempt at non-physical energies.
- `src/chemistry/lih-builder.ts → lowestInParticleSector`: column-major
  vs row-major eigenvector indexing (was reading row 0 of every column
  instead of column 0). The wrong indexing made the projection look
  like the global ground state at any sector — gave a misleading
  E_FCI of −7.716 Ha instead of the correct −7.843 Ha.

**Tests: 201 → 220** (+19 across integrals-shells / lih).
Typecheck clean. Lint warnings unchanged from baseline.

### Phase B v1 — publishable artifacts shipped (2026-05-05)

DMRG ground-state energy per site at the analytical thermodynamic limit
for both textbook 1D models, all the way to **N = 128 in a browser tab,
χ = 32**, on a single M2 Pro. Headline numbers:

**E18 — TFIM (h = 1, critical) vs Pfeuty −4/π = −1.273240**

| N | E/N | \|E/N − e_∞\| | rel | wall |
|---:|---:|---:|---:|---:|
| 16 | −1.251024 | 0.02222 | 1.74% | 3.2 s |
| 32 | −1.262010 | 0.01123 | 0.88% | 14.4 s |
| 48 | −1.265725 | 0.00751 | 0.59% | 55.5 s |
| 64 | −1.267593 | 0.00565 | 0.44% | 56.5 s |
| 80 | −1.268718 | 0.00452 | 0.36% | 75.8 s |
| 100 | −1.269619 | 0.00362 | 0.28% | 120.1 s |
| **128** | **−1.270409** | **0.00283** | **0.22%** | 165.9 s |

**E19 — Heisenberg AFM (J = 1) vs Bethe ¼ − ln 2 = −0.443147**

| N | E/N | \|E/N − e_∞\| | rel | wall |
|---:|---:|---:|---:|---:|
| 16 | −0.431984 | 0.01116 | 2.52% | 3.4 s |
| 32 | −0.437416 | 0.00573 | 1.29% | 18.4 s |
| 48 | −0.439291 | 0.00386 | 0.87% | 40.5 s |
| 64 | −0.440241 | 0.00291 | 0.66% | 59.2 s |
| 80 | −0.440815 | 0.00233 | 0.53% | 92.0 s |
| 100 | −0.441277 | 0.00187 | 0.42% | 121.5 s |
| **128** | **−0.441682** | **0.00146** | **0.33%** | 186.6 s |

Both: textbook 1/N convergence (each doubling of N halves \|Δ\|), exactly
what CFT predicts for OBC boundary corrections.

Off-ramp #1 from the roadmap is now reachable — *"longest 1D MPS in a
browser, analytically validated."* Artifacts at
`experiments/results/2026-05-05/level-2/E1{8,9}-…-publishable.json`.

Reproduce: `npx vite-node tools/run-phase-b.ts` (default Ns=[16..128],
χ=32). ~13 minutes wall-clock per model.

Bug fix in this run: `src/manybody/dmrg.ts → makeRandomMPS` had a
JS-bit-shift overflow at q ≥ 31 (`1 << 31` is negative); chi clamp now
uses a safe-shift helper. Was the gate that would have blocked any
N ≥ 32 DMRG run.

### Phase B v0 (2026-05-05)

- **Real-form XXZ / Heisenberg MPO** — `xxzMPOReal`, `heisenbergMPOReal`
  in `src/manybody/mpo.ts` rewrite XX + YY = 2(S⁺S⁻ + S⁻S⁺), so the
  bulk W has *zero* imaginary entries. Same operator as the imag-Y form
  to f64 precision (5 cross-checks in `tests/manybody/mpo.test.ts`).
  Lets the existing real-only DMRG + matrix-free Lanczos handle
  Heisenberg / XXZ with no complex-Hermitian extension.
- **DMRG on real-MPO Heisenberg / XXZ** validated against ITensor
  (`tests/manybody/dmrg-real.test.ts`): Heisenberg N=8 within 1e-4,
  XXZ N=16 (Δ=0.5 and Δ=1) within 1e-3 of the ITensor reference at
  χ=32, ~3 s per N=16 run.
- **Analytical thermodynamic limits** — `src/manybody/analytical.ts`:
  - `tfimPfeutyEnergyPerBond(J, h)` — Simpson's-rule numerical Pfeuty
    integral, exact at the special points (e_∞(λ=0)=−J, e_∞(λ=1)=−4J/π).
  - `heisenbergBetheEnergyPerBond(J)` — closed form J·(¼ − ln 2).
  - `xxBetheEnergyPerBond(J)` — closed form −2J/π for the free-fermion
    XX point (Δ = 0). Sanity test: more negative than Heisenberg.
- **E18 — DMRG TFIM vs Pfeuty** (`experiments/level-2-mps/E18-tfim-pfeuty.ts`).
  UI default: N ∈ {16, 24, 32, 48}, h ∈ {0.5, 1, 2}, χ=32.
  Publishable: pass `runE18({ Ns: [16,32,48,64,80,100,128], chiMax: 64 })`
  from devtools.
- **E19 — DMRG Heisenberg AFM vs Bethe** (`E19-heisenberg-bethe.ts`).
  UI default: N ∈ {16, 24, 32, 48}, J=1, χ=32. Same publishable opts.
- **`runLevel2()` runs E5/E6/E7 + E18 + E19**, dashboard panel updated
  ("Run E5–E7 + E18 + E19", expected 3–12 min). Level-2 e2e timeout
  bumped to 15 min.
- **Convergence sanity tests** (`tests/manybody/dmrg-pfeuty-bethe.test.ts`,
  ~19 s): both TFIM (h=1) and Heisenberg AFM show |E/N − e_∞| strictly
  decreasing as N grows from 8 → 16 → 24, hitting < 0.07 / < 0.06 at
  N=24 with χ=32. The boundary correction dominates at this size; the
  publishable artifact at N=128, χ=64 should hit ≤ 0.01 / ≤ 0.02.
- **Tests: 182 → 201** (+19 across mpo / dmrg-real / analytical /
  dmrg-pfeuty-bethe). Typecheck clean. Lint warnings unchanged from
  baseline.

### What's shipped since 2026-05-01

- **L3 Tier C** — 8×8 cascade fusion (5 ops → 1 dispatch), `src/three-qubit-dense.ts`,
  E12. **4.18× headline at N=15 D=80**, worst F=0.9999988.
- **L3 Tier D** — 16×16 cascade fusion (7 ops → 1 dispatch),
  `src/four-qubit-dense.ts`, E13. **3.14× plateau, honest negative**:
  per-block compute scales 4× per tier while memory traffic only 2×, so
  Tier D crosses into compute-bound territory. Tier C remains the
  bandwidth-bound sweet spot.
- **GPU MPS Phase 5 v0** — rectangular SVD via padding (lifts the
  square-matrix constraint). `src/shaders/mps-two-site-merge.wgsl` got
  `mStride`, `applyTwoSite` zero-pads to nMax × nMax.
- **GPU MPS Phase 5 fast-path** — `sigma-sort.wgsl` (single-thread
  insertion sort) + single-submit. Per-gate cost ↓ ~25% (432 → 326 μs).
- **GPU MPS Phase 5 v1** — canonical sweeps on GPU. New
  `mps-two-site-extract-uS.wgsl` + `mps-two-site-build-vh.wgsl` for
  the symmetric "push residual leftward" pipeline. `applyTwoSiteLeft` +
  `canonicalize(targetBond)` methods on `MPSGpu`. Arbitrary gate
  orderings now work, not just brick-wall.
- **GPU MPS Phase 6 v0** — `jacobi-svd-large.wgsl` (n ≤ 48, 37 KB
  workgroup storage; activates on adapters with the budget).
- **GPU MPS Phase 6 v1** — `jacobi-svd-storage.wgsl`. A and V live in
  global memory, only 4 active columns enter shared per (p,q) rotation.
  Workgroup footprint constant 2.5 KB. **n ≤ 64 on every adapter** →
  χ_max = 32 universally.
- **DMRG-v0** — `src/manybody/dmrg.ts`. Direct dense diagonalization +
  statevector → MPS conversion via SVD chain. ITensor cross-checked at
  N=8 to f64 precision. Real two-site DMRG with Lanczos is Phase A.
- **Mobile-first responsive** — viz / experiments / gpu-mps / landing /
  demo all validated at 390×844 in Playwright. `e2e/mobile-smoke.spec.ts`.
- **Public repo** — https://github.com/abgnydn/webgpu-q. MIT, CI green,
  CHANGELOG, README badges + screenshots, **v0.1.0 release** tagged.
- **Headline numbers on landing**: **4.18×** kernel fusion (Tier C),
  ITensor-validated, **160 tests** + 11 e2e specs.

### Test surface (current)

- `npm run test` → **160 / 160** (was 122). Adds Tier C/D math (16),
  three- and four-qubit cascade fusion, DMRG-v0 cross-checks (9),
  remaining many-body + chemistry coverage.
- `npx tsc --noEmit` → clean.
- `npm run lint` → clean (1 unused-eslint-disable warning, pre-existing).
- `npx playwright test` → **11 / 11** specs. Includes:
  - `e2e/level-{1,2,3,6}.spec.ts` — full ladder e2e
  - `e2e/gpu-mps.spec.ts` — Phases 1A / 1B / 2 / 4a / 4b/5 / 5 v1
  - `e2e/landing-smoke.spec.ts`, `e2e/viz-smoke.spec.ts`
  - `e2e/mobile-smoke.spec.ts` — 4 viewport tests at 390×844
  - `e2e/generate-og.spec.ts` — OG image regenerator

### Live deployment

Same: https://webgpu-q.vercel.app. Redeploy:
`vercel deploy --prod --scope ahmet-bar-gnaydns-projects`. **Standing
preference: do NOT auto-deploy** — test locally only; deploy only when
the user explicitly asks.

### Companion repos polished in the same pass

All 12 public repos at https://github.com/abgnydn now have LICENSE / CI /
CHANGELOG / README badges / v0.1.0 release. Same playbook applied
across `webgpu-dna`, `gpubench`, `zero-tvm`, `markview`, `safenpm`,
`webgpu-fusion-max`, `wgpu-adas-bench`, `webgpu-p2p-evolution`,
`webgpu-transformer-fusion`, `wgpu-native-bench`, plus the umbrella
`webgpu-kernel-fusion`. `safenpm` had its npm version reset from
1.0.0 → 0.1.0 (1.0.0 deprecated on the registry; `latest` dist-tag
moved to 0.1.0).

---

## Current state of play (snapshot from 2026-05-01, kept for reference)

### What's green

- `npm run test` → **122 / 122** — adds many-body suite (Hamiltonian
  invariants 6, ITensor reference 7, imaginary-time ground-state 3) on
  top of the chemistry + fusion + MPS unit tests already in place.
- `npx tsc --noEmit` → **clean**.
- `npm run lint` → **clean**.
- `npx playwright test` → **4 / 4 specs pass, ~1.4 min total** on M2 Pro
  (Chromium → Apple Metal-3). JSON artifacts auto-saved to
  `experiments/results/YYYY-MM-DD/level-N/`.
- Level 1 (statevector): **E1–E4 shipped + e2e**.
  - E1 gate fidelity vs CPU reference (worst F=0.999999116).
  - E2 bandwidth roofline (`apple metal-3` now in the lookup at 200 GB/s
    estimate; remaining noisy=true is timing variance, override exact
    peak via `?peak=N` URL param when needed).
  - E3 runtime scaling slope ≈ 1.0 (most recent fit: 0.988–1.040).
  - E4 dispatch-overhead α ≈ 22 μs, β ≈ 0.08 ns/amp.
- Level 2 (MPS): **E5–E7 shipped + e2e**.
  - E5 correctness: 180 / 180 cells pass after a Jacobi SVD stability fix
    (see "Hardened SVD" below).
  - E6 qubit ceiling: timing-noisy across N — environmental, not a bug.
  - E7 χ-scaling: published with status="fail" — this is an HONEST
    NEGATIVE RESULT, not a bug. The protocol expected slope ≈ 1, but at
    N = 16 entanglement entropy saturates around depth 4 (S ≤ N/2 bound),
    so log₂(χ_required) plateaus and the fit slope drops to 0.45. Re-running
    at larger N (≥ 24) would put more depths in the linear regime; that's
    the next paper-worthy follow-up, not a fix to ship.
- Level 3 (fusion): **E8–E11 shipped + e2e**. Both tiers of fusion live.
  - E8 correctness: 360 / 360 cells (worst F=0.999998).
  - E9 dispatch collapse: α_eff drops 38 μs (k=1) → 11 μs (k=64), 3.4× ratio.
  - E10 throughput (same-qubit chain fusion): best speedup 2.5× at
    N=20 D=160, fused peak ≈ 900 GB/s.
  - E11 brick-wall layer fusion (the real "kernel fusion" thesis):
    `(single_q ⊗ single_{q+1}) · CNOT(q, q+1)` fuses into one dense
    4×4 dispatch via `src/shaders/two-qubit-dense.wgsl` + the new
    `applyDense4x4` method on QuantumCircuit. **Best speedup 2.69×
    at N=20 D=80**, ≥ 2× on every cell from N=12 D=80 upwards;
    worst fidelity ≥ 1−1e-5 (f32 noise floor). Math verified by
    `tests/two-qubit-dense.test.ts` (fused 4×4 matches the 3-step
    sequence to FP precision).
  - Default `chainK` is **32** (was FUSED_CHAIN_MAX_K=64 — the protocol's
    default depth=40 violated the `depth ≥ chainK` preflight). Pass
    `chainK: 64` to opts when running the deeper sweep.
- Level 6 (chemistry): **E16 shipped, full dissociation curve + e2e**.
  - **Molecular integrals from scratch** (`src/chemistry/integrals.ts`,
    `src/chemistry/h2-builder.ts`): STO-3G H 1s contracted Gaussians,
    Boys F₀ + closed-form s-shell overlap / kinetic / nuclear-attraction /
    4-center ERIs. Symmetry MOs (σ_g / σ_u) → 16×16 dense H built by
    direct enumeration over the 16 occupation-number basis states with
    Jordan-Wigner sign bookkeeping. No Pauli decomposition needed for
    VQE — expectation = ψ†Hψ on the dense matrix.
  - At R = 0.7414 Å: integral-derived FCI = **−1.13727008 Ha** — matches
    canonical PySCF literature (−1.137270) to 7 decimals. The earlier
    OpenFermion-published Pauli table in `src/chemistry/h2.ts` is now
    just a cross-check; it differs by 2 mHa (FP rounding in the
    published coefficients, not a code bug).
  - **Full dissociation curve sweep**: 5 R-values × 10 random-init
    trials = 50 VQE optimizations. Hit-rate **50/50 within chemical
    accuracy**.

    | R (Å) | E_FCI (Ha) | median \|ΔE\| | max \|ΔE\| |
    |-------|-----------:|--------------:|-----------:|
    | 0.5    | −1.05516   | 0.000 mHa     | 0.000 mHa  |
    | 0.7414 | −1.13727   | 0.000 mHa     | 0.000 mHa  |
    | 1.0    | −1.10115   | 0.000 mHa     | 0.000 mHa  |
    | 1.5    | −0.99815   | 0.000 mHa     | 0.000 mHa  |
    | 2.5    | −0.93606   | 0.000 mHa     | 0.001 mHa  |

  - VQE config: HEA L=4 (20 params), Nelder-Mead 8000 max-iter,
    initial-state preparation = X gates on qubits 0, 1 — so the optimizer
    starts at the |HF⟩ closed-shell determinant. Without HF prep, HEA
    wanders out of N_e = 2 at long R and traps at unphysical minima
    100+ mHa above FCI. With it, every trial reaches FCI to FP precision.

### Hardened SVD (2026-05-01)

`src/linalg.ts → jacobiSweepPair` got three guards after the e2e suite
surfaced a deterministic V8-Chromium-only zero-norm bug at brick-depth-4
N=8 trial=3 (passed in node, failed in browser):

1. **Absolute-mag floor**: `mag ≤ tol · sqrt(max(‖p‖, ‖q‖))` skips
   rotations where the cross overlap is at FP noise relative to column
   norms — prevents 45° mixing of two near-zero columns from cancelling.
2. **NaN / Inf guards** on `(zr, zi)` and `(cj, sj)` — corrupting A or V
   with non-finite values is far worse than skipping a single rotation.
3. **Strict zero check** uses `!(app > 0)` so a NaN column norm bails
   the same way as a zero one (NaN > 0 is false; old `=== 0` missed it).

### Live deployment

The whole project is shipped at **https://webgpu-q.vercel.app**:

- `/` — landing page
- `/viz.html` — 4D hyperscope (chemistry density + pair density + spin chain + phase-transition sweep + quench light cone)
- `/experiments/` — research dashboard (E1–E16)

Vercel build via `npm run build` → `dist/`. Headers add COOP/COEP for SharedArrayBuffer-safe contexts. Redeploy: `vercel deploy --prod --scope ahmet-bar-gnaydns-projects`.

### Tier-1 + Tier-2 viz extensions (2026-05-01, even later)

- **Per-site Bloch arrows** in the bond-network panel: each site is a circle whose color encodes |⟨σ⟩| and contains an arrow showing the projection of (⟨σ_x⟩, ⟨σ_z⟩). Visible "all up" vs "all sideways" depending on TFIM phase.
- **Order-parameter sweep** auto-runs h ∈ [0, 2] with 25 points, plots |m_z| (orange) and S/S_max (cyan) vs h on a small SVG line chart in the controls column. h_c = 1 marker. The textbook QPT curve, generated live in your tab.
- **Quench-dynamics mode**: `recordTFIMQuench` evolves a product state under exp(-iτH), snapshots every Trotter step, and the bottom panel becomes a **sites × time light-cone heatmap**. Color = bond entropy. Lieb-Robinson cone visible as the heated triangle.
- **H_n linear chain chemistry** (`src/chemistry/hn-builder.ts`): generalizes H₂ integrals to n equally-spaced H atoms with Löwdin orthogonalization (S^{-1/2}) and full 4-index AO→OAO transform. Builds 2^(2n)-dim dense Hamiltonian. H_2 cross-checks the dedicated H₂ builder to 1e-8; H_3 (6 qubits) and H_4 (8 qubits) compute and stretch sensibly.

### Deferred (deliberately, with reasons)

- **Full two-site DMRG with Krylov local solver** — imag-time MPS already validates against ITensor to ≤ 5 mHa, so DMRG is purely an engineering / speed upgrade, not a correctness one. ~1-2 weeks to port from ITensor.
- **L4 swarm (WebRTC)** — protocol only. Two-process; needs a dedicated session.
- **L5 hardware (IBM)** — protocol only. Needs IBM Quantum token.
- **L6 E17 cross-sections** — needs G4EMLOW 8.8 tables from `webgpu-dna`.
- **L3 Tier C wider-window fusion** — 3-/4-qubit tile fusion.

### Many-body / DMRG-class extension (2026-05-01, late)

Added `src/manybody/` with Hamiltonian1D library (Heisenberg / TFIM / XXZ),
real-symmetric eigendecomposition (`dense-eig.ts`), matrix exponential
(`expm.ts`), imaginary-time-evolution ground-state solver (`ground-state.ts`).
Validated externally against ITensor DMRG: `tools/itensor-reference.jl`
generates `tests/manybody/itensor-reference.json` with energies for 19
configurations across the three models, our exact-diagonalization matches
ITensor to 1e-7 on N ≤ 8, and our imaginary-time MPS solver matches to ≤ 5 mHa
on N = 8.

Wired into the viz: `viz.html` bottom pane now has a model dropdown
(brick-wall circuit / TFIM ground state / Heisenberg ground state). For TFIM
mode, an `h/J` slider crosses the quantum phase transition — peak
entanglement entropy at h ≈ J is visible in real time as bonds thicken /
change color in the network panel.

### What's red / deferred

1. **L4 swarm (WebRTC)** — protocol only. Genuinely two-process; needs a
   dedicated session.
2. **L5 hardware (IBM)** — protocol only. Blocked on IBM Quantum token.
3. **L6 E17 cross-sections** — needs G4EMLOW 8.8 tables from sibling
   `webgpu-dna` repo. Cross-link not yet plumbed.
4. **L3 Tier C wider-window fusion** — E11 fuses 2 qubits per dispatch.
   The next ceiling is M-qubit fusion (M=3 or 4) where a 2^M × 2^M
   unitary collapses several brick-wall layers in one pass. Worth
   benchmarking if you want to push past the current 2.7× speedup.
5. **Full DMRG ground-state solver** — current code uses imaginary-time
   evolution which is good enough for textbook 1D models but slower than
   true two-site DMRG with a Krylov local solver. Porting ITensor's
   approach is ~2 weeks and would unlock χ ≥ 100 robustly.

### Known-good seeds

Added to `experiments/lib/seeds.ts`:
`E5_MPS_CORRECTNESS`, `E6_MPS_CEILING`, `E7_MPS_CHI_SCALING`, `E16_H2_VQE`.
Don't rename — artifacts reference them by string.

---

## Resume instructions (snapshot — superseded by the roadmap above)

**The "next step" is no longer wiring up E5 / E6 / E7 — those all shipped.
The current next step is Phase A of the roadmap above** (truncation
renormalization on GPU, MPO representation, real two-site DMRG with
Lanczos). Everything below is preserved for reference but reflects the
state circa 2026-04-22 when the project was still wiring Level 2.

### 1. Verify nothing rotted

```bash
cd /Users/ahmetbarisgunaydin2/webgpu-q
npm run test       # expect 160/160 (was 73/73 in old snapshot)
npm run typecheck  # expect clean
npm run lint       # clean on src/, tests/, experiments/
```

If any of these fail, diagnose before writing new code.

### 2. Dry-run E5 in the browser

E5 is written but never executed against a live GPU adapter. Temporary
wire-up path:

- Option A (fastest): add a throwaway button in `experiments/index.html` or a
  scratch `main-e5.ts` that calls `runE5({ trials: 2, Ns: [4, 6, 8] })`,
  open devtools, confirm F ≥ 0.999 per cell.
- Option B (correct): skip to step 4 (wire Level 2 properly) and run E5
  through the real orchestrator.

Goal: see one pass banner before writing E6 / E7. Catch bugs cheap.

### 3. Write E6 — qubit-count ceiling

File: `experiments/level-2-mps/E6-qubit-ceiling.ts`.

Template: mirror `E5-mps-correctness.ts` structure. Hypothesis from
`experiments/level-2-mps/protocol.md`:

> MPS reaches N ≥ 70 qubits in a single browser tab at χ = 32 for depth-4
> brick-wall circuits, sweep time ≤ 1.0 s median.

- Sweep N ∈ {32, 48, 64, 72, 96, 128}.
- Circuit: depth-4 brick-wall (same builder as E5).
- 20 trials per N (drop to 5 once median stabilizes — flag if IQR > 0.1·median).
- Record: sweep wall-seconds, peak resident χ (max bond dim), approximate
  RAM footprint (`N · 2 · χ² · 16 B`).
- Pass bar: no OOM at N=72, median sweep ≤ 1.0 s.
- **Negative result is evidence.** If it OOMs at N=72, publish the failure
  with the specific limit (heap vs step time vs SVD blowup).

### 4. Write E7 — χ scaling vs entanglement entropy

File: `experiments/level-2-mps/E7-chi-scaling.ts`.

- N = 16 fixed. Depth ∈ {1, 2, 3, 4, 5, 6, 8}.
- 50 Haar-random brick-wall seeds per depth.
- Per (depth, seed) sweep χ ∈ {2, 4, 8, 16, 32, 64, 128}, find smallest
  satisfying F ≥ 0.999 vs CPU statevector.
- Also record bipartite entanglement entropy S at mid-cut (from Schmidt
  spectrum of canonical form — `mps.bondDimensions()` + cached σ).
- Expected scaling in 1D brick-wall: χ_required ≈ 2^S, S ≈ depth. Fit
  log₂(χ_required) vs depth, slope within 20% of 1.0 = pass.

MPS currently exposes `bondDimensions()` but NOT raw σ arrays. If S is
needed, add `schmidtSpectrum(cut: number): Float64Array` — read from
tensors[cut] after a canonical sweep centered on that bond.

### 5. Wire Level 2 to UI

Two small files + one HTML edit:

1. Create `experiments/level-2-mps/run-all.ts` by copying Level 1's file
   structure. Export `runLevel2(opts?)` that runs E5, E6, E7 in order.
2. In `experiments/runner.ts` add
   `import { wireRunLevel2Button } from "./level-2-mps/run-all.js"; wireRunLevel2Button();`.
3. In `experiments/index.html`, replace the "awaits src/mps.ts" row with a
   panel matching Level 1's layout — button id `run-level-2`, results table
   `level-2-results`, banner `level-2-status`.

Browser sanity: open http://localhost:5175/experiments/, click
`Run E5–E7`, confirm all three go green, download the JSONs.

### 6. Commit checkpoint

```bash
npm run test && npm run typecheck && npm run lint
```

All three green → commit with a message like
"Level 2: MPS simulator + E5/E6/E7 experiments + UI wiring".

### 7. Pick a Level 3–6 to attempt next

Recommended order (easiest → hardest given current infra):

- **L3 fusion** (local, no hardware): `src/shaders/fused-chain.wgsl` that
  takes a list of gate ops and applies them in one dispatch per chain.
  Baseline: E2 bandwidth roofline. Target: beat unfused at N ≥ 16 by ≥ 2×.
- **L6 chemistry** (CPU, math-heavy): VQE for H₂ / HeH⁺ / LiH against FCI
  reference. Cross-link to `webgpu-dna` radiolysis is the demo moment.
- **L4 swarm** (network): WebRTC coordinator + pair-wise bond contraction
  across two peers. Hardest infra. Do last of the solo pieces.
- **L5 hardware** (external dep): IBM Quantum token, `qiskit-runtime`
  bridge script, shot-level cross-verify. Blocked on access.

Each level pattern: `src/` impl → `tests/` unit tests (local only) →
`experiments/level-N-*/E*.ts` with the research harness → `run-all.ts` →
UI wiring.

---

## Research-grade discipline (non-negotiable)

These come from `RESEARCH.md`. Every experiment enforces them.

### Reproducibility

- No `Math.random()` in any experiment path. Every random draw uses a named
  seed from `experiments/lib/seeds.ts` via `mulberry32(seed)`.
- Every JSON artifact records: git SHA (when available), `navigator.userAgent`,
  `adapter.info`, WebGPU limits, UTC ISO8601 timestamp, and echoes back
  `protocol`, `hypothesis`, `passBar`, `seed`, `warmup`, `trials`. See
  `experiments/lib/env.ts → captureEnv(device, adapter)`.
- Artifact shape is locked: `{ meta, env, rows, status, diagnosis }`. Do not
  add top-level keys without updating `experiments/lib/runner.ts` and the
  downstream dashboard.

### Timing

- `performance.now()` **with a forced GPU sync before AND after** — a mapped
  readback of a tiny buffer. `queue.submit` alone is non-blocking so raw
  timing is fiction. Harness: `experiments/lib/runner.ts → timedRun`.
- Discard 5 warmup samples. Retain 20 trials. Report median, p10, p90, p99,
  std, IQR — never single-shot.
- If `std/median > 0.1` on any cell, mark the artifact `"status": "noisy"`.

### Correctness

- Use **fidelity** F = |⟨ψ_ref | ψ_test⟩|², not max|Δp|. Two states can share
  a probability distribution and differ in phase — that kills any downstream
  controlled gate. Use `experiments/lib/fidelity.ts → stateMetrics`.
- Pass bar for f32-amplitude GPU paths: `F ≥ 1 − 1e-5`.
- Pass bar for f64 MPS vs f64 statevector: `F ≥ 0.999` (MPS has SVD
  truncation + accumulated Jacobi error, ~9 digits is realistic at χ = 64).
- Secondary: TVD, L1, L2, max|Δp|, ‖ψ_ref‖², ‖ψ_test‖² — always reported.

### Honest negative results

- If an experiment fails its pass bar, still commit the JSON with
  `"status": "fail"` and a `"diagnosis"` string naming the first failing
  cell and the smoking gun. **Failures are the evidence.** No silent
  rerunning until it passes.
- Example (MPS canonical-form bug, 2026-04-22): brick-wall F = 0.25 at depth
  2. Diagnosis: "non-monotonic two-site gate order breaks mixed-canonical
  invariant, local Frobenius norm ≠ global norm, renormalization distorts."
  Fix: `_canonicalizeBond(q)` before every `applyTwoSite`.

---

## Commands

```bash
npm install
npm run dev          # Vite dev server, http://localhost:5175
                     # experiments live at http://localhost:5175/experiments/
npm run test         # Vitest, ~500 ms (one outlier 5 s for the MPS bug repro)
npm run test:watch   # TDD loop
npm run typecheck    # tsc --noEmit (strict, noUncheckedIndexedAccess on)
npm run lint         # ESLint flat config, src/ tests/ experiments/
npm run build        # → dist/
npm run test:e2e     # Playwright, all 4 levels headless (~1.4 min on M2 Pro).
                     # Saves JSON artifacts to experiments/results/<date>/level-N/.
                     # Each level is also reachable via window.__webgpuq.runLevelN()
                     # in devtools at /experiments/.
npm run test:e2e:headed   # Same, but with a visible browser window.
```

---

## File layout

```
src/
  shaders/
    single-qubit.wgsl    # 1-q gate kernel, N/2 threads, 2×2 complex matrix via uniform
    two-qubit.wgsl       # controlled-U kernel, N/4 threads
  gates.ts               # H, X, Y, Z, S/Sdg, T/Tdg, Rx/Ry/Rz, P, matrixFloats()
  quantum.ts             # QuantumCircuit (GPU) + initGPU() with requiredLimits
  cpu-reference.ts       # CpuCircuit (Float64 TS reference, ground truth)
  circuits.ts            # bell, ghz, qft, deutschJozsa, randomCircuit builders
  linalg.ts              # ComplexMatrix, Jacobi complex SVD, matmul   — Level 2
  mps.ts                 # MPS class with canonical form + TEBD         — Level 2
  bench.ts               # GPU vs CPU throughput sweep (pre-research harness)
  main.ts                # Legacy browser demo entrypoint

tests/
  gates.test.ts          # Bell, GHZ, XX=I, HH=I, T⁴=Z, …
  fidelity.test.ts       # stateMetrics unit tests
  stats.test.ts          # median / percentile / IQR
  linalg.test.ts         # SVD round-trip, orthonormality, diagonal
  mps.test.ts            # Bell / GHZ / brick-wall / canonical / truncation

experiments/
  index.html             # Research dashboard (run buttons, result tables)
  runner.ts              # Dashboard entry point — wires each level's run-all
  lib/
    seeds.ts             # Named deterministic seeds (no Math.random)
    runner.ts            # timedRun harness + Artifact / ArtifactMeta schema
    env.ts               # captureEnv(device, adapter) → EnvBlock
    fidelity.ts          # stateMetrics, FIDELITY_PASS_BAR
    stats.ts             # stats() — median, p10/p90/p99, std, IQR
  level-1-statevector/
    protocol.md
    E1-gate-fidelity.ts
    E2-bandwidth-roofline.ts
    E3-scaling-law.ts
    E4-dispatch-overhead.ts
    run-all.ts           # runLevel1() + wireRunAllButton()
  level-2-mps/
    protocol.md
    E5-mps-correctness.ts   # ← in flight, not yet run
    (E6-qubit-ceiling.ts)   # ← TODO
    (E7-chi-scaling.ts)     # ← TODO
    (run-all.ts)            # ← TODO
  level-{3,4,5,6}-*/protocol.md   # protocols only, no code yet
  results/                 # JSON artifacts, organized YYYY-MM-DD/level-N/
```

---

## Architecture notes (carry forward)

### Statevector (Level 1)

- Amplitudes stored as `vec2<f32>` interleaved (re, im). Buffer = `2^(N+3)` B.
- Single-qubit gate: `N/2` threads, each processes the pair `(i, j)` where
  bit `q` is 0 and 1. Apply 2×2 complex matrix from uniform buffer.
- Two-qubit (controlled-U): `N/4` threads, index scattered around control
  + target bits, only control=1 is touched.
- `initGPU()` MUST request the adapter's max `maxBufferSize` and
  `maxStorageBufferBindingSize` via `requiredLimits`. Default 128 MiB cap
  silently truncates N ≥ 25 dispatches.
- No atomics needed — gate application is pair-local read / write, zero contention.

### MPS (Level 2)

- Tensor storage: `tensors[i]` is a `ComplexMatrix` of shape
  `(χ_L · 2, χ_R)` — left-grouped. Element `T[l, s, r]` at row `l·2 + s`,
  col `r`. Single-qubit gates apply cleanly this way.
- Statevector convention: qubit 0 is LSB of the index —
  `ψ[s_0 + 2·s_1 + 4·s_2 + …]`. `mps.statevector()` follows this for
  comparison with `CpuCircuit.psi`.
- Two-site gate order within the 4×4: `i = s_lo · 2 + s_hi` — site `q` is
  the MSB within the pair. Controlled-U needs the right ordering;
  see `buildControlledMatrix4(U, controlIsLo)`.
- **Canonical form invariant** (critical). Two-site TEBD needs
  `‖M‖_F² = ‖ψ‖²`, which requires left-canonical on sites `[0..q−1]` and
  right-canonical on `[q+2..N−1]`. `_canonicalizeBond(q)` does the sweep.
  Cost: O(N · χ³) per two-site gate. Trivial at N ≤ 20, χ ≤ 64.
- SVD is one-sided Jacobi on complex matrices: phase-align col q by
  e^(−iφ) so ⟨p, q⟩ is real, then apply the real Jacobi rotation. 60 sweep
  cap, TOL = 1e-14.
- `apply*` returns void (mutates). `statevector()` refuses `N > 24`.
- v1 constraint: `applyTwoSite` / `applyControlled` require `|c − t| = 1`.
  Non-adjacent two-qubit gates need SWAP ladders (not yet implemented).

### Research harness

- `experiments/lib/runner.ts → timedRun(device, fn, cfg)` is the only
  legitimate way to measure wall time on GPU paths. It owns the sync
  fence and the error-scope guards.
- `Artifact<Row>` is the JSON shape. `emitArtifact` logs; `downloadArtifact`
  serves it as a download from a click handler.
- Per-experiment logs use the `[artifact:protocol] status — diagnosis`
  prefix on stdout so CI greps can find pass/fail without parsing JSON.

---

## WebGPU gotchas (carry forward from webgpu-dna)

- `initGPU()` MUST pass `requiredLimits` for `maxStorageBufferBindingSize`
  and `maxBufferSize`. Default 128 MiB cap silently truncates large
  dispatches.
- `atomicAdd` only on `u32`. Not needed in statevector path (no contention).
- No recursive function calls in WGSL. All shaders are single-pass.
- Uniform buffers must be aligned.

---

## Hero-mode conventions for this repo

- Scope-honest. Most research tasks here = hours for a capable agent, not
  weeks. Attempt now; decompose only if truly large.
- Speculation labeled. "This should work" ≠ "tested". Benchmark > belief.
- Raw WGSL > framework. Dispatch ceremony is the enemy.
- Edge hardware underrated. The thesis is "no one has shipped this in a
  browser tab." Don't reinvent it; ship the numbers.

---

## Related repos / links

- **Sibling:** `/Users/ahmetbarisgunaydin2/Downloads/webgpu-dna/` —
  Geant4-DNA port. Has its own CLAUDE.md. Level 6 chemistry cross-links here.
- `kernelfusion.dev` — umbrella theory.
- `gpubench.dev` — WebGPU bench harness reuse pattern.
- Pan & Zhang 2021 (arXiv:2103.03074) — Sycamore tensor-network baseline.
- Karamitros 2011 — IRT chemistry, cross-link target.
- IBM Heron r2 (156q, 2025), Nighthawk (120q, Jan 2026) — E14 target.
- Schollwöck 2011 — MPS / DMRG review, χ-vs-error baseline.
- Vidal 2003 — iTEBD algorithm (what `applyTwoSite` implements).

---

## License

MIT (simulation). Research protocol and experiment artifacts: MIT.
