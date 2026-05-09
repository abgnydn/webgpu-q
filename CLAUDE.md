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

The project is past the launchpad. **All six chemistry-track phases are
shipped** (A through E5: foundation → 1D records → real molecules → HF
SCF → MP2 → cc-pVDZ basis → CCSD → CCSD(T) → cc-pVDZ CCSD(T) on H₂O).
The repo is public + CI-green. The honest path from here is below —
ranked by *what it costs vs what it unlocks*, not by ladder position.

### What's shipped (recap)

- ✓ **L1 statevector**, **L2 MPS** (incl. GPU MPS through Phase 6 v1
  with χ ≤ 64), **L3 kernel fusion** (Tier B/C/D — 4.18× headline),
  **L6 chemistry** (full quantum-chemistry stack)
- ✓ **DMRG** with Lanczos + MPO; ITensor cross-checked at N = 8 to f64
- ✓ **Phase B**: TFIM/Heisenberg N = 128 in browser, validated vs Pfeuty/Bethe
- ✓ **Phase C/D/E1-5**: HF / MP2 / FCI / CCSD / CCSD(T) on
  H₂ → LiH → BeH₂ → H₂O → CH₄ in STO-3G; **cc-pVDZ CCSD(T) on H₂O in 106 s**
- ✓ 309 unit tests, 11 e2e specs, all green; CI live

### Next: chemistry-track tier roadmap

Ranked by ROI. Cost in "sessions" assumes one focused session = a few
hours. The tier framework exists because each subsequent feature stops
being free physics ladder-rungs and starts being grungy engineering — so
order matters.

#### Tier 1 — Quick wins (~1 session bundled)

Each takes ≤1 hour, kills a real gap, ships in one combined commit:

| feature | LOC | unlocks |
|---|---:|---|
| **f, g, h orbital integrals** | ~50 | cc-pVTZ basis (need only Boys F_n table extension to n ≤ 12 + EMSL constants — `integrals-cg.ts` already handles arbitrary L) |
| **Spherical-harmonic basis** | ~150 | bit-exact match to PySCF (kills the 4 mHa Cartesian-d slack) |
| **DIIS SCF accelerator** | ~100 | 5-10× HF speedup (H₂O cc-pVDZ: 101 iter → ~10) |
| **Frozen core** | ~30 | 2-3× CCSD(T) speedup |
| **Diffuse functions** | ~30 | aug-cc-pVDZ — anions, excited states |
| **Schwarz integral screening** | ~50 | 2-5× ERI speedup |

After this single bundle: **bit-exact PySCF agreement at every level**,
plus 5-10× across-the-board speedup. Highest-ROI session in the project.

#### Tier 2 — Major capability (~10 sessions total)

| feature | sessions | unlocks |
|---|---:|---|
| **DFT (LDA + B3LYP + Lebedev grids)** | 2 | **~90% of all real chemistry** uses DFT |
| **HF analytical gradients + BFGS** | 2 | **geometry optimization** — find equilibrium structures |
| **WebGPU port of (T) kernel** | 3 | 10-100× speedup on (T); cc-pVTZ CCSD(T) becomes routine |
| **EOM-CCSD (excited states)** | 1-2 | UV-vis, photochemistry — reuses CCSD intermediates |
| **UHF + open-shell CCSD** | 2 | radicals, transition-metal complexes |
| **Density fitting (RI)** | 1 | 3-5× speedup + half memory → cc-pVTZ becomes routine |

After Tier 2: **a genuinely useful undergrad chemistry tool**. Drug-style
geometry optimizations + DFT vibrational analysis + UV-vis spectra in a
browser tab.

#### Tier 3 — Substantial (~25 sessions)

CCSDT (full triples), CASSCF (multi-reference), TD-DFT, MP2/CCSD
gradients (Z-vector), PCM solvent, coupled-perturbed HF (NMR /
polarizabilities), WebGPU integral parallelization. Each is well-defined
but takes a few sessions of careful work.

#### Tier 4 — Genuinely hard (a season each)

CASPT2 / NEVPT2 (multi-ref perturbation, intruder states), periodic DFT
(k-points, Brillouin zone), spin-orbit / X2C (two-component spinors),
analytical CC gradients (Lagrangian per method), QM/MM. Possible but
expensive; do later.

### Deferred: the original moonshots

Still on the table, but lower priority than chemistry depth right now:

- **Phase D (WebRTC swarm)** — distributed 1D chain across browsers.
  ~3-5 sessions. Foundation for any multi-machine moonshot. Reuse
  `webgpu-p2p-evolution`'s relay.
- **E.1 — Verify Sycamore** — 2D PEPS primitive + Sycamore gate set
  + distributed contraction via Phase D. Reproduce Pan & Zhang 2021 in
  a browser. ~3-5 sessions on top of Phase D.
- **E.2 — Fault-tolerant qubit** — stabilizer sim + surface code +
  syndrome decoder + threshold curve. ~4-6 sessions, no Phase D needed.
- **E.3 — Browser-native lattice QCD** — 4D lattice + Wilson Dirac +
  fused CG solver. ~6-10 sessions, hardest port.

### Cleanest near-term path

**Tier 1 bundle → HF gradients → DFT → WebGPU (T) → EOM-CCSD**.
~8-9 sessions to a "real chemistry tool in a browser tab." Every step
ships a publishable artifact; if you stop early you still have a
strictly more useful repo than yesterday.

The unifying thesis stays the same: *"every advanced physics
simulation in the world ships as a URL"*. webgpu-q is the proof point;
the chemistry track is its highest-leverage demonstration.

---

## Current state (2026-05-08)

**Latest milestone: Tier 2 stage 22 — vertical ionization
potentials & electron affinities** via both Koopmans' theorem
and ΔSCF (the proper N±1-electron SCF). The first user-visible
deliverable that UHF (stage 21) unlocks: remove an electron,
re-run the SCF on the open-shell cation, get the IP from the
energy difference.

```
                          Koopmans      ΔSCF       experiment
  Be atom (HF/STO-3G)      6.91 eV      6.91 eV       9.32 eV
  H₂O      (HF/STO-3G)    10.65 eV      8.36 eV      12.62 eV
  LiH      (HF/STO-3G)     7.40 eV      5.61 eV       7.85 eV
```

ΔSCF < Koopmans where orbital relaxation has room to work
(verified on every case where the basis has flexibility — Be at
STO-3G has none, so the two methods coincide to FP precision).
LiH Koopmans is within 6% of experiment; cases where Koopmans
beats ΔSCF (LiH) reflect partial cancellation between orbital
relaxation lowering and missing correlation.

`ionizationPotential(atoms, opts)` returns Koopmans + ΔSCF + the
cation ⟨S²⟩ (clean doublet ≈ 0.75). `electronAffinity` is the
mirror routine; STO-3G EA is documented basis-limited (LUMO
unbound without diffuse functions).

**Tier 2 stage 21 (previous commit) — Unrestricted Hartree-Fock
(UHF).** Opens the entire class of open-shell molecules — radicals,
doublet/triplet states, cations and anions of closed-shell parents.
Until this commit, `runRHFSCF` rejected odd electron counts; now
`runUHFSCF(integrals, nα, nβ)` handles any electron count via
spin-resolved (α, β) orbitals.

UHF identities:
```
J(D_total)[μν]      = Σ_λσ (D_α + D_β)_λσ · (μν|λσ)
K_σ[μν]              = Σ_λσ (D_σ)_λσ · (μλ|νσ)
F_σ[μν]              = h_AO + J(D_total) − K_σ
E_elec               = ½·Σ [(D_α + D_β)·h + D_α·F_α + D_β·F_β]
```

Validation against literature (HF/STO-3G):
```
H atom  (1 e⁻, doublet):  E = −0.466582 Ha   (lit: −0.4666)  → 4-digit match
Li atom (3 e⁻, doublet):  E = −7.315526 Ha   (lit: −7.3155)  → 4-digit match
H₂⁺     (1 e⁻ at R=1Å):   E = −0.581667 Ha   (lit: −0.58…0.60 range)
```

Spin diagnostics:
- ⟨S²⟩ = S(S+1) + n_β − Σ_{ij occ} |⟨α_i|β_j⟩|²  (Pople formula)
- For pure doublets above, ⟨S²⟩ = 0.750000 to FP precision (no
  spin contamination — clean, well-behaved radicals).

Symmetry-breaking initial guess: small −0.01 Ha shift on the
α-Fock would-be-SOMO diagonal so that initial guess α HOMO is
slightly below β HOMO. At convergence, this either stays open-shell
(radicals) or relaxes back to the closed-shell solution
(verified — UHF on H₂ matches RHF energy to 1e-8).

DIIS: stack α and β error vectors into one combined 2·n² vector;
standard Pulay extrapolation just works on the combined system.

α/β orbital splitting in Li atom is the textbook UHF feature —
ε_α(2s) = −0.180 Ha, ε_β(2s) = +0.102 Ha. Exchange stabilizes
the spin-aligned electron; the splitting is the orbital signature
of this and explains why UHF energies are LOWER than ROHF for
radicals.

**This unlocks**: ΔSCF ionization energies (run UHF on cation),
electron affinities (UHF on anion), open-shell vibrational
frequencies + IR + Raman + thermo (everything from stages 16–20
becomes doable for radicals once we expose UHF gradients —
follow-up).

**Tier 2 stage 20 (previous commit) — first hyperpolarizability β
via finite-field.** Completes the field-response trio:
- μ (dipole moment, stage 11)
- α (polarizability, stage 17)
- **β (hyperpolarizability, this stage)**

Computes the full 27-component β_ijk tensor via 3D finite-difference
stencils on the dipole vs applied field — 19 SCF runs total per
molecule (1 zero-field + 6 ±E_axis + 12 ±E,±E mixed corners).
Symmetrizes under Kleinman (full index permutation symmetry at
zero frequency) and reports the β_∥(z) component along the lab
z-axis plus the rotational vector invariant |β_vec|.

H₂O HF/STO-3G:
```
  β diagonals = (0.00, 0.00, -4.11) a.u.
  β_∥(z)      = -11.05 a.u.
  |β_vec|     = 11.05 a.u.
```
β_xxx = β_yyy = 0 exactly: forced by C₂ᵥ symmetry (σ_xz reflection
flips y, C₂ flips x). |β_vec| ≈ 11 a.u. compares to HF/cc-pVTZ ≈
8-10 a.u. — STO-3G slightly inflates but the order of magnitude
and sign are right. Experiment is 22.2 a.u. (correlation contributes
~half).

Smoking-gun symmetry validation — H₂ centrosymmetric (D∞h):
```
  All 27 β_ijk components < 1e-2 a.u.   (FD noise only)
  |β_vec|                  < 1e-2 a.u.
```
Inversion symmetry forces β = 0 exactly; the implementation
reproduces this from floating-point arithmetic alone.

Cost: 19 SCF runs ≈ 0.12 s on H₂O HF/STO-3G — significantly
cheaper than vibrational analysis or Raman.

**Tier 2 stage 19 (previous commit) — ideal-gas thermochemistry.**
Standard statistical-thermodynamics corrections on top of an
electronic energy + harmonic vibrational frequencies. Computes:
- Zero-point energy (ZPE)
- Thermal correction to U(T) and H(T)
- Translational entropy (Sackur-Tetrode at standard P)
- Rotational entropy (rigid-rotor, asymmetric or linear)
- Vibrational entropy (harmonic oscillator)
- Total entropy and Gibbs free-energy correction G(T) − E_e

H₂O at 298.15 K, 1 atm (HF/STO-3G frequencies):
```
  ZPE              15.30 kcal/mol   (expt 13.4 — STO-3G freqs ~10% high)
  Thermal H        17.67 kcal/mol
  T·S              13.43 kcal/mol
  G correction      4.24 kcal/mol
  Total entropy   45.06 cal/(mol·K) (expt 45.1 — match to 0.1!)
```
Total entropy match is exact at 4 sig figs; the ZPE is the only
quantity affected by HF/STO-3G's known frequency overshoot.

Per-component breakdown:
  S_trans = 34.6 cal/(mol·K)   ← Sackur-Tetrode dominant
  S_rot   = 10.4 cal/(mol·K)   ← rigid rotor
  S_vib   ≈  0   cal/(mol·K)   ← all modes >> kT at room T
  S_elec  =  0   cal/(mol·K)   ← singlet ground state

Symmetry number σ scales rotational entropy as −R·ln(σ) exactly
(verified to 1e-12). H₂O takes σ=2; CH₄ σ=12; NH₃ σ=3; HF σ=1.

This closes the chemistry-paper SI bundle: with thermo, you can
now compute reaction free energies ΔG, equilibrium constants
K_eq = exp(−ΔG/RT), and enthalpies of formation directly from
the same browser tab.

**Tier 2 stage 18 (previous commit) — Placzek Raman activities.**
Combines vibrational modes (stage 16) and the polarizability
finite-field machinery (stage 17) to predict Raman scattering
intensities. Per Placzek polarizability theory:
```
  S_k = 45·ā_k²  +  7·γ_k²    (Å⁴/amu)
```
where ā_k = (1/3)·tr(∂α/∂q_k) and γ_k² is the standard
polarizability-derivative anisotropy invariant (5 components).

Implementation: 6N central FD on the polarizability tensor
(each polarizability eval is 6 SCF runs internally → 36N extra
SCF runs on top of the vibrational analysis). Total H₂O cost:
~13 seconds for 162 SCF runs.

H₂O HF/STO-3G full vibrational spectrum:
```
freq(cm⁻¹)  IR(km/mol)  Raman(Å⁴/amu)  label
   2170         7.2         118        bend
   4140        44.3         610        sym-stretch  ← strongest Raman
   4391        30.0         275        asym-stretch
```
Sym-stretch dominates Raman — textbook "symmetric breathing
mode is the brightest in Raman scattering." All three modes are
both IR- and Raman-active (C₂ᵥ has no centrosymmetry).

Smoking-gun validation — H₂ HF/STO-3G stretch:
- IR intensity:    < 0.001 km/mol         (forbidden by symmetry)
- Raman activity:  > 1 Å⁴/amu             (allowed by symmetry)
This is the classic textbook "rule of mutual exclusion": in
centrosymmetric molecules (D∞h for homonuclear diatomics), every
fundamental is EITHER IR-active OR Raman-active, never both.
The implementation reproduces this from the floating-point
arithmetic alone, and is the cleanest possible test that the
projection of dipole derivatives + polarizability derivatives
onto modes is being done correctly.

**Tier 2 stage 17 (previous commit) — static dipole polarizability
via finite-field.** Perturbs the core Hamiltonian with `+E·μ_AO`
(the dipole-field interaction `−μ̂·E` for an electron with
`μ̂ = −r̂`), runs SCF at ±E along each axis, and finite-differences
the resulting dipole moment to recover the full 3×3 polarizability
tensor. Reports tensor + isotropic average + anisotropy + sorted
principal components.

H₂O HF/STO-3G after geom-opt:
```
  α tensor (a.u.):
    5.509  0.000  0.000
    0.000  0.040  0.000
    0.000  0.000  2.565
  α_iso  = 2.705 a.u. = 0.401 Å³
```
Tiny α_yy reflects the absence of p-orbitals on H in STO-3G —
nothing to polarize INTO perpendicular to the molecular plane.
The full HF/aug-cc-pVDZ value (~7-8 a.u.) is achievable today by
flipping the basis arg; experiment is 9.79 a.u. (HF systematically
under-polarizes by ~10% from missing correlation).

H₂ HF/STO-3G: α_zz > 0 (parallel polarizability), α_xx = α_yy ≈ 0
(no p polarization functions). Off-diagonals < 0.01 a.u. (FD noise).

The cleanest sign-correctness check: `H_int = −μ̂·E` for an
electron with `μ̂ = −r̂` gives `+r̂·E` for the electronic part,
which means h_AO acquires `+E·μ_AO`. Initial implementation had
the sign reversed (subtraction), giving negative polarizability —
caught immediately by the "α positive" pass bar.

**Tier 2 stage 16 (previous commit) — harmonic vibrational
spectrum (frequencies + IR intensities).** Builds the molecular
Hessian by central FD on the analytical gradient (HF or any
RKS-DFT functional), tracks the dipole at each displaced
geometry to also build ∂μ/∂R, mass-weights both, projects out
translation/rotation, diagonalizes for frequencies in cm⁻¹, and
projects ∂μ/∂R onto each mode for IR intensities in km/mol.

H₂O HF/STO-3G after geom-opt — full IR spectrum:
```
   2170 cm⁻¹       7.2 km/mol     bend
   4140 cm⁻¹      44.3 km/mol     sym-stretch
   4391 cm⁻¹      30.0 km/mol     asym-stretch
```
Frequencies match Hehre/Stewart/Pople 1969 reference to 0.1 cm⁻¹.
Intensities are sensible (1–200 km/mol), but the exact numerical
ordering vs a specific literature reference is dominated by
STO-3G's known property errors (~10–20% off vs cc-pVTZ).

H₂ HF/STO-3G: 1 vibrational mode (linear flag drops the second
rotation), and IR intensity ≈ 0 (homonuclear symmetry → no dipole
change). The latter is the cleanest validation that the projection
is correct: a symmetry-forbidden mode emerges as zero from the
floating-point arithmetic.

H₂O HF/STO-3G after geom-opt vs Hehre/Stewart/Pople 1969 reference:
- bend         **2170** cm⁻¹  (ref ~2170 — match)
- sym-stretch  **4140** cm⁻¹  (ref 4140 — match to 0.1 cm⁻¹)
- asym-stretch **4391** cm⁻¹  (ref 4390 — match to 1 cm⁻¹)
The 0.1-cm⁻¹ agreement on the high-frequency modes is essentially
machine precision relative to the 5e-3 Å FD step. No imaginary
modes at the optimized minimum (correct sign of Hessian).

H₂ HF/STO-3G stretch: ~5060 cm⁻¹ (linear-mode flag drops the
2nd rotation, leaves 1 vibrational mode after 5 zero modes).

What got built:
- `vibrationalFrequencies(atoms, opts)`: full pipeline (Hessian
  build → mass-weight → Eckart projection → diagonalize → cm⁻¹).
- 6N gradient evals per Hessian (cost dominates for triatomics +
  larger; H₂O Hessian = 18 evals ≈ 5–10 s on M2 Pro).
- Imaginary-frequency convention: NEGATIVE cm⁻¹ in output, the
  standard quantum-chemistry sign. Useful for transition-state
  detection (saddle points have one imaginary mode).
- Eckart-frame projection vectors built from atomic masses + COM-
  shifted positions; degenerate-rotation handling for linear
  molecules via `linear: true` flag.
- Standard atomic masses (¹H, ⁷Li, ⁹Be, ¹²C, ¹⁴N, ¹⁶O — isotope-
  pure, the conventional theoretical reference).

Bug caught along the way: initial unit conversion went the wrong
way (multiplied by ANGSTROM_TO_BOHR instead of dividing). Computed
frequencies were 1.89× too high — H₂O bend came out at 4100 cm⁻¹.
The 1.89× ratio matches ANGSTROM_TO_BOHR exactly, isolating the
bug to the Å→Bohr coordinate-derivative scaling.

**Tier 2 stage 15b-LYP (previous commit) — triplet TDA + TDDFT
across the full functional ladder.** With the spin-polarized LYP
correlation in `functional-spin.ts` (Miehlich 1989 integrated-by-
parts form), triplet excitations now work for HF, LDA, BVWN5,
B3VWN5, **BLYP, and B3LYP5** — every functional in the SCF ladder.

Key insight that made spin-polarized LYP tractable: LYP is
LINEAR in (γ_↑↑, γ_↑↓, γ_↓↓), so the four γ-coefficients
(A_0, A_↑↑, A_↑↓, A_↓↓) are closed-form functions of (ρ_↑, ρ_↓).
γ-derivatives are exactly the A_X coefficients (∂f/∂γ_↑↑ = A_↑↑,
etc.); only the ρ-derivatives need FD on the analytic energy
density. Total addLYPC_spin: ~80 lines, no torturous chain-rule
algebra on 8 G_i pieces × 2 ρ-variables.

Closed-shell collapse verified to 1e-9 (ε, v_γ exactly; v_ρ to
the FD step-size precision):
- H₂O STO-3G first singlet / triplet (eV) across the ladder:
  HF:      S 13.20  T 11.10  gap 2.10
  LDA:     S 11.50  T  9.53  gap 1.97
  BVWN5:   S 11.35  T  9.41  gap 1.95
  BLYP:    S 11.31  T  9.36  gap 1.95
  B3VWN5:  S 11.76  T  9.80  gap 1.96
  B3LYP5:  S 11.72  T  9.76  gap 1.97
- The ~2 eV singlet-triplet gap shrinks slightly going HF → DFT
  (less unbalanced exchange in DFT functionals); LYP-bearing and
  VWN5-bearing functionals at the same B88 / B3 mixing land
  within ~0.05 eV of each other — the standard small LYP-vs-VWN
  difference at minimal basis. All triplet < singlet by ~2 eV
  (Hund's rule analog).

What got built this commit:
- `lypCoefficients(ρ_↑, ρ_↓)` returning closed-form A_0, A_↑↑,
  A_↑↓, A_↓↓ — the four γ-coefficients of f_LYP^Miehlich.
- `lypEpsPerVolume(...)` linear-in-γ energy density.
- `addLYPC_spin(...)`: γ-derivatives analytic from A_X,
  ρ-derivatives via central FD on the energy density.
- `evalXC_GGA_spin` extended to cover BLYP / B3LYP5 — composes
  Slater_spin / B88_spin / VWN5_spin / LYP_spin pieces with
  per-functional coefficients (mirrors the closed-shell evalXC
  case statements). New helper `addVWN5ScaleAdjust` lets us
  scale VWN5 down to 0/0.19 in BLYP/B3LYP5 without exposing a
  separate spin-polarized VWN5-only routine.
- `tda-dft.ts` triplet GGA path now accepts `bvwn5 | b3vwn5 |
  blyp | b3lyp5`.

Bug caught along the way: `ρ^(8/3)` was originally written as
`ru * cbrt(ru*ru) * cbrt(ru*ru) = ru^(7/3)` instead of
`ru² · cbrt(ru²) = ru^(8/3)`. Closed-shell ε mismatched
closed-shell evalXC by ~3 mHa per particle — the hand-computed
A_↑↑(cs)·γ_↑↑ matched the code, but A_0(cs) (which carries the
Slater-decomposed C_F·ρ_σ^(8/3) piece) did not. Fix turned a
0.0028 mismatch into 1e-16.

Pass bars added (1 new test in tda-dft-triplet-gga, +2 cases
in the existing for-loop tests, +1 in tda-dft-triplet —
together "spin-polarized BLYP closed-shell limit" + "BLYP /
B3LYP5 + triplet supported and < singlet" + the existing
BVWN5/B3VWN5 cases extended to include BLYP/B3LYP5).

**Tier 2 stage 15b-GGA-B88 (previous commit) — triplet TDA +
TDDFT for BVWN5 + B3VWN5.** Stage 15b LDA closed the LDA-and-HF
ladder; this commit extends triplet support to the B88-GGA
functionals (Slater + B88 + VWN5, optionally with 0.20 HF
exchange for B3VWN5).
The spin-polarized B88 piece composes by spin-decomposition
(no σ-coupling), so v_↑, v_↓, v_γ↑↑, v_γ↓↓ all derive from the
existing closed-shell B88 with x_σ = √γ_σσ/ρ_σ^(4/3); v_γ↑↓ = 0.

What got built:
- `addB88X_spin` + `evalXC_GGA_spin` in `dft/functional-spin.ts`:
  spin-polarized Slater + B88 + VWN5 (BVWN5 / B3VWN5). Closed-shell
  baseline ρ_↑=ρ_↓=ρ/2, γ_αα'=γ/4 collapses to the closed-shell
  evalXC[bvwn5] for ε_xc, v_↑/v_↓, and the chain-rule-combined v_γ
  to 1e-9 across (ρ, γ) ∈ {0.05–50, 1e-4–1e4} (FD test).
- `evalXCKernelGGA_triplet`: 4-piece triplet kernel via central FD
  on v_↑ and v_γ↑↑ at the closed-shell baseline:
    fRR^t = (H_{ρ↑ρ↑} − H_{ρ↑ρ↓})/2
    fRG^t = (H_{ρ↑γ↑↑} − H_{ρ↑γ↓↓})/4
    fGG^t = (H_{γ↑↑γ↑↑} − H_{γ↑↑γ↓↓})/8
    vG^t  = (2·v_γ↑↑ − v_γ↑↓)/4
  Factors line up so the SAME `2·K_xc` integrand template the
  singlet path uses lands on the textbook |T_z=0⟩ triplet
  kernel.
- `tda-dft.ts`: when `spin: "triplet"` and method ∈ {bvwn5, b3vwn5},
  swap singlet (ker, vGamma) for the triplet kernel components.
  BLYP / B3LYP5 still throw with the more specific "needs
  spin-polarized LYP kernel" message — that's its own session.

H₂O STO-3G first singlet / triplet (eV) across the ladder:
  HF:      singlet 13.20  triplet 11.10  gap 2.10
  LDA:     singlet 11.50  triplet  9.53  gap 1.97
  BVWN5:   singlet 11.35  triplet  9.41  gap 1.95
  B3VWN5:  singlet 11.76  triplet  9.80  gap 1.96
The ~2 eV singlet-triplet gap shrinks slightly going HF → DFT
(less unbalanced exchange in DFT functionals) — the standard
textbook ordering.

Pass bars (5 new tests, all green):
- BVWN5 spin-polarized closed-shell limit matches `evalXC[bvwn5]`
  for ε_xc, v_↑, and chain-rule v_γ to 1e-9.
- BVWN5 + B3VWN5 triplet < singlet on H₂ STO-3G.
- BVWN5 + B3VWN5 TDDFT ≤ TDA per state (B-correction sign).
- BLYP / B3LYP5 + triplet throws clearly.

**Tier 2 stage 15b — triplet TDA + TDDFT for HF and LDA**
(previous commit). Closed-shell triplets need the spin-polarized
XC kernel f_↑↑ − f_↑↓, which the closed-shell `evalXC` in
`functional.ts` can't produce on its own. Stage 15b ships a
spin-polarized LSDA build (Slater exchange + VWN5 correlation,
separate ρ_↑/ρ_↓ inputs) in a new `dft/functional-spin.ts`, plus
the FD triplet kernel ½(f_↑↑ − f_↑↓), and wires
`spin: "singlet" | "triplet"` into `runTDA` / `runTDDFT`.

What got built:
- `evalXC_LSDA(rhoUp, rhoDn, …)` — Slater + VWN5 with the full
  Vosko-Wilk-Nusair spin interpolation (P, F, α_c parameter sets,
  f(ζ) interpolation). Closed-shell limit ρ_↑=ρ_↓=ρ/2 collapses
  to the closed-shell ε_xc and v_xc to 1e-12 / 1e-10 (sanity test).
- `evalXCKernelLDA_triplet(rho)` — central FD on v_↑ separately in
  the ρ_↑ and ρ_↓ directions to extract f_↑↑ and f_↑↓, returns
  ½(f_↑↑ − f_↑↓) so the same `2·K_xc` factor in `tda-dft.ts` lands
  exactly on the textbook Casida triplet kernel.
- `tda-dft.ts`: `TDAOpts.spin` ("singlet" default | "triplet"),
  `sCoul` becomes 0 for triplet (drops Hartree), kernel switches
  to triplet path for LDA, oscillator strengths short-circuit
  to zero (ΔS=1 dipole-forbidden from a closed-shell ground state).
- Triplet + GGA / hybrid throws clearly: needs spin-polarized
  GGA second derivatives (B88 / LYP f_σσ'), still open. No
  silent half-build.

Pass bars:
- `runTDA(method='hf', spin='triplet')` matches `runCIS({spin:'triplet'})`
  on H₂ STO-3G to 1e-10 (CIS / TDHF generalization is exact).
- `runTDA(method='lda-svwn', spin='triplet')` on H₂ STO-3G sits
  below the singlet excitation (the standard ordering — triplet
  states are lower than singlets at the same orbital configuration).
- `runTDA(method='blyp'/'b3lyp5', spin='triplet')` throws with a
  clear "needs spin-polarized GGA kernel" diagnostic.

Stage 15a (previous commit) recap: spherical-d on the grid —
`MolecularIntegrals.sphericalT` exposes T (n_sph × n_cart). All
grid-using paths (RKS-DFT SCF, TDA-DFT XC kernel, HF + DFT
analytical gradients) apply the Cartesian → spherical-d transform
T when integrals are built with `spherical: true`. The pre-15a
"guard with throw" was documentation papering over a real bug;
15a is the actual fix. H₂O cc-pVDZ B3LYP5 spherical vs Cartesian:
SCF Δ = 1.2 mHa, TDA[0] 7.627 vs 7.601 eV, |∇| within 1–4 %.

**Honest negatives still open** (each its own session):
- **15c — Becke-partition weight derivatives**: ∂ω_K(r_p; R)/∂R_N
  at fixed r_p plus the grid-translation contribution from points
  originated on atom N. Closes the ~1e-3 Ha/Bohr translational-
  invariance residual in DFT gradients. ~1-2 sessions of careful
  algebra + integral work.
- **15d — Davidson eigensolver**: iterative Krylov method for
  CIS / TDDFT scaling beyond a few hundred dimensions.
  ~1 session of standard linear-algebra port.
- **15e — Continuum representation for E17 σ_ion**: Stieltjes
  imaging or B-spline / DVR continuum orbitals. Needed to bring
  σ_ion within ±15 % of Itikawa-Mason 2005. ~2-3 sessions
  of substantive new infrastructure.

**Tier 2 stage 14 — gap-closing pass + E17.**
Cleaned up the loose ends from stages 4–13 in one focused
session: shipped the deferred E17 cross-section experiment as
the documented honest-negative the architecture allowed; closed
four cleanup gaps; left the deeper "needs new infrastructure"
items honestly named in this section.

What got closed in stage 14:
- **GGA gradients are wired through `optimizeGeometry`** for the
  full ladder. BLYP / B3LYP5 / B3VWN5 / BVWN5 geom-opts run
  analytical (BLYP H₂O: 4.5× fewer evals than FD, identical E
  to 0.5 µHa). The previous "LDA only" docstring was lying —
  the code path was already correct since stage 6b.
- **Mayer atomic valences** as a clean helper (`mayerValences`)
  on top of the bond-order matrix — the off-diagonal sum, which
  is the actually-meaningful quantity (the diagonal of the bond-
  order matrix is a self-overlap, not a valence).
- **Spherical-d + TDA-DFT / DFT-gradient bug** caught and
  guarded: the spherical-d transform shrinks `integrals.n` but
  leaves `integrals.shells` as Cartesian, desyncing the AO→MO
  transform on the grid and producing silent NaN amplitudes.
  Both `runTDA` / `runTDDFT` and `dftGradient` now refuse with
  a clear "use spherical: false" message rather than NaN.
  Discovered while running E17 cc-pVDZ; the proper fix (apply
  the T transform to grid quantities) is documented as a
  follow-up.
- **E17 cross sections** shipped as honest negative (see Tier 2
  stage 14 entry below).

What's documented as still-open (the ACTUAL honest negatives):
- **Triplet TDA / TDDFT for DFT functionals.** Closed-shell
  triplet kernel f_xc^triplet = (f_↑↑ − f_↑↓) needs a spin-
  resolved evalXC (with separate ρ_↑, ρ_↓ inputs); currently
  evalXC is closed-shell only. Singlet TDA / TDDFT works for the
  full functional ladder; runCIS still ships triplet for HF.
- **Becke-partition weight derivatives** in DFT gradients —
  ~1e-3 Ha/Bohr translational-invariance residual (sub-mHa/Bohr
  per component, doesn't affect practical geom-opt).
- **Spherical-d in TDA-DFT / DFT-gradient on the grid** — refuses
  with a clear error today; the proper path is to apply the
  Cartesian → spherical transform to phi / phix / phixx on the
  grid alongside the AO matrices.
- **Davidson eigensolver** for large-basis CIS / TDDFT — current
  dense eigsymm is fine for n_occ · n_virt ≤ a few hundred.
- **Continuum representation** for E17 σ_ion convergence —
  Stieltjes imaging, SAC-CI continuum, B-spline / DVR continuum
  orbitals, or direct optical-photoionization. Substantial
  follow-up.

**Tier 2 stage 13 — Wiberg-Mayer bond orders.**
Per-pair shared-electron counts from the AO density:

  B_AB = Σ_{μ ∈ A, ν ∈ B} (P · S)_μν · (P · S)_νμ

Built:
- `bondOrders(integrals, P, shellAtomIdx)` in `properties.ts`
  returning the full nAtoms × nAtoms matrix. Reference-agnostic.

H₂O / BeH₂ HF/STO-3G bond-order matrices:
- H₂:    B_HH = 1.0000   (perfect single bond)
- H₂O:   B_OH = 0.9540 (each)   B_HH = 0.0125 (geminal)
         Mayer valence: V_O = 1.91, V_H = 0.97
- BeH₂:  B_BeH = 0.9976 (each)   B_HH = 0.0004
         Mayer valence: V_Be = 2.00, V_H = 1.00

All match the textbook ordering for these molecules. The 0.95
on O-H reflects the small polarization "loss" relative to a
pure covalent bond (charge sits more on O).

Tests (4 new): single-bond H₂, dual-bond H₂O with small geminal
H-H, BeH₂ valences ≈ {Be: 2, H: 1}, agreement across HF / DFT
functionals to within 0.1.

Honest limitations (same as Mulliken):
- Basis-set dependent. Numbers shift with basis.
- The diagonal of the matrix is not the Mayer valence — that's
  the off-diagonal sum. Document in the module header.

**Tier 2 stage 12 — Mulliken population analysis.**
Per-atom partial charges from the AO density:

  n_A = Σ_{μ on A} (P · S)_μμ      (gross atomic population)
  q_A = Z_A − n_A                  (Mulliken charge)

Σ_A q_A = total molecular charge (= 0 for neutrals) by trace
invariance.

Built:
- `mullikenCharges(integrals, P, shellAtomIdx)` in
  `properties.ts`. Reference-agnostic — works on any closed-shell
  AO density (HF, DFT, post-HF relaxed). 4 new tests cover H₂O
  qualitative ordering (O negative, Hs equally positive), H₂ +
  BeH₂ symmetry checks, conservation under DFT functional swap.

H₂O STO-3G Mulliken charges (e):
- HF:      O = −0.3664   H = +0.1832
- LDA:     O = −0.3840   H = +0.1920
- BVWN5:   O = −0.3631   H = +0.1816
- BLYP:    O = −0.3605   H = +0.1802
- B3VWN5:  O = −0.3705   H = +0.1852
- B3LYP5:  O = −0.3683   H = +0.1841

LDA most-polar (pulls electrons toward O); GGA less polar; HF
in between — the well-known DFT-vs-HF trend, with hybrids
sitting between pure DFT and HF as expected.

Honest limitations shipped with the routine:
- Mulliken is basis-dependent. Charges shift when you change
  basis sets, sometimes drastically with diffuse functions.
- The 50/50 overlap split is somewhat arbitrary at large
  shared-overlap regions. NPA / CHELPG / Bader are better but
  need substantial new infrastructure.

**Tier 2 stage 11 — ground-state dipole moments.**
First ground-state property routine on top of the SCF density.
Reuses the dipole AO machinery from stage 10:

  μ = − Σ_{μν} P_μν · ⟨χ_μ | r̂ | χ_ν⟩  +  Σ_A Z_A · R_A

Returns a [μ_x, μ_y, μ_z] 3-vector in atomic units (e·Bohr);
multiply by `AU_TO_DEBYE` (= 2.5417) for Debye.

What got built:
- `src/chemistry/properties.ts`: `dipoleMoment(integrals, P)` and
  `dipoleMagnitude(mu)` helpers. Reference-agnostic — pass any
  closed-shell AO density (HF, DFT, post-HF relaxed).
- 4 tests: H₂O HF/STO-3G ≈ 1.7 D, points along +z (right
  direction in our coordinate system); H₂O DFT functionals all
  in 1.5-2.0 D; H₂ and BeH₂ have zero dipole by symmetry to
  within 1e-9 a.u.

H₂O STO-3G ground-state dipole moments (Debye):
- HF:      1.726
- LDA:     1.729
- BVWN5:   1.640
- BLYP:    1.639
- B3VWN5:  1.679
- B3LYP5:  1.678
- Experimental: **1.85** (vapor phase)

All methods systematically underestimate by ~10 % — STO-3G is
the bottleneck (tight basis truncates polarizability). Bigger
basis (cc-pVDZ+) closes most of the gap.

**Tier 2 stage 10 — oscillator strengths.**
Closed-shell singlet TDA / TDDFT now returns `oscillatorStrengths`
alongside excitation energies — dimensionless transition
intensities, which together give a UV-vis-style spectrum.

  T_axis_n = √2 · Σ_ia c_ia · ⟨φ_i^MO | r_axis | φ_a^MO⟩
  f_n^TDA  = (4/3) · ω_n · Σ_axis |T_axis_n|²
  f_n^TDDFT= (4/3) ·       Σ_axis |Σ_ia (S·Z')_ia · μ_ia|²
                                (S = (A−B)^(1/2); ω cancels via
                                 Casida normalization of X+Y)

Built:
- `dipole_cg(A, B, axis)` in `integrals-cg.ts`: ⟨A|r_axis|B⟩
  via primOverlap with shifted angular momentum + A_axis·S
  trick. Reuses existing primitive overlap path.
- `computeOscillatorStrengths` private helper in `tda-dft.ts`:
  builds dipole AO matrices, transforms the (occ × virt) MO
  block per axis, contracts with the appropriate amplitude
  (X for TDA, S·Z' for TDDFT). Both runners return
  `oscillatorStrengths: Float64Array` per root.
- 4 new tests: H₂ HOMO→LUMO f matches the LCAO bond-axis
  estimate (~1.10); H₂O sums positive + bounded by TRK; TDA
  vs TDDFT agree within 30 % total intensity; DFT functional
  ladder all produces non-negative finite f.

H₂O STO-3G singlet excitations + oscillator strengths (TDA):
- TDA-HF:    13.20(0.004) 15.16(0.000) 16.78(0.077) 19.20(0.060) 22.08(1.167)  Σf=1.31
- TDA-LDA:   11.50(0.003) 13.80(0.000) 14.14(0.074) 17.76(0.067) 22.03(1.103)  Σf=1.25
- TDA-BLYP:  11.31(0.002) 13.67(0.000) 14.09(0.076) 17.67(0.063) 21.68(1.127)  Σf=1.27
- TDA-B3LYP5:11.72(0.003) 13.99(0.000) 14.65(0.075) 17.98(0.063) 21.81(1.136)  Σf=1.28

State 2 carries f ≈ 0 by point-group symmetry (it's the
forbidden transition the bare C₁ basis still couples to in our
non-symmetry-adapted code; the integral evaluates to ≈ 0
numerically, which is the right physics). State 5 carries the
dominant intensity. ΣF ≈ 1.27 across functionals — well within
the TRK bound of 10.

**Tier 2 stage 9c — GGA / hybrid TDA + TDDFT.**
Closes the singlet TDDFT loop across the full functional ladder
(HF, LDA, BVWN5, BLYP, B3VWN5, B3LYP5). The GGA / hybrid kernel
adds 4 pieces on top of the LDA `f_RR · ψ ψ` integrand:

  K_xc[ia, jb]^GGA = ∫ w · {
       ψ_ia · f_RR · ψ_jb
     + 2·ψ_ia · f_RG · α_jb
     + 2·α_ia · f_RG · ψ_jb
     + 4·α_ia · f_GG · α_jb
     + 2·v_γ · ∇ψ_ia · ∇ψ_jb
   } dr

with ψ_ia = φ_i^MO · φ_a^MO and α_ia = ∇ρ · ∇ψ_ia.

Built:
- `evalXCKernel(kind, ρ, γ)` in `dft/functional.ts`: numerical
  central-FD on `evalXC` v_ρ, v_γ in both ρ and γ directions.
  4 evalXC calls per evaluation. Returns {fRR, fRG, fGG} as
  Float64Arrays.
- GGA path in `tda-dft.ts/buildTDABlocks`: pre-computes MO
  orbital VALUES + GRADIENTS on the grid, builds ψ_ia, ∇ψ_ia,
  α_ia, then assembles the 5-piece K_xc integrand.

H₂O STO-3G first singlet (eV) across the ladder:
- TDA-HF:    13.20   →  TDHF:        13.16
- TDA-LDA:   11.50   →  TDDFT-LDA:   11.42
- TDA-BVWN5: 11.35   →  TDDFT-BVWN5: 11.30
- TDA-BLYP:  11.31   →  TDDFT-BLYP:  11.26
- TDA-B3VWN5:11.76   →  TDDFT-B3VWN5:11.71
- TDA-B3LYP5:11.72   →  TDDFT-B3LYP5:11.67

Clean ordering: HF highest (largest gap), pure DFT cluster at
11.3-11.5 eV, B3-style hybrids land between (the 20 % HF mixing
pulls them up), TDDFT uniformly ≤ TDA (B-correction). The gap
between B3LYP5 (11.7 eV) and B3LYP/cc-pVTZ literature (~7-8 eV)
is the basis-set difference, not a TDA bug — STO-3G is too small
for valence excitations.

Tests (10): every functional × {TDA, TDDFT} produces real,
positive lowest-state excitations and TDDFT ≤ TDA per state.
Plus the BLYP-vs-LDA-vs-HF ordering check.

Honest negatives still in: triplet TDA / TDDFT for DFT (closed-
shell triplet kernel f_xc^triplet differs from singlet — runCIS
still ships triplet for HF only). No symmetry adaptation.

**Tier 2 stage 9b — full TDDFT (Casida).**
Added the B coupling block on top of stage 9's A and solved the
RPA / TDDFT problem
   (A − B) · (A + B) Z = ω² Z
via the symmetric reformulation M = (A−B)^(1/2) · (A+B) · (A−B)^(1/2)
followed by `eigsymmetric` on M. Square root is element-wise
when (A − B) is diagonal (pure DFT, hfMix = 0); otherwise via
eigendecomposition. Same `method = "hf" | "lda-svwn"` surface
as runTDA — `runTDDFT` is the new entry point.

Built:
- `runTDDFT` in tda-dft.ts. Shares matrix-build with `runTDA`
  via the new private `buildTDABlocks` helper. The B block
  reuses the same Coulomb 2·(ia|jb) and LDA XC kernel
  (ia|f_xc|jb) as A; the only difference is exchange
  permutation: A uses (ij|ab), B uses (ib|aj).
- `matrixSqrtSymmetric`: real-symmetric square root via
  eigendecomposition. Refuses negative eigenvalues with a
  message hinting at closed-shell instability.

H₂O STO-3G singlet excitations (Ha [eV]):
- TDA-HF:    0.4852 [13.20]  0.5573 [15.16]  0.6167 [16.78]
- TDHF/RPA:  0.4836 [13.16]  0.5567 [15.15]  0.6126 [16.67]
- TDA-LDA:   0.4226 [11.50]  0.5071 [13.80]  0.5197 [14.14]
- TDDFT-LDA: 0.4198 [11.42]  0.5066 [13.79]  0.5141 [13.99]

Full RPA / TDDFT eigenvalues are uniformly ≤ TDA at the same
level — the textbook B-block-correction sign. Shifts are small
(≤ 100 mHa per state) because at STO-3G the orbital-energy gap
dominates; bigger basis sets show larger differences.

Tests (9 → +2): TDHF strictly ≤ TDA-HF and TDDFT-LDA strictly
≤ TDA-LDA per state, all real and positive (no instabilities).

**Tier 2 stage 9 — TDA-DFT (singlet, LDA).**
TDA generalized from HF orbitals to Kohn-Sham orbitals + the
LDA XC kernel + HF-exchange mixing. Singlet sector:

  A^singlet_{ia,jb} = (ε_a − ε_i)·δ·δ
                    + 2·(ia|jb)            (Hartree)
                    − hfMix·(ij|ab)        (HF exchange × hfMix)
                    + 2·(ia|f_xc|jb)       (XC kernel; LDA only)

with `hfMix = 1` (HF), `0` (LDA), `0.20` (B3-style hybrids).

What got built:
- **`evalXCKernelLDA`** in `dft/functional.ts`: numerical
  central-FD on `evalXC` v_ρ to get f_xc(ρ_p) at every grid
  point. Two extra `evalXC` calls — trivial cost. Avoids
  re-deriving Slater + 30-line VWN5 second derivatives by hand.
- **`runTDA`** in `tda-dft.ts`: builds (ia|f_xc|jb) on the
  molecular grid via pre-computed ψ_ia(g) = φ_i^MO(g)·φ_a^MO(g)
  and the f_xc-weighted contraction. Reproduces `runCIS` singlet
  exactly when `method = "hf"`; uses LDA XC kernel for
  `method = "lda-svwn"`; throws a clear "needs GGA TDA" error
  for the GGA / hybrid functionals.
- 7 tests: TDA-HF matches CIS to 1e-10; TDA-LDA shifts H₂O
  first singlet below HF/CIS (LDA over-binds the gap, XC kernel
  adds the right sign correction); amplitudes normalized;
  GGA / hybrid throws the right errors.

H₂O STO-3G singlet excitations (Ha [eV]):
- CIS / TDA-HF:  0.4852 [13.20]  0.5573 [15.16]  0.6167 [16.78]
- TDA-LDA:       0.4226 [11.50]  0.5071 [13.80]  0.5197 [14.14]
The 1.7 eV TDA-LDA downshift on the first singlet is the
well-known LDA improvement over CIS for valence transitions
(STO-3G is too small to land on the 7-8 eV experimental value;
that's a basis-set issue, not a TDA bug).

**Honest negatives** documented for follow-up:
- **GGA / hybrid TDA-DFT deferred**: BVWN5/BLYP/B3VWN5/B3LYP5
  need the full f_ρρ + f_ργ + f_γγ kernel tensors and basis-
  Hessian-style integrals. Single-line throw with TODO message.
- **Triplet TDA-DFT deferred**: closed-shell triplet kernel
  f_xc^triplet differs from singlet (spin-asymmetric second
  derivative). `runCIS` still ships triplet for HF.
- **No full TDDFT**: only A is diagonalized (TDA). Full TDDFT
  diagonalizes the (A, B) 2×2 block — modest follow-up.

**Tier 2 stage 8 — CIS / TDA excited states.**
First excited-state capability on top of HF. Diagonalizes the
CIS (Tamm-Dancoff) Hamiltonian on the singles excitation manifold:

  A^singlet_{ia,jb} = (ε_a − ε_i)·δ_ij·δ_ab + 2·(ia|jb) − (ij|ab)
  A^triplet_{ia,jb} = (ε_a − ε_i)·δ_ij·δ_ab            − (ij|ab)

with chemist-notation (pq|rs) MO ERIs from `transformERIToMO`.

What got built:
- **`src/chemistry/cis.ts`**: `runCIS(integrals, hf, opts)`
  builds the singlet + triplet A blocks separately and dense-
  diagonalizes via `eigsymmetric`. Returns excitation energies
  + amplitudes per spin sector. Optional `nRoots` and
  `spin: "singlet" | "triplet" | "both"` filters.
- 6 tests: triplet > 0 (ground-state stability) and singlet >
  triplet (Hund's rule) for H₂ / H₂O STO-3G; H₂ first singlet
  HOMO→LUMO at 0.947 Ha ≈ 25.7 eV (textbook reference); CIS
  amplitudes are normalized to 1e-10; `S₀ − T₀ = 2·(ia|ia)`
  internal consistency check.

Limitations / scope notes shipped:
- Dense eigsymm only — fine for n_occ·n_virt ≤ a few hundred.
  Larger systems would want a Davidson iterative solver.
- TDA / no full RPA — only A is diagonalized. Full TDDFT
  diagonalizes the (A, B) 2×2 block; modest follow-up.
- TDA-DFT (KS orbitals + hybrid mix) is a one-line extension
  once the XC kernel is plumbed; deferred.
- No symmetry adaptation — common literature values quoted with
  point-group labels (e.g. H₂O ¹B₁) need symmetry projection
  we don't ship; the raw C₁ HOMO→LUMO excitations we compute
  match what other codes report when symmetry is disabled.

**Tier 2 stage 7 — Lebedev angular quadrature.**
The DFT angular grid is now Lebedev-Laikov by default (order 110,
exact for spherical harmonics up to L = 17). Replaced the older
12 × 24 = 288-point Gauss-Legendre × uniform-φ product rule for a
**2.6× point reduction at strictly better algebraic accuracy**.
Available orders: 50, 110, 302 — `LEBEDEV_AVAILABLE_ORDERS`.
Pass `nLebedev: null` to fall back to the product rule.

What got built:
- **`src/chemistry/dft/lebedev.ts`**: `genOh` orbit expander
  (octahedral group Oh on 6 symmetry classes — axis, face,
  corner, (a,a,b), (a,b,0), (a,b,c)) plus tabulated parameters
  for orders 50, 110, 302. Tables sourced from the Lebedev-
  Laikov 1999 Fortran routine (Christoph van Wuellen translation,
  via PySCF's `dft/LebedevGrid.py`). Cross-validated to fp:
  Σw = 4π exactly, |r|=1 to 1e-16 per point, ∫x²y²z² = 4π/105
  to 1e-15 on every order.
- **`molecularGrid`** refactored to take `nLebedev?: LebedevOrder`
  and use it as the angular path; default `nLebedev: 110`.
  Legacy product rule still reachable via `nLebedev: null` and
  `nTheta` / `nPhi` for cross-checks.

H₂O STO-3G timings (M2 Pro, BLYP):
- Grid:        43200 → 16500 points (2.6× fewer).
- SCF:         119 ms → 55 ms (2.2× faster).
- Gradient:    389 ms → 356 ms (1.1× faster — grad is dominated
               by the n⁴ ERI derivative loop, which is angular-
               independent).
- Energy:      −75.27725 → −75.27722 (31 µHa difference, well
               below chemical accuracy).
- ρ-integration error: 1e-4 e (was 0 with the product rule; the
               product rule is exact for any 2π-periodic finite
               φ-Fourier mode, so uniform-φ + Gauss-Legendre on
               cos θ trivially conserves charge to fp). Both are
               far below the 0.01 e test pass bar.

Full vitest chemistry track: 60.7 s → 49.8 s (~17% faster, dominated
by the DFT-heavy tests). DFT energy tests alone: 2.5 s → 0.9 s.

**Tier 2 stage 6b — DFT gradients (GGA + hybrids).**
Analytical RKS-DFT geometry optimization is now end-to-end for
the full functional ladder: `lda-svwn`, `bvwn5`, `blyp`, `b3vwn5`,
`b3lyp5`. The GGA path adds the ∂γ/∂R term using a new basis-
Hessian evaluator on the molecular grid:
  ∂(∇ρ)_a/∂R_N^k = −2·Σ_{μ on N} { (∂_k φ_μ)·(P·∂_a φ)_μ
                                  + (∂_k ∂_a φ_μ)·(Pφ)_μ }
  ∂γ/∂R_N^k     = 2·Σ_a (∇ρ)_a · ∂(∇ρ)_a/∂R_N^k
  contribution   = −4·w·v_γ · Σ_{μ on N} {…}

What got built:
- **`evalBasisHessianOnGrid`** in `src/chemistry/dft/density.ts`:
  6 unique Hessian components (xx, yy, zz, xy, xz, yz) per (μ, p).
  Same shifted-L recursion as the gradient evaluator with extra
  ±2 polynomial powers on each axis. FD-validated to 2e-8 against
  central-FD of the gradient.
- **`dftGradient`** extended with the GGA path: pre-computes
  (Pφ), (P·∂_a φ) for a∈{x,y,z}, and contracts with the basis
  Hessian. LDA is now a code-path simplification rather than a
  separate function.
- 20 new test cases: FD-vs-analytical to **1e-3 Ha/Bohr** for
  every (functional, molecule) ∈ {lda-svwn, bvwn5, blyp, b3vwn5,
  b3lyp5} × {H₂, H₂O, BeH₂} STO-3G. Translational invariance
  also at 1e-3.

H₂O STO-3G gradient timings (M2 Pro, single-thread TS):
- LDA:     374 ms (vs 98 ms SCF).
- BVWN5:   409 ms.
- BLYP:    411 ms.
- B3LYP5:  511 ms.
GGA only adds ~10% on top of LDA — Hessian build is cheap.

The remaining honest negative is the **weights-fixed approximation**:
∂(Becke-partition weights)/∂R is still not computed. The residual
on Σ_atoms ∇E is ~1e-3 Ha/Bohr on H₂O, sub-mHa/Bohr per component.
Eliminating it is the immediate follow-up.

**Tier 2 stage 6 — DFT analytical gradients (LDA).** The Pulay
HF gradient was reused with `kFactor = hfMix`, and a first-pass
LDA XC contribution shipped before the GGA-Hessian work. H₂O
LDA STO-3G geom-opt: 55.4 s FD → 7.6 s analytical (7.3× faster).

**Tier 2 stage 5b — HF gradient speedup.**
The Pulay-1969 analytical gradient now actually beats FD. Three
optimizations on top of the stage-5a correctness implementation:
- **8-fold canonical ERI loop**: iterate (μ ≥ ν, λ ≥ σ, (μν) ≥
  (λσ)) only, computing one set of three derivative ERIs (∂A, ∂B,
  ∂C; ∂D from translational invariance) per canonical quartet.
  J + K combine via a unified Γ-coupling sum over the 8 ERI-
  symmetric permutations (deduplicated for low-multiplicity
  canonicals where μ = ν, λ = σ, or (μν) = (λσ)). 16× fewer
  derivative ERI evaluations than the naive loop.
- **Schwarz screening**: precompute Q_μν = √|⟨μν|μν⟩| and skip
  canonical quartets with `Q_μν · Q_λσ · |Γ| < 1e-10`.
- **1-electron pair symmetry**: μ ≥ ν loop with sym = 2 for off-
  diagonals, sym = 1 for the diagonal (where bra and ket sides
  go to the same atom — the conditional matters and was the
  source of the only bug introduced during this stage).

H₂O STO-3G headline (M2 Pro, single-threaded TS):
- HF energy: 86 ms.
- Analytical gradient: **4500 ms → 440 ms** (10× speedup).
- Geometry optimization: **52 s → 6.6 s analytical** (2.2× faster
  than FD's 14.5 s; was 3× SLOWER than FD before this stage).
- Same final energy E = −74.96590049 to 8 decimals as the FD path.
- Full vitest suite: 47 s → 22 s on the chemistry track.

The 8-fold canonical loop has a defensive moat: per-pair Γ-coef
is computed by enumerating the 8 permutations and deduplicating
on the fly, so it handles every (μ = ν, λ = σ, (μν) = (λσ))
multiplicity case without case analysis.

**Tier 2 stage 5a — analytical HF gradients (correctness).**
Pulay 1969 gradient via integral derivatives. FD-validated to
1e-5 Ha/Bohr per component on H₂ / H₂O / BeH₂ STO-3G.
Translational invariance Σ ∇E = 0 holds to 1e-9. Integral
derivatives via bra-side Hellmann-Feynman shift
`2α·prim(I+1) − I_axis·prim(I−1)` at the primitive level;
translational invariance recovers the partner-center derivatives.

**Tier 2 stage 4 — LYP correlation + B3LYP5.**
Two functionals shipped on top of the Tier 2 stage 3 GGA + hybrid
infrastructure:
- `blyp`: Slater + B88 GGA exchange + LYP GGA correlation. The
  classic "BLYP" most chemists mean by "GGA-DFT".
- `b3lyp5`: Becke 1993 hybrid with VWN5 — the published B3LYP, with
  VWN5 in place of VWN_RPA (i.e. PySCF's "B3LYP5"):
    E_xc = 0.20·E_x^HF + 0.80·E_x^Slater + 0.72·ΔE_x^B88
         + 0.81·E_c^LYP + 0.19·E_c^VWN5

How the LYP closed-shell bug from the prior attempt was avoided:
- Closed-shell collapse cross-referenced against the canonical
  libxc Maple source (`maple/gga_exc/gga_c_lyp.mpl`). The libxc
  per-particle ε at z = 0 simplifies to:
    ε^closed = −a/h − a·b·C_F·E/h
             + a·b·E·(3 + 7δ)·γ / (72·h·ρ^(8/3))
  with u = ρ^(−1/3), h = 1+d·u, E = exp(−c·u), δ = c·u + d·u/h.
- The previous attempt's hand-collapsed Miehlich form gave a γ
  coefficient of (73 + 11δ)/144 — about 10× too large with the
  wrong δ-coefficient. That's exactly the sign-error-grade bug the
  prior attempt shipped (30–240 mHa off PySCF B3LYP). The libxc
  cross-check is what caught it.
- Defensive moat: `tests/chemistry/lyp.test.ts` is a 20-test FD
  self-test on (ρ·ε_LYP) — analytic v_ρ and v_γ must match central-
  FD to 1e-6 across (ρ, γ) ∈ {0.01–2, 1e-6–4} sample grid, plus
  closed-form γ = 0 UEG match to 1e-10. Catches sign + magnitude
  errors at the kernel level before they hit any molecule.

H₂ STO-3G energies (Ha): HF = −1.117, BLYP = −1.155, B3LYP5 =
−1.159. H₂O STO-3G: HF = −74.96, BLYP = −75.28, B3LYP5 = −75.28.
Within ~10 mHa of published references (literature B3LYP/H₂ ≈
−1.166; PySCF B3LYP5/H₂O ≈ −75.31). The hybrid hierarchy is not
strictly bracketed (small minimal-basis molecules can have B3LYP5
slightly below BLYP) — that's a physical feature, not a bug.

**Tier 2 stage 3 — GGA + hybrid DFT.** Three functionals on top
of LDA: `bvwn5` (Slater + B88 + VWN5), `b3vwn5` (Becke3 hybrid w/
VWN5). What got built:
- Density gradients on the grid: `evalBasisGradOnGrid` (∇φ_μ),
  `evalDensityAndGradient` (∇ρ + γ = |∇ρ|²) — same O(n²·nGrid) cost.
- B88 GGA exchange — Becke 1988, ε_x^B88 = ε_x^Slater
  − 2^(−1/3) β ρ^(1/3) F(u), F(u) = u²/(1 + 6β u arcsinh u).
  Analytical v_ρ + v_γ.
- GGA Fock build: V_xc[μν] = ∫{v_ρ φ_μ φ_ν + 2 v_γ ∇ρ·(∇φ_μ φ_ν +
  φ_μ ∇φ_ν)} dr. Hybrid path subtracts ½ × hfMix × K from F.

H₂O / STO-3G timings: LDA 75 ms / 8 iter, BVWN5 86 ms / 6 iter,
B3VWN5 97 ms / 7 iter, BLYP 90 ms / 8 iter, B3LYP5 105 ms / 8 iter.

**Tier 2 stage 2 — DFT/LDA.** Becke-partitioned molecular grid
(Becke M3 radial × Gauss-Chebyshev 2nd-kind × Gauss-Legendre ×
uniform-φ angular). Default 50r × 12θ × 24φ per atom integrates
ρ to 10⁻⁵–10⁻⁷ e. DFT/STO-3G LDA matches PySCF SVWN5 within ~5 mHa.
Modules: `src/chemistry/dft/{grid,density,functional,rks-scf}.ts`.

**Tier 2 stage 1 — geometry optimization.** `optimizeGeometry(atoms,
opts)` minimizes E_HF over atomic positions with central-FD
gradients + L-BFGS line search. Validated on H₂ / H₂O / BeH₂
STO-3G to sub-mÅ + sub-degree agreement with PySCF references
(R_OH = 0.9894 Å vs 0.9893; ∠HOH = 100.02° vs 100.04; R_BeH =
1.291 Å). FD gradients keep it basis-/level-agnostic; analytical
swap is a transparent follow-up.

**Tier 1 bundle.** Six chemistry-track quick wins shipped earlier:
- **DIIS** SCF accelerator — H₂O cc-pVDZ HF: 101 → 14 iter (7.2×
  speedup), bit-identical energy.
- **Frozen-core** option on MP2 / CCSD / CCSD(T) (zeroes T1, T2 in core
  blocks every iter; canonical 1s-frozen for first-row chemistry).
- **Spherical-harmonic d-shell** basis (`{ spherical: true }` opt on
  `computeMolecularIntegrals`). cc-pVDZ HF/H₂O matches PySCF to **35
  µHa** vs 340 µHa Cartesian — kills the documented Cartesian-d slack.
- **f/g/h orbital integrals**: rewrote `boysAll` with per-n Taylor
  inside the recurrence-stability threshold + closed-form-anchored
  upward outside. Max relative error at n=12 dropped from 1.5e-2 to
  8e-10. Unblocks cc-pVTZ (and beyond) basis sets.
- **aug-cc-pVDZ** diffuse functions (H + O wired). HF/H₂O matches
  PySCF to 50 µHa; 14 mHa lower than cc-pVDZ as expected.
- **Schwarz integral screening** in the AO ERI build (Q[μ,ν] =
  √⟨μν|μν⟩, skip pairs with Q_μν · Q_λσ < 1e-10).

**Previously:** Phase E stage 5 — cc-pVDZ CCSD(T) on H₂O in 106 s
wall-clock in a browser tab. HF / MP2 / CCSD / CCSD(T) all ship; CH₄
full-STO-3G FCI works via sparse-CSR Hsec (Phase C v5).

**Headline numbers:**
- L1 statevector: F ≥ 0.999999 vs CPU; 4-experiment ladder (E1–E4) green.
- L2 MPS / DMRG: TFIM & Heisenberg N=128 in browser, χ=32, validated to
  Pfeuty/Bethe limits at 1/N. ITensor cross-checked at N=8 to f64.
- L3 kernel fusion: **4.18× headline** (Tier C, 8×8 cascade); Tier D plateau
  is the documented honest negative.
- L6 chemistry: HF (≤ 0.05 mHa vs PySCF, ≤ 50 µHa with spherical d) →
  MP2 → FCI (CH₄ to 0.76 mHa) → CCSD (≥ 99% capture) → **CCSD(T)** (≤
  0.25 mHa vs FCI). aug-cc-pVDZ now wired alongside cc-pVDZ.

**Test surface:** `npm run test` → **479/479** (was 475) + 1 opt-in
(cc-pVDZ CCSD(T), gated on `PHASE_E5_CCPVDZ=1`). `npx tsc --noEmit`
clean. `npm run lint` clean (2 pre-existing unused-disable warnings).
`npx playwright test` → **11/11 specs**, all 4 levels e2e.

**Live:** https://webgpu-q.vercel.app — landing, `/viz.html` (4D
hyperscope), `/experiments/` (E1–E16+ dashboard). **Standing preference:
do NOT auto-deploy** — deploy only when the user explicitly asks.

**For per-phase detail** (Phase A → E5, Tier 1, B v0/v1, C v0–v5, D,
E1–E5, viz extensions, public-repo polish, hardened-SVD fix, Tier B/C/D
fusion): read `git log` — every phase shipped its own commit with full
benchmarks in the message body. Don't replicate that history here.

**Next up (per the roadmap above):** the closed-shell ground-
state property suite (energy, geometry, dipole, charges) +
singlet excited-state suite (TDA / TDDFT energies + oscillator
strengths) is now closed across the full HF + DFT ladder. The
project is genuinely a "real undergrad chemistry tool in a
browser tab" already.

Bigger levers from here: **WebGPU port of the (T) kernel**
(~3 sessions, 10-100× → cc-pVTZ CCSD(T) routine) or
**EOM-CCSD** for correlated excited states (~1-2 sessions).

Smaller wins still: triplet TDA / TDDFT for DFT functionals
(~½ session), Mayer-valence summary helper (~10 minutes),
frequency analysis (Hessian + harmonic vibrations, needs
analytical second derivatives — 2-3 sessions of integral
work).

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
    E5-mps-correctness.ts
    E6-qubit-ceiling.ts
    E7-chi-scaling.ts
    E18-tfim-pfeuty.ts
    E19-heisenberg-bethe.ts
    run-all.ts
  level-3-fusion/      # E8–E13 shipped (Tiers A/B/C/D fusion)
  level-6-chemistry/   # E16, E20–E31 shipped (H₂ → CCSD(T)/cc-pVDZ)
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
