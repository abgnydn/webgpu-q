# Level 6 — Chemistry → webgpu-dna

## Thesis fragment
> VQE on small molecules (H₂, HeH⁺, LiH) converges to FCI within
> chemical accuracy; track-structure cross sections agree with Geant4.

## Why this level
This level closes the loop with the sibling project `webgpu-dna`
(Geant4-DNA ported to WebGPU). A simulator that can drive VQE at
textbook-accuracy, and whose downstream radiobiology output matches
Geant4 cross sections, is the capstone demonstration.

## Status
Protocol only. Cross-linked to `webgpu-dna`:
https://github.com/abgnydn/webgpu-dna (sibling repo in this workspace).

## Baselines

- **Full configuration interaction (FCI)** — exact ground-state energy
  in a given basis. Reported by PySCF / Psi4.
- **Hartree–Fock** — upper bound; VQE must beat HF by the correlation
  energy.
- **Geant4-DNA (G4EMLOW 8.8)** — cross-section and G-value ground truth,
  already used in `webgpu-dna` validation.

## Experiments

### E16 — VQE ground-state energy, UCCSD ansatz
- **Hypothesis:** For H₂ at bond lengths R ∈ {0.5, 0.74, 1.0, 1.5, 2.5} Å
  in STO-3G, VQE with hardware-efficient UCCSD ansatz converges to
  chemical accuracy (|ΔE| ≤ 1.6 mHa = 1 kcal/mol) vs FCI.
- **Method:** 10 VQE runs per R with different random classical
  initialisations. Reference energies from PySCF FCI. Record final
  |ΔE|, wall-clock, shot count, optimizer iterations.
- **Pass bar:** median |ΔE| ≤ 1.6 mHa AND max |ΔE| ≤ 5 mHa across the
  scan.
- **Secondary:** Reports the full bond-dissociation curve.

### E17 — Cross-section comparison (Geant4 ↔ webgpu-q-derived)
- **Hypothesis:** For electron-on-water ionization and excitation at
  E ∈ {100 eV, 300 eV, 1 keV, 3 keV, 10 keV}, cross sections derived
  from a Bethe-Born sum over the TDA-B3LYP5 oscillator-strength
  spectrum agree with Itikawa-Mason 2005 reference values (the
  upstream source for Geant4-DNA's G4EMLOW 8.8 tables) within ±15%.
- **Method:** Five energies × two channels (σ_ion, σ_exc). Run
  RKS-DFT/B3LYP5 + TDA on H₂O at the experimental geometry, then
  σ_inel(T) = (4π/T)·Σ_n(f_n/ω_n)·ln(αT/ω_n) with α = 4. Partition
  by I_p(H₂O) = 12.62 eV.
- **Pass bar:** |σ_ours − σ_ref| / σ_ref ≤ 0.15 in each cell.
- **Status (shipped 2026-05-07):** **FAIL — honest negative.**
  TRK sum rule satisfied at cc-pVDZ (Σf ≈ 10.77 vs N_e = 10) but the
  discrete TDA spectrum piles oscillator strength near I_p, and
  Bethe-Born's (f/ω)·ln(αT/ω) weight amplifies low-ω contributions,
  so σ_ion OVERSHOOTS by 2-6× (STO-3G to cc-pVDZ). σ_exc undershoots
  in STO-3G (1 state below I_p) and overshoots in cc-pVDZ (4 states).
  Resolution requires genuine continuum representation — Stieltjes
  imaging, SAC-CI in a continuum basis, B-spline / DVR continuum
  orbitals, or direct optical-photoionization cross sections.
  Substantial new infrastructure; deferred.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-6/E{16,17}-*.json`

E17 artifacts include a one-row-per-energy-channel table plus a
provenance block pointing at the exact G4EMLOW tarball checksum used
on the Geant4 side.
