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
  from webgpu-q-accelerated CI calculations agree with the tabulated
  Geant4-DNA values (G4EMLOW 8.8, Emfietzoglou + Born) within ±15%.
- **Method:** Five energies × two channels. Compare to
  `sigma_ionisation_e_emfietzoglou` and `sigma_excitation_e_born`
  tables.
- **Pass bar:** |σ_ours − σ_g4| / σ_g4 ≤ 0.15 in each cell.
- **Cross-link:** Validation script lives in
  `webgpu-dna/validation/compare.py` (already used for CSDA + G-values).

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-6/E{16,17}-*.json`

E17 artifacts include a one-row-per-energy-channel table plus a
provenance block pointing at the exact G4EMLOW tarball checksum used
on the Geant4 side.
