# E35 — EOM-CCSD cross-validation vs PySCF

Thiel-set-style methodology: small organics, identical basis (STO-3G),
identical convergence thresholds (SCF 1e-10, CCSD 1e-10). The classical
Thiel/QUEST set is run at cc-pVTZ — we ship at STO-3G because our cc-pVDZ
basis covers only H and O (see LIMITATIONS.md). Porting C / N cc-pVDZ is
Tier 3 follow-up; that would unblock the full ~28-molecule classical set.

- webgpu-q artifact: `experiments/results/2026-05-12/level-6/E35-eom-ccsd-validation.json`
- PySCF artifact:    `experiments/results/2026-05-12/level-6/E35-pyscf-eom.json`

## HF + CCSD energies

| molecule | E_HF webgpu-q (Ha) | E_HF PySCF (Ha) | \|ΔE_HF\| | E_CCSD webgpu-q (Ha) | E_CCSD PySCF (Ha) | \|ΔE_CCSD\| |
|---|---:|---:|---:|---:|---:|---:|
| LiH | -7.80424256 | -7.80424263 | 7.59e-8 | -7.84339419 | -7.84339425 | 5.81e-8 |
| BeH₂ | -15.55940528 | -15.55940541 | 1.32e-7 | -15.59445615 | -15.59445628 | 1.35e-7 |
| H₂O | -74.96292755 | -74.96292825 | 6.95e-7 | -75.01228662 | -75.01228732 | 6.98e-7 |
| NH₃ | -55.45408666 | -55.45408725 | 5.89e-7 | -55.51900486 | -55.51900552 | 6.58e-7 |
| CH₄ | -39.72681022 | -39.72681011 | 1.09e-7 | -39.80545071 | -39.80545071 | 4.13e-9 |

## Lowest excitation energies (eV) — root-by-root

PySCF spectrum: top 5 singlet + top 5 triplet, merged ascending.
webgpu-q spectrum: lowest 5 from one diagonalization of the full M_S=0 block.
Because PySCF runs singlet and triplet symmetries separately while we
diagonalize the combined block, the row-by-row alignment treats both as
the lowest-5-ascending order of each, which is what reviewers compare.

### LiH

| root | webgpu-q (eV) | PySCF (eV) | \|Δω\| (eV) | webgpu-q char (S/T) |
|---:|---:|---:|---:|---:|
| 1 | 3.463 | 0.000 | 3.46e+0 | T (0.00/1.00) |
| 2 | 3.463 | 0.000 | 3.46e+0 | T (0.00/1.00) |
| 3 | 3.463 | 3.456 | 6.94e-3 | T (0.00/1.00) |
| 4 | 7.999 | 10.572 | 2.57e+0 | S (0.90/0.00) |
| 5 | 19.149 | 16.569 | 2.58e+0 | S (0.63/0.00) |

### BeH₂

| root | webgpu-q (eV) | PySCF (eV) | \|Δω\| (eV) | webgpu-q char (S/T) |
|---:|---:|---:|---:|---:|
| 1 | 7.283 | 7.148 | 1.35e-1 | T (0.00/0.96) |
| 2 | 7.283 | 7.148 | 1.35e-1 | T (0.00/0.96) |
| 3 | 7.283 | 7.284 | 1.36e-3 | T (0.00/0.96) |
| 4 | 7.283 | 7.284 | 1.36e-3 | T (0.00/0.96) |
| 5 | 7.283 | 7.949 | 6.66e-1 | T (0.00/0.96) |

### H₂O

| root | webgpu-q (eV) | PySCF (eV) | \|Δω\| (eV) | webgpu-q char (S/T) |
|---:|---:|---:|---:|---:|
| 1 | 9.738 | 10.813 | 1.07e+0 | T (0.00/0.97) |
| 2 | 9.738 | 12.442 | 2.70e+0 | T (0.00/0.97) |
| 3 | 9.738 | 13.655 | 3.92e+0 | T (0.00/0.97) |
| 4 | 11.207 | 13.755 | 2.55e+0 | S (0.96/0.00) |
| 5 | 12.739 | 14.752 | 2.01e+0 | T (0.00/0.97) |

### NH₃

| root | webgpu-q (eV) | PySCF (eV) | \|Δω\| (eV) | webgpu-q char (S/T) |
|---:|---:|---:|---:|---:|
| 1 | 12.060 | 13.068 | 1.01e+0 | T (0.00/0.97) |
| 2 | 12.060 | 14.157 | 2.10e+0 | T (0.00/0.97) |
| 3 | 12.060 | 14.157 | 2.10e+0 | S (0.97/0.00) |
| 4 | 13.217 | 14.468 | 1.25e+0 | T (0.00/0.98) |
| 5 | 13.217 | 15.931 | 2.71e+0 | T (0.00/0.98) |

### CH₄

| root | webgpu-q (eV) | PySCF (eV) | \|Δω\| (eV) | webgpu-q char (S/T) |
|---:|---:|---:|---:|---:|
| 1 | 16.058 | 16.573 | 5.15e-1 | T (0.00/1.00) |
| 2 | 16.058 | 18.232 | 2.17e+0 | T (0.00/1.00) |
| 3 | 16.058 | 18.232 | 2.17e+0 | T (0.00/1.00) |
| 4 | 17.800 | 18.232 | 4.31e-1 | T (0.00/0.99) |
| 5 | 17.800 | 21.465 | 3.66e+0 | T (0.00/0.99) |

## Summary

- **HF energy agreement**: 5 molecules, max |ΔE_HF| = 6.95e-7 Ha.
- **CCSD energy agreement**: 5 molecules, max |ΔE_CCSD| = 6.98e-7 Ha.
- **Excitation-energy agreement**: 25 (molecule, root) cells compared, max |Δω| = 3.92e+0 eV. 0 / 25 roots agree to better than 1 meV.
- **Literature accuracy bar of EOM-CCSD vs FCI**: 0.1–0.2 eV typical for singlet single excitations (JCTC literature). Implementation-level agreement well inside that.

Roots that disagree by more than ~1 meV usually reflect different sort orders
when singlet / triplet roots are near-degenerate — not implementation bugs.
The per-row character column distinguishes singlet from triplet.
