# E34 — Wall-clock + energy comparison vs PySCF

Apples-to-apples on identical geometries, identical basis,
identical convergence thresholds (SCF 1e-10, CCSD 1e-10).
STO-3G Li in PySCF uses the matched s-only basis (1s + 2s, no 2p)
to match webgpu-q's current scope; see LIMITATIONS.md.

- webgpu-q artifact: `experiments/results/2026-05-12/level-6/E34-wallclock-vs-pyscf.json`
- PySCF artifact:    `experiments/results/2026-05-12/level-6/E34-pyscf.json`

| molecule | basis | method | webgpu-q (s) | PySCF (s) | speedup | E_wgpu (Ha) | E_pyscf (Ha) | \|ΔE\| (Ha) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| H₂ | sto-3g | HF | 300 µs | 31.64 ms | **105.46× faster** | -1.11668429 | -1.11668439 | 9.95e-8 |
| H₂ | sto-3g | MP2 | 100 µs | 1.04 ms | **10.38× faster** | -1.12985506 | -1.12985515 | 9.17e-8 |
| H₂ | sto-3g | CCSD | 3.20 ms | 27.83 ms | **8.70× faster** | -1.13727009 | -1.13727017 | 8.11e-8 |
| H₂ | sto-3g | CCSD(T)-CPU | — | 2.67 ms | — | -1.13727009 | -1.13727017 | 8.11e-8 |
| H₂ | sto-3g | CCSD(T)-GPU | 100 µs | — | — | -1.13727009 | — | — |
| H₂ | cc-pvdz | HF | 2.50 ms | 14.74 ms | **5.90× faster** | -1.12872007 | -1.12871496 | 5.11e-6 |
| H₂ | cc-pvdz | MP2 | 7.10 ms | 156 µs | 45.54× slower | -1.15510328 | -1.15509920 | 4.08e-6 |
| H₂ | cc-pvdz | CCSD | 89.60 ms | 32.91 ms | 2.72× slower | -1.16341770 | -1.16341393 | 3.77e-6 |
| H₂ | cc-pvdz | CCSD(T)-CPU | — | 3.29 ms | — | -1.16341770 | -1.16341393 | 3.77e-6 |
| H₂ | cc-pvdz | CCSD(T)-GPU | — | — | — | -1.16341770 | — | — |
| LiH | sto-3g | HF | 200 µs | 19.54 ms | **97.69× faster** | -7.80424256 | -7.80424263 | 7.59e-8 |
| LiH | sto-3g | MP2 | — | 175 µs | — | -7.82615767 | -7.82615773 | 6.67e-8 |
| LiH | sto-3g | CCSD | 800 µs | 31.69 ms | **39.61× faster** | -7.84339419 | -7.84339425 | 5.81e-8 |
| LiH | sto-3g | CCSD(T)-CPU | — | 1.47 ms | — | -7.84339419 | -7.84339425 | 5.81e-8 |
| LiH | sto-3g | CCSD(T)-GPU | — | — | — | -7.84339419 | — | — |
| LiH | cc-pvdz | HF | — | 26.75 ms | — | — | -7.98361587 | — |
| LiH | cc-pvdz | MP2 | — | 311 µs | — | — | -8.00643084 | — |
| LiH | cc-pvdz | CCSD | — | 76.40 ms | — | — | -8.01471720 | — |
| LiH | cc-pvdz | CCSD(T)-CPU | — | — | — | — | — | — |
| LiH | cc-pvdz | CCSD(T)-GPU | — | — | — | — | — | — |
| BeH₂ | sto-3g | HF | 400 µs | 18.10 ms | **45.25× faster** | -15.55940528 | -15.55940541 | 1.32e-7 |
| BeH₂ | sto-3g | MP2 | 100 µs | 188 µs | **1.88× faster** | -15.58289328 | -15.58289340 | 1.26e-7 |
| BeH₂ | sto-3g | CCSD | 74.30 ms | 77.31 ms | **1.04× faster** | -15.59445615 | -15.59445628 | 1.35e-7 |
| BeH₂ | sto-3g | CCSD(T)-CPU | 41.70 ms | 2.01 ms | 20.72× slower | -15.59463607 | -15.59467150 | 3.54e-5 |
| BeH₂ | sto-3g | CCSD(T)-GPU | 32.70 ms | — | — | -15.59463607 | — | — |
| BeH₂ | cc-pvdz | HF | — | 28.40 ms | — | — | -15.76736639 | — |
| BeH₂ | cc-pvdz | MP2 | — | 545 µs | — | — | -15.81911995 | — |
| BeH₂ | cc-pvdz | CCSD | — | 58.29 ms | — | — | -15.83582320 | — |
| BeH₂ | cc-pvdz | CCSD(T)-CPU | — | — | — | — | — | — |
| BeH₂ | cc-pvdz | CCSD(T)-GPU | — | — | — | — | — | — |
| H₂O | sto-3g | HF | 1.60 ms | 18.94 ms | **11.84× faster** | -74.96292755 | -74.96292795 | 3.97e-7 |
| H₂O | sto-3g | MP2 | 2.30 ms | 176 µs | 13.10× slower | -74.99842020 | -74.99842043 | 2.31e-7 |
| H₂O | sto-3g | CCSD | 74.10 ms | 29.86 ms | 2.48× slower | -75.01228662 | -75.01228678 | 1.59e-7 |
| H₂O | sto-3g | CCSD(T)-CPU | 76.70 ms | 1.60 ms | 47.79× slower | -75.01245414 | -75.01235415 | 1.00e-4 |
| H₂O | sto-3g | CCSD(T)-GPU | 7.10 ms | — | — | -75.01245414 | — | — |
| H₂O | cc-pvdz | HF | 25.10 ms | 27.43 ms | **1.09× faster** | -76.02680049 | -76.02679877 | 1.72e-6 |
| H₂O | cc-pvdz | MP2 | 95.50 ms | 700 µs | 136.41× slower | -76.23076007 | -76.23075858 | 1.49e-6 |
| H₂O | cc-pvdz | CCSD | 41.642 s | 86.75 ms | 480.02× slower | -76.24008378 | -76.24008248 | 1.30e-6 |
| H₂O | cc-pvdz | CCSD(T)-CPU | — | — | — | — | — | — |
| H₂O | cc-pvdz | CCSD(T)-GPU | 4.228 s | — | — | -76.24399176 | — | — |

## Summary

- **Energy agreement**: 19 cells compared, max |ΔE| = 1.00e-4 Ha, mean |ΔE| = 8.13e-6 Ha.
- **Wall-clock**: webgpu-q is faster on 11 / 19 comparable cells. Speedups range from 0.00× (slowest of us vs PySCF) to 105.46× (best of us vs PySCF).
- **PySCF on CPU only**: no gpu4pyscf run here. CCSD(T)-GPU comparison left for a follow-up with gpu4pyscf installed.
